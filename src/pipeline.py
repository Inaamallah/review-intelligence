"""
Review Intelligence Pipeline
=============================
End-to-end NLP pipeline for customer review analysis.
Takes any CSV of product reviews and produces a stakeholder dashboard
showing which product features are most liked and most disliked.

Usage:
    from src.pipeline import ReviewIntelligencePipeline

    pipeline = ReviewIntelligencePipeline()
    pipeline.run(
        csv_path="your_reviews.csv",
        text_column="review_text",
        output_dir="outputs/"
    )

What gets saved to output_dir after run() completes:
    ├── reviews_analyzed.csv          <- per-review aspect + sentiment data
    ├── aspect_summary.csv            <- aggregated satisfaction scores
    ├── stakeholder_dashboard.html    <- open this in any browser
    └── stakeholder_report.txt        <- plain-English written report
"""

import pandas as pd
import numpy as np
import re
import os
import sys
import ast
import nltk
from dotenv import load_dotenv

# Fix Windows encoding — cp1252 cannot handle emoji characters (✅, ⚠️, etc.)
# This must happen before any print() call that uses Unicode symbols.
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Load API keys from .env file (keeps secrets out of source code)
load_dotenv()
from collections import defaultdict, Counter
from transformers import pipeline as hf_pipeline

# Download required NLTK data silently on first run
for pkg in ['punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger_eng']:
    nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer


# ── SECTION 1: TEXT PREPROCESSING ──────────────────────────────────────────
# Two separate preprocessing pipelines exist because different downstream
# consumers need different things from the same raw text.
# Transformer models (BERT) need natural English with punctuation and
# stopwords preserved — they were trained on complete sentences.
# Statistical tools (word clouds, frequency counters) need stopwords
# removed and words lemmatized so counting is clean and meaningful.

STOP_WORDS  = set(stopwords.words('english'))
LEMMATIZER  = WordNetLemmatizer()
AUXILIARY_VERBS = {
    'is','are','was','were','am','been','being','do','does','did',
    'have','has','had','will','would','could','should','might',
    'may','shall','must','can','need','dare'
}

def remove_html(text):
    """Remove HTML tags (<br/>) and HTML entities (&amp;) from review text.
    These are computer encoding artifacts that were never real language
    and would confuse any downstream model."""
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'&\w+;', ' ', text)
    return text

def expand_contractions(text):
    """Expand contractions before lowercasing — "can't" becomes "cannot".
    This must happen before lowercasing because the contractions library
    works more reliably on naturally cased text."""
    try:
        import contractions
        return contractions.fix(text)
    except ImportError:
        # Fallback manual patterns if the library is not installed
        patterns = [
            (r"can\'t", "cannot"), (r"won\'t", "will not"),
            (r"n\'t",   " not"),   (r"\'re",  " are"),
            (r"\'ve",   " have"),  (r"\'ll",  " will"),
            (r"\'d",    " would"), (r"\'m",   " am")
        ]
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text)
        return text

def preprocess_for_transformers(text):
    """Minimal preprocessing for BERT-based models.

    Philosophy: only remove what was never real language.
    Keep punctuation (! and ? are sentiment signals for BERT),
    keep stopwords (BERT needs 'not' and 'but' for grammar),
    keep natural sentence structure (BERT was trained on complete sentences).
    """
    if pd.isna(text) or text == '':
        return ''
    text = str(text)
    text = remove_html(text)           # strips <br/>, &amp; etc.
    text = expand_contractions(text)   # can't → cannot
    text = text.lower()                # normalize casing
    text = ' '.join(text.split())      # collapse extra whitespace
    return text

def get_wordnet_pos(tag):
    """Translate NLTK Treebank POS tags to WordNet format.
    This translation is necessary because the lemmatizer uses a different
    tag system than the POS tagger — without this, all words default to
    noun form, causing 'draining' to stay as 'draining' instead of 'drain'."""
    if tag.startswith('V'): return wordnet.VERB
    elif tag.startswith('N'): return wordnet.NOUN
    elif tag.startswith('J'): return wordnet.ADJ
    elif tag.startswith('R'): return wordnet.ADV
    else: return wordnet.NOUN

def lemmatize_with_pos(text):
    """POS-aware lemmatization with auxiliary verb protection.
    Without POS context, 'is' becomes 'be' and 'was' becomes 'be' because
    the lemmatizer wrongly treats them as regular verbs. We protect all
    auxiliary verbs by skipping lemmatization for them entirely."""
    words    = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    result   = []
    for word, tag in pos_tags:
        # Auxiliary verbs and very short words keep their original form
        if word.lower() in AUXILIARY_VERBS or len(word) <= 2:
            result.append(word)
        else:
            result.append(LEMMATIZER.lemmatize(word, get_wordnet_pos(tag)))
    return ' '.join(result)

def preprocess_for_statistics(text):
    """Aggressive preprocessing for word counting and visualization.

    Philosophy: reduce text to its meaningful keywords only.
    Remove punctuation (so 'battery.' and 'battery' count as one word),
    remove stopwords (so 'the' doesn't dominate frequency charts),
    lemmatize (so 'batteries' and 'battery' count together).
    """
    if pd.isna(text) or text == '':
        return ''
    text = str(text)
    text = remove_html(text)
    text = expand_contractions(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)   # remove punctuation
    text = lemmatize_with_pos(text)                  # drain/draining → drain
    text = ' '.join([w for w in text.split() if w not in STOP_WORDS])
    text = ' '.join(text.split())
    return text


# ── SECTION 2: ASPECT EXTRACTION ───────────────────────────────────────────
# Zero-shot classification is used here because it requires no labeled
# training data and adapts to any product domain by changing only the
# taxonomy dictionary. The model reads the aspect description and decides
# whether the review semantically entails that description.

DEFAULT_TAXONOMY = {
    "product quality":    "product build and material quality",
    "value for money":    "pricing and value for money",
    "functionality":      "battery life, performance, and functionality",
    "customer experience":"shipping, delivery, packaging, and customer service",
    "content quality":    "quality of story, writing, film, or music",
    "design and aesthetics": "visual design and aesthetics"
}

def load_aspect_classifier():
    """Load the zero-shot NLI model for aspect detection.
    cross-encoder/nli-MiniLM2-L6-H768 is chosen because it is lightweight
    enough for Colab's free GPU while producing reliable confidence scores."""
    import torch
    device = 0 if torch.cuda.is_available() else -1
    return hf_pipeline(
        "zero-shot-classification",
        model="cross-encoder/nli-MiniLM2-L6-H768",
        device=device
    )

def extract_aspects(text, classifier, taxonomy=None, threshold=0.50):
    """Detect which product aspects a review discusses.

    multi_label=True is critical — it allows one review to match multiple
    aspects simultaneously (e.g. battery AND sound quality both mentioned).
    Only aspects where the model is at least 'threshold' confident are kept;
    everything below is treated as 'not discussed in this review'.
    """
    if taxonomy is None:
        taxonomy = DEFAULT_TAXONOMY
    if not text or len(str(text).split()) < 4:
        return {}

    aspect_descriptions = list(taxonomy.values())
    reverse_map         = {v: k for k, v in taxonomy.items()}

    try:
        result = classifier(
            str(text)[:512],
            candidate_labels=aspect_descriptions,
            multi_label=True,
            hypothesis_template="This review discusses {}."
        )
        detected = {}
        for label, score in zip(result['labels'], result['scores']):
            if score >= threshold:
                detected[reverse_map[label]] = round(score, 3)
        return detected
    except Exception:
        return {}


# ── SECTION 3: SENTIMENT CLASSIFICATION ────────────────────────────────────
# We classify sentiment at the aspect level, not the whole-review level.
# A review saying "sound quality is incredible but battery is terrible"
# is mixed overall — whole-review sentiment would be meaningless.
# Instead, we extract the most relevant sentence for each aspect and
# classify sentiment on that focused context only.

ASPECT_KEYWORDS = {
    "product quality":    ["quality","build","material","durable","broke","cheap","sturdy"],
    "value for money":    ["price","money","worth","value","expensive","cost","overpriced"],
    "functionality":      ["work","function","perform","battery","charge","connect","feature"],
    "customer experience":["ship","deliver","package","arrived","late","service","return"],
    "content quality":    ["story","writing","plot","character","book","film","music","song"],
    "design and aesthetics":["look","design","color","appearance","beautiful","style"]
}

def get_relevant_context(text, aspect):
    """Find the sentence in a review most likely to discuss a given aspect.

    We score each sentence by how many aspect keywords it contains and
    return the highest-scoring one. If no sentence contains any keywords,
    we fall back to the full review text so the sentiment model always
    gets something meaningful to classify rather than an empty string.
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if not sentences:
        return text[:512]
    keywords = ASPECT_KEYWORDS.get(aspect, [])
    scored   = [(sum(1 for kw in keywords if kw in s.lower()), s) for s in sentences]
    scored.sort(reverse=True)
    # If best score is zero, no sentence matched any keyword — use full text
    return scored[0][1] if scored[0][0] > 0 else text[:512]

def load_sentiment_classifier():
    """Load the sentiment classification model.
    distilbert-base-uncased-finetuned-sst-2-english is fine-tuned on consumer
    reviews — exactly our domain — and is lightweight enough for free Colab GPU."""
    import torch
    device = 0 if torch.cuda.is_available() else -1
    return hf_pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device, truncation=True, max_length=512
    )

def classify_sentiments(text, detected_aspects, sentiment_model):
    """Classify sentiment for each detected aspect using contextual windows.

    If the model's confidence is below 0.65, we label it NEUTRAL rather
    than forcing a binary positive/negative decision. A low-confidence
    prediction on a genuinely ambiguous sentence is better labeled as
    uncertain than forced into a category that could mislead stakeholders.
    """
    if not detected_aspects:
        return {}
    results = {}
    for aspect in detected_aspects.keys():
        context = get_relevant_context(str(text), aspect)
        try:
            result     = sentiment_model(context[:512])[0]
            label      = result['label']
            confidence = result['score']
            results[aspect] = 'NEUTRAL' if confidence < 0.65 else label
        except Exception:
            results[aspect] = 'NEUTRAL'
    return results


# ── SECTION 4: AGGREGATION ──────────────────────────────────────────────────

def build_summary(df):
    """Aggregate per-review sentiments into aspect-level satisfaction scores.

    Satisfaction rate = POSITIVE / (POSITIVE + NEGATIVE).
    NEUTRAL mentions are excluded from the rate calculation because they
    represent ambiguous sentiment that neither helps nor hurts the picture.
    Including them would dilute the signal from decisive mentions.
    """
    aspect_data = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0})
    for _, row in df.iterrows():
        sentiments = row.get('sentiments_parsed', {})
        if isinstance(sentiments, dict):
            for aspect, sentiment in sentiments.items():
                key = sentiment.lower() if sentiment else 'neutral'
                if key in ('positive', 'negative', 'neutral'):
                    aspect_data[aspect][key] += 1

    rows = []
    for aspect, counts in aspect_data.items():
        pos      = counts['positive']
        neg      = counts['negative']
        neu      = counts['neutral']
        total    = pos + neg + neu
        decisive = pos + neg
        sat_rate = (pos / decisive * 100) if decisive > 0 else 0

        if sat_rate >= 75:   signal = "STRONG POSITIVE"
        elif sat_rate >= 55: signal = "POSITIVE"
        elif sat_rate >= 45: signal = "NEUTRAL"
        elif sat_rate >= 25: signal = "NEGATIVE"
        else:                signal = "STRONG NEGATIVE"

        rows.append({
            'Aspect':            aspect,
            'Total Mentions':    total,
            'Positive':          pos,
            'Negative':          neg,
            'Neutral':           neu,
            'Satisfaction Rate': round(sat_rate, 1),
            'Signal':            signal
        })
    return pd.DataFrame(rows).sort_values('Satisfaction Rate', ascending=False)
    
# ── SECTION 4.5: AUTOMATIC TAXONOMY DISCOVERY ──────────────────────────────

# This section is what makes the pipeline truly plug-and-play.
# Instead of requiring the user to manually define what aspects matter
# for their specific dataset, we sample a portion of their reviews,
# send them to Claude, and ask it to read them like a product analyst would.
# Claude identifies the recurring themes and writes rich descriptions for each,
# returning a taxonomy that feeds directly into the zero-shot classifier.
# The quality is indistinguishable from what a domain expert would write manually,
# and it takes about 15 seconds.

def auto_discover_taxonomy(df, text_column, sample_size=200, n_aspects=6):
    """
    Reads a sample of reviews and uses Claude to discover what aspects
    customers actually talk about — no human input required.
    
    Returns a taxonomy dictionary ready to plug into the pipeline,
    with the same format as the manual taxonomy you would write yourself.
    """
    import json

    # We only need a sample to discover the patterns — 200 reviews gives
    # a statistically representative picture of what customers write about most.
    # Using the full dataset would waste API tokens without improving the result.
    sample = (
        df[text_column]
        .dropna()
        .sample(min(sample_size, len(df)), random_state=42)
        .tolist()
    )

    # Format the sampled reviews as a numbered list for the prompt.
    # We cap each review at 200 characters because we only need enough text
    # for Claude to identify the topic — not the full review content.
    reviews_text = "\n".join([
        f"{i+1}. {str(r)[:200]}"
        for i, r in enumerate(sample)
    ])

    # This prompt is carefully designed to produce taxonomy descriptions
    # that are rich enough for the zero-shot classifier to use accurately.
    # The key instruction is to include synonyms and related vocabulary
    # in each description — because the zero-shot model uses semantic
    # similarity to match review sentences against these descriptions,
    # and richer descriptions produce more accurate matches.
    prompt = f"""You are a senior product analyst reviewing customer feedback.
I will give you {len(sample)} customer reviews for a product or service.

Read them carefully and identify the {n_aspects} most important aspects 
that customers repeatedly discuss. Base this entirely on what customers 
actually write about — do not assume aspects that are not present in the data.

For each aspect:
- NAME: short, stakeholder-readable (2-4 words, all lowercase)
- DESCRIPTION: a rich phrase including synonyms and related vocabulary 
  that a semantic similarity model can use to match relevant sentences.
  Include the kinds of language customers actually use.

Return ONLY a valid JSON object. No explanation, no markdown formatting,
no backticks — just the raw JSON object itself.

Example of the format to return:
{{"food quality": "taste, flavor, freshness of food, quality of ingredients, how dishes are prepared, delicious or disappointing meals",
  "service speed": "waiting time, how fast staff respond, slow or quick service, time to be seated or served"}}

Here are the {len(sample)} reviews to analyze:

{reviews_text}

Return the JSON taxonomy now:"""

    # Call the  API 
    # and writes the taxonomy. The model receives all sampled reviews at
    # once and synthesizes them into a coherent set of aspect categories.
    try:
        # API Changed to Google Gemini
        import google.generativeai as genai
        
        # Configure the Gemini library
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Generate the taxonomy
        response = model.generate_content(prompt)
        raw = response.text.strip()

        # Strip any accidental markdown formatting
        raw = raw.replace("```json", "").replace("```", "").strip()

        taxonomy = json.loads(raw)

        # Print the discovered taxonomy to the console so the user can
        # see exactly what was found before the pipeline continues
        print(f"\n  ✅ Auto-discovered {len(taxonomy)} aspects "
              f"from {len(sample)} sampled reviews:\n")
        for aspect, description in taxonomy.items():
            print(f"     • {aspect}")
            print(f"       {description[:90]}...")
            print()

        return taxonomy

    except Exception as e:
        # If anything goes wrong — network error, API key missing, 
        # JSON parse failure — we warn the user and fall back gracefully
        # to the default taxonomy rather than crashing the whole pipeline.
        print(f"\n  ⚠️  Auto-discovery failed: {e}")
        print("  Falling back to DEFAULT_TAXONOMY — consider setting ANTHROPIC_API_KEY")
        return DEFAULT_TAXONOMY


# ── SECTION 5: DASHBOARD GENERATION ────────────────────────────────────────
# The dashboard is a single self-contained HTML file that any stakeholder
# can open in a browser with no Python, no installation, and no internet
# connection required. It combines charts, word pills, customer quotes,
# and plain-English recommendations into one shareable artifact.

def _get_word_intelligence(df, aspect_name, top_n=12):
    """Extract the most common words from positive vs negative mentions
    of a given aspect. Uses text_for_stats (stopwords removed, lemmatized)
    so the counts reflect meaningful content words only."""
    pos_words, neg_words = [], []
    for _, row in df.iterrows():
        sentiments = row.get('sentiments_parsed', {})
        if not isinstance(sentiments, dict):
            continue
        if aspect_name in sentiments:
            # Filter out very short words and the aspect name words themselves
            words = [
                w for w in str(row.get('text_for_stats', '')).split()
                if len(w) > 3 and w not in aspect_name.split()
            ]
            if sentiments[aspect_name] == 'POSITIVE':
                pos_words.extend(words)
            elif sentiments[aspect_name] == 'NEGATIVE':
                neg_words.extend(words)

    return {
        'positive': Counter(pos_words).most_common(top_n),
        'negative': Counter(neg_words).most_common(top_n)
    }

def _get_representative_quotes(df, aspect_name):
    """Find the most confident positive and negative review quotes for an aspect.
    Sorted by detection confidence so the clearest examples appear first.
    We use the original raw text here — not preprocessed — because stakeholders
    need to read natural human language, not NLP-cleaned keyword bags."""
    pos_q, neg_q = [], []
    for _, row in df.iterrows():
        asp  = row.get('aspects_parsed', {})
        sent = row.get('sentiments_parsed', {})
        if not isinstance(asp, dict) or not isinstance(sent, dict):
            continue
        if aspect_name in asp and aspect_name in sent:
            score = asp[aspect_name]
            text  = str(row.get('text', ''))[:280]
            if sent[aspect_name] == 'POSITIVE':
                pos_q.append((score, text))
            elif sent[aspect_name] == 'NEGATIVE':
                neg_q.append((score, text))

    pos_q.sort(reverse=True)
    neg_q.sort(reverse=True)
    return {
        'positive': [t for _, t in pos_q[:1]],
        'negative': [t for _, t in neg_q[:1]]
    }

def _color(rate):
    """Map a satisfaction rate to a hex color for the dashboard."""
    if rate >= 75: return '#16a34a'
    if rate >= 55: return '#22c55e'
    if rate >= 45: return '#d97706'
    if rate >= 25: return '#dc2626'
    return '#991b1b'

def _bg_color(rate):
    """Map a satisfaction rate to a light background color."""
    if rate >= 75: return '#f0fdf4'
    if rate >= 55: return '#f0fdf4'
    if rate >= 45: return '#fffbeb'
    if rate >= 25: return '#fef2f2'
    return '#fef2f2'

def _build_aspect_card(aspect, row, wi, quotes):
    """Build the HTML card for a single aspect. Each card shows the
    satisfaction rate, a sentiment breakdown bar, the most common
    positive and negative vocabulary, and one representative customer
    quote in each direction."""
    sat   = row['Satisfaction Rate']
    pos   = row['Positive']
    neg   = row['Negative']
    neu   = row.get('Neutral', 0)
    tot   = row['Total Mentions']
    dec   = pos + neg
    pp    = (pos / dec * 100) if dec > 0 else 0
    np_   = (neg / dec * 100) if dec > 0 else 0

    # Build colored word pills for positive vocabulary
    pos_pills = ' '.join([
        f'<span style="background:#dcfce7;color:#15803d;padding:3px 10px;'
        f'border-radius:20px;font-size:11px;display:inline-block;margin:2px">'
        f'{w} <strong>({c})</strong></span>'
        for w, c in wi['positive'][:8]
    ]) or '<span style="color:#9ca3af;font-size:12px">insufficient data</span>'

    # Build colored word pills for negative vocabulary
    neg_pills = ' '.join([
        f'<span style="background:#fee2e2;color:#b91c1c;padding:3px 10px;'
        f'border-radius:20px;font-size:11px;display:inline-block;margin:2px">'
        f'{w} <strong>({c})</strong></span>'
        for w, c in wi['negative'][:8]
    ]) or '<span style="color:#9ca3af;font-size:12px">insufficient data</span>'

    # Build customer quote blocks
    pos_quote = ''.join([
        f'<blockquote style="background:#f0fdf4;border-left:3px solid #16a34a;'
        f'padding:10px 14px;margin:8px 0;border-radius:0 6px 6px 0;font-style:italic;'
        f'font-size:13px;color:#374151;line-height:1.5">"{q[:230]}..."</blockquote>'
        for q in quotes['positive']
    ])
    neg_quote = ''.join([
        f'<blockquote style="background:#fef2f2;border-left:3px solid #dc2626;'
        f'padding:10px 14px;margin:8px 0;border-radius:0 6px 6px 0;font-style:italic;'
        f'font-size:13px;color:#374151;line-height:1.5">"{q[:230]}..."</blockquote>'
        for q in quotes['negative']
    ])

    return f"""
    <div style="background:{_bg_color(sat)};border:1px solid {_color(sat)};
                border-radius:14px;padding:26px;margin-bottom:24px;
                box-shadow:0 1px 6px rgba(0,0,0,0.06)">

        <!-- Header row: aspect name left, big score right -->
        <div style="display:flex;justify-content:space-between;
                    align-items:flex-start;margin-bottom:18px">
            <div>
                <h3 style="margin:0 0 4px 0;color:#111827;font-size:17px;
                           text-transform:capitalize;letter-spacing:0.01em">
                    {aspect}
                </h3>
                <div style="font-size:12px;color:#6b7280">{tot} customer mentions</div>
            </div>
            <div style="text-align:center;min-width:80px">
                <div style="font-size:38px;font-weight:800;color:{_color(sat)};
                            line-height:1">{sat:.0f}%</div>
                <div style="font-size:10px;color:#6b7280;margin-top:2px;
                            text-transform:uppercase;letter-spacing:0.05em">
                    satisfaction
                </div>
            </div>
        </div>

        <!-- Sentiment breakdown bar -->
        <div style="margin-bottom:18px">
            <div style="font-size:11px;color:#6b7280;margin-bottom:5px;
                        text-transform:uppercase;letter-spacing:0.04em">
                Sentiment Breakdown
            </div>
            <div style="display:flex;height:10px;border-radius:5px;
                        overflow:hidden;background:#e5e7eb">
                <div style="width:{pp:.1f}%;background:#16a34a;
                            transition:width 0.6s ease"></div>
                <div style="width:{np_:.1f}%;background:#dc2626;
                            transition:width 0.6s ease"></div>
            </div>
            <div style="display:flex;justify-content:space-between;
                        font-size:11px;color:#6b7280;margin-top:5px">
                <span>✅ {pos} positive &nbsp;({pp:.0f}%)</span>
                <span>❌ {neg} negative &nbsp;({np_:.0f}%)</span>
                <span>➖ {neu} neutral</span>
            </div>
        </div>

        <!-- Positive vocabulary -->
        <div style="margin-bottom:14px">
            <div style="font-size:11px;font-weight:700;color:#15803d;
                        text-transform:uppercase;letter-spacing:0.05em;margin-bottom:7px">
                💚 Words used positively
            </div>
            <div style="line-height:2">{pos_pills}</div>
        </div>

        <!-- Negative vocabulary -->
        <div style="margin-bottom:14px">
            <div style="font-size:11px;font-weight:700;color:#b91c1c;
                        text-transform:uppercase;letter-spacing:0.05em;margin-bottom:7px">
                🔴 Words used negatively
            </div>
            <div style="line-height:2">{neg_pills}</div>
        </div>

        <!-- Customer quotes -->
        <div>
            <div style="font-size:11px;font-weight:700;color:#374151;
                        text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px">
                💬 Customer voice
            </div>
            {pos_quote}
            {neg_quote}
        </div>
    </div>"""


def generate_dashboard(df, summary, output_dir):
    """Generate the full HTML stakeholder dashboard and plain-text written report.

    This function is called automatically at the end of pipeline.run().
    Both files are completely self-contained — no internet, no Python, no
    dependencies required to open them. Share by email or Google Drive link.

    Outputs:
        {output_dir}/stakeholder_dashboard.html  ← open in any browser
        {output_dir}/stakeholder_report.txt      ← paste into slide deck / email
    """
    # ── Build supporting data structures ──────────────────────────────────
    print("     Building word intelligence per aspect...")
    word_intel = {
        asp: _get_word_intelligence(df, asp)
        for asp in summary['Aspect']
    }
    quotes = {
        asp: _get_representative_quotes(df, asp)
        for asp in summary['Aspect']
    }

    # ── Compute header statistics ──────────────────────────────────────────
    total_reviews     = len(df)
    with_aspects      = df['aspects_parsed'].apply(len).gt(0).sum()
    total_detections  = sum(
        len(a) for a in df['aspects_parsed'] if isinstance(a, dict)
    )
    avg_sat = summary['Satisfaction Rate'].mean()
    best    = summary.nlargest(1,  'Satisfaction Rate').iloc[0]
    worst   = summary.nsmallest(1, 'Satisfaction Rate').iloc[0]

    # ── Build per-aspect cards ─────────────────────────────────────────────
    print("     Building aspect cards...")
    cards_html = ""
    for _, row in summary.sort_values('Satisfaction Rate', ascending=False).iterrows():
        cards_html += _build_aspect_card(
            row['Aspect'], row,
            word_intel.get(row['Aspect'], {'positive': [], 'negative': []}),
            quotes.get(row['Aspect'], {'positive': [], 'negative': []})
        )

    # ── Build priority actions section ────────────────────────────────────
    worst2 = summary[summary['Satisfaction Rate'] < 70].nsmallest(2, 'Satisfaction Rate')

    # Only flag as "Protect Strength" if satisfaction is genuinely high (above 70%)
    best2  = summary[summary['Satisfaction Rate'] >= 70].nlargest(2, 'Satisfaction Rate')

    if worst2.empty:
        worst2_html = "<p style='color:#6b7280;font-size:13px'>All aspects performing well — no urgent fixes needed.</p>"
    else:
        worst2_html = ''.join([
            f'<div style="margin:8px 0"><strong style="color:#374151">'
            f'{r["Aspect"].title()}</strong> — '
            f'<strong style="color:#dc2626">{r["Satisfaction Rate"]:.0f}%</strong>'
            f' satisfaction across {r["Total Mentions"]} mentions</div>'
            for _, r in worst2.iterrows()
        ])
    
    if best2.empty:
        best2_html = "<p style='color:#6b7280;font-size:13px'>Insufficient data to identify clear strengths yet.</p>"
    else:
        best2_html = ''.join([
            f'<div style="margin:8px 0"><strong style="color:#374151">'
            f'{r["Aspect"].title()}</strong> — '
            f'<strong style="color:#16a34a">{r["Satisfaction Rate"]:.0f}%</strong>'
            f' satisfaction — promote this in marketing</div>'
            for _, r in best2.iterrows()
        ])

    priority_html = f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:32px">
        <div style="background:#fffbeb;border:1px solid #d97706;
                    border-radius:12px;padding:20px">
            <h4 style="color:#b45309;margin:0 0 12px 0;font-size:14px;
                       text-transform:uppercase;letter-spacing:0.05em">
                ⚡ Fix First — Lowest Satisfaction
            </h4>
            {''.join([
                f'<div style="margin:8px 0"><strong style="color:#374151">'
                f'{r["Aspect"].title()}</strong> — '
                f'<strong style="color:#dc2626">{r["Satisfaction Rate"]:.0f}%</strong>'
                f' satisfaction across {r["Total Mentions"]} mentions</div>'
                for _, r in worst2.iterrows()
            ])}
        </div>
        <div style="background:#f0fdf4;border:1px solid #16a34a;
                    border-radius:12px;padding:20px">
            <h4 style="color:#15803d;margin:0 0 12px 0;font-size:14px;
                       text-transform:uppercase;letter-spacing:0.05em">
                💪 Protect Your Strengths
            </h4>
            {''.join([
                f'<div style="margin:8px 0"><strong style="color:#374151">'
                f'{r["Aspect"].title()}</strong> — '
                f'<strong style="color:#16a34a">{r["Satisfaction Rate"]:.0f}%</strong>'
                f' satisfaction — promote this in marketing</div>'
                for _, r in best2.iterrows()
            ])}
        </div>
    </div>"""

    # ── Assemble the full HTML document ───────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Product Intelligence Dashboard</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
      font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
      background: #f3f4f6;
      color: #111827;
      line-height: 1.6;
  }}
  .page {{ max-width: 980px; margin: 0 auto; padding: 36px 20px; }}
  .stat-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 14px;
      margin-bottom: 32px;
  }}
  .stat-card {{
      background: white;
      border-radius: 12px;
      padding: 20px;
      text-align: center;
      box-shadow: 0 1px 4px rgba(0,0,0,0.08);
  }}
  .stat-number {{ font-size: 30px; font-weight: 800; }}
  .stat-label  {{ font-size: 12px; color: #6b7280; margin-top: 4px; }}
  h2 {{ color: #111827; margin-bottom: 20px; font-size: 18px; }}
  @media (max-width: 640px) {{
      .stat-grid {{ grid-template-columns: repeat(2, 1fr); }}
  }}
</style>
</head>
<body>
<div class="page">

  <!-- ── Header ── -->
  <div style="background:linear-gradient(135deg,#0f172a 0%,#1d4ed8 100%);
              color:white;border-radius:16px;padding:38px 40px;
              margin-bottom:30px;text-align:center">
      <div style="font-size:11px;letter-spacing:0.15em;text-transform:uppercase;
                  opacity:0.7;margin-bottom:10px">NLP Pipeline Output</div>
      <h1 style="font-size:26px;font-weight:800;margin-bottom:8px">
          📊 Product Intelligence Dashboard
      </h1>
      <p style="opacity:0.8;font-size:14px">
          Customer Review Analysis — Aspect-Based Sentiment Pipeline
      </p>
      <p style="opacity:0.55;font-size:12px;margin-top:6px">
          Preprocessing → Aspect Extraction → Sentiment Analysis → Aggregation
      </p>
  </div>

  <!-- ── Overview Stats ── -->
  <div class="stat-grid">
      <div class="stat-card">
          <div class="stat-number" style="color:#1d4ed8">{total_reviews:,}</div>
          <div class="stat-label">Reviews Analyzed</div>
      </div>
      <div class="stat-card">
          <div class="stat-number" style="color:#7c3aed">{with_aspects:,}</div>
          <div class="stat-label">With Aspects Found</div>
      </div>
      <div class="stat-card">
          <div class="stat-number" style="color:#d97706">{total_detections:,}</div>
          <div class="stat-label">Aspect Mentions</div>
      </div>
      <div class="stat-card">
          <div class="stat-number"
               style="color:{'#16a34a' if avg_sat >= 55 else '#dc2626'}">
              {avg_sat:.0f}%
          </div>
          <div class="stat-label">Avg Satisfaction</div>
      </div>
  </div>

  <!-- ── Priority Actions ── -->
  <h2>🚨 Priority Actions for Product Team</h2>
  {priority_html}

  <!-- ── Aspect Cards ── -->
  <h2>📋 Aspect-by-Aspect Breakdown</h2>
  {cards_html}

  <!-- ── Footer ── -->
  <div style="text-align:center;color:#9ca3af;font-size:12px;
              margin-top:32px;padding-top:20px;border-top:1px solid #e5e7eb">
      Generated by Review Intelligence Pipeline &nbsp;·&nbsp;
      Python · HuggingFace Transformers · NLTK · Pandas
  </div>

</div>
</body>
</html>"""

    # ── Save HTML dashboard ────────────────────────────────────────────────
    html_path = os.path.join(output_dir, 'stakeholder_dashboard.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    size_kb = os.path.getsize(html_path) / 1024
    print(f"     ✅ Dashboard:  {html_path}  ({size_kb:.1f} KB)")

    # ── Build and save written report ─────────────────────────────────────
    report_lines = [
        "PRODUCT INTELLIGENCE REPORT — Customer Review Sentiment Analysis",
        "=" * 62, "",
        "EXECUTIVE SUMMARY",
        "-" * 18,
        f"Reviews analyzed:  {total_reviews:,}",
        f"Reviews with insights: {with_aspects:,}  ({with_aspects/total_reviews*100:.1f}%)",
        f"Average satisfaction:  {avg_sat:.1f}%",
        f"Overall health:  {'POSITIVE ✅' if avg_sat >= 55 else 'NEEDS ATTENTION ⚠️' if avg_sat >= 45 else 'CRITICAL 🚨'}",
        "", "=" * 62,
        "SATISFACTION RANKING  (best → worst)",
        "=" * 62,
    ]
    for _, row in summary.sort_values('Satisfaction Rate', ascending=False).iterrows():
        bar = '█' * int(row['Satisfaction Rate'] / 5)
        report_lines.append(
            f"  {row['Aspect'].upper():<28}  {row['Satisfaction Rate']:>5.1f}%"
            f"  |{bar:<20}|  ({row['Total Mentions']} mentions)"
        )
    report_lines += [
        "", "=" * 62, "KEY FINDINGS", "=" * 62, "",
        f"STRENGTH — {best['Aspect'].upper()}  ({best['Satisfaction Rate']:.1f}% satisfaction)",
        f"Customers consistently respond positively to {best['Aspect']} across",
        f"{best['Total Mentions']} mentions. Protect this in future versions",
        "and use it as a marketing differentiator.",
    ]
    wi_best = word_intel.get(best['Aspect'], {'positive': []})
    if wi_best['positive']:
        report_lines.append(
            f"  Positive language: {', '.join(w for w, _ in wi_best['positive'][:6])}"
        )
    report_lines += [
        "",
        f"PRIORITY FIX — {worst['Aspect'].upper()}  ({worst['Satisfaction Rate']:.1f}% satisfaction)",
        f"{100 - worst['Satisfaction Rate']:.0f}% of customers who mention {worst['Aspect']}",
        "are dissatisfied. This is your most urgent improvement area.",
    ]
    wi_worst = word_intel.get(worst['Aspect'], {'negative': []})
    if wi_worst['negative']:
        report_lines.append(
            f"  Complaint language: {', '.join(w for w, _ in wi_worst['negative'][:6])}"
        )
    report_lines += [
        "", "=" * 62, "PRIORITIZED ACTION LIST", "=" * 62,
    ]
    for rank, (_, row) in enumerate(
        summary.sort_values('Satisfaction Rate').iterrows(), 1
    ):
        tag = (
            "🚨 URGENT"   if row['Satisfaction Rate'] < 45 else
            "⚠️  MONITOR" if row['Satisfaction Rate'] < 60 else
            "✅ MAINTAIN"
        )
        report_lines.append(
            f"  {rank}. [{tag}]  {row['Aspect'].title():<28}"
            f"  {row['Satisfaction Rate']:.1f}%  ({row['Total Mentions']} mentions)"
        )
    report_lines += [
        "", "=" * 62, "METHODOLOGY NOTE", "=" * 62,
        "Aspect detection: cross-encoder/nli-MiniLM2-L6-H768  (zero-shot NLI)",
        "Sentiment model:  distilbert-base-uncased-finetuned-sst-2-english",
        f"Aspect threshold: confidence ≥ 0.50 required for detection",
        f"Coverage:         {with_aspects}/{total_reviews} reviews ({with_aspects/total_reviews*100:.1f}%)",
        "=" * 62,
    ]

    report_text = '\n'.join(report_lines)
    txt_path = os.path.join(output_dir, 'stakeholder_report.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"     ✅ Report:     {txt_path}")

    return html_path, txt_path


# ── SECTION 6: MAIN PIPELINE CLASS ─────────────────────────────────────────

class ReviewIntelligencePipeline:
    """End-to-end pipeline for customer review analysis.

    This class is the single entry point for the entire system.
    Calling run() executes all 7 steps in sequence and saves four output
    files to the specified output directory.

    To adapt the pipeline to a new product domain, pass a custom taxonomy
    dictionary to __init__. Everything else requires zero changes.

    Example — general product reviews (default taxonomy):
        pipe = ReviewIntelligencePipeline()
        pipe.run("reviews.csv", text_column="review_text")

    Example — restaurant reviews (custom taxonomy):
        restaurant_taxonomy = {
            "food quality":   "taste, flavor, and freshness of the food",
            "service":        "staff attitude and speed of service",
            "value for money":"pricing and whether it was worth the cost",
            "ambiance":       "atmosphere, decor, and restaurant environment"
        }
        pipe = ReviewIntelligencePipeline(taxonomy=restaurant_taxonomy)
        pipe.run("restaurant_reviews.csv", text_column="review_text")
    """

    def __init__(self, taxonomy=None):
        self.taxonomy             = taxonomy or DEFAULT_TAXONOMY
        self.aspect_classifier    = None
        self.sentiment_classifier = None
        print("✅ ReviewIntelligencePipeline initialized")
        print(f"   Aspects configured: {list(self.taxonomy.keys())}")

    def _load_models(self):
        """Load both NLP models lazily — only when run() is called.
        Lazy loading means initializing the class is instant; the
        30-60 second model download only happens when you actually run."""
        if self.aspect_classifier is None:
            print("     Loading aspect detection model (cross-encoder/nli-MiniLM2)...")
            self.aspect_classifier = load_aspect_classifier()
        if self.sentiment_classifier is None:
            print("     Loading sentiment model (distilbert-sst2)...")
            self.sentiment_classifier = load_sentiment_classifier()
        print("     ✅ Both models ready")

    def run(self, csv_path, text_column="text", output_dir="outputs/",
        sample_size=None, threshold=0.50, auto_discover=True, n_aspects=6):
        """Run the complete 7-step pipeline on a reviews CSV file.

        Parameters
        ----------
        csv_path    : str   — path to your reviews CSV file
        text_column : str   — name of the column containing review text
        output_dir  : str   — directory where all four output files are saved
        sample_size : int   — optionally limit to N reviews (None = use all)
        threshold   : float — min model confidence to count an aspect detection

        Output files saved to output_dir
        ---------------------------------
        reviews_analyzed.csv          per-review data with aspects + sentiments
        aspect_summary.csv            aggregated satisfaction scores per aspect
        stakeholder_dashboard.html    visual dashboard — open in any browser
        stakeholder_report.txt        plain-English recommendations document
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n{'='*58}")
        print(f"  REVIEW INTELLIGENCE PIPELINE  —  STARTING")
        print(f"{'='*58}")

        # ── Step 1: Load data ──────────────────────────────────────────────
        print(f"\n[1/7] Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        if sample_size:
            df = df.sample(min(sample_size, len(df)), random_state=42)
        df = df.dropna(subset=[text_column]).reset_index(drop=True)
        print(f"      {len(df):,} reviews loaded")

        # ── Step 1.5: Auto-discover taxonomy if none was explicitly provided ──
        # We check whether the taxonomy is still the default one, which tells us
        # the user did not pass a custom taxonomy to __init__. If that is the case
        # and auto_discover is enabled, we read the data and build the taxonomy now.
        # This must happen after loading the data (we need reviews to analyze)
        # but before preprocessing (the taxonomy shapes what we are looking for).
        if auto_discover and self.taxonomy is DEFAULT_TAXONOMY:
            print("\n[AUTO] No taxonomy provided — discovering aspects from your data...")
            print("       Sampling reviews and consulting Claude...\n")
            self.taxonomy = auto_discover_taxonomy(
                df          = df,
                text_column = text_column,
                sample_size = min(200, len(df)),
                n_aspects   = n_aspects
            )
            print("  Pipeline will now use the auto-discovered taxonomy above.\n")


        # ── Step 2: Preprocess ─────────────────────────────────────────────
        print("\n[2/7] Preprocessing text...")
        print("      Creating text_for_models (transformer-ready)...")
        df['text_for_models'] = df[text_column].apply(preprocess_for_transformers)
        print("      Creating text_for_stats  (statistics-ready)...")
        df['text_for_stats']  = df[text_column].apply(preprocess_for_statistics)
        # Remove reviews that are too short to extract meaningful aspects from
        df = df[df['text_for_models'].apply(lambda x: len(str(x).split()) >= 5)]
        df = df.reset_index(drop=True)
        print(f"      {len(df):,} reviews after filtering short text")

        # ── Step 3: Load NLP models ────────────────────────────────────────
        print("\n[3/7] Loading NLP models...")
        self._load_models()

        # ── Step 4: Aspect extraction ──────────────────────────────────────
        # Zero-shot classification — no labeled training data required.
        # The model reads each aspect description and decides whether
        # the review text semantically entails discussing that topic.
        print("\n[4/7] Extracting aspects from reviews...")
        try:
            from tqdm import tqdm
            aspect_iter = tqdm(df['text_for_models'], desc="      Aspects")
        except ImportError:
            aspect_iter = df['text_for_models']

        df['detected_aspects'] = [
            extract_aspects(
                text, self.aspect_classifier, self.taxonomy, threshold
            )
            for text in aspect_iter
        ]
        df['aspects_parsed'] = df['detected_aspects']

        with_asp = df['aspects_parsed'].apply(len).gt(0).sum()
        print(f"      {with_asp:,} / {len(df):,} reviews had aspects detected "
              f"({with_asp/len(df)*100:.1f}%)")

        # ── Step 5: Sentiment classification ──────────────────────────────
        # Aspect-level sentiment — not whole-review sentiment.
        # We extract the most relevant sentence per aspect and classify
        # sentiment on that focused context rather than the full review.
        print("\n[5/7] Classifying sentiment per aspect...")
        try:
            sentiment_iter = tqdm(
                zip(df['text_for_models'], df['aspects_parsed']),
                total=len(df), desc="      Sentiment"
            )
        except ImportError:
            sentiment_iter = zip(df['text_for_models'], df['aspects_parsed'])

        df['aspect_sentiments'] = [
            classify_sentiments(text, aspects, self.sentiment_classifier)
            for text, aspects in sentiment_iter
        ]
        df['sentiments_parsed'] = df['aspect_sentiments']

        # ── Step 6: Aggregate ──────────────────────────────────────────────
        print("\n[6/7] Aggregating results...")
        summary = build_summary(df)
        df.to_csv(     os.path.join(output_dir, 'reviews_analyzed.csv'), index=False)
        summary.to_csv(os.path.join(output_dir, 'aspect_summary.csv'),   index=False)
        print(f"      Saved reviews_analyzed.csv  and  aspect_summary.csv")

        # ── Step 7: Generate dashboard and report ─────────────────────────
        # This was the step missing in the original pipeline — now integrated.
        # Produces a browser-ready HTML file and a plain-English report,
        # both saved to output_dir alongside the two CSV files.
        print("\n[7/7] Generating stakeholder dashboard and report...")
        generate_dashboard(df, summary, output_dir)

        # ── Final summary printout ─────────────────────────────────────────
        print(f"\n{'='*58}")
        print(f"  PIPELINE COMPLETE")
        print(f"{'='*58}")
        print(f"  Reviews analyzed : {len(df):,}")
        print(f"  Outputs saved to : {output_dir}")
        print(f"  ├── reviews_analyzed.csv")
        print(f"  ├── aspect_summary.csv")
        print(f"  ├── stakeholder_dashboard.html  ← open in browser")
        print(f"  └── stakeholder_report.txt")
        print(f"\n  Satisfaction Ranking:")
        for _, row in summary.iterrows():
            bar = '█' * int(row['Satisfaction Rate'] / 5)
            print(
                f"    {row['Aspect']:<28}"
                f"  {row['Satisfaction Rate']:>5.1f}%"
                f"  |{bar:<20}|"
                f"  ({row['Total Mentions']} mentions)"
            )
        print(f"{'='*58}\n")

        return df, summary