# 📊 Review Intelligence Pipeline

> An end-to-end NLP system that transforms raw customer reviews into actionable business intelligence — automatically discovering what product features customers love and hate, then delivering a professional stakeholder dashboard.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange?logo=huggingface&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 What Problem Does This Solve?

A company receives thousands of customer reviews every day. Reading them manually is impossible. Standard sentiment analysis only tells you *"70% positive"* — which is useless because it tells you nothing about **what** to fix or **what** to double down on.

This pipeline answers a far more valuable question:

> *"Customers love your sound quality (87% positive, 2,300 mentions) but are frustrated with battery life (34% positive, 1,800 mentions)."*

That is **actionable intelligence**. That is what this project delivers — automatically, for any product domain, from any CSV of reviews.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🤖 **Auto Taxonomy Discovery** | Uses Google Gemini to read your reviews and discover what aspects customers actually discuss — no manual configuration needed |
| 🔍 **Zero-Shot Aspect Detection** | Detects which product features each review mentions without any labeled training data |
| 💬 **Aspect-Level Sentiment** | Classifies sentiment per feature, not per whole review — so mixed reviews are understood correctly |
| 📊 **Stakeholder Dashboard** | Generates a self-contained HTML dashboard that anyone can open in a browser |
| 📄 **Written Report** | Produces a plain-English text report with prioritised recommendations |
| 🌐 **Web Application** | Drag-and-drop Streamlit interface — no code required to use it |
| 🔁 **Any Domain** | Works on restaurant reviews, hotel reviews, app reviews, product reviews — just upload your CSV |

---

## 🏗️ Architecture Overview

```
Raw Reviews CSV  (any domain, any size)
        │
        ▼
┌─────────────────────────────────┐
│  STEP 1 — Auto Taxonomy         │  Google Gemini reads a sample of your
│  Discovery (Gemini API)         │  reviews and discovers what aspects
│                                 │  customers write about, with keywords
└──────────────┬──────────────────┘
               │  taxonomy + keywords
               ▼
┌─────────────────────────────────┐
│  STEP 2 — Text Preprocessing    │  Two pipelines:
│                                 │  • text_for_models  (for BERT — natural English)
│                                 │  • text_for_stats   (for counting — clean keywords)
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  STEP 3 — Aspect Extraction     │  Zero-shot NLI model reads each review
│  (cross-encoder/nli-MiniLM2)    │  and flags which aspects are discussed
│                                 │  with a confidence score ≥ threshold
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  STEP 4 — Sentiment Analysis    │  For each detected aspect, extracts the
│  (distilbert-sst2)              │  most relevant sentence and classifies
│                                 │  it as POSITIVE / NEGATIVE / NEUTRAL
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  STEP 5 — Aggregation           │  Computes satisfaction rate per aspect
│                                 │  across all reviews:
│                                 │  satisfaction = POSITIVE / (POS + NEG)
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  STEP 6 — Dashboard Generation  │  Produces four output files:
│                                 │  • stakeholder_dashboard.html
│                                 │  • stakeholder_report.txt
│                                 │  • reviews_analyzed.csv
│                                 │  • aspect_summary.csv
└─────────────────────────────────┘
```

---

## 📁 Project Structure

```
review_intelligence/
│
├── src/
│   ├── pipeline.py          ← The complete NLP pipeline (all 6 sections)
│   └── app.py               ← Streamlit web application
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_aspect_extraction.ipynb
│   ├── 04_sentiment_analysis.ipynb
│   ├── 05_aggregation.ipynb
│   ├── 06_dashboard.ipynb
│   ├── 07_pipeline_integration.ipynb
│   ├── 08_evaluation.ipynb
│   └── 09_web_application.ipynb
│
├── data/
│   ├── raw/                 ← Original data (never modified)
│   └── processed/           ← Cleaned and analyzed data
│
├── outputs/                 ← Generated dashboards and reports
│
├── requirements.txt
└── README.md
```

---

## 🧠 NLP Concepts Demonstrated

This project covers the following NLP techniques, applied in a real end-to-end system:

- **Text Preprocessing** — HTML removal, contraction expansion, lowercasing, whitespace normalisation
- **Lemmatisation with POS Tagging** — Reduces words to root forms using grammatical context, with auxiliary verb protection to prevent "is" → "be" errors
- **Two-Pipeline Strategy** — Separate preprocessing for transformer models vs statistical tools
- **Zero-Shot Classification** — Aspect detection using Natural Language Inference without labeled training data
- **Aspect-Based Sentiment Analysis (ABSA)** — Sentiment classified per product feature, not per whole review
- **Contextual Window Extraction** — Isolates the most relevant sentence per aspect before sentiment classification
- **Auto Taxonomy Discovery** — Uses an LLM to read your reviews and discover domain-specific aspects automatically
- **Aggregation Engine** — Converts per-review predictions into dataset-level satisfaction statistics
- **Evaluation** — Accuracy measured against star ratings as ground truth

---

## 🛠️ Models Used

| Model | Role | Why This Model |
|---|---|---|
| `gemini-1.5-flash` | Taxonomy discovery | Fast, capable LLM that reads reviews and generates structured aspect taxonomies |
| `cross-encoder/nli-MiniLM2-L6-H768` | Aspect extraction | Lightweight NLI model with reliable confidence scores, runs on free GPU |
| `distilbert-base-uncased-finetuned-sst-2-english` | Sentiment classification | Fine-tuned on consumer reviews, 97% the quality of BERT at 40% the size |

---

## 🚀 Getting Started

### Option 1 — Use the Web Application (Recommended)

The easiest way to use this project requires no coding knowledge.

**Step 1:** Open Google Colab and mount your Drive

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/review_intelligence')
```

**Step 2:** Set your API keys

```python
import os
os.environ["GOOGLE_API_KEY"] = "your-gemini-api-key-here"
```

> Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com)

**Step 3:** Install dependencies

```bash
pip install streamlit pyngrok google-generativeai transformers torch \
            nltk contractions python-dotenv tqdm pandas scikit-learn
```

**Step 4:** Launch the web application

```python
import subprocess, time
from pyngrok import ngrok

# Authenticate ngrok (get your token at ngrok.com)
!ngrok authtoken YOUR_NGROK_TOKEN

# Start Streamlit
subprocess.Popen([
    "streamlit", "run", "src/app.py",
    "--server.port", "8501",
    "--server.headless", "true"
], stdout=open("streamlit.log", "w"), stderr=subprocess.STDOUT)

time.sleep(5)
public_url = ngrok.connect(8501)
print(f"\n🌐 Your app is live at: {public_url}")
```

**Step 5:** Open the URL, upload your reviews CSV, enter your API key in the sidebar, and click **Run Analysis Pipeline**.

---

### Option 2 — Use the Python API

For developers who want to integrate the pipeline into their own code:

```python
from src.pipeline import ReviewIntelligencePipeline

# Fully automatic — Gemini discovers aspects from your data
pipe = ReviewIntelligencePipeline()
df, summary = pipe.run(
    csv_path    = "your_reviews.csv",
    text_column = "review_text",      # column name in your CSV
    output_dir  = "outputs/",
    threshold   = 0.45               # aspect detection confidence cutoff
)
```

For a specific product domain with a manual taxonomy:

```python
hotel_taxonomy = {
    "room cleanliness": "cleanliness, tidiness, hygiene of the room and bathroom",
    "staff service":    "helpfulness, friendliness, professionalism of hotel staff",
    "breakfast":        "quality, variety, and value of the breakfast offering",
    "location":         "proximity to attractions, transport links, neighbourhood",
    "value for money":  "whether the price reflects the quality and experience"
}

pipe = ReviewIntelligencePipeline(taxonomy=hotel_taxonomy)
df, summary = pipe.run("hotel_reviews.csv", text_column="review")
```

---

## 📥 Input Format

Your CSV needs at minimum one column containing review text. Any column name works — you select it in the app or pass it as `text_column`. A rating column is optional but enables accuracy validation.

| review_text | rating |
|---|---|
| The food was absolutely delicious but service was slow | 3 |
| Terrible experience, would not return | 1 |
| Perfect in every way, highly recommend | 5 |

---

## 📤 Output Files

After running, four files are saved to your `output_dir`:

| File | Description |
|---|---|
| `stakeholder_dashboard.html` | Visual dashboard — open in any browser, no internet required |
| `stakeholder_report.txt` | Plain-English written report with prioritised recommendations |
| `reviews_analyzed.csv` | Per-review data: detected aspects + sentiment labels |
| `aspect_summary.csv` | Aggregated satisfaction scores per aspect |

---

## 📊 Example Dashboard Output

The HTML dashboard shows:

- **Overview stats** — total reviews, coverage, aspect mentions, average satisfaction
- **Priority actions** — which aspects to fix first and which strengths to protect in marketing
- **Per-aspect cards** — satisfaction rate, sentiment breakdown bar, most common positive and negative vocabulary, representative customer quotes

---

## 🔧 Configuration Options

All settings are exposed in the Streamlit sidebar. When using the Python API directly:

| Parameter | Default | Description |
|---|---|---|
| `text_column` | `"text"` | Column name containing review text |
| `threshold` | `0.50` | Minimum confidence to count an aspect detection. Lower = more detections, less precise |
| `sample_size` | `None` (all) | Limit reviews processed. Use 500–1000 during testing |
| `auto_discover` | `True` | Whether to use Gemini to discover taxonomy automatically |
| `n_aspects` | `6` | How many aspects Gemini should discover |

---

## 🎯 Accuracy

The pipeline's sentiment classifier was evaluated using star ratings as ground truth (5-star = POSITIVE ground truth, 1–2 star = NEGATIVE ground truth). Mid-range ratings (3–4 stars) were excluded as genuinely ambiguous.

| Metric | Value |
|---|---|
| Overall Sentiment Accuracy | ~75–85% (varies by dataset) |
| Evaluation Method | Star rating as proxy ground truth |
| Aspect Detection | Estimated 70–85% precision via manual spot-check |

> Accuracy improves significantly on focused single-domain datasets (e.g. only restaurant reviews) vs the mixed-domain Amazon dataset used during development.

---

## ⚙️ How Gemini Auto-Discovery Works

When no custom taxonomy is provided, the pipeline automatically:

1. Samples up to 200 reviews from your dataset
2. Sends them to Gemini with a structured prompt asking it to identify recurring themes
3. Receives back a JSON array with aspect names, rich descriptions, and domain-specific keywords
4. Uses those descriptions as candidate labels for zero-shot classification
5. Uses those keywords for sentence-level context extraction during sentiment analysis

This means the pipeline **adapts to your specific domain automatically** — restaurant keywords for restaurant data, technical terms for software reviews, medical vocabulary for healthcare feedback.

You can verify Gemini ran by checking the console output, which prints the discovered aspects before any NLP processing begins:

```
✅ Auto-discovered 5 aspects from 200 reviews:

   • food quality
     Description: taste, flavor, freshness of ingredients, quality of dishes...
     Keywords: food, taste, flavor, delicious, bland...

   • service speed
     Description: waiting time, how fast staff respond, slow or quick service...
     Keywords: wait, slow, fast, minutes, waiter...
```

---

## 📚 Development Journey

This project was built over 7 days as a learning exercise covering the full NLP engineering lifecycle:

| Day | Focus |
|---|---|
| Day 1 | Data collection and exploratory data analysis |
| Day 2 | Text preprocessing pipeline (two separate strategies) |
| Day 3 | Aspect extraction using zero-shot classification |
| Day 4 | Aspect-level sentiment analysis with contextual windows |
| Day 5 | Aggregation engine and satisfaction scoring |
| Day 6 | HTML dashboard and written report generation |
| Day 7 | Full pipeline integration, domain testing, README |
| Day 8 | Accuracy evaluation methodology |
| Day 9 | Streamlit web application |

---

## 🔑 Requirements

```
pandas
numpy
transformers
torch
nltk
scikit-learn
streamlit
pyngrok
google-generativeai
contractions
python-dotenv
tqdm
datasets
```

Install all at once:

```bash
pip install -r requirements.txt
```

NLTK data (downloaded automatically on first run):

```python
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
```

---

## ⚠️ Known Limitations

- **Small datasets** (fewer than 50 reviews per aspect): Satisfaction rates are sensitive to individual classification errors. Use results directionally, not as precise measurements.
- **Mixed-domain data**: The Amazon Polarity dataset spans 18 product categories. Single-domain datasets produce more accurate and meaningful results.
- **Manual taxonomy keywords**: When a custom taxonomy is provided manually (not auto-discovered), sentiment context extraction falls back to full-review text for aspects not in the default keyword list.
- **Colab session limits**: Free Colab sessions time out after inactivity. The Streamlit URL becomes unavailable when the session ends.

---

## 🧑‍💻 Author

**Inaamallah**  
AI Engineering Student — Bahria University (CGPA 3.63/4.00)  
Freelance Data Analyst — Upwork  

Skills demonstrated in this project: Python · NLP · HuggingFace Transformers · Zero-Shot Learning · LLM API Integration · Streamlit · Data Pipeline Design · HTML/CSS

---

## 📄 License

This project is open source under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [HuggingFace](https://huggingface.co) for the transformer models and datasets library
- [Google Gemini](https://aistudio.google.com) for the taxonomy discovery API
- [Streamlit](https://streamlit.io) for the web application framework
- [McAuley Lab, UCSD](https://cseweb.ucsd.edu/~jmcauley/) for the Amazon Reviews dataset
- [Stanford NLP](https://nlp.stanford.edu) for the Amazon Polarity dataset
```
