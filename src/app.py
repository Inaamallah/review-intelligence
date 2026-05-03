import streamlit as st
import pandas as pd
import os
import sys
import tempfile
import json
import time
import base64
from pathlib import Path

# ── Page configuration — must be the first Streamlit call ─────────────────
st.set_page_config(
    page_title="Review Intelligence Pipeline",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Add project root to Python path so we can import pipeline.py ──────────
# Automatically detect the project root from this file's location.
# This file lives in  <project_root>/src/app.py  →  parent.parent = project root
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)

# ── Custom CSS — makes the app look professional ──────────────────────────
# We inject custom CSS to override Streamlit defaults with our own styling.
# This is how you customize Streamlit's appearance beyond its built-in themes.
st.markdown("""
<style>
    /* Main background and font */
    .stApp { background-color: #f8f9fa; }
    
    /* Upload box styling */
    .uploadedFile { border-radius: 10px; }
    
    /* Progress steps */
    .step-box {
        background: white;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 8px 0;
        border-left: 4px solid #1d4ed8;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        font-size: 14px;
    }
    
    /* Result metric boxes */
    .metric-box {
        background: white;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 1px 6px rgba(0,0,0,0.08);
    }
    
    /* Download button styling */
    .download-btn {
        display: inline-block;
        background: #1d4ed8;
        color: white !important;
        padding: 10px 24px;
        border-radius: 8px;
        text-decoration: none;
        font-weight: 600;
        margin: 6px;
    }
    
    /* Header gradient banner */
    .header-banner {
        background: linear-gradient(135deg, #0f172a, #1d4ed8);
        color: white;
        padding: 32px 40px;
        border-radius: 16px;
        margin-bottom: 28px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper: create a downloadable link for a file ─────────────────────────
def create_download_link(file_path, link_text, mime_type="text/html"):
    """
    Streamlit does not natively support downloading generated files with
    custom filenames, so we encode the file as base64 and embed it in an
    HTML anchor tag. When the user clicks it, their browser downloads the file.
    This is a standard pattern for file downloads in Streamlit apps.
    """
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    filename = Path(file_path).name
    href = (
        f'<a href="data:{mime_type};base64,{b64}" '
        f'download="{filename}" class="download-btn">'
        f'{link_text}</a>'
    )
    return href


# ── Main application layout ────────────────────────────────────────────────
def main():
    
    # Header banner
    st.markdown("""
    <div class="header-banner">
        <h1 style="font-size:28px;font-weight:800;margin-bottom:8px">
            📊 Review Intelligence Pipeline
        </h1>
        <p style="opacity:0.85;font-size:15px;margin:0">
            Upload any customer reviews CSV — get a full stakeholder dashboard automatically
        </p>
        <p style="opacity:0.6;font-size:12px;margin-top:8px">
            Powered by NLP: Preprocessing · Aspect Extraction · Sentiment Analysis · Aggregation
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar: Configuration options ────────────────────────────────────
    # The sidebar lets users customize the pipeline without exposing code.
    # We collect all settings here and pass them to pipeline.run() later.
    with st.sidebar:
        st.markdown("### ⚙️ Pipeline Settings")
        
        st.markdown("**API Configuration**")
        google_api_key = st.text_input(
            "Google API Key (for auto taxonomy discovery)",
            type="password",    # hides the key as the user types
            placeholder="AIza...",
            help="Required for automatic aspect discovery. Get yours at console.cloud.google.com"
        )
        
        st.markdown("---")
        st.markdown("**Detection Settings**")
        
        threshold = st.slider(
            "Aspect Detection Threshold",
            min_value=0.30, max_value=0.70, value=0.45, step=0.05,
            help="Lower = detects more aspects but less precise. Higher = fewer but more confident detections."
        )
        
        sample_size = st.select_slider(
            "Reviews to Analyze",
            options=[100, 500, 1000, 2500, 5000, "All"],
            value=1000,
            help="Larger samples give more reliable insights but take longer to process."
        )
        
        n_aspects = st.slider(
            "Number of Aspects to Discover",
            min_value=3, max_value=8, value=6,
            help="How many product feature categories to identify in your reviews."
        )
        
        st.markdown("---")
        st.markdown("**Optional: Custom Taxonomy**")
        custom_taxonomy_input = st.text_area(
            "Paste JSON taxonomy (leave blank for auto-discovery)",
            placeholder='{"food quality": "taste and freshness...",\n"service": "staff attitude..."}',
            height=120,
            help="Advanced: define your own aspects. If blank, the pipeline discovers them automatically."
        )

    # ── Main area: File upload ─────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📂 Upload Your Reviews File")
        uploaded_file = st.file_uploader(
            "Drag and drop your CSV file here, or click to browse",
            type=["csv"],
            help="Your CSV must have at least one column containing review text."
        )
    
    with col2:
        st.markdown("### 📋 Expected Format")
        st.dataframe(
            pd.DataFrame({
                "review_text": ["Great product!", "Terrible quality..."],
                "rating": [5, 1]
            }),
            hide_index=True,
            use_container_width=True
        )
        st.caption("Only the review text column is required. Rating is optional.")

    # ── If a file was uploaded, show preview and run button ───────────────
    if uploaded_file is not None:
        
        # Read and preview the uploaded file
        df_preview = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)   # reset file pointer after reading for preview
        
        st.markdown(f"**✅ File loaded:** `{uploaded_file.name}`  "
                    f"— {len(df_preview):,} rows, {len(df_preview.columns)} columns")
        
        with st.expander("👁️ Preview first 5 rows"):
            st.dataframe(df_preview.head(), use_container_width=True)
        
        # Let user specify which column contains the review text
        text_columns = df_preview.columns.tolist()
        text_column = st.selectbox(
            "Which column contains the review text?",
            options=text_columns,
            index=0,
            help="Select the column from your CSV that contains the customer review text."
        )
        
        st.markdown("---")
        
        # ── Run button ────────────────────────────────────────────────────
        run_button = st.button(
            "🚀 Run Analysis Pipeline",
            type="primary",
            use_container_width=True
        )
        
        if run_button:
            # Validate that required settings are present
            if not google_api_key and not custom_taxonomy_input:
                st.warning(
                    "⚠️ No Google API key provided and no custom taxonomy defined. "
                    "The pipeline will use the default taxonomy, which may not match "
                    "your specific product domain. For best results, provide an API key."
                )
            
            # ── Set up environment ────────────────────────────────────────
            if google_api_key:
                os.environ["GOOGLE_API_KEY"] = google_api_key
            
            # Save uploaded file to a temporary location so the pipeline can read it
            # Streamlit gives us the file as bytes in memory — we write it to disk
            # because pipeline.run() expects a file path, not a file object.
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".csv", dir=tempfile.gettempdir()
            ) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_csv_path = tmp_file.name
            
            # Create a timestamped output directory for this specific run
            # Using a timestamp ensures multiple runs do not overwrite each other
            import datetime
            timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(PROJECT_ROOT, "outputs", "webapp_runs", timestamp)
            os.makedirs(output_dir, exist_ok=True)
            
            # ── Run the pipeline with live progress display ───────────────
            # We display each step as it completes so the user knows the
            # pipeline is running and approximately how far along it is.
            # Streamlit updates the UI in real time as you call st.markdown().
            
            progress_placeholder = st.empty()   # placeholder we will update
            status_placeholder   = st.empty()
            
            with st.status("🔄 Running analysis pipeline...", expanded=True) as status:
                
                try:
                    # Import the pipeline — this loads your entire pipeline.py
                    from src.pipeline import ReviewIntelligencePipeline
                    
                    st.write("✅ Pipeline module loaded")
                    
                    # Parse custom taxonomy if provided
                    taxonomy = None
                    if custom_taxonomy_input.strip():
                        try:
                            taxonomy = json.loads(custom_taxonomy_input)
                            st.write(f"✅ Custom taxonomy loaded: {list(taxonomy.keys())}")
                        except json.JSONDecodeError:
                            st.error("❌ Invalid JSON in custom taxonomy. Using auto-discovery instead.")
                            taxonomy = None
                    
                    # Initialize the pipeline
                    pipe = ReviewIntelligencePipeline(taxonomy=taxonomy)
                    st.write("✅ Pipeline initialized")
                    
                    # Resolve sample size
                    actual_sample = None if sample_size == "All" else int(sample_size)
                    
                    # ── Run pipeline step-by-step with real progress ──────
                    # Instead of calling pipe.run() as one blocking call,
                    # we execute each stage individually so Streamlit can
                    # update the UI between steps. This way the user sees
                    # real progress instead of a frozen spinner.
                    
                    from src.pipeline import (
                        preprocess_for_transformers, preprocess_for_statistics,
                        load_aspect_classifier, load_sentiment_classifier,
                        extract_aspects, classify_sentiments,
                        build_summary, generate_dashboard,
                        auto_discover_taxonomy, DEFAULT_TAXONOMY
                    )
                    
                    # Step 1: Load data
                    st.write("📂 **Step 1/7:** Loading data...")
                    df = pd.read_csv(tmp_csv_path)
                    if actual_sample:
                        df = df.sample(min(actual_sample, len(df)), random_state=42)
                    df = df.dropna(subset=[text_column]).reset_index(drop=True)
                    st.write(f"   ✅ {len(df):,} reviews loaded")
                    # Add this line right after:  df = df.dropna(subset=[text_column]).reset_index(drop=True)
                    # It creates a standardised 'text' column that all dashboard functions can rely on,
                    # regardless of what the user's original column was called.
                    df['text'] = df[text_column].astype(str)
                    
                    # Step 1.5: Auto-discover taxonomy
                    used_taxonomy = pipe.taxonomy
                    if taxonomy is None and used_taxonomy is DEFAULT_TAXONOMY:
                        st.write("🔍 **Auto-Discovery:** Asking Gemini to discover aspects from your data...")
                        used_taxonomy = auto_discover_taxonomy(
                            df, text_column,
                            sample_size=min(200, len(df)),
                            n_aspects=n_aspects
                        )
                        pipe.taxonomy = used_taxonomy
                        st.write(f"   ✅ Discovered aspects: {list(used_taxonomy.keys())}")
                    
                    # Step 2: Preprocess
                    st.write("🔧 **Step 2/7:** Preprocessing text...")
                    df['text_for_models'] = df[text_column].apply(preprocess_for_transformers)
                    df['text_for_stats']  = df[text_column].apply(preprocess_for_statistics)
                    df = df[df['text_for_models'].apply(lambda x: len(str(x).split()) >= 5)]
                    df = df.reset_index(drop=True)
                    st.write(f"   ✅ {len(df):,} reviews after preprocessing")
                    
                    # Step 3: Load NLP models (this is the slow download step)
                    st.write("⬇️ **Step 3/7:** Loading NLP models (first run downloads ~500MB)...")
                    aspect_clf = load_aspect_classifier()
                    st.write("   ✅ Aspect detection model loaded")
                    sentiment_clf = load_sentiment_classifier()
                    st.write("   ✅ Sentiment classification model loaded")
                    
                    # Step 4: Aspect extraction
                    st.write(f"🔎 **Step 4/7:** Extracting aspects from {len(df):,} reviews...")
                    aspects_list = []
                    aspect_progress = st.progress(0, text="Extracting aspects...")
                    for i, text in enumerate(df['text_for_models']):
                        aspects_list.append(
                            extract_aspects(text, aspect_clf, pipe.taxonomy, threshold)
                        )
                        if (i + 1) % max(1, len(df) // 20) == 0 or i == len(df) - 1:
                            aspect_progress.progress(
                                (i + 1) / len(df),
                                text=f"Aspects: {i+1}/{len(df)} reviews processed"
                            )
                    df['detected_aspects'] = aspects_list
                    df['aspects_parsed'] = df['detected_aspects']
                    aspect_progress.empty()
                    with_asp = df['aspects_parsed'].apply(len).gt(0).sum()
                    st.write(f"   ✅ {with_asp:,}/{len(df):,} reviews had aspects detected")
                    
                    # Step 5: Sentiment classification
                    st.write(f"💬 **Step 5/7:** Classifying sentiment per aspect...")
                    sentiments_list = []
                    sent_progress = st.progress(0, text="Classifying sentiment...")
                    for i, (text, aspects) in enumerate(
                        zip(df['text_for_models'], df['aspects_parsed'])
                    ):
                        sentiments_list.append(
                            classify_sentiments(text, aspects, sentiment_clf)
                        )
                        if (i + 1) % max(1, len(df) // 20) == 0 or i == len(df) - 1:
                            sent_progress.progress(
                                (i + 1) / len(df),
                                text=f"Sentiment: {i+1}/{len(df)} reviews processed"
                            )
                    df['aspect_sentiments'] = sentiments_list
                    df['sentiments_parsed'] = df['aspect_sentiments']
                    sent_progress.empty()
                    st.write("   ✅ Sentiment classification complete")
                    
                    # Step 6: Aggregate
                    st.write("📊 **Step 6/7:** Aggregating results...")
                    summary = build_summary(df)
                    result_df = df
                    os.makedirs(output_dir, exist_ok=True)
                    df.to_csv(os.path.join(output_dir, 'reviews_analyzed.csv'), index=False)
                    summary.to_csv(os.path.join(output_dir, 'aspect_summary.csv'), index=False)
                    st.write("   ✅ Summary tables saved")
                    
                    # Step 7: Generate dashboard
                    st.write("📋 **Step 7/7:** Generating dashboard and report...")
                    generate_dashboard(df, summary, output_dir)
                    st.write("   ✅ Dashboard and report generated")
                    
                    status.update(label="✅ Analysis complete!", state="complete")
                    
                    # ── Display results summary ───────────────────────────
                    st.markdown("---")
                    st.markdown("## 📊 Results Summary")
                    
                    # Top metrics in 4 columns
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Reviews Analyzed", f"{len(result_df):,}")
                    with m2:
                        with_asp = result_df['aspects_parsed'].apply(
                            lambda x: len(x) > 0 if isinstance(x, dict) else False
                        ).sum()
                        st.metric("With Aspects", f"{with_asp:,}")
                    with m3:
                        avg_sat = summary['Satisfaction Rate'].mean()
                        st.metric("Avg Satisfaction", f"{avg_sat:.0f}%",
                                  delta="Overall" )
                    with m4:
                        st.metric("Aspects Found", len(summary))
                    
                    # Satisfaction table
                    st.markdown("### Aspect Satisfaction Ranking")
                    display_summary = summary[[
                        'Aspect', 'Total Mentions',
                        'Positive', 'Negative', 'Satisfaction Rate', 'Signal'
                    ]].copy()
                    
                    st.dataframe(
                        display_summary,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # ── Download buttons ──────────────────────────────────
                    st.markdown("### 📥 Download Your Results")
                    
                    html_path = os.path.join(output_dir, 'stakeholder_dashboard.html')
                    txt_path  = os.path.join(output_dir, 'stakeholder_report.txt')
                    csv_path  = os.path.join(output_dir, 'aspect_summary.csv')
                    
                    d1, d2, d3 = st.columns(3)
                    
                    with d1:
                        if os.path.exists(html_path):
                            st.markdown(
                                create_download_link(
                                    html_path,
                                    "📊 Download Dashboard (HTML)"
                                ),
                                unsafe_allow_html=True
                            )
                            st.caption("Open in any browser — no internet required")
                    
                    with d2:
                        if os.path.exists(txt_path):
                            st.markdown(
                                create_download_link(
                                    txt_path,
                                    "📄 Download Report (TXT)",
                                    mime_type="text/plain"
                                ),
                                unsafe_allow_html=True
                            )
                            st.caption("Copy into slide decks or emails")
                    
                    with d3:
                        if os.path.exists(csv_path):
                            st.markdown(
                                create_download_link(
                                    csv_path,
                                    "📈 Download Summary (CSV)",
                                    mime_type="text/csv"
                                ),
                                unsafe_allow_html=True
                            )
                            st.caption("Use in Excel or Power BI")
                    
                    # Preview of the HTML dashboard inside the app
                    st.markdown("### 👁️ Dashboard Preview")
                    st.info("For the best experience, download the HTML file and open it in your browser.")
                    
                    if os.path.exists(html_path):
                        with open(html_path, 'r', encoding='utf-8') as f:
                            dashboard_html = f.read()
                        st.components.v1.html(dashboard_html, height=600, scrolling=True)
                
                except Exception as e:
                    status.update(label="❌ Pipeline failed", state="error")
                    st.error(f"An error occurred: {str(e)}")
                    st.exception(e)   # shows full traceback for debugging
                
                finally:
                    # Always clean up the temporary file
                    if os.path.exists(tmp_csv_path):
                        os.remove(tmp_csv_path)
    
    else:
        # Show instructions when no file is uploaded yet
        st.markdown("---")
        st.info(
            "👆 Upload a CSV file above to get started. "
            "The pipeline will automatically discover what aspects your customers "
            "write about and produce a full satisfaction dashboard."
        )
        
        # Show example use cases
        st.markdown("### 💡 What This Tool Does")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📥 Input**
            Any CSV with customer reviews — product reviews, restaurant reviews, 
            hotel reviews, app reviews, any text feedback
            """)
        
        with col2:
            st.markdown("""
            **⚙️ Process**
            Auto-discovers product aspects → classifies sentiment per aspect → 
            aggregates across all reviews → generates insights
            """)
        
        with col3:
            st.markdown("""
            **📤 Output**
            Visual HTML dashboard + plain-English report showing which features 
            customers love and which need urgent improvement
            """)

if __name__ == "__main__":
    main()
