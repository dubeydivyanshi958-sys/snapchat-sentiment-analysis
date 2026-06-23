# Snapchat Review Sentiment Analysis

NLP project analyzing 280,000+ Google Play Store reviews of Snapchat — sentiment classification, complaint-theme extraction, and a decade of sentiment trends, shipped as a Jupyter notebook **and** a live interactive dashboard.

**[Live Dashboard →](#)** &nbsp;|&nbsp; **[Analysis Notebook →](notebook/snapchat_sentiment_analysis.ipynb)**

---

## What this project does

- Classifies review sentiment (positive / neutral / negative) using rating-derived labels, cross-validated against VADER lexicon-based sentiment
- Detects review language across a global, multilingual user base (60+ languages)
- Trains and compares TF-IDF + classical ML (Logistic Regression, Naive Bayes) against a DistilBERT transformer baseline
- Extracts recurring complaint themes from negative reviews via n-gram analysis — the kind of signal a product team could act on directly
- Tracks sentiment trends across a 10-year review history (2016–2026)
- Ships as an interactive Streamlit dashboard with a live sentiment classifier you can type into

## Key findings

| Finding | Detail |
|---|---|
| Sentiment mix | ~57% positive, ~34% negative, ~9% neutral |
| Rating vs. text agreement | Only ~60–65% agreement between star rating and VADER text sentiment — star ratings alone are an imperfect ground truth |
| Model performance | TF-IDF + Naive Bayes: 78.5% accuracy · Logistic Regression: 73.1% accuracy, stronger F1 balance across classes |
| Hardest class | Neutral reviews are consistently hardest to classify — a known, documented challenge in sentiment analysis, not a modeling flaw |
| Top complaint themes | App updates, dark mode requests, account deletion friction |

## Tech stack

`Python` · `pandas` · `scikit-learn` · `NLTK` · `langdetect` · `VADER` · `HuggingFace Transformers` · `Streamlit` · `Plotly` · `WordCloud`

## Repo structure

```
.
├── notebook/
│   └── snapchat_sentiment_analysis.ipynb   # Full analysis, executed with outputs
├── app/
│   ├── app.py                              # Streamlit dashboard
│   ├── requirements.txt
│   ├── .streamlit/config.toml              # Theme config
│   ├── data/reviews_dashboard.parquet      # Cleaned data used by the dashboard
│   └── models/                             # Trained TF-IDF vectorizer + classifier
└── data/
    └── snapchat_reviews.csv                # Raw dataset (see note below)
```

## Running locally

**Notebook:**
```bash
conda create -n snapchat-sentiment python=3.11
conda activate snapchat-sentiment
pip install pandas numpy matplotlib seaborn wordcloud scikit-learn nltk langdetect vaderSentiment transformers torch jupyter
jupyter notebook notebook/snapchat_sentiment_analysis.ipynb
```
The DistilBERT comparison cell downloads a pre-trained model from HuggingFace on first run (~270MB, cached afterward) — requires internet access.

**Dashboard:**
```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

## Data note

The raw and intermediate CSVs in `data/` are large (~50–70MB each). If you're cloning this repo and only want to run the dashboard, you don't need them — `app/data/reviews_dashboard.parquet` is the only file the app reads. For full notebook reproducibility, consider [Git LFS](https://git-lfs.com/) if re-hosting this repo with the raw data included.

Reviewer usernames were dropped from the dashboard's dataset for privacy; they remain in the raw CSV used by the notebook.

## Possible extensions

- Fine-tune a transformer directly on the 3-class labels for stronger neutral-class performance
- Correlate sentiment dips with specific `appVersion` releases to flag problematic updates
- Add topic modeling (LDA/BERTopic) for deeper theme discovery beyond n-grams
