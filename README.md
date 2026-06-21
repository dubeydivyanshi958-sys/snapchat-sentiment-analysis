# Snapchat App Store Review Sentiment Analysis

NLP portfolio project analyzing 280,000+ Google Play Store reviews of Snapchat to classify
sentiment, compare classical ML vs. transformer-based approaches, and extract actionable
complaint themes.

## What's inside

- `notebook/snapchat_sentiment_analysis.ipynb` — the full analysis, fully executed with outputs
- `data/snapchat_reviews.csv` — raw dataset (280,539 reviews)
- `data/snapchat_reviews_with_lang.csv` — dataset with detected language per review (cached, since
  language detection on the full dataset takes several minutes)
- `data/snapchat_reviews_clean.csv` — cleaned dataset used for modeling

## Key techniques

- Rating-derived sentiment labeling, cross-validated against VADER lexicon-based sentiment
- Language detection across 280K+ multilingual reviews
- TF-IDF + Logistic Regression / Naive Bayes classification
- DistilBERT transformer comparison (requires internet access to download model — see notebook
  note)
- Word clouds, n-gram complaint theme extraction, sentiment trend analysis over a 10-year period

## Running it yourself

```bash
conda create -n app-review-sentiment python=3.11
conda activate app-review-sentiment
pip install pandas numpy matplotlib seaborn wordcloud scikit-learn nltk langdetect vaderSentiment transformers torch jupyter
jupyter notebook notebook/snapchat_sentiment_analysis.ipynb
```

The DistilBERT comparison cell (Section 8) requires internet access to download the pre-trained
model from HuggingFace on first run (~270MB, cached afterward).

## Key findings

- Reviews skew positive (~57%) but with substantial negative volume (~34%) for complaint analysis
- Star rating and text-based sentiment agree only ~60-65% of the time — a useful caveat against
  treating ratings as perfectly clean ground truth
- Classical ML performs well on clear positive/negative reviews; the neutral class remains hardest
  to classify (a known challenge in sentiment analysis, not a modeling flaw)
- Top complaint themes: app updates, dark mode requests, account deletion friction
