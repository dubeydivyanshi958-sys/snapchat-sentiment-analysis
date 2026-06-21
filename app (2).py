import re
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# ---------------------------------------------------------------------------
# Page config & theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Snapchat Review Sentiment",
    page_icon="👻",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#F5D54B"
POSITIVE = "#6FCF97"
NEUTRAL = "#9B9FA8"
NEGATIVE = "#EF6F6C"
BG = "#13151A"
SURFACE = "#1C1F26"
TEXT = "#F2F0E9"
SUBTEXT = "#A7ABB5"

SENTIMENT_COLORS = {"positive": POSITIVE, "neutral": NEUTRAL, "negative": NEGATIVE}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: {TEXT};
}}
.stApp {{
    background-color: {BG};
}}
h1, h2, h3 {{
    font-family: 'Sora', sans-serif !important;
    font-weight: 700 !important;
    color: {TEXT} !important;
}}
[data-testid="stSidebar"] {{
    background-color: {SURFACE};
    border-right: 1px solid #2A2E37;
}}
[data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', monospace;
    color: {ACCENT};
    font-size: 1.9rem !important;
}}
[data-testid="stMetricLabel"] {{
    color: {SUBTEXT} !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
[data-testid="stMetric"] {{
    background-color: {SURFACE};
    border: 1px solid #2A2E37;
    border-radius: 10px;
    padding: 14px 18px;
}}
.hero {{
    border-left: 4px solid {ACCENT};
    padding: 6px 0 6px 18px;
    margin-bottom: 1.2rem;
}}
.hero-sub {{
    color: {SUBTEXT};
    font-size: 1.02rem;
    margin-top: -6px;
}}
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    background-color: {SURFACE};
    border-radius: 8px 8px 0 0;
    color: {SUBTEXT};
    font-family: 'Sora', sans-serif;
    font-weight: 600;
}}
.stTabs [aria-selected="true"] {{
    color: {ACCENT} !important;
    border-bottom: 2px solid {ACCENT} !important;
}}
.insight-box {{
    background-color: {SURFACE};
    border: 1px solid #2A2E37;
    border-left: 3px solid {ACCENT};
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    color: {TEXT};
}}
</style>
""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = go.layout.Template()
PLOTLY_TEMPLATE.layout = go.Layout(
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    font=dict(color=TEXT, family="Inter"),
    title_font=dict(family="Sora", color=TEXT, size=18),
    legend=dict(bgcolor=SURFACE),
    xaxis=dict(gridcolor="#2A2E37", zerolinecolor="#2A2E37"),
    yaxis=dict(gridcolor="#2A2E37", zerolinecolor="#2A2E37"),
)

# ---------------------------------------------------------------------------
# Data & model loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_parquet("data/reviews_dashboard.parquet")
    df["at"] = pd.to_datetime(df["at"])
    return df

@st.cache_resource
def load_model():
    vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
    model = joblib.load("models/sentiment_model.joblib")
    return vectorizer, model

@st.cache_resource
def get_text_tools():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    return set(stopwords.words("english")), WordNetLemmatizer()

def clean_text(text, stop_words, lemmatizer):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

df = load_data()
vectorizer, model = load_model()
stop_words, lemmatizer = get_text_tools()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Filters")

year_min, year_max = int(df["year"].min()), int(df["year"].max())
year_range = st.sidebar.slider("Year range", year_min, year_max, (year_min, year_max))

langs_available = df["language"].value_counts()
top_langs = ["All"] + langs_available[langs_available.index != "too_short"].head(12).index.tolist()
lang_choice = st.sidebar.selectbox("Language", top_langs)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"<span style='color:{SUBTEXT}; font-size:0.85rem;'>"
    "Dataset: 262,717 cleaned Google Play Store reviews of Snapchat (2016–2026). "
    "TF-IDF + Logistic Regression model trained on rating-derived sentiment labels."
    "</span>", unsafe_allow_html=True
)

mask = (df["year"] >= year_range[0]) & (df["year"] <= year_range[1])
if lang_choice != "All":
    mask &= df["language"] == lang_choice
fdf = df[mask]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='hero'><h1>Snapchat Review Sentiment Dashboard</h1>"
    "<div class='hero-sub'>NLP analysis of 260K+ Google Play Store reviews — sentiment, "
    "complaint themes, and trends over a decade of releases.</div></div>",
    unsafe_allow_html=True
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Reviews in view", f"{len(fdf):,}")
c2.metric("Avg. rating", f"{fdf['score'].mean():.2f} ★")
c3.metric("Positive", f"{(fdf['sentiment']=='positive').mean()*100:.1f}%")
c4.metric("Negative", f"{(fdf['sentiment']=='negative').mean()*100:.1f}%")
c5.metric("Neutral", f"{(fdf['sentiment']=='neutral').mean()*100:.1f}%")

st.write("")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Sentiment Deep-Dive", "Complaint Themes", "Trends Over Time", "Try It Yourself"]
)

# ---- Tab 1: Overview ----
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        rating_counts = fdf["score"].value_counts().sort_index()
        fig = px.bar(
            x=rating_counts.index, y=rating_counts.values,
            labels={"x": "Star Rating", "y": "Number of Reviews"},
            title="Star Rating Distribution",
            color=rating_counts.index,
            color_continuous_scale=[NEGATIVE, NEUTRAL, ACCENT, POSITIVE, POSITIVE],
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sent_counts = fdf["sentiment"].value_counts().reindex(["positive", "neutral", "negative"])
        fig = px.pie(
            values=sent_counts.values, names=sent_counts.index,
            title="Sentiment Mix",
            color=sent_counts.index, color_discrete_map=SENTIMENT_COLORS,
            hole=0.45,
        )
        fig.update_layout(template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        yearly = fdf.groupby("year").size()
        fig = px.bar(
            x=yearly.index, y=yearly.values,
            labels={"x": "Year", "y": "Review Count"},
            title="Review Volume by Year",
        )
        fig.update_traces(marker_color=ACCENT)
        fig.update_layout(template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        lang_counts = fdf[fdf["language"] != "too_short"]["language"].value_counts().head(10)
        fig = px.bar(
            x=lang_counts.values, y=lang_counts.index, orientation="h",
            labels={"x": "Number of Reviews", "y": "Language"},
            title="Top Detected Languages",
        )
        fig.update_traces(marker_color=ACCENT)
        fig.update_layout(template=PLOTLY_TEMPLATE, yaxis=dict(autorange="reversed", gridcolor="#2A2E37"))
        st.plotly_chart(fig, use_container_width=True)

# ---- Tab 2: Sentiment Deep-Dive ----
with tab2:
    st.markdown(
        f"<div class='insight-box'>Sentiment labels are derived from star ratings "
        "(1-2★ negative, 3★ neutral, 4-5★ positive). Cross-checked against VADER lexicon-based "
        "sentiment on the full notebook analysis, the two methods agree only ~60-65% of the time — "
        "a reminder that star ratings are an imperfect proxy for what reviewers actually say.</div>",
        unsafe_allow_html=True
    )

    st.markdown("#### Word Clouds by Sentiment")
    wc_cols = st.columns(3)
    cmap = {"positive": "Greens", "neutral": "Greys", "negative": "Reds"}
    for col, sentiment in zip(wc_cols, ["positive", "neutral", "negative"]):
        with col:
            subset = fdf[fdf["sentiment"] == sentiment]["clean_content"].dropna()
            if len(subset) > 0:
                sample_n = min(8000, len(subset))
                text = " ".join(subset.sample(sample_n, random_state=1))
                if text.strip():
                    wc = WordCloud(
                        width=420, height=320, background_color=SURFACE,
                        colormap=cmap[sentiment], max_words=60
                    ).generate(text)
                    fig, ax = plt.subplots(figsize=(4.2, 3.2))
                    fig.patch.set_facecolor(SURFACE)
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")
                    ax.set_title(sentiment.capitalize(), color=TEXT, fontsize=13, fontfamily="sans-serif")
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.info(f"No {sentiment} reviews in current filter")

    st.markdown("#### Sample Reviews")
    sample_sentiment = st.selectbox("Show examples of:", ["positive", "neutral", "negative"], key="sample_sel")
    samples = fdf[fdf["sentiment"] == sample_sentiment].nlargest(5, "thumbsUpCount")[["content", "score", "thumbsUpCount"]]
    for _, row in samples.iterrows():
        st.markdown(
            f"<div class='insight-box'>{'★'*int(row['score'])}{'☆'*(5-int(row['score']))} "
            f"&nbsp;·&nbsp; 👍 {int(row['thumbsUpCount'])}<br>{row['content']}</div>",
            unsafe_allow_html=True
        )

# ---- Tab 3: Complaint Themes ----
with tab3:
    st.markdown(
        f"<div class='insight-box'>Top recurring phrases in negative reviews — the kind of "
        "signal a product or support team would use to prioritize fixes.</div>",
        unsafe_allow_html=True
    )
    neg_text = fdf[fdf["sentiment"] == "negative"]["clean_content"].dropna()
    if len(neg_text) > 20:
        cv = CountVectorizer(ngram_range=(2, 3), max_features=25, stop_words="english")
        try:
            X_ngrams = cv.fit_transform(neg_text)
            freqs = X_ngrams.sum(axis=0).A1
            terms = cv.get_feature_names_out()
            top = sorted(zip(terms, freqs), key=lambda x: -x[1])[:18]
            complaint_df = pd.DataFrame(top, columns=["phrase", "count"]).iloc[::-1]
            fig = px.bar(
                complaint_df, x="count", y="phrase", orientation="h",
                title="Top Phrases in Negative Reviews",
            )
            fig.update_traces(marker_color=NEGATIVE)
            fig.update_layout(template=PLOTLY_TEMPLATE, height=550)
            st.plotly_chart(fig, use_container_width=True)
        except ValueError:
            st.info("Not enough negative review text in this filter to extract themes.")
    else:
        st.info("Not enough negative reviews in this filter to extract themes.")

# ---- Tab 4: Trends Over Time ----
with tab4:
    yearly_sentiment = fdf.groupby(["year", "sentiment"]).size().unstack(fill_value=0)
    yearly_pct = yearly_sentiment.div(yearly_sentiment.sum(axis=1), axis=0) * 100
    yearly_pct = yearly_pct.reindex(columns=["negative", "neutral", "positive"], fill_value=0)

    fig = go.Figure()
    for sentiment in ["negative", "neutral", "positive"]:
        fig.add_trace(go.Scatter(
            x=yearly_pct.index, y=yearly_pct[sentiment],
            mode="lines", stackgroup="one", name=sentiment,
            line=dict(color=SENTIMENT_COLORS[sentiment]),
        ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, title="Sentiment Mix by Year (%)",
        xaxis_title="Year", yaxis_title="Share of Reviews (%)"
    )
    st.plotly_chart(fig, use_container_width=True)

    avg_rating_yearly = fdf.groupby("year")["score"].mean()
    fig2 = px.line(
        x=avg_rating_yearly.index, y=avg_rating_yearly.values,
        labels={"x": "Year", "y": "Average Rating"},
        title="Average Star Rating by Year", markers=True,
    )
    fig2.update_traces(line_color=ACCENT, marker=dict(size=8))
    fig2.update_layout(template=PLOTLY_TEMPLATE, yaxis_range=[1, 5])
    st.plotly_chart(fig2, use_container_width=True)

# ---- Tab 5: Try It Yourself ----
with tab5:
    st.markdown("#### Try the sentiment classifier")
    st.markdown(
        f"<span style='color:{SUBTEXT}'>Type a review-style sentence and see how the trained "
        "TF-IDF + Logistic Regression model classifies it.</span>", unsafe_allow_html=True
    )
    user_text = st.text_area("Your review text:", placeholder="e.g. the new update keeps crashing my phone", height=100)

    if st.button("Classify sentiment", type="primary"):
        if user_text.strip():
            cleaned = clean_text(user_text, stop_words, lemmatizer)
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]
            proba = model.predict_proba(vec)[0]
            classes = model.classes_

            color = SENTIMENT_COLORS[pred]
            st.markdown(
                f"<h2 style='color:{color}'>{pred.upper()}</h2>", unsafe_allow_html=True
            )

            prob_df = pd.DataFrame({"sentiment": classes, "probability": proba})
            fig = px.bar(
                prob_df, x="probability", y="sentiment", orientation="h",
                color="sentiment", color_discrete_map=SENTIMENT_COLORS,
                title="Model Confidence",
            )
            fig.update_layout(template=PLOTLY_TEMPLATE, showlegend=False, height=300)
            fig.update_xaxes(range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Type something first!")

st.markdown(
    f"<div style='text-align:center; color:{SUBTEXT}; padding-top:20px; font-size:0.85rem;'>"
    "Built with Streamlit + Plotly · TF-IDF/Logistic Regression model trained on 262K+ labeled reviews"
    "</div>", unsafe_allow_html=True
)
