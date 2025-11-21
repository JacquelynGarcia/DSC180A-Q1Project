import pandas as pd
import spacy
from newspaper import Article
from urllib.parse import urlparse
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from src.predictive_models import (
    predict_frequency_model,
    predict_echo_chamber_model,
    predict_sensationalism_model,
    predict_credibility_model
)

# Load spaCy English language model for NLP processing
nlp = spacy.load("en_core_web_sm")

def fetch_article(url):
    """
    Fetch and parse an article from a given URL.
    """
    article_data = {
        "url": url,
        "title": "",
        "text": "",
        "authors": [],
        "publish_date": None,
        "source": "",
        "keywords": [],
        "error": None
    }

    # Use newspaper3k library to download and parse the article
    article = Article(url)
    article.download()
    article.parse()

    # extract data from parsed article
    article_data["title"] = article.title or ""
    article_data["text"] = article.text or ""
    article_data["authors"] = article.authors or []
    article_data["source"] = urlparse(url).netloc # extracts data source from within the given url
    article_data["publish_date"] = (
        pd.to_datetime(article.publish_date).date().isoformat()
        if article.publish_date else None
    )

    # get all people, organizations, and places from the article
    doc = nlp(article_data["title"] + " " + article_data["text"])
    keywords = [ent.text for ent in doc.ents if ent.label_ in {"ORG", "GPE", "PERSON"}]
    article_data["keywords"] = list(set(keywords))  

    return article_data


def train_job_model(df_train):
    """
    Train a classifier to predict job title from text.
    """
    # Filter out rows with missing job or statement data
    job_df = df_train.dropna(subset=["job", "statement"])
    job_le = LabelEncoder()
    y_job = job_le.fit_transform(job_df["job"])

    job_clf = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
        ("logreg", LogisticRegression(max_iter=300, class_weight="balanced"))
    ])

    job_clf.fit(job_df["statement"], y_job)
    return job_clf, job_le


def predict_job(text, job_clf, job_le):
    """
    Predict job title from text.
    """
    probs = job_clf.predict_proba([text])[0]
    pred_label = job_le.inverse_transform([probs.argmax()])[0]
    return pred_label


def train_models(df_train):
    """
    Train job classifier
    """
    job_clf, job_le = train_job_model(df_train)
    return (job_clf, job_le)

def prepare_article_for_models(article, job_clf, job_le):
    """
    Convert article data into the format expected by the trained frequency, echo chamber,
    sensationalism, and credibility models.
    """
    # Extract basic article information
    text = article.get("text", "").strip()
    title = article.get("title", "")
    source = article.get("source", "")
    authors = article.get("authors", [])
    author = authors[0] if authors else "" 
    publish_date = article.get("publish_date", "")
    keywords = article.get("keywords", [])

    # Extract named entities from the text using spaCy
    doc = nlp(text)
    ents = [ent.text for ent in doc.ents if ent.label_ in {"PERSON","ORG","GPE"}]
    subject = ", ".join(sorted(set(keywords + ents)))[:300]
    t = text.lower()
    job = predict_job(text, job_clf, job_le)
    context = f"Article from {source} published {publish_date}"
    is_political = int("politic" in t or "election" in t or "senate" in t or "president" in t)
    subject_length = len(subject.split(",")) if subject else 0

    df = pd.DataFrame([{
        "id": "custom_001",                    # Unique identifier
        "statement": text,                     # Main text content
        "subject": subject,                    # Topic/subject of article
        "speaker": author,                     # Article author
        "job": job,                           # Predicted job title
        "context": context,                   # Publication context
        "subject_length": subject_length,     # Subject complexity metric
        "is_political": is_political,         # Political content
        "source": source,                     # Source domain
        "publish_date": publish_date          # Publication date
    }])

    return df

def evaluate_article(url, freq_model, echo_model,sens_model, cred_model, job_model):
    """
    Evaluate an article using all trained models
    """
    article = fetch_article(url)
    job_clf, job_le = job_model
    df = prepare_article_for_models(article, job_clf, job_le)

    model_freq, tfidf_freq, count_vec_freq, token_dict_freq, buzzwords_freq, le_freq = freq_model
    model_echo, vectorizer_echo, le_echo, concentration_map = echo_model
    sens_pipeline, sens_meta, sens_num = sens_model
    cred_pipeline, party_enc_cred = cred_model

    freq_result = predict_frequency_model(df, model_freq, tfidf_freq, count_vec_freq, token_dict_freq, buzzwords_freq, le_freq).iloc[0]
    echo_result = predict_echo_chamber_model(df, model_echo, vectorizer_echo, le_echo, concentration_map).iloc[0]
    sens_result = predict_sensationalism_model(df, sens_pipeline, sens_meta, sens_num).iloc[0]
    cred_result = predict_credibility_model(df, cred_pipeline, party_enc_cred).iloc[0]

    return {
        "url": url, # Article metadata
        "source": article.get("source", ""),
        "title": article.get("title", ""),
        "predicted_label": freq_result["predicted_label"], # Frequency heuristic results
        "frequency_heuristic_score": freq_result["frequency_heuristic_score"],
        "predicted_echo_class": echo_result["predicted_echo_class"], # Echo chamber results
        "echo_chamber_score": echo_result["echo_chamber_score"],
        "predicted_sensationalism": sens_result["predicted_sensationalism"], # Sensationalism results
        "sensationalism_score": sens_result["sensationalism_score"],
        "predicted_credibility": cred_result["predicted_credibility"], # Credibility results
        "credibility_score": cred_result["credibility_score"]
    }


