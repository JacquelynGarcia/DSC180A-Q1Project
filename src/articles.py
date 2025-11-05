import pandas as pd
import spacy
from newspaper import Article
from urllib.parse import urlparse
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from predictive_models import (
    predict_frequency_model,
    predict_echo_chamber_model,
    predict_sensationalism_model,
    predict_credibility_model
)

nlp = spacy.load("en_core_web_sm")

def fetch_article(url):

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

    # use Article library to parse url
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

# train classifiers to extract the "job" and "party" feature from new articles
def train_party_model(df_train):
    party_df = df_train.dropna(subset=["party", "statement"])
    party_le = LabelEncoder()
    y_party = party_le.fit_transform(party_df["party"])

    party_clf = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
        ("logreg", LogisticRegression(max_iter=300, class_weight="balanced"))
    ])

    party_clf.fit(party_df["statement"], y_party)

    return party_clf, party_le


def train_job_model(df_train):
    job_df = df_train.dropna(subset=["job", "statement"])
    job_le = LabelEncoder()
    y_job = job_le.fit_transform(job_df["job"])

    job_clf = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
        ("logreg", LogisticRegression(max_iter=300, class_weight="balanced"))
    ])

    job_clf.fit(job_df["statement"], y_job)
    return job_clf, job_le

def predict_party(text, party_clf, party_le):
    probs = party_clf.predict_proba([text])[0]
    max_prob = probs.max()
    pred_label = party_le.inverse_transform([probs.argmax()])[0]
    return pred_label


def predict_job(text, job_clf, job_le):
    probs = job_clf.predict_proba([text])[0]
    max_prob = probs.max()
    pred_label = job_le.inverse_transform([probs.argmax()])[0]
    return pred_label


def train_party_and_job_models(df_train, save_dir=None):
    party_clf, party_le = train_party_model(df_train)
    job_clf, job_le = train_job_model(df_train)
    return (party_clf, party_le), (job_clf, job_le)

def prepare_article_for_models(article, job_clf, job_le, party_clf, party_le):
    text = article.get("text", "").strip()
    title = article.get("title", "")
    source = article.get("source", "")
    authors = article.get("authors", [])
    author = authors[0] if authors else ""
    publish_date = article.get("publish_date", "")
    keywords = article.get("keywords", [])

    doc = nlp(text)
    ents = [ent.text for ent in doc.ents if ent.label_ in {"PERSON","ORG","GPE"}]
    subject = ", ".join(sorted(set(keywords + ents)))[:300]
    t = text.lower()

    job = predict_job(text, job_clf, job_le)
    party = predict_party(text, party_clf, party_le)

    context = f"Article from {source} published {publish_date}"

    is_political = int("politic" in t or "election" in t or "senate" in t or "president" in t)
    subject_length = len(subject.split(",")) if subject else 0

    df = pd.DataFrame([{
        "id": "custom_001",
        "statement": text,
        "subject": subject,
        "speaker": author,
        "job": job,
        "party": party,
        "context": context,
        "subject_length": subject_length,
        "is_political": is_political,
        "source": source,
        "publish_date": publish_date
    }])

    return df

def evaluate_article(url, freq_model, echo_model, sens_model, cred_model, job_party_model):
    """
    Evaluate an article using all trained models
    
    Args:
        url: URL of the article to analyze
        freq_model: tuple of (model_freq, tfidf_freq, count_vec_freq, token_dict_freq, buzzwords_freq, le_freq)
        echo_model: tuple of (model_echo, vectorizer_echo, le_echo, concentration_map)
        sens_model: tuple of (sens_pipeline, sens_preproc, sens_meta, sens_num)
        cred_model: tuple of (cred_pipeline, party_enc_cred)
        job_party_model: tuple of (job_clf, job_le, party_clf, party_le)
    """
    article = fetch_article(url)
    job_clf, job_le, party_clf, party_le = job_party_model
    df = prepare_article_for_models(article, job_clf, job_le, party_clf, party_le)

    model_freq, tfidf_freq, count_vec_freq, token_dict_freq, buzzwords_freq, le_freq = freq_model
    model_echo, vectorizer_echo, le_echo, concentration_map = echo_model
    sens_pipeline, sens_preproc, sens_meta, sens_num = sens_model
    cred_pipeline, party_enc_cred = cred_model  

    freq_result = predict_frequency_model(df, model_freq, tfidf_freq, count_vec_freq, token_dict_freq, buzzwords_freq, le_freq).iloc[0]
    echo_result = predict_echo_chamber_model(df, model_echo, vectorizer_echo, le_echo, concentration_map).iloc[0]
    sens_result = predict_sensationalism_model(df, sens_pipeline, sens_preproc, sens_meta, sens_num).iloc[0]
    cred_result = predict_credibility_model(df, cred_pipeline, party_enc_cred).iloc[0]

    return {
        "url": url,
        "source": article.get("source", ""),
        "title": article.get("title", ""),
        "predicted_label": freq_result["predicted_label"],
        "frequency_heuristic_score": freq_result["frequency_heuristic_score"],
        "predicted_echo_class": echo_result["predicted_echo_class"],
        "echo_chamber_score": echo_result["echo_chamber_score"],
        "predicted_sensationalism": sens_result["predicted_sensationalism"],
        "sensationalism_score": sens_result["sensationalism_score"],
        "predicted_credibility": cred_result["predicted_credibility"],
        "credibility_score": cred_result["credibility_score"],
    }


