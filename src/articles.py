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

# Load spaCy English language model for NLP processing
nlp = spacy.load("en_core_web_sm")

def fetch_article(url):
    """
    Fetch and parse an article from a given URL.
    
    Args:
        url (str): The URL of the article to fetch
        
    Returns:
        dict: Article data containing title, text, authors, publish_date, 
              source, keywords, and error information
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

# train classifiers to extract the "job" and "party" feature from new articles
def train_party_model(df_train):
    """
    Train a classifier to predict political party from text.
    Args:
        df_train: Training dataset with 'party' and 'statement' columns
        
    Returns:
        tuple: (trained_classifier, label_encoder)
    """
    # Filter out rows with missing party or statement data
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
    """
    Train a classifier to predict job title from text.
    
    Args:
        df_train: Training dataset with 'job' and 'statement' columns
        
    Returns:
        tuple: (trained_classifier, label_encoder)
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

def predict_party(text, party_clf, party_le):
    """
    Predict political party affiliation from text.
    
    Args:
        text (str): Text to analyze
        party_clf: Trained party classifier
        party_le: Label encoder for party labels
        
    Returns:
        str: Predicted party label
    """
    probs = party_clf.predict_proba([text])[0]
    pred_label = party_le.inverse_transform([probs.argmax()])[0]
    return pred_label


def predict_job(text, job_clf, job_le):
    """
    Predict job title from text.
    
    Args:
        text: Text to analyze
        job_clf: Trained job classifier
        job_le: Label encoder for job labels
        
    Returns:
        str: Predicted job label
    """
    probs = job_clf.predict_proba([text])[0]
    pred_label = job_le.inverse_transform([probs.argmax()])[0]
    return pred_label


def train_party_and_job_models(df_train):
    """
    Train both party and job classification models.
    
    Args:
        df_train: Training dataset
        
    Returns:
        tuple: ((party_classifier, party_encoder), (job_classifier, job_encoder))
    """
    party_clf, party_le = train_party_model(df_train)
    job_clf, job_le = train_job_model(df_train)
    return (party_clf, party_le), (job_clf, job_le)

def prepare_article_for_models(article, job_clf, job_le, party_clf, party_le):
    """
    Convert article data into the format expected by the trained frequency, echo chamber,
    sensationalism, and credibility models.
    
    Args:
        article: Article data from fetch_article()
        job_clf: Trained job classifier
        job_le: Job label encoder
        party_clf: Trained party classifier
        party_le: Party label encoder
        
    Returns:
        Single-row DataFrame with model features
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
    party = predict_party(text, party_clf, party_le)
    context = f"Article from {source} published {publish_date}"
    is_political = int("politic" in t or "election" in t or "senate" in t or "president" in t)
    subject_length = len(subject.split(",")) if subject else 0

    df = pd.DataFrame([{
        "id": "custom_001",                    # Unique identifier
        "statement": text,                     # Main text content
        "subject": subject,                    # Topic/subject of article
        "speaker": author,                     # Article author
        "job": job,                           # Predicted job title
        "party": party,                       # Predicted party affiliation
        "context": context,                   # Publication context
        "subject_length": subject_length,     # Subject complexity metric
        "is_political": is_political,         # Political content
        "source": source,                     # Source domain
        "publish_date": publish_date          # Publication date
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


