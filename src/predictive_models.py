import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from scipy.sparse import hstack
from sklearn.utils import resample
from textblob import TextBlob


# Load in full datasets
def load_datasets(train_path, val_path, test_path):
    """
    Load merged LiarPLUS/Politifact datasets (train, validation, test).
    """

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    return train_df, val_df, test_df

# Frequency Heuristic Model
def build_frequency_model(df_train):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix_train = tfidf.fit_transform(df_train['statement'])

    count_vec = CountVectorizer(stop_words='english')
    count_matrix_train = count_vec.fit_transform(df_train['statement'])
    token_freq = np.asarray(count_matrix_train.sum(axis=0)).ravel()
    token_dict = {w: token_freq[i] for i, w in enumerate(count_vec.get_feature_names_out())}

    buzzwords = {'always','never','everyone','nobody','millions','billions','every',
                 'no one','thousands','people say','experts agree'}

    def avg_word_freq(text):
        words = [w for w in text.lower().split() if w in token_dict]
        return np.mean([token_dict[w] for w in words]) if words else 0

    def buzzword_score(text):
        return sum(b in text.lower() for b in buzzwords)

    def repetition_score(text):
        tokens = text.lower().split()
        return 1 - len(set(tokens)) / len(tokens) if tokens else 0

    X_train = pd.DataFrame({
        "tfidf_mean": tfidf_matrix_train.mean(axis=1).A1,
        "word_freq_mean": df_train['statement'].apply(avg_word_freq),
        "buzzword_score": df_train['statement'].apply(buzzword_score),
        "repetition_score": df_train['statement'].apply(repetition_score)
    }).fillna(0)

    le = LabelEncoder()
    y_train = le.fit_transform(df_train['label'])

    model = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=200, random_state=42))])
    model.fit(X_train, y_train)

    return model, tfidf, count_vec, token_dict, buzzwords, le

def predict_frequency_model(df, model, tfidf, count_vec, token_dict, buzzwords, le):
    def avg_word_freq(text):
        words = [w for w in text.lower().split() if w in token_dict]
        return np.mean([token_dict[w] for w in words]) if words else 0

    def buzzword_score(text):
        return sum(b in text.lower() for b in buzzwords)

    def repetition_score(text):
        tokens = text.lower().split()
        return 1 - len(set(tokens)) / len(tokens) if tokens else 0

    tfidf_matrix = tfidf.transform(df['statement'])
    X = pd.DataFrame({
        "tfidf_mean": tfidf_matrix.mean(axis=1).A1,
        "word_freq_mean": df['statement'].apply(avg_word_freq),
        "buzzword_score": df['statement'].apply(buzzword_score),
        "repetition_score": df['statement'].apply(repetition_score)
    }).fillna(0)

    preds = model.predict(X)
    probs = model.predict_proba(X).max(axis=1)
    pred_labels = le.inverse_transform(preds)

    label_to_score = {
        "true": 2,
        "mostly-true": 2,
        "half-true": 1,
        "barely-true": 0,
        "false": 0,
        "pants-on-fire": 0
    }

    freq_scores = [label_to_score.get(lbl, 1) for lbl in pred_labels]

    return pd.DataFrame({
        "id": df["id"],
        "statement": df["statement"],
        "predicted_frequency_heuristic": freq_scores,
        "frequency_heuristic_score": probs
    })


# Sensationalism Model
def build_sensationalism_model(df_train, df_val, df_test, n_estimators=300, max_depth=6,
                               learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42):
    """
    Build a sensationalism detection model. Identifies sensationalism by analyzing text
    features, sentiment analysis, and metadata features
    """
    def map_sensationalism_from_counts(row):
        """
        Map LiarPLUS rating counts to sensationalism levels. Higher sensationalism 
        score = more extreme/false ratings.
        """
        total = row["barely_true"] + row["false"] + row["half_true"] + row["mostly_true"] + row["pants_on_fire"]
        if total == 0:
            return 1
        score = (
            2 * row["barely_true"] +
            3 * row["false"] +
            1 * row["half_true"] +
            0.5 * row["mostly_true"] +
            4 * row["pants_on_fire"] 
        ) / total
        if score < 1.5:
            return 0  # Low sensationalism
        elif score < 2.5:
            return 1  # Moderate sensationalism
        else:
            return 2  # High sensationalism

    for df in [df_train, df_val, df_test]:
        df["sensationalism"] = df.apply(map_sensationalism_from_counts, axis=1)

    def extract_text_features(text):
        text = str(text)
        exclaim = text.count("!")
        allcaps = len(re.findall(r"\b[A-Z]{2,}\b", text))
        sens_words = sum(1 for w in [
            "shocking","unbelievable","incredible","amazing","outrageous",
            "disaster","terrifying","massive","horrifying","explosive",
            "record-breaking","unprecedented","urgent","worst","best"
        ] if w in text.lower())
        blob = TextBlob(text)
        return exclaim, allcaps, sens_words, abs(blob.sentiment.polarity), blob.sentiment.subjectivity

    feats = df_train["statement"].apply(extract_text_features)
    df_train[["exclaim","allcaps","sens_words","polarity","subjectivity"]] = pd.DataFrame(feats.tolist(), index=df_train.index)

    text_col = "statement"
    meta_features = ["speaker", "context", "job"]
    numeric_features = ["exclaim", "allcaps", "sens_words", "polarity", "subjectivity"]

    preprocessor = ColumnTransformer([
        ("text", TfidfVectorizer(max_features=5000, stop_words="english"), text_col),
        ("cat", OneHotEncoder(handle_unknown="ignore"), meta_features),
        ("num", StandardScaler(), numeric_features)
    ])

    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        eval_metric="mlogloss"
    )

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    X_train = df_train[[text_col] + meta_features + numeric_features]
    y_train = df_train["sensationalism"]

    pipeline.fit(X_train, y_train)

    return pipeline, meta_features, numeric_features

def predict_sensationalism_model(df, pipeline, meta_features, numeric_features):
    """
    Make predictions using the sensationalism model.
    """
    df = df.copy()

    def extract_text_features(text):
        text = str(text)
        exclaim = text.count("!")
        allcaps = len(re.findall(r"\b[A-Z]{2,}\b", text))
        sens_words = sum(1 for w in [
            "shocking","unbelievable","incredible","amazing","outrageous",
            "disaster","terrifying","massive","horrifying","explosive",
            "record-breaking","unprecedented","urgent","worst","best"
        ] if w in text.lower())
        blob = TextBlob(text)
        return exclaim, allcaps, sens_words, abs(blob.sentiment.polarity), blob.sentiment.subjectivity

    feats = df["statement"].apply(extract_text_features)
    df[["exclaim","allcaps","sens_words","polarity","subjectivity"]] = pd.DataFrame(feats.tolist(), index=df.index)

    X = df[["statement"] + meta_features + numeric_features]
    
    # Make predictions
    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X).max(axis=1)

    return pd.DataFrame({
        "id": df["id"],
        "statement": df["statement"],
        "predicted_sensationalism": preds,
        "sensationalism_score": probs
    })


# Malicious Account
def build_malicious_account_model(df_train):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix_train = tfidf.fit_transform(df_train['statement'])

    def avg_token_length(text):
        tokens = text.split()
        return np.mean([len(t) for t in tokens]) if tokens else 0

    def repetition_score(text):
        tokens = text.lower().split()
        return 1 - len(set(tokens)) / len(tokens) if tokens else 0

    def link_count(text):
        return text.count('http') + text.count('www.')

    def hashtag_mention_count(text):
        return text.count('#') + text.count('@')

    def punctuation_ratio(text):
        punct = sum(1 for c in text if c in "!?.,")
        return punct / len(text) if len(text) > 0 else 0

    def uppercase_ratio(text):
        upper = sum(1 for c in text if c.isupper())
        return upper / len(text) if len(text) > 0 else 0

    X_train = pd.DataFrame({
        "tfidf_mean": tfidf_matrix_train.mean(axis=1).A1,
        "avg_token_length": df_train['statement'].apply(avg_token_length),
        "repetition_score": df_train['statement'].apply(repetition_score),
        "link_count": df_train['statement'].apply(link_count),
        "hashtag_mention_count": df_train['statement'].apply(hashtag_mention_count),
        "punctuation_ratio": df_train['statement'].apply(punctuation_ratio),
        "uppercase_ratio": df_train['statement'].apply(uppercase_ratio)
    }).fillna(0)

    le = LabelEncoder()
    y_train = le.fit_transform(df_train['label'])

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    model.fit(X_train, y_train)

    return model, tfidf, le

def predict_malicious_account_model(df, model, tfidf, le):
    def avg_token_length(text):
        tokens = text.split()
        return np.mean([len(t) for t in tokens]) if tokens else 0

    def repetition_score(text):
        tokens = text.lower().split()
        return 1 - len(set(tokens)) / len(tokens) if tokens else 0

    def link_count(text):
        return text.count('http') + text.count('www.')

    def hashtag_mention_count(text):
        return text.count('#') + text.count('@')

    def punctuation_ratio(text):
        punct = sum(1 for c in text if c in "!?.,")
        return punct / len(text) if len(text) > 0 else 0

    def uppercase_ratio(text):
        upper = sum(1 for c in text if c.isupper())
        return upper / len(text) if len(text) > 0 else 0

    tfidf_matrix = tfidf.transform(df['statement'])
    X = pd.DataFrame({
        "tfidf_mean": tfidf_matrix.mean(axis=1).A1,
        "avg_token_length": df['statement'].apply(avg_token_length),
        "repetition_score": df['statement'].apply(repetition_score),
        "link_count": df['statement'].apply(link_count),
        "hashtag_mention_count": df['statement'].apply(hashtag_mention_count),
        "punctuation_ratio": df['statement'].apply(punctuation_ratio),
        "uppercase_ratio": df['statement'].apply(uppercase_ratio)
    }).fillna(0)

    preds = model.predict(X)
    probs = model.predict_proba(X).max(axis=1)
    pred_labels = le.inverse_transform(preds)

    label_to_score = {
        "true": 2,
        "mostly-true": 2,
        "half-true": 1,
        "barely-true": 0,
        "false": 0,
        "pants-on-fire": 0
    }

    malicious_scores = [label_to_score.get(lbl, 1) for lbl in pred_labels]

    return pd.DataFrame({
        "id": df["id"],
        "statement": df["statement"],
        "predicted_malicious_account": malicious_scores,
        "malicious_account_score": probs
    })

# Naive Realism
def map_naive_realism_from_sentiment(text):
    blob = TextBlob(str(text))
    subj = blob.sentiment.subjectivity
    polarity = abs(blob.sentiment.polarity)
    score = subj + polarity

    if score < 0.4:
        return 0   # balanced / open-minded
    elif score < 0.8:
        return 1   # somewhat naive-realist
    else:
        return 2   # strongly naive-realist
    
def extract_naive_realism_features(text):

    text = str(text)
    words = text.lower().split()

    cautious_words = ["maybe", "perhaps", "possibly", "likely", "suggests", "could", "might"]
    absolutes = ["always", "never", "everyone", "nobody", "clearly", "undeniably"]
    cautious_ratio = sum(w in cautious_words for w in words) / max(len(words), 1)
    absolute_ratio = sum(w in absolutes for w in words) / max(len(words), 1)

    dismissive_terms = ["idiot", "fool", "biased", "brainwashed", "fake", "delusional"]
    dismissive_count = sum(w in text.lower() for w in dismissive_terms)

    return (
        absolute_ratio,
        cautious_ratio,
        dismissive_count
    )

def build_naive_realism_model(df_train, df_val, df_test):
    for df in [df_train, df_val, df_test]:
        df["naive_realism"] = df["statement"].apply(map_naive_realism_from_sentiment)

    feats = df_train["statement"].apply(extract_naive_realism_features)
    df_train[["absolute_ratio", "cautious_ratio", "dismissive_count"]] = pd.DataFrame(
        feats.tolist(), index=df_train.index
    )

    text_col = "statement"
    meta_features = ["speaker", "context", "job"]
    numeric_features = ["absolute_ratio", "cautious_ratio", "dismissive_count"]

    preprocessor = ColumnTransformer([
        ("text", TfidfVectorizer(max_features=5000, stop_words="english"), text_col),
        ("cat", OneHotEncoder(handle_unknown="ignore"), meta_features),
        ("num", StandardScaler(), numeric_features)
    ])

    model = XGBClassifier(
        num_class=3,
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss"
    )

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    X_train = df_train[[text_col] + meta_features + numeric_features]
    y_train = df_train["naive_realism"]
    pipeline.fit(X_train, y_train)

    return pipeline, meta_features, numeric_features

def predict_naive_realism_model(df, pipeline, meta_features, numeric_features):
    df = df.copy()

    feats = df["statement"].apply(extract_naive_realism_features)
    df[["absolute_ratio", "cautious_ratio", "dismissive_count"]] = pd.DataFrame(
        feats.tolist(), index=df.index
    )

    X = df[["statement"] + meta_features + numeric_features]
    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X).max(axis=1)

    return pd.DataFrame({
        "id": df["id"],
        "statement": df["statement"],
        "predicted_naive_realism": preds,
        "naive_realism_score": probs
    })





