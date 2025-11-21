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


# Load in full LiarPLUS datasets
def load_datasets(train_path, val_path, test_path):
    """
    Load and preprocess the LIAR-PLUS datasets (training, validation, and test sets).
    """
    # Define column names for the LiarPLUS dataset
    cols = [
    "index", "id", "label", "statement", "subject", "speaker", "job", "state",
    "party", "barely_true", "false", "half_true", "mostly_true", "pants_on_fire",
    "context", "justification"
    ]

    dfs = []

    # Process each dataset file
    for path in [train_path, val_path, test_path]:
        # Load TSV file without header
        df = pd.read_csv(path, sep="\t", header=None)
        df.columns = cols
        
        # Remove unnecessary index column
        df = df.drop(columns=["index"])
        
        # Clean ID field by removing .json extension
        df["id"] = df["id"].str.replace(".json", "", regex=False)
        
        # Strip whitespace from all text columns
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].str.strip()
            
        # Reset index for clean indexing
        df.reset_index(drop=True, inplace=True)
        dfs.append(df)

    return dfs

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

# # Echo Chamber Model
# def build_echo_chamber_model(df_train, n_estimators=300, max_depth=6, learning_rate=0.1,
#                              subsample=0.8, colsample_bytree=0.8, random_state=42):
#     """
#     Build an echo chamber detection model. Identifies political bias by analyzing party 
#     concentration, political content indicators, and text features.
#     """
#     topic_party_counts = (df_train.groupby(["subject", "party"]).size().unstack(fill_value=0))

#     # Calculate party concentration for each topic 
#     topic_party_counts["party_concentration"] = (topic_party_counts.max(axis=1) / topic_party_counts.sum(axis=1))

#     concentration_map = topic_party_counts["party_concentration"].to_dict()

#     df_train["echo_chamber"] = df_train["subject"].map(concentration_map)

#     def categorize_echo(value):
#         """Convert concentration to categorical classes."""
#         if value <= 0.4:
#             return 0
#         elif value <= 0.6:
#             return 1
#         elif value <= 0.8:
#             return 2
#         else:
#             return 3

#     df_train["echo_chamber_4class"] = df_train["echo_chamber"].apply(categorize_echo)

#     # Balance classes using upsampling
#     dfs = []
#     max_size = df_train["echo_chamber_4class"].value_counts().max()
#     for cls in df_train["echo_chamber_4class"].unique():
#         df_cls = df_train[df_train["echo_chamber_4class"] == cls]
#         df_cls_upsampled = resample(df_cls, replace=True, n_samples=max_size, random_state=42)
#         dfs.append(df_cls_upsampled)
#     df_train_bal = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)

#     df_train_bal["subject_length"] = df_train_bal["subject"].apply(lambda x: len(str(x).split(",")))
#     df_train_bal["is_political"] = df_train_bal["subject"].apply(
#         lambda x: int("politics" in str(x).lower() or "election" in str(x).lower())
#     )

#     le = LabelEncoder()
#     df_train_bal["party_encoded"] = le.fit_transform(df_train_bal["party"].fillna("none"))

#     vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
#     X_train_tfidf = vectorizer.fit_transform(df_train_bal["statement"])

#     X_train_full = hstack([
#         X_train_tfidf, 
#         df_train_bal[["subject_length", "is_political", "party_encoded"]].values
#     ])

#     y_train = df_train_bal["echo_chamber_4class"]

#     model = XGBClassifier(
#         n_estimators=n_estimators,
#         max_depth=max_depth,
#         learning_rate=learning_rate,
#         subsample=subsample,
#         colsample_bytree=colsample_bytree,
#         random_state=random_state,
#         eval_metric="mlogloss",
#     )

#     model.fit(X_train_full, y_train)

#     return model, vectorizer, le, concentration_map

# def predict_echo_chamber_model(df, model, vectorizer, le, concentration_map):
#     """
#     Make predictions using the echo chamber model.
#     """
#     df = df.copy()
#     df["echo_chamber"] = df["subject"].map(concentration_map)

#     def categorize_echo(value):
#         """Convert continuous concentration to categorical classes."""
#         if value <= 0.4:
#             return 0
#         elif value <= 0.6:
#             return 1
#         elif value <= 0.8:
#             return 2
#         else:
#             return 3

#     df["echo_chamber_4class"] = df["echo_chamber"].apply(categorize_echo)

#     df["subject_length"] = df["subject"].apply(lambda x: len(str(x).split(",")))
#     df["is_political"] = df["subject"].apply(lambda x: int("politics" in str(x).lower() or "election" in str(x).lower()))
#     df["party_encoded"] = le.fit_transform(df["party"].fillna("none"))

#     X_tfidf = vectorizer.transform(df["statement"])
#     X_full = hstack([X_tfidf, df[["subject_length", "is_political", "party_encoded"]].values])

#     # Make predictions
#     preds = model.predict(X_full)
#     probs = model.predict_proba(X_full).max(axis=1)

#     return pd.DataFrame(
#         {
#             "id": df["id"],
#             "statement": df["statement"],
#             "predicted_echo_class": preds,
#             "echo_chamber_score": probs,
#         }
#     )

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

# # Credibility Model
# def build_credibility_model(df_train, df_val, df_test, n_estimators=300, max_depth=6,
#                             learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
#                             random_state=42):
#     """
#     Build a credibility assessment model by analyzing expertise levels, subjectivity, 
#     political party affiliations, and text.
#     """
#     # Map LiarPLUS labels to credibility levels
#     credibility_map = {
#         "pants-fire": 0,    # Low credibility
#         "false": 0,         # Low credibility
#         "barely-true": 1,   # Medium credibility
#         "half-true": 1,     # Medium credibility
#         "mostly-true": 2,   # High credibility
#         "true": 2           # High credibility
#     }

#     for df in [df_train, df_val, df_test]:
#         df["credibility"] = df["label"].map(credibility_map)

#     # Extract subjectivity as a credibility indicator
#     # More subjective content may be less credible
#     for df in [df_train, df_val, df_test]:
#         df["subjectivity"] = df["statement"].apply(lambda t: TextBlob(str(t)).sentiment.subjectivity)

#     def encode_expertise(job):
#         """
#         Encode job titles into expertise levels.
#         Higher levels indicate more credible sources.
#         """
#         job = str(job).lower()
#         if any(w in job for w in ["professor","scientist","researcher","doctor","expert"]):
#           return 4  # Highest expertise
#         elif any(w in job for w in ["senator","governor","mayor","politician","president"]):
#           return 3  # High expertise
#         elif any(w in job for w in ["journalist","reporter","editor"]):
#           return 2  # Medium expertise 
#         elif any(w in job for w in ["actor","comedian","celebrity"]):
#           return 1  # Low expertise 
#         else:
#           return 0  # Unknown

#     for df in [df_train, df_val, df_test]:
#         df["expertise_level"] = df["job"].apply(encode_expertise)

#     le_party = LabelEncoder()
#     all_parties = pd.concat([df_train["party"], df_val["party"], df_test["party"]]).fillna("unknown")
#     le_party.fit(all_parties)
#     for df in [df_train, df_val, df_test]:
#         df["party_encoded"] = le_party.transform(df["party"].fillna("unknown"))

#     text_col = "statement"
#     cat_features = ["party_encoded", "expertise_level"]  
#     num_features = ["subjectivity"]  

#     preprocessor = ColumnTransformer(
#         transformers=[
#             ("text", TfidfVectorizer(max_features=5000, stop_words="english"), text_col),
#             ("num", StandardScaler(), num_features),
#             ("cat", "passthrough", cat_features)
#         ],
#         remainder="drop"
#     )

#     model = XGBClassifier(
#         n_estimators=n_estimators,
#         learning_rate=learning_rate,
#         max_depth=max_depth,
#         subsample=subsample,
#         colsample_bytree=colsample_bytree,
#         eval_metric="mlogloss",
#         random_state=random_state
#     )

#     pipeline = Pipeline([
#         ("features", preprocessor),
#         ("clf", model)
#     ])

#     X_train = df_train[[text_col] + cat_features + num_features]
#     y_train = df_train["credibility"]

#     pipeline.fit(X_train, y_train)
#     return pipeline, le_party

# def predict_credibility_model(df, pipeline, le_party):
#     """
#     Make predictions using the credibility model.
#     """
#     df = df.copy()

#     df["subjectivity"] = df["statement"].apply(lambda t: TextBlob(str(t)).sentiment.subjectivity)

#     def encode_expertise(job):
#         job = str(job).lower()
#         if any(w in job for w in ["professor","scientist","researcher","doctor","expert"]):
#             return 4
#         elif any(w in job for w in ["senator","governor","mayor","politician","president"]):
#             return 3
#         elif any(w in job for w in ["journalist","reporter","editor"]):
#             return 2
#         elif any(w in job for w in ["actor","comedian","celebrity"]):
#             return 1
#         else:
#             return 0
#     df["expertise_level"] = df["job"].apply(encode_expertise)
#     df["party_encoded"] = le_party.transform(df["party"].fillna("unknown"))

#     X = df[["statement","party_encoded","expertise_level","subjectivity"]]
    
#     # Make predictions
#     preds = pipeline.predict(X)
#     probs = pipeline.predict_proba(X).max(axis=1)

#     return pd.DataFrame({
#         "id": df["id"],
#         "statement": df["statement"],
#         "predicted_credibility": preds,
#         "credibility_score": probs
#     })

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


    



