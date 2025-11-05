import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def load_datasets(train_path, val_path, test_path):
    cols = [
    "index", "id", "label", "statement", "subject", "speaker", "job", "state",
    "party", "barely_true", "false", "half_true", "mostly_true", "pants_on_fire",
    "context", "justification"
    ]

    dfs = []

    for path in [train_path, val_path, test_path]:
        df = pd.read_csv(path, sep="\t", header=None)
        df.columns = cols
        df = df.drop(columns=["index"])
        df["id"] = df["id"].str.replace(".json", "", regex=False)
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].str.strip()
        df.reset_index(drop=True, inplace=True)
        dfs.append(df)

    return dfs

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

    model = Pipeline([
        ("scaler", StandardScaler()), 
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
    ])
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

    return pd.DataFrame({
        "id": df["id"],
        "statement": df["statement"],
        "predicted_label": le.inverse_transform(preds),
        "frequency_heuristic_score": probs
    })
