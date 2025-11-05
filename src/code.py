import pandas as pd

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
