import json
from code import load_datasets

def main():
    with open("src/config.json", "r") as f:
        cfg = json.load(f)

    train_path = cfg["train_path"]
    val_path = cfg["val_path"]
    test_path = cfg["test_path"]

    df_train, df_val, df_test = load_datasets(train_path, val_path, test_path)



if __name__ == "__main__":
    main()