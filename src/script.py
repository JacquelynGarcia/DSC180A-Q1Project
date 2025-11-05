import json
from code import load_datasets, build_frequency_model, predict_frequency_model

def main():
    with open("src/config.json", "r") as f:
        cfg = json.load(f)

    train_path = cfg["train_path"]
    val_path = cfg["val_path"]
    test_path = cfg["test_path"]

    df_train, df_val, df_test = load_datasets(train_path, val_path, test_path)

    model_freq, tfidf_freq, count_vec_freq, token_dict_freq, buzzwords_freq, le_freq = build_frequency_model(df_train)

    val_results = predict_frequency_model(df_val, model_freq, tfidf_freq, count_vec_freq, token_dict_freq, buzzwords_freq, le_freq)
    test_results = predict_frequency_model(df_test, model_freq, tfidf_freq, count_vec_freq, token_dict_freq, buzzwords_freq, le_freq)


if __name__ == "__main__":
    main()