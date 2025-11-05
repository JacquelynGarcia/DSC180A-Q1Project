import json
from code import (
    load_datasets, 
    build_frequency_model, 
    predict_frequency_model,
    build_echo_chamber_model,
    predict_echo_chamber_model,
    build_sensationalism_model,
    predict_sensationalism_model,
    build_credibility_model,
    predict_credibility_model
)

def main():
    with open("src/config.json", "r") as f:
        cfg = json.load(f)

    data_cfg = cfg["data"]
    df_train, df_val, df_test = load_datasets(
        data_cfg["train_path"], data_cfg["val_path"], data_cfg["test_path"]
    )
    print("Datasets loaded successfully.")
    print(df_train.head())

    freq_cfg = cfg["models"]["frequency"]
    model_freq, tfidf_freq, count_vec_freq, token_dict_freq, buzzwords_freq, le_freq = build_frequency_model(df_train, **freq_cfg)
    val_results_freq = predict_frequency_model(df_val, model_freq, tfidf_freq, count_vec_freq, token_dict_freq, buzzwords_freq, le_freq)
    test_results_freq = predict_frequency_model(df_test, model_freq, tfidf_freq, count_vec_freq, token_dict_freq, buzzwords_freq, le_freq)
    print("Frequency model complete.")

    echo_cfg = cfg["models"]["echo_chamber"]
    model_echo, vectorizer_echo, le_echo, concentration_map = build_echo_chamber_model(df_train, **echo_cfg)
    val_results_echo = predict_echo_chamber_model(df_val, model_echo, vectorizer_echo, le_echo, concentration_map)
    test_results_echo = predict_echo_chamber_model(df_test, model_echo, vectorizer_echo, le_echo, concentration_map)
    print("Echo Chamber model complete.")

    sens_cfg = cfg["models"]["sensationalism"]
    sens_pipeline, sens_preproc, sens_meta, sens_num = build_sensationalism_model(
        df_train, df_val, df_test,**sens_cfg)
    val_results_sens = predict_sensationalism_model(df_val, sens_pipeline, sens_preproc, sens_meta, sens_num)
    test_results_sens = predict_sensationalism_model(df_test, sens_pipeline, sens_preproc, sens_meta, sens_num)
    print("Sensationalism model complete.")

    cred_cfg = cfg["models"]["credibility"]
    cred_pipeline, party_enc_cred = build_credibility_model(df_train, df_val, df_test, **cred_cfg)
    val_results_cred = predict_credibility_model(df_val, cred_pipeline,party_enc_cred)
    test_results_cred = predict_credibility_model(df_test, cred_pipeline,party_enc_cred)
    print("Credibility model complete.")


if __name__ == "__main__":
    main()