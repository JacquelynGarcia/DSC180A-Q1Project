import json
from src.predictive_models import (
    load_datasets, 
    build_frequency_model, 
    build_echo_chamber_model,
    build_sensationalism_model,
    build_credibility_model,
)

from src.articles import (
    train_party_and_job_models,
    evaluate_article
)

def main():
    with open("src/config.json", "r") as f:
        cfg = json.load(f)

    data_cfg = cfg["data"]
    df_train, df_val, df_test = load_datasets(
        data_cfg["train_path"], data_cfg["val_path"], data_cfg["test_path"]
    )
    print("Datasets loaded successfully.")

    freq_cfg = cfg["models"]["frequency"]
    model_freq, tfidf_freq, count_vec_freq, token_dict_freq, buzzwords_freq, le_freq = build_frequency_model(df_train, **freq_cfg)
    print("Frequency Heuristic model trained.")

    # echo_cfg = cfg["models"]["echo_chamber"]
    # model_echo, vectorizer_echo, le_echo, concentration_map = build_echo_chamber_model(df_train, **echo_cfg)
    # print("Echo Chamber model trained.")

    sens_cfg = cfg["models"]["sensationalism"]
    sens_pipeline, sens_meta, sens_num = build_sensationalism_model(
        df_train, df_val, df_test,**sens_cfg)
    print("Sensationalism model trained.")

    # cred_cfg = cfg["models"]["credibility"]
    # cred_pipeline, party_enc_cred = build_credibility_model(df_train, df_val, df_test, **cred_cfg)
    # print("Credibility model trained.")

    (party_clf, party_le), (job_clf, job_le) = train_party_and_job_models(df_train)

    freq_model = (model_freq, tfidf_freq, count_vec_freq, token_dict_freq, buzzwords_freq, le_freq)
    # echo_model = (model_echo, vectorizer_echo, le_echo, concentration_map)
    sens_model = (sens_pipeline, sens_meta, sens_num)
    # cred_model = (cred_pipeline, party_enc_cred)
    job_party_model = (job_clf, job_le, party_clf, party_le)

    # Use trained models to predict scores for new articles
    print("\nAnalyzing article...")
    url = "https://www.cnn.com/2025/10/20/politics/trump-no-kings-protests-vance-cia-analysis"  
    

    article_analysis = evaluate_article(
        url, 
        freq_model, 
        # echo_model, 
        sens_model, 
        # cred_model, 
        job_party_model
    )
    
    print(f"\nArticle Analysis Results:")
    print(f"URL: {article_analysis['url']}")
    print(f"Title: {article_analysis['title']}")
    print(f"Source: {article_analysis['source']}")
    print(f"Predicted Label: {article_analysis['predicted_label']}")
    print(f"Frequency Heuristic Score: {article_analysis['frequency_heuristic_score']:.3f}")
    print(f"Echo Chamber Class: {article_analysis['predicted_echo_class']}")
    print(f"Echo Chamber Score: {article_analysis['echo_chamber_score']:.3f}")
    print(f"Sensationalism Level: {article_analysis['predicted_sensationalism']}")
    print(f"Sensationalism Score: {article_analysis['sensationalism_score']:.3f}")
    print(f"Credibility Level: {article_analysis['predicted_credibility']}")
    print(f"Credibility Score: {article_analysis['credibility_score']:.3f}")


if __name__ == "__main__":
    main()