import json
from src.predictive_models import (
    load_datasets, 
    build_frequency_model, 
    build_sensationalism_model,
    build_malicious_account_model,
    build_naive_realism_model
)

from src.articles import (
    train_job_model,
    evaluate_article
)

def main():
    with open("src/config.json", "r") as f:
        cfg = json.load(f)

    train_path = "../data/train2.tsv"
    val_path = "../data/val2.tsv"
    test_path = "../data/test2.tsv"
    df_train, df_val, df_test = load_datasets(
        train_path, val_path, test_path
    )
    print("Datasets loaded successfully.")

    freq_cfg = cfg["models"]["frequency"]
    freq_model = build_frequency_model(df_train, **freq_cfg)
    print("Frequency Heuristic model trained.")

    sens_cfg = cfg["models"]["sensationalism"]
    sens_model = build_sensationalism_model(
        df_train, df_val, df_test,**sens_cfg)
    print("Sensationalism model trained.")

    # job_clf, job_le = train_job_model(df_train)

    # job_party_model = (job_clf, job_le)

    # Use trained models to predict scores for new articles
    print("\nAnalyzing article...")
    article_text = "https://www.cnn.com/2025/10/20/politics/trump-no-kings-protests-vance-cia-analysis"  
    

    article_analysis = evaluate_article(
        url, 
        freq_model, 
        sens_model, 
        job_party_model
    )
    
    print(f"\nArticle Analysis Results:")
    print(f"URL: {article_analysis['url']}")
    print(f"Title: {article_analysis['title']}")
    print(f"Source: {article_analysis['source']}")
    print(f"Predicted Label: {article_analysis['predicted_label']}")
    print(f"Frequency Heuristic Score: {article_analysis['frequency_heuristic_score']:.3f}")
    print(f"Sensationalism Level: {article_analysis['predicted_sensationalism']}")
    print(f"Sensationalism Score: {article_analysis['sensationalism_score']:.3f}")


if __name__ == "__main__":
    main()