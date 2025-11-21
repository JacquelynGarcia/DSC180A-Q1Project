import pandas as pd
from src.predictive_models import (
    predict_frequency_model,
    predict_sensationalism_model,
    predict_malicious_account_model, 
    predict_naive_realism_model
)

def prepare_article(article_text):
    return pd.DataFrame({
        "statement": [article_text.strip()]
    })


def evaluate_article(article_text, freq_model, sens_model, ma_model, nr_model):
    """
    Evaluate an article using all trained models
    """
    df = prepare_article(article_text)

    model_freq, tfidf_freq, count_vec_freq, token_dict_freq, buzzwords_freq, le_freq = freq_model
    sens_pipeline, sens_num = sens_model
    model_malicious, tfidf_malicious, le_malicious = ma_model
    naive_pipeline, naive_num = nr_model

    freq_result = predict_frequency_model(df, model_freq, tfidf_freq, count_vec_freq, token_dict_freq, buzzwords_freq, le_freq).iloc[0]
    sens_result = predict_sensationalism_model(df, sens_pipeline, sens_num).iloc[0]
    ma_result = predict_malicious_account_model(df, model_malicious, tfidf_malicious, le_malicious).iloc[0]
    nr_result = predict_naive_realism_model(df, naive_pipeline, naive_num).iloc[0]

    return {
        "predicted_frequency_heuristic": freq_result["predicted_frequency_heuristic"], # Frequency heuristic results
        "frequency_heuristic_score": freq_result["frequency_heuristic_score"],
        "predicted_sensationalism": sens_result["predicted_sensationalism"], # Sensationalism results
        "sensationalism_score": sens_result["sensationalism_score"],
        "predicted_malicious_account": ma_result["predicted_malicious_account"], # Malicious account results
        "malicious_account_score": ma_result["malicious_account_score"],
        "predicted_naive_realism": nr_result["predicted_naive_realism"], # Naive realism results
        "naive_realism_score": nr_result["naive_realism_score"]
    }


