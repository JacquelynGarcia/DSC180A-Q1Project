from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
from google import genai
from google.genai import types
import json
import pandas as pd
import uuid
import sys

# Add project root to Python path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

# Importing models
from src.predictive_models import (
    load_datasets,
    build_frequency_model, predict_frequency_model,
    build_sensationalism_model, predict_sensationalism_model,
    build_malicious_account_model, predict_malicious_account_model,
    build_naive_realism_model, predict_naive_realism_model,
)

load_dotenv()

app = Flask(__name__)


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Initializing Models - Runs at Start of Flask
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train_set.csv")
VAL_PATH   = os.path.join(DATA_DIR, "val_set.csv")
TEST_PATH  = os.path.join(DATA_DIR, "test_set.csv")

print("Loading LIAR-PLUS datasets...")
train_df, val_df, test_df = load_datasets(TRAIN_PATH, VAL_PATH, TEST_PATH)

print("Building Frequency model...")
freq_model, freq_tfidf, freq_count_vec, freq_token_dict, freq_buzzwords, freq_le = \
    build_frequency_model(train_df)

print("Building Sensationalism model...")
sens_pipeline, sens_numeric_features = build_sensationalism_model(train_df)

print("Building Malicious Account model...")
mal_model, mal_tfidf, mal_le = build_malicious_account_model(train_df)

print("Building Naive Realism model...")
naive_pipeline, naive_numeric_features = build_naive_realism_model(train_df)

print("All models initialized.")

# Getting Model Scores
def get_model_scores(article_text: str) -> dict:
    """
    Run all factuality ML models on the given article text.

    Args:
        article_text: Full article text pasted by the user.

    Returns:
        A dict with model-derived scores for each factuality factor.
        Each factor has:
          - model_score: discrete level (0, 1, 2)
          - model_confidence: probability [0, 1]
    """
    # Wrap the article in a one-row DataFrame matching your model API
    df = pd.DataFrame({"statement": [article_text]})

    # Frequency
    freq_df = predict_frequency_model(
        df,
        freq_model, freq_tfidf, freq_count_vec,
        freq_token_dict, freq_buzzwords, freq_le
    )
    freq_level = int(freq_df["predicted_frequency_heuristic"].iloc[0])
    freq_conf  = float(freq_df["frequency_heuristic_score"].iloc[0])

    # Sensationalism
    sens_df = predict_sensationalism_model(df, sens_pipeline, sens_numeric_features)
    sens_level = int(sens_df["predicted_sensationalism"].iloc[0])
    sens_conf  = float(sens_df["sensationalism_score"].iloc[0])

    # Malicious Account
    mal_df = predict_malicious_account_model(df, mal_model, mal_tfidf, mal_le)
    mal_level = int(mal_df["predicted_malicious_account"].iloc[0])
    mal_conf  = float(mal_df["malicious_account_score"].iloc[0])

    # Naive Realism
    naive_df = predict_naive_realism_model(df, naive_pipeline, naive_numeric_features)
    naive_level = int(naive_df["predicted_naive_realism"].iloc[0])
    naive_conf  = float(naive_df["naive_realism_score"].iloc[0])

    return {
        "frequency_heuristic": {
            "model_score": freq_level,
            "model_confidence": freq_conf,
        },
        "sensationalism": {
            "model_score": sens_level,
            "model_confidence": sens_conf,
        },
        "malicious_account": {
            "model_score": mal_level,
            "model_confidence": mal_conf,
        },
        "naive_realism": {
            "model_score": naive_level,
            "model_confidence": naive_conf,
        },
    }

# Prompting
def factuality_score(article_text):
    """
    Ask Gemini to analyze the article AND optionally call get_model_scores() as a tool.
    """

    prompt = f"""
    You are an expert in misinformation and disinformation detection, scoring, and ranking. Your task is to analyze the given article and 
    score how strongly it exhibits each factuality factor. 
    ---

    ### Factuality Factors: 
    1. Frequency Heuristic
    - Look for repetition, repeated claims, or appeals to popularity (“many people are saying…”).
    - Score: 0 = minimal/no repetition; 1 = some repetitive or consensus-based framing; 2 = heavy repetition or appeal to consensus.

    2. Malicious Account
    - Assess whether the source or referenced accounts appear suspicious, coordinated, inauthentic, or unreliable.
    - Score: 0 = credible source; 1 = mildly suspicious; 2 = deceptive, coordinated, or malicious.

    3. Sensationalism
    - Examine whether the text uses dramatic, exaggerated, emotional, or fear-based language.
    - Score: 0 = neutral tone; 1 = mildly sensationalized; 2 = highly dramatic or emotional.

    4. Naive Realism
    - Evaluate whether the article presents its viewpoint as the only truth, dismisses alternatives, or discourages nuance.
    - Score: 0 = balanced; 1 = somewhat one-sided; 2 = dogmatic or absolutist.

    ### Article to Evaluate
    {article_text}

    ### Task
    Provide a score (0–2), justification, and confidence level on a scale from 0-100% for each of the four factuality factors.
    RETURN ONLY VALID JSON. DO NOT USE MARKDOWN. DO NOT USE ```json OR ANY CODE FENCES. OUTPUT ONLY A JSON OBJECT.

    ### Output Format:
    {{
        "frequency_heuristic": {{
            "score": 0|1|2,
            "reasoning": "Explanation",
            "confidence": 0-100
        }},
        "malicious_account": {{
            "score": 0|1|2,
            "reasoning": "Explanation",
            "confidence": 0-100
        }},
        "sensationalism": {{
            "score": 0|1|2,
            "reasoning": "Explanation",
            "confidence": 0-100
        }},
        "naive_realism": {{
            "score": 0|1|2,
            "reasoning": "Explanation",
            "confidence": 0-100
        }}
    }}


    """

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[get_model_scores],
        ),
    )

    return response.text.strip()

@app.route("/")
def index():
    return render_template("index.html")

def save_to_csv(article_url, parsed):
    run_id = str(uuid.uuid4())

    freq_conf = parsed["frequency_heuristic"]["confidence"]
    mal_conf = parsed["malicious_account"]["confidence"]
    sens_conf = parsed["sensationalism"]["confidence"]
    naive_conf = parsed["naive_realism"]["confidence"]

    overall_conf = (freq_conf + mal_conf + sens_conf + naive_conf) / 4

    row = {
        "id": run_id,
        "url": article_url,

        "freq_score": parsed["frequency_heuristic"]["score"],
        "freq_reason": parsed["frequency_heuristic"]["reasoning"],
        "freq_confidence": freq_conf,

        "mal_score": parsed["malicious_account"]["score"],
        "mal_reason": parsed["malicious_account"]["reasoning"],
        "mal_confidence": mal_conf,

        "sens_score": parsed["sensationalism"]["score"],
        "sens_reason": parsed["sensationalism"]["reasoning"],
        "sens_confidence": sens_conf,

        "naive_score": parsed["naive_realism"]["score"],
        "naive_reason": parsed["naive_realism"]["reasoning"],
        "naive_confidence": naive_conf,

        "overall_confidence": overall_conf
    }

    df_row = pd.DataFrame([row])
    csv_path = "base_outputs.csv"

    if os.path.exists(csv_path):
        df_row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(csv_path, index=False)


@app.route("/score", methods=["POST"])
def score():
    article_text = request.form.get("article", "")
    article_url = request.form.get("article_url", "")

    if not article_text.strip():
        return jsonify({"error": "No article text provided"}), 400

    try:
        raw_output = factuality_score(article_text)

        try:
            clean = raw_output.strip()

            start = clean.find("{")
            end = clean.rfind("}")

            if start == -1 or end == -1:
                return jsonify({"error": "Model returned no JSON", "raw": raw_output}), 500

            json_str = clean[start:end+1]

            parsed = json.loads(json_str)

            save_to_csv(article_url, parsed)

        except Exception:
            return jsonify({"error": "Model returned invalid JSON", "raw": raw_output}), 500

        return jsonify(parsed)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
