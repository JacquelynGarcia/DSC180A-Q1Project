from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
import google.genai as genai

load_dotenv()

app = Flask(__name__)


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")


def factuality_score(article_text):
    """
    Replace this with your actual pipeline:
    - call your factuality factor agents
    - run your scoring functions
    """

    prompt = f"""
    You are a factuality evaluator. 
    Read the following article and return a factuality score from 0 to 1.
    Article:
    {article_text}

    Return ONLY a number.
    """

    response = model.generate_content(prompt)
    return response.text.strip()


# --------------------------
# ROUTES
# --------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/score", methods=["POST"])
def score():
    article_text = request.form["article"]

    if not article_text.strip():
        return jsonify({"error": "No article text provided"}), 400

    try:
        score = factuality_score(article_text)
        return jsonify({"score": score})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(debug=True)
