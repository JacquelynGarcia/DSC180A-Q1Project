from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
from google import genai
import json
import pandas as pd
import uuid

load_dotenv()

app = Flask(__name__)


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def factuality_score(article_text):
    """
    Placeholder scoring function — you will later replace with your
    factuality factor pipeline or multi-agent scoring.
    """

    prompt = f"""
    You are an expert in misinformation and disinformation detection, scoring, and ranking. Your task is to analyze the given article and 
    score how strongly it exhibits each factuality factor. 
    ---

    ### Factuality Factors: 
    1. **Frequency Heuristic** 
    - *Repetition Analysis*: Observe how often a claim or narrative is echoed across the text or across references. 
    - *Origin Tracing*: Determine whether frequently repeated information is traced to a credible or questionable source. 
    - *Evidence Verification*: Evaluate if the text implies truth merely due to repetition or popularity of a claim. 
    - **Scoring:** 0 = none/minimal repetition; 1 = moderate repetition or common-belief phrasing; 2 = heavy repetition or appeal to consensus. 
    
    2. **Malicious Account** - *Account Analysis*: If the text references or cites accounts, consider whether 
    their creation dates or activity patterns suggest inauthentic behavior. 
    - *Interaction Patterns*: Evaluate if the content originates from or interacts with accounts resembling coordinated or bot-like behavior. 
    - *Content Review*: Assess if the account or source repeatedly spreads false or harmful information. 
    - **Scoring:** 0 = credible source; 1 = slightly suspicious or biased source; 2 = clearly deceptive, coordinated, or malicious source. 
    
    3. **Sensationalism** - *Language Intensity*: Examine the text for overly dramatic or exaggerated claims. 
    - *Tone Comparison*: Compare emotional tone of headlines versus main content. 
    - *Shock vs. Substance*: Determine whether the article prioritizes shock value over factual reporting. 
    - **Scoring:** 0 = neutral/objective; 1 = mildly emotional or dramatic; 2 = highly sensationalized 
    
    4. **Naive Realism** - *Perspective Analysis*: Evaluate whether the content presents its view as the only correct one. 
    - *Dissenting View Checks*: Analyze whether differing views are acknowledged or dismissed. 
    - *Isolation Analysis*: Determine whether the article attempts to isolate readers from alternative perspectives. 
    - **Scoring:** 0 = balanced and nuanced; 1 = somewhat one-sided; 2 = fully dogmatic. 
    
    ### Examples 
    **Example 1:** Article: The Federal Reserve announced a 0.25% interest rate increase following its quarterly meeting. 
    Economists had predicted the move based on recent inflation data. 
    Frequency Heuristic: 0 — Single statement of fact with proper attribution and no repetitive framing. 
    Malicious Account: 0 — Credible financial news source citing official Federal Reserve announcement. 
    Sensationalism: 0 — Neutral, technical language appropriate for economic reporting. 
    Naive Realism: 0 — Presents information objectively without claiming singular truth. 
    
    **Example 2:** Article: WAKE UP PEOPLE! Everyone knows the government is controlling us through 5G towers! 
    Thousands are saying it online. The mainstream media won't tell you the TRUTH because they're in on it! 
    Frequency Heuristic: 2 — Heavy appeals to popularity ('everyone knows,' 'thousands are saying') to establish credibility through repetition. 
    Malicious Account: 2 — Anonymous source spreading debunked conspiracy theories. 
    Sensationalism: 2 — All-caps words, exclamation marks, fear-mongering language designed to provoke emotional response. 
    Naive Realism: 2 — Presents conspiracy as absolute truth, dismisses all mainstream sources, and attempts to isolate readers. 
    
    **Example 3:** Article: Is your food secretly KILLING you? Experts reveal the hidden dangers lurking in your kitchen. You won't believe what we found! 
    Frequency Heuristic: 1 — Uses populist framing ('hidden dangers') which suggests knowledge. 
    Malicious Account: 1 — Clickbait journalism from established outlet but not malicious. 
    Sensationalism: 2 — Clickbait with dramatic language and mystery ('you won't believe'). 
    Naive Realism: 1 — Implies hidden truth without presenting the actual complexity of food safety. 
    
    **Example 4:** Article: Tech company announces its AI will revolutionize healthcare and transform diagnosis forever. 
    The CEO called it a 'game-changing breakthrough' during yesterday's product launch. 
    Frequency Heuristic: 1 — Repeats transformative framing. 
    Malicious Account: 0 — Corporate PR from legitimate company so promotional but not malicious. 
    Sensationalism: 2 — Extreme claims ('revolutionize,' 'transform forever,' 'game-changing'). 
    Naive Realism: 1 — Presents corporate claims as reality without acknowledging uncertainty or limitations. 
    
    **Example 5:** Article: According to multiple sources familiar with the matter, the investigation is ongoing. 
    Officials declined to comment on specifics, citing the active nature of the case. 
    Frequency Heuristic: 0 — Reports lack of information honestly without speculation. 
    Malicious Account: 0 — Standard practice of protecting sources while noting limitations. 
    Sensationalism: 0 — Neutral tone and restrained language. 
    Naive Realism: 0 — Explicitly communicates uncertainty and ongoing nature of situation. 
    
    **Example 6:** Article: 'Many people are saying the new policy is unfair. It's becoming clear that something needs to change. 
    More and more citizens are questioning the decision every day.' 
    Frequency Heuristic: 2 — Repeating claims of rising support that lack clear attribution. 
    Malicious Account: 1 — Lacks verification. 
    Sensationalism: 1 — Emotionally charged framing ('unfair') and builds urgency but not extreme language. 
    Naive Realism: 1 — Implies growing consensus makes position correct without presenting other arguments. 
    
    **Example 7:** Article: The new immigration bill sparked heated debate. Republican lawmakers argue it strengthens border security, 
    while Democratic representatives contend it lacks humanitarian protections. Legal experts are divided on constitutional questions. 
    Frequency Heuristic: 0 — Multiple perspective stated once with attribution to distinct groups. 
    Malicious Account: 0 — Mainstream news outlet citing political figures and experts. 
    Sensationalism: 1 — Word 'heated' adds mild emotional tone but otherwise is balanced. 
    Naive Realism: 0 — Acknowledges multiple perspectives across parties and expert opinion. 
    
    **Example 8:** Article: Shocking allegations have emerged against the senator. Anonymous sources describe a pattern of misconduct, 
    though the senator's team calls the claims 'politically motivated lies' and notes no formal charges have been filed. 
    Frequency Heuristic: 1 — 'Pattern of misconduct' suggests multiple mentions. 
    Malicious Account: 1 — Anonymous sourcing. 
    Sensationalism: 2 — 'Shocking allegations' is designed to attract attention. 
    Naive Realism: 0 — Presents both accusation and denial. Also acknowledges lack of formal charges. 
    
    **Example 9:** Article: As commonly understood by most experts in the field, renewable energy will replace fossil 
    fuels within two decades. Some outlier researchers dispute this timeline, but the consensus is clear. 
    Frequency Heuristic: 1 — Frames majority view as established fact. 
    Malicious Account: 0 — Appears to be legitimate climate journalism citing expert community. 
    Sensationalism: 0 — Factual tone without emotional manipulation or exaggeration. 
    Naive Realism: 2 — Dismisses dissenting views and presents prediction as inevitable. 
    
    **Example 10:** Article: Influencer @TrendyLifestyle123 claims this detox tea cured her chronic illness. 
    The product has 50K followers and hundreds of testimonials, though medical professionals warn these supplements are unregulated. 
    Frequency Heuristic: 1 — Implies truth through follower count and testimonials, but includes medical warning. 
    Malicious Account: 2 — Suspicious account name and health claims. 
    Sensationalism: 1 — 'Cured chronic illness' is dramatic claim but is presented as a quote from influencer. 
    Naive Realism: 1 — Amplifies an unverified claim but includes expert skepticism. 
    
    ### Article to Evaluate: {article_text} 
    ### Tool Usage Requirement: You also have access to tools that return a predictive model score for each factuality factor. 
    Treat these model scores as informative context, NOT ground truth. You must reason independently.

    ### Instructions: 
    1. Think step-by-step about the article’s tone, evidence, framing, and intent. 
    2. Identify linguistic cues that signal bias, repetition, exaggeration, or malicious intent. 
    3. Use your factuality factor analysis and the tool outputs to provide a numeric score, a justification for why that score was chosen, and your confidence
    level in that assessment on a scale of 0-100%. If your score is different than the predictive model score, you must explain why you disagree. 
    4. RETURN ONLY VALID JSON. DO NOT USE MARKDOWN. DO NOT USE ```json OR ANY CODE FENCES. OUTPUT ONLY A JSON OBJECT.

    ### Output Format: {{ "frequency_heuristic": {{ "score": 0|1|2, "reasoning": "Explanation", "confidence": 0-100 }}, "malicious_account": {{ "score": 0|1|2, "reasoning": "Explanation", "confidence": 0-100 }}, "sensationalism": {{ "score": 0|1|2, "reasoning": "Explanation", "confidence": 0-100 }}, "naive_realism": {{ "score": 0|1|2, "reasoning": "Explanation", "confidence": 0-100 }} }}
    """

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt
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
    csv_path = "data_outputs.csv"

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
