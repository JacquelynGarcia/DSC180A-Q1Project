# DSC180A Capstone Project - Narrative Integrity Analyzer

## Overview
This project aims to mitigate the gap of truth identification in an era where bias and sensationalism are prevalent in the political landscape through the use of mutliple factuality factors.
We use the LIAR-PLUS dataset to train separate models for:

- Frequency Heuristics - detecting repetition, buzzwords, and linguistic patterns that artificially enhance “truthiness.”
- Sensationalism - identifying emotionally charged, exaggerated, or dramatic rhetoric.
- Malicious Account – detecting linguistic markers commonly associated with bot-like or spammy messaging.
- Naive Realism – capturing absolutist, polarized, or dismissive language that reflects cognitive bias

Each model captures a unique dimension of factuality, contributing to a broader framework for automated fact-checking.

## Repository Structure
```text
DSC180A-Q1Project/
├── data/
│   ├── article.txt                   # Article text for ingestion
│   ├── ground_truth.csv              # Hand-labeled articles with appropriate factuality factor scores
│   ├── politifact.csv                # Manually scraped data from Politifact.org to augment LIAR-PLUS dataset
│   ├── train_set.csv                 # Training set with new scraped data
│   ├── train2.tsv                    # Original LiarPLUS train set
│   ├── val_set.csv                   # Validation set with new scraped data
│   ├── val2.tsv                      # Original LiarPLUS validation set
│   ├── test_set.csv                  # Test set with new scraped data
│   └── test2.tsv                     # Original LiarPLUS test set
│
├── notebooks/
│   ├── eda_visualization.ipynb       # Exploratory visualizations for LIAR-PLUS dataset
│   ├── metrics.ipynb                 # Accuracy scores for predictive models and LLM models
│   ├── model_accuracy.ipynb          # Various performance metrics for baseline predictive models
│   ├── prompting.ipynb               # Contains 20 incremental prompts refining the model 
│   └── scraped_data.ipynb            # Additonal scraped data from Politifact added to LiarPLUS
│
├── src/                              # Core project source code
│   ├── __init__.py
│   ├── articles.py                   # Article ingestion, preprocessing
│   ├── predictive_models.py          # ML/LLM-based prediction pipeline
│   ├── script.py                     # Main script to run full pipeline
│   └── config.json                   # Config settings for models and pipeline
│
├── webapp/                           # Flask-based UI for demo interactions
│   ├── app.py                        # Flask entrypoint
|   ├── prompts/
│   │   ├── base.txt                      # Base prompt for LLM 
│   │   ├── chain_of_thought.txt          # LLM prompt incorporating chain of thought
│   │   └── factal_chain_of_thought.txt   # LLM prompt incorporating fractal chain of thought
|   ├── results/
│   │   ├── base_outputs.csv              # Outputs saved from running LLM with base prompt
│   │   ├── cot_outputs.csv               # Outputs saved from running LLM with chain of thought prompt
│   │   └── fcot_outputs.csv              # Outputs saved from running LLM with fractal chain of thought prompt
│   ├── static/
│   │   └── style.css                 # CSS styling for UI
│   └── templates/
│       └── index.html                # Main UI page
│
├── .gitignore
├── README.md
└── requirements.txt                  # Requirements for environment
```

## Dataset
This project uses the LIAR-PLUS dataset, an extended version of the original LIAR dataset. We augmented this dataset with more recent
scraped data from PolitiFact.
This includes labeled political statements along with metadata such as subjects, speakers, party affiliations, and justifications.

### Dataset Summary
| Split | File | Description |
|-------|------|--------------|
| **Train** | `train_set.csv` | Used to train all factuality models. |
| **Validation** | `val_set.csv` | Used for tuning and intermediate evaluation. |
| **Test** | `test_set.csv` | Used for final evaluation and analysis. |

## Installation

### Clone the repository
```bash
git clone https://github.com/JacquelynGarcia/DSC180A-Q1Project.git
cd DSC180A-Q1Project
```

### Create virtual environment
```bash
python -m venv venv

source venv/bin/activate # Mac/Linux
venv\Scripts\activate # Windows
```

### Install dependencies
```bash
pip install -r requirements.txt
```

You're ready to go!

## Requirements
To ensure this project is reproducible, this project uses the following Python libraries and versions:

```lua
pandas==2.2.3
numpy==1.26.4
scikit-learn==1.5.2
xgboost==2.1.1
textblob==0.18.0
scipy==1.14.1
wordcloud==1.9.2
feedparser==6.0.10
sentence-transformers==2.7.0
transformers==4.44.2
newspaper3k==0.2.8
chromadb==0.4.24
beautifulsoup4==4.12.2
requests==2.31.0
spacy==3.7.2
en-core-web-sm==3.7.1
matplotlib==3.7.2
seaborn==0.12.2
lxml_html_clean==0.1.0
google-genai
python-dotenv
flask
torch
```

If you encounter an OpenMP error on macOS when running XGBoost, ensure homebrew is installed and install the OpenMP runtime:
```bash
brew install libomp
```

## Running the Pipeline
Once the dataset and environment are set up, you can execute all four factuality models in sequence using the main script.

### Run the project
From the project root directory, run:

```bash
python -m src.script
```

If properly installed, the example console output should contain the following:
```
Datasets loaded successfully.
Frequency Heuristic model trained.
Sensationalism model trained.
Malicious Account model trained.
Naive Realism model trained.

Analyzing article...

Article Analysis Results:
Frequency Heuristic Level: 0
Frequency Heuristic Score: 0.305
Sensationalism Level: 2
Sensationalism Score: 0.835
Malicious Account Level: 0
Malicious Account Score: 0.225
Naive Realism Level: 1
Naive Realism Score: 0.566
```

## Running the Web Application
The Flask UI lets you do the following:
- Paste an article link
- Paste article text
- View scores for each factuality factor
- View reasoning and confidence percentages
- Use a Clear button to analyze a new article
- Automatically saves every analysis to `results/data_outputs.csv`

### 1. Navigate to the webapp directory
```bash
cd webapp
```

### 2. Start the Flask server
```bash
python app.py
```

### 3. Opent the UI in your browser
```cpp
http://127.0.0.1:5000
```
Your interactive Narrative Integrity Analyzer is now running!

## Model Summaries
Each model focuses on a different factuality factor within political statements, capturing linguistic, contextual, or behavioral patterns associated with truthfulness and bias.

### Frequency Heuristic Model
Goal: Detect linguistic cues that may indicate exaggeration or misinformation through overuse of buzzwords and repetition.  

Features:
- TF-IDF mean  
- Average word frequency  
- Buzzword count  
- Repetition ratio  

Model: `RandomForestClassifier`  
Outputs:
- `predicted_label`
- `frequency_heuristic_score` - probability of label confidence  


### Malicious Account Model
Goal: Goal: Identify linguistic and behavioral traces aligned with inauthentic or “malicious account” behavior

Features:
- TF-IDF mean  
- Average token length
- Repitition score
- Link count
- Hashtag + mention count
- Punctuation ratio
- Uppercase ratio

Model: `RandomForestClassifier` within a `scikit-learn` Pipeline using `StandardScaler` 

Outputs:
- `predicted_malicious_account`
- `malicious_account_score` - probability of label confidence

### Sensationalism Model
Goal: Identify emotional, exaggerated, or dramatic tones that make a statement "sensational".

Features:
- Exclamation count (`!`)  
- Number of ALLCAPS words  
- Sensational keywords
- Sentiment polarity and subjectivity 
- Metadata such as `speaker`, `party`, and `context`  

Model: `XGBoost` within a `scikit-learn` Pipeline using `ColumnTransformer`  
Outputs:
- `predicted_sensationalism`
- `sensationalism_score` - probability of label confidence

### Naive Realism Model
Goal: Measure how strongly a statement presents opinion as fact through absolutist phrasing, lack of hedging, and dismissive language.

Features:
- Absolute-language ratio
- Cautious-language ratio
- Dismissive term count

Model: `XGBoost` Pipeline  
Outputs:
- `predicted_naive_realism`
- `naive_realism_score` - probability of label confidence

## Note on Exploratory Models and Methodology Evolution

During the early stages of this project, our team explored a broader set of factuality factors through exploratory data analysis and experimental modeling found within the `eda_visualization.ipynb` notebook. This notebook includes preliminary implementations of **Credibility** and **Echo Chamber** models, along with earlier versions of the **Frequency Heuristic** and **Sensationalism** models.

As our methodology matured, we refined our framework to focus on four primary factuality dimensions for the final production pipeline which included **Frequency Heuristic**, **Sensationalism**, and the newly added **Naive Realism**, and **Malicious Account**.

While **Credibility** and **Echo Chamber** are not included in the finalized model pipeline, exploratory versions of these models remain available in the `notebooks/` directory. Users who wish to extend the project, compare modeling strategies, or incorporate additional factuality factors are welcome to experiment with these earlier models.