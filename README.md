# DSC180A Capstone Project - Narrative Integrity Analyzer

## Overview
This project aims to mitigate the gap of truth identification in an era where bias and sensationalism are prevalent in the political landscape through the use of mutliple factuality factors.
We use the LIAR-PLUS dataset to train separate models for:

- Frequency Heuristics - detecting overuse of buzzwords and repetition.  
- Echo Chambers - measuring topic and party alignment.  
- Sensationalism - identifying exaggeration and emotional language.  
- Credibility - assessing the speaker's overall reliability.

Each model captures a unique dimension of factuality, contributing to a broader framework for automated fact-checking.

## Repository Structure
```text
DSC180A-Q1Project/
├── data/
│   ├── article.txt                   # Article text for ingestion
│   ├── ground_truth.csv              # Hand-labeled articles with appropriate factuality factor scores
│   ├── politifact.csv                # Manually scraped data from Politifact.org to augment LIAR-PLUS dataset
│   ├── train_set.csv / train2.tsv
│   ├── val_set.csv   / val2.tsv
│   └── test_set.csv  / test2.tsv
│
├── notebooks/
│   ├── eda_visualization.ipynb       # Exploratory visualizations for LIAR-PLUS dataset
│   ├── model_accuracy.ipynb          # Various accuracy scores for baseline models
│   ├── prompting.ipynb               # Contains 20 incremental prompts refining the model and ending with function calling
│   ├── scraped_data.ipynb            # Additonal scraped data from outside sources
│   └── hand_labels_template.csv
│
├── research/                         # Final capstone research report
│   └── DSC180A_Capstone_Report.pdf
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
│   ├── data_outputs.csv              # Outputs saved from UI interactions
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
This project uses the LIAR-PLUS dataset, an extended version of the original LIAR dataset.
This includes labeled political statements along with metadata such as subjects, speakers, party affiliations, and justifications.

### Dataset Summary
| Split | File | Description |
|-------|------|--------------|
| **Train** | `train2.tsv` | Used to train all factuality models. |
| **Validation** | `val2.tsv` | Used for tuning and intermediate evaluation. |
| **Test** | `test2.tsv` | Used for final evaluation and analysis. |

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
Frequency model trained.
Echo Chamber model trained.
Sensationalism model trained.
Credibility model trained.

Analyzing article...

Article Analysis Results:
URL: https://www.cnn.com/2025/10/20/politics/trump-no-kings-protests-vance-cia-analysis
Title: AI sewage video: Trump’s response to ‘No Kings’ marches only proved the protesters’ point
Source: www.cnn.com
Predicted Label: mostly-true
Frequency Score: 0.355
Echo Chamber Class: 3
Echo Chamber Score: 0.603
Sensationalism Level: 0
Sensationalism Score: 0.797
Credibility Level: 1
Credibility Score: 0.851
```

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


### Echo Chamber Model
Goal: Measure how concentrated topics are across political affiliations, simulating the "echo chamber" effect seen in partisan speech.  

Features:
- TF-IDF embeddings of statements  
- Subject length  
- Party alignment ratio per topic  
- Political relevance flag (`is_political`)  

Model: `XGBClassifier`

Outputs:
- `predicted_echo_class`
- `echo_chamber_score` - probability-based score of ideological clustering  

### Sensationalism Model
Goal: Identify emotional, exaggerated, or dramatic tones that make a statement "sensational."

Features:
- Exclamation count (`!`)  
- Number of ALLCAPS words  
- Sensational keywords
- Sentiment polarity and subjectivity 
- Metadata such as `speaker`, `party`, and `context`  

Model: `XGBoost` within a `scikit-learn` Pipeline using `ColumnTransformer`  
Outputs:
- `predicted_sensationalism`
- `sensationalism_score` - numeric probability for sensational tone  

### Credibility Model
Goal: Assess the trustworthiness of a statement based on speaker background, expertise, and tone.  

Features:
- TF-IDF representation of the statement text  
- Speaker expertise level  
- Political party encoding  
- Subjectivity score 

Model: `XGBoost` Pipeline  
Outputs:
- `predicted_credibility`
- `credibility_score` - predicted probability of a credible statement  
