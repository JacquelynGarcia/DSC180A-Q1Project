# DSC180A Capstone Project - Beyond the Buzz

## Overview
This project aims to mitigate the gap of truth identification in an era where bias and sensationalism are prevalent in the political landscape through the use of mutliple factuality factors.
We use the LIAR-PLUS dataset to train separate models for:

- Frequency Heuristics - detecting overuse of buzzwords and repetition.  
- Echo Chambers - measuring topic and party alignment.  
- Sensationalism - identifying exaggeration and emotional language.  
- Credibility - assessing the speaker's overall reliability.

Each model captures a unique dimension of factuality, contributing to a broader framework for automated fact-checking.

## Repository Structure
- src/ - Contains all source code for model building, configuration, and execution.
- article.py - Contains main ingestion for articles.
- predictive_models.py - Core logic for each factuality model, including training and prediction functions.  
- script.py - The main entry point that loads the dataset, builds models, runs predictions, and outputs results.  
- config.json - Central configuration file specifying dataset paths and hyperparameters for each model.  

- notebooks/ - Jupyter notebooks for exploratory data analysis and model evaluation.  
- eda_visualization.ipynb - Exploratory Data Analysis of the training dataset; includes distribution plots, text patterns, and insight visualizations.  
- ModelAccuracy.ipynb - Visualizations and metrics to compare each model's performance (accuracy, precision, recall, F1-score, etc.).  

- data/ - Local folder containing the LIAR-PLUS dataset (`train2.tsv`, `val2.tsv`, `test2.tsv`).

- requirements.txt - Lists all Python dependencies needed to reproduce the environment.  

- .gitignore - Ensures unnecessary files are not pushed to GitHub.

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
