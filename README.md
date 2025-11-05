# DSC180A Capstone Project - Beyond the Buzz

## Overview
This project aims to mitigate the gap of truth identification in an era where bias and sensationalism are prevalent in the political landscape through the use of mutliple factuality factors.
We use the LIAR-PLUS dataset to train separate models for:

- Frequency Heuristics - detecting overuse of buzzwords and repetition.  
- Echo Chambers - measuring topic and party alignment.  
- Sensationalism: identifying exaggeration and emotional language.  
- Credibility - assessing the speaker's overall reliability.

Each model captures a unique dimension of factuality, contributing to a broader framework for automated fact-checking.

## Repository Structure
- src/ - Contains all source code for model building, configuration, and execution.  
- code.py - Core logic for each factuality model, including training and prediction functions.  
- script.py - The main entry point that loads the dataset, builds models, runs predictions, and outputs results.  
- config.json - Central configuration file specifying dataset paths and hyperparameters for each model.  

- data/ - Local folder containing the LIAR-PLUS dataset (`train2.tsv`, `val2.tsv`, `test2.tsv`).  
*Note: this folder is excluded from GitHub via `.gitignore` for size and licensing reasons.*  

- requirements.txt - Lists all Python dependencies needed to reproduce the environment.  

- .gitignore - Ensures unnecessary files are not pushed to GitHub.

## Installation

### Clone the repository
```bash
git clone https://github.com/JacquelynGarcia/DSC180A-Q1Project.git
cd DSC180A-Q1Project
```

### Download the LIAR-PLUS dataset
This project relies on the [LIAR-PLUS dataset](https://www.kaggle.com/datasets/saketchaturvedi/liarplus), which contains labeled political statements with metadata such as subjects, speakers, and contexts.

1. Visit the Kaggle dataset page:  
   🔗 https://www.kaggle.com/datasets/saketchaturvedi/liarplus  
2. Download the dataset ZIP file.  
3. Extract it and locate the following files:
   - `train2.tsv`
   - `val2.tsv`
   - `test2.tsv`
4. Place these files in a new folder named `data/` inside the project root:

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