# Among Us Impostor Predictor

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-Bedrock-232F3E?style=flat-square&logo=amazonaws&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow?style=flat-square)

> Because "it was probably red" is not a data-driven strategy.

A machine learning classifier that predicts whether a player was the Impostor in a game of Among Us — built from real player game logs, with a focus on honest modeling and leakage-free feature engineering.

---

## The Problem

In Among Us, identifying the Impostor comes down to gut instinct and social deduction. This project asks: can a machine learn to spot suspicious behavior from raw game stats alone — without ever seeing a kill?

---

## Technical Approach

### Data
- 2,227 game logs combined from 30 CSV files of real Among Us sessions
- Features engineered from raw strings: game length and task time converted from `"07m 04s"` format to seconds
- Yes/No columns encoded as binary
- Identified and removed **3 sources of data leakage** that directly revealed player role before training

### Modeling
- Baseline: `DecisionTreeClassifier`
- Improved: `RandomForestClassifier`
- Evaluation: accuracy score, confusion matrix, and feature importance analysis

### Leakage Removed

| Feature | Reason Removed |
|---|---|
| `Imposter Kills` | Always 0 for Crewmates — directly identifies role |
| `Outcome` | Game result unknown at prediction time |
| `Task Completed` | Always 0 for Impostors — directly identifies role |

---

## Tech Stack

```
Language   Python 3.11
Libraries  pandas · scikit-learn · matplotlib · jupyter
Cloud      AWS Bedrock (coming soon)
```

---

## Getting Started

**Clone the repo**
```bash
git clone https://github.com/adandemraju/amongus-imposter-predictor.git
cd amongus-imposter-predictor
```

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Run the notebooks**
```bash
jupyter notebook
```

Open `notebooks/exploration.ipynb` for data cleaning, then `notebooks/model.ipynb` for model training and evaluation.

---

## Project Structure

```
amongus-imposter-predictor/
├── data/
│   ├── *.csv                 # Raw game logs (30 files)
│   └── cleaned_data.csv      # Processed dataset
├── notebooks/
│   ├── exploration.ipynb     # Data cleaning and feature engineering
│   └── model.ipynb           # Model training and evaluation
└── README.md
```

---

## Roadmap

- [x] Data cleaning and feature engineering
- [x] Leakage identification and removal
- [x] Decision Tree baseline
- [x] Random Forest comparison
- [ ] Hyperparameter tuning
- [ ] `class_weight='balanced'` for imbalanced classes
- [ ] AWS Bedrock integration for natural language game summaries
- [ ] Interactive web app deployment

---

## Data Source

<!-- Add dataset link here -->
