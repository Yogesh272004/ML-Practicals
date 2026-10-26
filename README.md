# Machine Learning Practicals

This repository contains six beginner-friendly ML practicals and small illustrative CSV datasets.

## Experiments
1. Linear Regression on an illustrative Boston Housing-style dataset
2. Logistic Regression for Titanic survival classification
3. Decision Tree classification on Adult Census Income-style data
4. Random Forest classification on Adult Census Income-style data
5. AdaBoost classification on Adult Census Income-style data
6. Hierarchical Clustering on Wholesale Customers-style data

## Important dataset note
The included CSVs are tiny, illustrative teaching datasets (10 rows each), not the original full datasets. Their scores are not meaningful for real-world evaluation. Use official datasets for formal research or when your instructor requires the original data.

## Requirements
Install Python 3.10+ and Git, then open a terminal in the cloned repository.

```bash
python -m pip install -r requirements.txt
```

On Windows, if `python` is not recognized, try `py` instead.

## Run an experiment
Run commands from the repository root folder so the relative dataset paths work.

```bash
python experiments/experiment1_boston.py
python experiments/experiment2_titanic.py
python experiments/experiment3_decision_tree.py
python experiments/experiment4_random_forest.py
python experiments/experiment5_boosting.py
python experiments/experiment6_hierarchical.py
```

## Clone this repository
After you upload this folder to your own GitHub repository:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
python -m pip install -r requirements.txt
python experiments/experiment4_random_forest.py
```
