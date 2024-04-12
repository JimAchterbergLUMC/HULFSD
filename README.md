# HULFSD

## Repository for High Utility Low Fidelity Synthetic Data 

Computes synthetic embeddings to be used in prediction tasks.

## Datasets
datasets.json files stores retrievable information on the different datasets.

## Usage
The main.py file runs everything; generates all forms of synthetic data, calculates utility and privacy metrics, generates plots, etc.

In the utils folder you can find all the helper functions:
- inference.py: utility scoring function, attribute inference scoring function, sample authenticity scoring function
- fidelity.py: feature-wise plot generating function
- pred_models.py: retrieve prediction models and their respective parameter spaces
- preprocess.py: general and dataset specific preprocessing
- sd.py: synthetic data generating functions.

