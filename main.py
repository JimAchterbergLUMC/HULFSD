from ucimlrepo import fetch_ucirepo
from utils import preprocess, models, gen_synthetic, embed
import json
import os
import pandas as pd


# open configuration of datasets
with open("datasets.json", "r") as f:
    config = json.load(f)

# set which dataset we are using (and assert it is one of the options)
ds = "adult"
assert ds in ["adult"]

# fetch real dataset
dataset = fetch_ucirepo(id=config[ds]["id"])
# load features and target
X = dataset.data.features
y = dataset.data.targets

# dataset specific preprocessing
if ds == "adult":
    X, y = preprocess.preprocess_adult(X, y)
else:
    pass

# drop unnecessary features
X = X.drop(config[ds]["drop_features"], axis=1)

# generate regular synthetic dataset
syn_X, syn_y = gen_synthetic.generate(X, y, model="copula")


# general preprocessing: one hot encoding, normalization
X_train, X_test, y_train, y_test = preprocess.preprocess(
    X,
    y,
    cat_features=config[ds]["cat_features"],
    num_features=config[ds]["num_features"],
)
syn_X_train, syn_X_test, syn_y_train, syn_y_test = preprocess.preprocess(
    syn_X,
    syn_y,
    cat_features=config[ds]["cat_features"],
    num_features=config[ds]["num_features"],
)

# create real embedding dataset
emb_X = pd.DataFrame(
    embed.fa(pd.concat([X_train, X_test], axis=0)), columns=X_train.columns
)
emb_X_train, emb_X_test, _, _ = preprocess.split_embedding_set(emb_X, y)

# create synthetic embedding dataset from real embedding dataset
syn_emb_X, syn_emb_y = gen_synthetic.generate(emb_X, y, model="copula", embedded=True)
syn_emb_X_train, syn_emb_X_test, syn_emb_y_train, syn_emb_y_test = (
    preprocess.split_embedding_set(syn_emb_X, syn_emb_y)
)
# get the different models and corresponding parameter search spaces for Bayesian hyperparameter tuning
models_, param_search_spaces = models.get_models_()

# setup result directory
result_path = "results"
if not os.path.exists(result_path):
    os.makedirs(result_path)

with open(os.path.join(result_path, "results.txt"), "w") as file:
    # loop over different possible models and parameter search spaces
    for model, param_space in zip(models_, param_search_spaces):
        best_model, test_score = models.get_best_model_(
            X_train, X_test, y_train, y_test, model, param_space
        )

        syn_best_model, syn_test_score = models.get_best_model_(
            syn_X_train, X_test, syn_y_train, y_test, model, param_space
        )

        emb_best_model, emb_test_score = models.get_best_model_(
            emb_X_train, emb_X_test, y_train, y_test, model, param_space
        )

        syn_emb_best_model, syn_emb_test_score = models.get_best_model_(
            syn_emb_X_train, emb_X_test, syn_emb_y_train, y_test, model, param_space
        )

        print(best_model)
        print(test_score)
        file.write(f"Real: Model: {best_model} Best test score: {test_score} \n")
        print(emb_best_model)
        print(emb_test_score)
        file.write(
            f"Embedded: Real: Model: {emb_best_model} Best test score: {emb_test_score} \n"
        )
        print(syn_best_model)
        print(syn_test_score)
        file.write(
            f"Synthetic: Model: {syn_best_model} Best test score: {syn_test_score} \n"
        )
        print(syn_emb_best_model)
        print(syn_emb_test_score)
        file.write(
            f"Embedded: Synthetic: Model: {syn_emb_best_model} Best test score: {syn_emb_test_score} \n"
        )
