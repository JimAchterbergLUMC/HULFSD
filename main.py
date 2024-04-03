from ucimlrepo import fetch_ucirepo
from utils import preprocess, models
import json
import os

# open configuration of datasets
with open("datasets.json", "r") as f:
    config = json.load(f)

# set which dataset we are using (and assert it is one of the options)
ds = "adult"
assert ds in ["adult"]

# fetch dataset
dataset = fetch_ucirepo(id=config[ds]["id"])
# load features and target
X = dataset.data.features
y = dataset.data.targets

# dataset specific preprocessing
if ds == "adult":
    X, y = preprocess.preprocess_adult(X, y)
else:
    pass

# general preprocessing: one hot encoding, normalization, dropping features
X_train, X_test, y_train, y_test = preprocess.preprocess(
    X,
    y,
    cat_features=config[ds]["cat_features"],
    num_features=config[ds]["num_features"],
    drop_features=config[ds]["drop_features"],
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
        print(best_model)
        print(test_score)
        file.write(f"Model: {best_model} Best test score: {test_score} \n")
