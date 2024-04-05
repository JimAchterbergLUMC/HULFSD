from ucimlrepo import fetch_ucirepo
from utils import pred_models, preprocess, project, sd
import json
import os
import pandas as pd


# open configuration of datasets
with open("datasets.json", "r") as f:
    config = json.load(f)

# set which dataset we are using (and assert it is one of the options)
ds = "adult"
assert ds in ["adult", "bank", "credit", "shopping"]

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

# generate regular synthetic dataset (based on training set from same indices as later)
X_train, _, y_train, _ = preprocess.traintest_split(X, y)
syn_X, syn_y = sd.generate(X_train, y_train, sample_size=X.shape[0], model="copula")

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

# train the encoder-decoder network
encoder, decoder, ae = project.encoder_decoder(
    X_train,
    y_train,
    latent_size=X_train.shape[1],
    model_args={"batch_size": 128, "epochs": 100},
    activation="relu",
)
# project data through the encoder
emb_X_train = pd.DataFrame(
    encoder.predict(X_train),
    columns=["column_" + str(x) for x in list(range(X_train.shape[1]))],
)
# note that the test data was not seen during encoder training
emb_X_test = pd.DataFrame(
    encoder.predict(X_test),
    columns=emb_X_train.columns,
)

# create synthetic embedding dataset from (full) real embedding dataset
# we might need to change full dataset to only training set? But then we should also do this the first time...
syn_emb_X, syn_emb_y = sd.generate(
    emb_X_train, y_train, model="copula", sample_size=X.shape[0], embedded=True
)
# rows are randomly sampled from the SD generating model. so for train test split we can just take the first k rows as train set.
syn_emb_X_train, syn_emb_X_test, syn_emb_y_train, syn_emb_y_test = (
    syn_emb_X[: emb_X_train.shape[0]],
    syn_emb_X[emb_X_train.shape[0] :],
    syn_emb_y[: emb_X_train.shape[0]],
    syn_emb_y[emb_X_train.shape[0] :],
)

# get the different models and corresponding parameter search spaces for Bayesian hyperparameter tuning
models_, param_search_spaces = pred_models.get_models_()

# setup result directory
result_path = "results"
if not os.path.exists(result_path):
    os.makedirs(result_path)

with open(os.path.join(result_path, "results.txt"), "w") as file:
    # loop over different possible models and parameter search spaces
    for model, param_space in zip(models_, param_search_spaces):
        best_model, test_score = pred_models.get_best_model_(
            X_train, X_test, y_train, y_test, model, param_space
        )

        syn_best_model, syn_test_score = pred_models.get_best_model_(
            syn_X_train, X_test, syn_y_train, y_test, model, param_space
        )

        emb_best_model, emb_test_score = pred_models.get_best_model_(
            emb_X_train, emb_X_test, y_train, y_test, model, param_space
        )

        syn_emb_best_model, syn_emb_test_score = pred_models.get_best_model_(
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
