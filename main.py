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

# this dictionary will hold all the data we want to get results for
data = {}

# drop unnecessary features
X = X.drop(config[ds]["drop_features"], axis=1)

# generate regular synthetic dataset (based on training set from same indices as later)
# X_train, _, y_train, _ = preprocess.traintest_split(X, y)
# syn_X, syn_y = sd.generate(X_train, y_train, sample_size=X.shape[0], model="copula")

# general preprocessing: one hot encoding, normalization
X_train, X_test, y_train, y_test = preprocess.preprocess(
    X,
    y,
    cat_features=config[ds]["cat_features"],
    num_features=config[ds]["num_features"],
)
data["real"] = [X_train, X_test, y_train, y_test]

# syn_X_train, syn_X_test, syn_y_train, syn_y_test = preprocess.preprocess(
#     syn_X,
#     syn_y,
#     cat_features=config[ds]["cat_features"],
#     num_features=config[ds]["num_features"],
# )
# data["synthetic"] = [syn_X_train, syn_X_test, syn_y_train, y_test]

# train the encoder network
encoder, _, _ = project.encoder_network(
    X_train,
    y_train,
    latent_size=X_train.shape[1],
    model_args={"batch_size": 128, "epochs": 5},
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
data["projected_real"] = [emb_X_train, emb_X_test, y_train, y_test]

# create synthetic projections
# syn_emb_X, syn_emb_y = sd.generate(
#     emb_X_train, y_train, model="copula", sample_size=X.shape[0], projected=True
# )
# # split into train and test set
# syn_emb_X_train, syn_emb_X_test, syn_emb_y_train, syn_emb_y_test = (
#     preprocess.traintest_split(syn_emb_X, syn_emb_y)
# )
# data["projected_synthetic"] = [syn_emb_X_train, syn_emb_X_test, syn_emb_y_train, y_test]

# pass projections through flipped encoder
flipped_encoder = project.flip_encoder(encoder)

# check the reconstruction
test_rec = flipped_encoder.predict(encoder.predict(X_train))
import numpy as np

print(np.round(X_train, 3))
print(pd.DataFrame(np.round(test_rec, 3), columns=X_train.columns))
exit()

dec_X_train = flipped_encoder.predict(emb_X_train)
dec_X_test = flipped_encoder.predict(emb_X_test)
data["decoded_real"] = [dec_X_train, dec_X_test, y_train, y_test]

syn_dec_X_train = flipped_encoder.predict(syn_emb_X_train)
syn_dec_X_test = flipped_encoder.predict(syn_emb_X_test)
data["decoded_synthetic"] = [syn_dec_X_train, syn_dec_X_test, syn_emb_y_train, y_test]

# get the different models and corresponding parameter search spaces for Bayesian hyperparameter tuning
models_, param_search_spaces = pred_models.get_models_()

# setup result directory
result_path = "results"
if not os.path.exists(result_path):
    os.makedirs(result_path)


with open(os.path.join(result_path, "results.txt"), "w") as file:
    # loop over the different datasets for which we want to get results
    for name, (X_tr, X_te, y_tr, y_te) in data.items():
        # loop over different possible models and parameter search spaces
        for model, param_space in zip(models_, param_search_spaces):
            best_model, test_score = pred_models.get_best_model_(
                X_tr, X_te, y_tr, y_te, model, param_space
            )
            file.write(
                f"Dataset: {name} \n Model: {best_model} \n Best test score: {test_score} \n \n"
            )
            file.flush()

    # after looping through all the 'regular' models, also check the decoders prediction performance on a test set
