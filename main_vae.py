from ucimlrepo import fetch_ucirepo
from utils import fidelity, preprocess, sd, inference, vae
import json
import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

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
    X, y = preprocess.preprocess_adult(
        X, y, config[ds]["num_features"], config[ds]["drop_features"]
    )
else:
    pass


# this dictionary will hold all the data we want to get results for
data = {}

# drop unnecessary features
# X = X.drop(config[ds]["drop_features"], axis=1)

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
data["real"] = [X_train, X_test, y_train, y_test]

(
    syn_X_train,
    syn_X_test,
    syn_y_train,
    syn_y_test,
) = preprocess.preprocess(
    syn_X,
    syn_y,
    cat_features=config[ds]["cat_features"],
    num_features=config[ds]["num_features"],
)
data["synthetic"] = [syn_X_train, X_test, syn_y_train, y_test]


# reshuffle data according to original dataframe
X_train = preprocess.reshuffle(ori_X=X, prepr_X=X_train)
X_test = preprocess.reshuffle(ori_X=X, prepr_X=X_test)
syn_X_train = preprocess.reshuffle(ori_X=X, prepr_X=syn_X_train)
syn_X_test = preprocess.reshuffle(ori_X=X, prepr_X=syn_X_test)

# find num_classes list based on original data
class_list = preprocess.get_num_classes_(X)

# retrieve VAE (and its individual fitted components)
latent_dim = 60
hulf_vae = vae.VAE(
    latent_dim=latent_dim,
    compression_layers=[100],
    decompression_layers=[100],
    num_classes=class_list,
    loss_factor=1,  # how much more important is performance than regularization
)

# KL loss is added as regularization loss inside the model
hulf_vae.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])

# fit the VAE
sd.fit_model(
    model=hulf_vae,
    X=X_train,
    y=y_train,
    model_args={"epochs": 100, "batch_size": 128},
    monitor="val_loss",
)
# retrieve fitted instances of the encoder, decoder and predictor
encoder, decoder, predictor = hulf_vae.encoder, hulf_vae.decoder, hulf_vae.predictor

# generate data from the VAE and split into train and test
sample_size = (X.shape[0], latent_dim)
z = np.random.normal(loc=0.0, scale=1.0, size=sample_size)
decoded = decoder.predict(z)
vae_X = pd.DataFrame(decoded, columns=X_train.columns).astype(float)
# vae_y = (
#     pd.DataFrame(predictor.predict(vae_X), columns=[y_train.name])
#     .astype(float)
#     .round()
#     .astype(int)
# ).squeeze()

# print(vae_X)
# print(vae_y)

vae_X = preprocess.decode_datatypes(
    data=vae_X,
    cat_features=config[ds]["cat_features"],
)

# lets try to use regular y for training as well, so we are not synthetically generating this!
vae_X_train, vae_X_test, vae_y_train, vae_y_test = preprocess.traintest_split(
    X=vae_X, y=y
)

data["vae"] = [vae_X_train, X_test, vae_y_train, y_test]

# setup result directory
result_path = os.path.join("results", ds, "vae")
if not os.path.exists(result_path):
    os.makedirs(result_path)

# get results and write them to a text file
with open(os.path.join(result_path, "results.txt"), "w") as file:
    # check fidelity of regular synthetic and decoded synthetic projections (perhaps also add income as feature)
    print("generating plots")

    plots = fidelity.get_column_plots(
        real_data=data["real"][0],
        regular_synthetic=data["synthetic"][0],
        decoded_synthetic=preprocess.sklearn_preprocessor(
            "normalize",
            data=data["vae"][0],
            features=config[ds]["num_features"],
        ),
        cat_features=config[ds]["cat_features"],
        num_features=config[ds]["num_features"],
    )

    plots.savefig(os.path.join(result_path, "fidelity.png"))

    # loop over the different datasets for which we want to get results
    for name, (X_tr, X_te, y_tr, y_te) in data.items():
        print(f"at dataset {name}")
        print("calculating utility")
        file.write(f"Dataset: {name} \n")
        best_models, best_scores = inference.utility(
            X_train=X_tr, X_test=X_te, y_train=y_tr, y_test=y_te
        )

        result = pd.DataFrame(best_scores, columns=best_models, index=["test_score"])
        # also check decoder performance on test set?
        file.write(f"Utility result: \n")
        file.write(result.to_string(header=True, index=True))
        file.write("\n")

        # for synthetic and decoded synthetic projections, perform attribute inference on each possible sensitive field
        if name in ["synthetic", "vae"]:
            # attribute inference is done by predicting sensitive fields through leaked fields (all non-target fields, might change?)
            real_data = data["real"][0]
            sensitive_fields = config[ds]["sensitive_fields"]
            for sens in sensitive_fields:
                print(f"calculating attribute inference for {sens}")

                # do we need to process data back to non-onehot?
                best_models, best_scores = inference.attribute_inference(
                    real_data=real_data,
                    synthetic_data=X_tr,
                    sensitive_field=sens,
                )
                result = pd.DataFrame(
                    best_scores,
                    columns=best_models,
                    index=[f"{sens} inference accuracy"],
                )
                file.write(f"Privacy against inference result: \n")
                file.write(result.to_string(header=True, index=True))
                file.write("\n")

            # compute authenticity score for regular synthetic and decoded synthetic projections
            print("calculating sample authenticity")
            auth_labels = inference.authenticity(
                real_data=real_data, synthetic_data=X_tr, metric="euclidean"
            )
            result = pd.DataFrame(
                auth_labels.mean(), columns=["Average"], index=["Authenticity score"]
            )
            file.write(f"Authenticity score result: \n")
            file.write(result.to_string(header=True, index=True))
            file.write("\n")

        # end of current data loop
        file.write("\n \n")
        file.flush()
