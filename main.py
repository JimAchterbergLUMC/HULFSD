from ucimlrepo import fetch_ucirepo
from utils import fidelity, preprocess, sd, inference
import json
import os
import pandas as pd
from matplotlib import pyplot as plt

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
X_train, _, y_train, _ = preprocess.traintest_split(X, y)
syn_X, syn_y = sd.generate(X_train, y_train, sample_size=X.shape[0], model="vae")


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


# train the encoder network
encoder, _, _ = sd.encoder_network(
    X=X_train,
    y=y_train,
    latent_size=len(X_train.columns),
    model_args={"batch_size": 128, "epochs": 100},
)
# project data through the encoder

emb_X_train = pd.DataFrame(
    encoder.predict(X_train),
    columns=["column_" + str(x) for x in list(range(len(X_train.columns)))],
)
emb_X_test = pd.DataFrame(
    encoder.predict(X_test),
    columns=["column_" + str(x) for x in list(range(len(X_test.columns)))],
)

# note that the test data was not seen during encoder training
data["projected_real"] = [emb_X_train, emb_X_test, y_train, y_test]

# create synthetic projections
syn_emb_X, syn_emb_y = sd.generate(
    emb_X_train, y_train, model="copula", sample_size=X.shape[0], projected=True
)
# split into train and test set
syn_emb_X_train, syn_emb_X_test, syn_emb_y_train, syn_emb_y_test = (
    preprocess.traintest_split(syn_emb_X, syn_emb_y)
)
data["projected_synthetic"] = [syn_emb_X_train, emb_X_test, syn_emb_y_train, y_test]

# visualize projections to check if they are accurate
tsne_plot = fidelity.tsne_projections(emb_X_train, syn_emb_X_train)
tsne_plot.savefig(os.path.join("results", "adult", "tsne.png"))
tsne_plot.show()
exit()

# pass projections through flipped encoder
flipped_encoder = sd.flip_encoder(encoder)
syn_dec_X_train = pd.DataFrame(
    flipped_encoder.predict(syn_emb_X_train), columns=X_train.columns
)


# ensure data types are correct
syn_dec_X_train = preprocess.decode_projection_datatypes(
    projections=syn_dec_X_train,
    num_features=config[ds]["num_features"],
    cat_features=config[ds]["cat_features"],
)

print(syn_dec_X_train.iloc[0, :].to_list())

data["decoded_synthetic"] = [syn_dec_X_train, X_test, syn_emb_y_train, y_test]


# setup result directory
result_path = os.path.join("results", ds)
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
            data=data["decoded_synthetic"][0],
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
        if name in ["synthetic", "decoded_synthetic"]:
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
