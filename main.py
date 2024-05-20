from utils import preprocess, sd, inference
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from joblib import Parallel, delayed


def exec__(
    ds: str,
    sd_model: str,
    sd_model_args: dict,
    proj_model_args: dict,
):
    """
    Function to retrieve results of the paper: High Utility Low Fidelity Synthetic Data. Retrieves results
    for a specific dataset in the UCI ML repository, configurated in the datasets.json file, and a specific
    synthetic data generating model from the Synthetic Data Vault.

    """
    assert ds in ["adult", "bank", "credit", "shopping"]
    assert sd_model in ["copula", "gan", "vae"]
    print(f"Working on dataset: {ds} and synthetic data model: {sd_model}")
    random_state = 999  # for reproducible random data splitting

    result_path = os.path.join("results", ds, sd_model)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # load data
    with open("datasets.json", "r") as f:
        config = json.load(f)
    config = config[ds]
    dataset = fetch_ucirepo(id=config["id"])
    X = dataset.data.features
    y = dataset.data.targets

    # setup data dictionary storing all data
    data = {}

    # fully preprocess real data
    print("preprocessing data")
    X_train, X_test, y_train, y_test = preprocess.preprocess(
        X=X, y=y, config=config, random_state=random_state
    )
    data["real"] = [X_train, X_test, y_train, y_test]

    # generate regular synthetic data
    print("generating regular synthetic samples")
    X_train, X_test, y_train, y_test = preprocess.sd_pre_preprocess(
        X=X, y=y, config=config, random_state=random_state
    )
    syn_X, syn_y = sd.sdv_generate(
        X=X_train,
        y=y_train,
        sample_size=X.shape[0],
        model=sd_model,
        model_args=sd_model_args,
    )
    # preprocess synthetic data
    syn_X_train, syn_X_test, syn_y_train, syn_y_test = preprocess.sd_preprocess(
        syn_X=syn_X,
        real_X=pd.concat([X_train, X_test]),
        syn_y=syn_y,
        config=config,
        random_state=random_state,
    )
    data["regular_synthetic"] = [
        syn_X_train,
        data["real"][1],
        syn_y_train,
        data["real"][3],
    ]

    # generate synthetic projections
    print("generating synthetic projections")
    latent_dim = data["real"][0].shape[1]

    proj_model = sd.EncoderModel(
        latent_dim=latent_dim, input_dim=latent_dim, compression_layers=[]
    )

    proj_model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[sd.Keras_binary_MCC()],
    )

    sd.fit_model(
        model=proj_model,
        X=data["real"][0],
        y=data["real"][2],
        model_args=proj_model_args,
        monitor="val_MCC",
    )

    encoder, predictor = proj_model.encoder, proj_model.predictor
    proj_X_train = pd.DataFrame(
        encoder.predict(data["real"][0]),
        columns=["column_" + str(x) for x in list(range(latent_dim))],
    )
    proj_X_test = pd.DataFrame(
        encoder.predict(data["real"][1]),
        columns=["column_" + str(x) for x in list(range(latent_dim))],
    )
    data["real_projected"] = [
        proj_X_train,
        proj_X_test,
        data["real"][2],
        data["real"][3],
    ]

    syn_proj_X, syn_proj_y = sd.sdv_generate(
        X=proj_X_train,
        y=data["real"][2],
        sample_size=X.shape[0],
        model=sd_model,
        model_args=sd_model_args,
        projected=True,
    )

    syn_proj_X_train, syn_proj_X_test, syn_proj_y_train, syn_proj_y_test = (
        train_test_split(
            syn_proj_X,
            syn_proj_y,
            stratify=syn_proj_y,
            test_size=0.3,
            random_state=random_state,
        )
    )
    data["synthetic_projected"] = [
        syn_proj_X_train,
        proj_X_test,
        syn_proj_y_train,
        data["real"][3],
    ]

    # generate decoded synthetic projections
    print("generating decoded synthetic projections")
    decoder = encoder.flip()
    syn_dec_X_train = pd.DataFrame(
        decoder.predict(data["synthetic_projected"][0]), columns=data["real"][0].columns
    )
    syn_dec_X_train = preprocess.postprocess_projections(
        data=syn_dec_X_train, config=config
    )
    data["synthetic_decoded"] = [
        syn_dec_X_train,
        data["real"][1],
        syn_proj_y_train,
        data["real"][3],
    ]  # or should we just use y_train?

    # find fidelity
    print("Making column plots")
    fid_plots = inference.get_column_plots_(
        data["real"][0],
        data["regular_synthetic"][0],
        data["synthetic_decoded"][0],
        config,
    )
    fid_plots.savefig(os.path.join(result_path, "fidelity_plot.png"))
    print("Making tSNE plot")
    fid_tsne = inference.get_projection_plot_(
        real=data["real_projected"][0],
        synthetic=data["synthetic_projected"][0],
        type="tsne",
    )
    fid_tsne.savefig(os.path.join(result_path, "tsne_plot.png"))

    # find utility
    print("getting utility")
    ut = inference.get_utility_(data=data)
    ut.to_csv(os.path.join(result_path, "utility.csv"), sep="\t")

    # find privacy
    print("getting attribute inference")
    aia = inference.get_attribute_inference_(
        data=data, sd_sets=["regular_synthetic", "synthetic_decoded"], config=config
    )
    aia.to_csv(os.path.join(result_path, "attribute_inference.csv"), sep="\t")
    print("getting authenticity")
    auth = inference.get_authenticity_(
        data=data,
        sd_sets=["regular_synthetic", "synthetic_decoded"],
    )
    auth.to_csv(os.path.join(result_path, "authenticity.csv"), sep="\t")


if __name__ == "__main__":
    datasets = ["adult"]
    sd_models = ["copula", "vae", "gan"]
    sd_model_args = {}
    proj_model_args = {"epochs": 300, "batch_size": 512}

    # get results
    for ds in datasets:
        for sd_model in sd_models:
            exec__(
                sd_model=sd_model,
                ds=ds,
                sd_model_args=sd_model_args,
                proj_model_args=proj_model_args,
            )
        # Parallel(
        #     n_jobs=-1,
        # )(
        #     delayed(exec__)(
        #         sd_model=sd_model,
        #         ds=ds,
        #         sd_model_args=sd_model_args,
        #         proj_model_args=proj_model_args,
        #     )
        #     for sd_model in sd_models
        # )
