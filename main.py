from ucimlrepo import fetch_ucirepo
from utils import preprocess, sd, inference, enc
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def exec__(ds: str, sd_model: str):
    assert ds in ["adult", "bank", "credit", "shopping"]
    assert sd_model in ["copula", "gan", "vae"]
    random_state = 999  # for reproducible random data splitting

    # load data
    with open("datasets.json", "r") as f:
        config = json.load(f)
    config = config[ds]
    dataset = fetch_ucirepo(id=config["id"])
    X = dataset.data.features
    y = dataset.data.targets

    # setup data dictionary storing all data
    data = {}

    # generate regular synthetic data
    X_train, X_test, y_train, y_test = preprocess.sd_preprocess(
        X=X, y=y, config=config, random_state=random_state
    )
    syn_X, syn_y = sd.generate(
        X=X_train, y=y_train, sample_size=X.shape[0], model=sd_model, model_args={}
    )

    # fully preprocess real data
    X_train, X_test, y_train, y_test = preprocess.preprocess(
        X=X, y=y, config=config, random_state=random_state
    )
    data["real"] = [X_train, X_test, y_train, y_test]

    # preprocess synthetic data
    syn_X_train, syn_X_test, syn_y_train, syn_y_test = preprocess.preprocess(
        X=syn_X, y=syn_y, config=config, random_state=random_state
    )
    data["regular_synthetic"] = [syn_X_train, X_test, syn_y_train, y_test]

    # generate synthetic projections
    latent_dim = X_train.shape[1]
    proj_model = enc.EncoderModel(
        latent_dim=latent_dim, input_dim=latent_dim, compression_layers=[]
    )
    proj_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
    sd.fit_model(
        model=proj_model,
        X=X_train,
        y=y_train,
        model_args={"batch_size": 512, "epochs": 100},
        monitor="val_AUC",
    )
    encoder, predictor = proj_model.encoder, proj_model.predictor
    proj_X_train = pd.DataFrame(
        encoder.predict(X_train),
        columns=["column_" + str(x) for x in list(range(latent_dim))],
    )
    proj_X_test = pd.DataFrame(
        encoder.predict(X_test),
        columns=["column_" + str(x) for x in list(range(latent_dim))],
    )
    data["real_projected"] = [proj_X_train, proj_X_test, y_train, y_test]
    syn_proj_X, syn_proj_y = sd.generate(
        X=proj_X_train,
        y=y_train,
        sample_size=X.shape[0],
        model=sd_model,
        model_args={},
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
        y_test,
    ]

    # generate decoded synthetic projections
    decoder = encoder.flip()
    syn_dec_X_train = pd.DataFrame(
        decoder.predict(syn_proj_X_train), columns=X_train.columns
    )
    syn_dec_X_train = preprocess.postprocess_projections(
        data=syn_dec_X_train, config=config
    )
    data["synthetic_decoded"] = [
        syn_dec_X_train,
        X_test,
        syn_proj_y_train,
        y_test,
    ]  # or should we just use y_train?

    # find fidelity
    fid_plots = inference.get_column_plots(
        data["real"][0],
        data["regular_synthetic"][0],
        data["synthetic_decoded"][0],
        config,
    )
    fid_tsne = inference.tsne_projections(
        real=data["real_projected"][0], synthetic=data["synthetic_projected"][0]
    )

    # find utility
    ut = inference.get_utility_(data=data)

    # find privacy
    aia = inference.get_attribute_inference_(
        data=data, sd_sets=["regular_synthetic", "synthetic_decoded"], config=config
    )
    auth = inference.get_authenticity_(
        data=data, sd_sets=["regular_synthetic", "synthetic_decoded"], config=config
    )

    return fid_plots, fid_tsne, ut, aia, auth


if __name__ == "__main__":
    ds = "adult"
    sd_model = "copula"

    # get results
    fid_plots, fid_tsne, ut, aia, auth = exec__(ds=ds, sd_model=sd_model)

    # save results for specific dataset and specific synthetic data generating model
    result_path = os.path.join("results", ds, sd_model)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    fid_plots.savefig(os.path.join(result_path, "fidelity_plot.png"))
    fid_tsne.savefig(os.path.join(result_path, "tsne_plot.png"))
    ut.to_csv(os.path.join(result_path, "utility.csv"))
    aia.to_csv(os.path.join(result_path, "attribute_inference.csv"))
    auth.to_csv(os.path.join(result_path, "authenticity.csv"))
