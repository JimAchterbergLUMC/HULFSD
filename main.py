from utils import preprocess, sd, inference
import json
import os
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import StratifiedKFold


def exec__(
    ds: str, sd_model: str, sd_model_args: dict, proj_model_args: dict, n_splits: int
):
    """
    Function to retrieve results of the paper: High Utility Low Fidelity Synthetic Data. Retrieves results
    for a specific dataset in the UCI ML repository, configurated in the datasets.json file, and a specific
    synthetic data generating model from the Synthetic Data Vault.

    """
    assert ds in ["adult", "credit", "diabetes"]
    assert sd_model in ["copula", "gan", "vae"]
    print(f"Working on dataset: {ds} and synthetic data model: {sd_model}")
    random_state = 123  # for reproducible random data splitting

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

    # encode categories (i.e. to higher-order), drop features, cast datatypes
    X, y = preprocess.preprocess(X, y, config)

    # make k splits
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (train, test) in enumerate(cv.split(X, y)):

        # real data: one hot encode (together) and normalize (separately)
        X_ = preprocess.sklearn_preprocessor(
            processor="one-hot", data=X, features=config["cat_features"]
        )
        X_tr = preprocess.sklearn_preprocessor(
            processor="normalize", data=X_.iloc[train], features=config["num_features"]
        )
        X_te = preprocess.sklearn_preprocessor(
            processor="normalize", data=X_.iloc[test], features=config["num_features"]
        )
        data["Real"] = [
            X_tr.copy(),
            X_te.copy(),
            y.iloc[train].copy(),
            y.iloc[test].copy(),
        ]

        # regular synthetic: train model, then one hot encode (together with real data) and normalize (separately)
        print(
            f"generating regular synthetic for fold {fold} for sd model {sd_model} for dataset {ds}"
        )
        syn_X, syn_y = sd.sdv_generate(
            X=X.iloc[train],
            y=y.iloc[train],
            sample_size=X.shape[0],
            model=sd_model,
            model_args=sd_model_args,
        )
        syn_X = preprocess.sklearn_preprocessor(
            processor="one-hot",
            data=pd.concat([X, syn_X], ignore_index=True),
            features=config["cat_features"],
        )
        syn_X = syn_X[-X.shape[0] :]
        X_tr = preprocess.sklearn_preprocessor(
            processor="normalize",
            data=syn_X.iloc[train],
            features=config["num_features"],
        )
        data["Regular Synthetic"] = [
            X_tr.copy(),
            data["Real"][1].copy(),
            syn_y.iloc[train].copy(),
            data["Real"][3].copy(),
        ]

        # synthetic decoded projections: instantiate, compile, train, get encoder, project, generate synthetic, split, flip, decode synthetic, normalize and cast to categoricals
        print(
            f"generating synthetic decoded data for fold {fold} for sd model {sd_model} for dataset {ds}"
        )
        latent_dim = int(2 * data["Real"][0].shape[1])
        proj_model = sd.EncoderModel(
            latent_dim=latent_dim,
            input_dim=data["Real"][0].shape[1],
            compression_layers=[],
            activation="linear",
        )

        sd.fit_model(
            model=proj_model,
            X=data["Real"][0],
            y=data["Real"][2],
            loss="binary_crossentropy",
            metric="accuracy",
            model_args=proj_model_args,
        )
        encoder, predictor = proj_model.encoder, proj_model.predictor
        X_tr = pd.DataFrame(
            encoder.predict(data["Real"][0]),
            columns=["column_" + str(x) for x in list(range(latent_dim))],
        )
        X_te = pd.DataFrame(
            encoder.predict(data["Real"][1]),
            columns=["column_" + str(x) for x in list(range(latent_dim))],
        )
        data["Real Projected"] = [X_tr, X_te]
        syn_X, _ = sd.sdv_generate(
            X=X_tr,
            y=pd.Series(0, index=list(range(X_tr.shape[0])), name="y"),
            # y=data["Real"][2],
            sample_size=X.shape[0],
            model=sd_model,
            model_args=sd_model_args,
            projected=True,
        )
        syn_y = pd.DataFrame(
            predictor.predict(syn_X), columns=[data["Real"][2].name]
        ).squeeze()
        syn_y = (syn_y.round()).astype(int)
        data["Synthetic Projected"] = [syn_X.iloc[train], syn_X.iloc[test]]

        decoder = sd.Decoder(output_dim=data["Real"][0].shape[1])
        sd.fit_model(
            model=decoder,
            X=data["Real Projected"][0],
            y=data["Real"][0],
            loss="mean_squared_error",
            metric="mean_absolute_error",
            model_args=proj_model_args,
        )

        # decoder = encoder.flip()
        X_tr = pd.DataFrame(
            decoder.predict(syn_X.iloc[train]), columns=data["Real"][0].columns
        )
        # normalize, get multiclass categories, round binaries.
        X_tr = preprocess.postprocess_projections(
            data=X_tr, config=config, normalize=False
        )

        data["Synthetic Decoded"] = [
            X_tr.copy(),
            data["Real"][1].copy(),
            syn_y.iloc[train].copy(),
            data["Real"][3].copy(),
        ]

        # get fidelity
        print(
            f"getting fidelity for fold: {fold} for sd model: {sd_model} for dataset {ds}"
        )
        if fold in [0, 4, 9]:
            fid_tsne = inference.get_projection_plot_(
                real=data["Real Projected"][0],
                synthetic=data["Synthetic Projected"][0],
                type="tsne",
                n_neighbours=35,
            )
            fid_tsne.savefig(os.path.join(result_path, f"tsne_plot_{fold}.png"))

        # remove projections data, no longer required
        data.pop("Real Projected")
        data.pop("Synthetic Projected")

        fid_plots = inference.get_column_plots_(
            data,
            config,
        )
        fid_plots.savefig(os.path.join(result_path, f"fidelity_plot_{fold}.png"))

        # get utility, aia, authenticity for current fold
        print(
            f"getting utility for fold: {fold} and sd model: {sd_model} for dataset {ds}"
        )
        # get utility for either/both encoder/sklearn models
        utility = inference.get_utility_(
            data=data, encoder=False, encoder_args=proj_model_args, sklearn=True
        )
        print(f"getting AIA for fold: {fold} and sd model: {sd_model}")
        # attribute inference will search for a good tabular model through cross-validation
        aia = inference.get_attribute_inference_(
            data=data, config=config, sd_sets=["Regular Synthetic", "Synthetic Decoded"]
        )
        print(
            f"getting authenticity for fold: {fold} and sd model: {sd_model} for dataset {ds}"
        )
        auth = inference.get_authenticity_(
            data=data, sd_sets=["Regular Synthetic", "Synthetic Decoded"]
        )
        # update output stats tables with results from this fold
        if fold == 0:
            # at first fold initialize running stats with zeros
            ut_stats = inference.RunningStats(df=utility)
            aia_stats = inference.RunningStats(df=aia)
            auth_stats = inference.RunningStats(df=auth)
        ut_stats.update(utility)
        aia_stats.update(aia)
        auth_stats.update(auth)
        # during each fold write stats tables to csv to see current outputs
        ut_stats.mean_value.to_csv(
            os.path.join(result_path, "utility_mean.csv"), sep="\t"
        )
        ut_stats.standard_deviation.to_csv(
            os.path.join(result_path, "utility_std.csv"), sep="\t"
        )
        aia_stats.mean_value.to_csv(os.path.join(result_path, "aia_mean.csv"), sep="\t")
        aia_stats.standard_deviation.to_csv(
            os.path.join(result_path, "aia_std.csv"), sep="\t"
        )
        auth_stats.mean_value.to_csv(
            os.path.join(result_path, "authenticity_mean.csv"), sep="\t"
        )
        auth_stats.standard_deviation.to_csv(
            os.path.join(result_path, "authenticity_std.csv"), sep="\t"
        )


if __name__ == "__main__":
    datasets = ["credit"]
    sd_models = ["vae"]
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
                n_splits=10,
            )
