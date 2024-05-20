from utils import preprocess
from scipy.spatial import distance
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

# filter out convergence warnings
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(action="ignore", category=ConvergenceWarning)


def utility(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
):
    """
    Finds utility score. Utility score is defined as the prediction performance on a test set.
    The unit of this score depends on the datatype of the target, i.e. binary, categorical, continuous,
    and is automatically inferred. Bayesian hyperparameter selection is performed.
    returns: lists of best models and corresponding best scores.

    """

    # infer datatype of target
    dt = preprocess.infer_data_type(y_train)
    assert dt in ["binary", "multiclass", "continuous"]

    models, param_search_spaces, scoring = get_pred_models_(dt=dt)

    best_models, best_scores = [], []

    # for the different types of models, find the best hyperparameters and return score on a test set
    for model, param_space in zip(models, param_search_spaces):
        print(f"we are now at model: {model}")
        best_model, best_score = get_best_model_(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            model=model,
            param_space=param_space,
            scoring=scoring,
        )
        # append not the fitted model instance, but the string of the name plus all hyperparameters
        best_models.append(f"{best_model.__class__.__name__}_{best_model.get_params()}")
        best_scores.append(best_score)

    return best_models, best_scores


def attribute_inference(
    real_data: pd.DataFrame, synthetic_data: pd.DataFrame, sensitive_field: str
):
    """
    Perform an attribute inference attack using synthetic data, by training an inference model on synthetic data and inferring on a real test set.
    In the real test set, non-sensitive key fields are known for inference.
    Note that this is the same as the utility function, only for sensitive targets and different predictors.

    """

    # key fields are all fields except for the sensitive field
    key_fields = [x for x in real_data.columns if x != sensitive_field]

    best_models, best_scores = utility(
        X_train=synthetic_data[key_fields],
        X_test=real_data[key_fields],
        y_train=synthetic_data[sensitive_field],
        y_test=real_data[sensitive_field],
    )
    return best_models, best_scores


def authenticity(
    real_data: pd.DataFrame, synthetic_data: pd.DataFrame, metric: str = "euclidean"
):
    """
    Privacy check from Alaa et al (2022). Outputs 1 for synthetic samples closer to real training set, than any other sample in the real training set.
    Intuition: will output mostly 1 if synthetic samples are real samples memorized, with some small noise added.
    For a good dataset, score should not be >>.5, since we expect closest samples in the real training set to be synthetic or real with equal probability.

    """
    # compute distances of closest synthetic to real records
    dist_real_syn = distance.cdist(real_data, synthetic_data, metric=metric).min(axis=1)

    # compute distances of closest real to real records
    dist_real_real = distance.cdist(real_data, real_data, metric=metric)
    # set diagonal to infinity, else every sample will be closest to itself
    dist_real_real[range(len(dist_real_real)), range(len(dist_real_real))] = float(
        "inf"
    )
    dist_real_real = dist_real_real.min(axis=1)

    # labels are 1 if the closest record to a real record is synthetic
    labels = (dist_real_syn <= dist_real_real).astype(int)

    return labels.mean()


def plot_dataframe(df: pd.DataFrame, ncols: int, rotate_labels: bool):
    """
    Loop over columns and plot in a single figure using subfigures. Histogram for numericals, countplot else.
    """

    nrows = int(np.ceil(len(df.columns) / ncols))
    fig = plt.figure(figsize=(16, 16))

    for i, col in enumerate([x for x in df.columns if x != "dataset"]):
        ax = plt.subplot(nrows, ncols, i + 1)
        dt = preprocess.infer_data_type(df[col])
        if dt == "continuous":
            sns.histplot(data=df, x=col, hue="dataset", alpha=0.5, ax=ax)
        else:
            sns.countplot(
                data=df,
                x=col,
                hue="dataset",
                stat="count",
                ax=ax,
            )

        # Remove individual legends from subplots
        if ax.get_legend():
            handles, labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()

        if rotate_labels:
            ax.tick_params(axis="x", labelrotation=45)

    # Extract legend from one subplot and place it outside the subplots
    fig.legend(handles, labels, fontsize=11, loc="upper right")
    fig.suptitle("Feature distributions", fontsize=11)

    plt.tight_layout()

    return plt


def get_projection_plot_(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    type: str = "tsne",
    n_neighbours: int = 35,
):
    X = pd.concat([real, synthetic], axis=0)
    X = X.reset_index(drop=True)

    if type == "tsne":
        emb = TSNE(
            n_components=2,
            learning_rate="auto",
            init="pca",
            perplexity=n_neighbours,
            verbose=0,
            n_iter=1000,
        ).fit_transform(X)
    else:
        raise Exception("incorrect type of projection")

    emb = pd.DataFrame(emb, index=np.arange(emb.shape[0]), columns=["comp1", "comp2"])

    labels = pd.concat(
        [
            pd.Series("Real", index=np.arange(real.shape[0])),
            pd.Series(
                "Synthetic",
                index=np.arange(real.shape[0], synthetic.shape[0] + real.shape[0]),
            ),
        ]
    )

    labels = labels.reset_index(drop=True)
    emb = emb.reset_index(drop=True)
    emb = pd.concat([emb, labels], axis=1)
    emb.columns = ["comp1", "comp2", "labels"]
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=emb, x="comp1", y="comp2", hue="labels", alpha=0.1)
    plt.title(f"tSNE plot ({n_neighbours} perplexity)", fontsize=11)
    plt.tight_layout()
    return plt


def get_utility_(data: dict):
    """
    Retrieves utility scores for a dictionary of datasets and formats output. Dictionary of datasets is in the format:
    key:[X_train,X_test,y_train,y_test]. See utility() function for what utility score means.
    """
    output = []
    for name, (X_tr, X_te, y_tr, y_te) in data.items():
        print(f"we are at subdataset: {name}")
        best_models, best_scores = utility(X_tr, X_te, y_tr, y_te)
        names = np.repeat(name, len(best_models))
        res = np.column_stack((names, best_models, best_scores))
        result = pd.DataFrame(res, columns=["dataset", "best models", "best scores"])
        if isinstance(output, pd.DataFrame):
            output = pd.concat([output, result], ignore_index=True)
        else:
            output = result.copy()
    return output


def get_column_plots_(
    real_data: pd.DataFrame,
    regular_synthetic: pd.DataFrame,
    decoded_synthetic: pd.DataFrame,
    config: dict,
):
    """
    Get plots of real data, regular synthetic data, and decoded synthetic projections.
    All columns are plotted in a single plot.
    """
    real_data = real_data.copy()
    regular_synthetic = regular_synthetic.copy()
    decoded_synthetic = decoded_synthetic.copy()
    cat_features = config["cat_features"]

    # recode data back to original size (remove one hot encoding)
    real_data = preprocess.decoding_onehot(real_data, cat_features)
    regular_synthetic = preprocess.decoding_onehot(regular_synthetic, cat_features)
    decoded_synthetic = preprocess.decoding_onehot(decoded_synthetic, cat_features)

    # remove feature string from categories to shorten categories for plotting
    def remove_feature_str(cat):
        keep = cat.split("_")[1:]
        return "_".join(keep)

    real_data = real_data.map(
        lambda x: remove_feature_str(x) if isinstance(x, str) else x
    )
    regular_synthetic = regular_synthetic.map(
        lambda x: remove_feature_str(x) if isinstance(x, str) else x
    )
    decoded_synthetic = decoded_synthetic.map(
        lambda x: remove_feature_str(x) if isinstance(x, str) else x
    )

    # concatenate dataframes with a hue column for easy plotting in same figure
    full_df = pd.concat(
        [
            real_data.assign(dataset="Real"),
            regular_synthetic.assign(dataset="Regular Synthetic"),
            decoded_synthetic.assign(dataset="Decoded Synthetic"),
        ]
    )

    plot = plot_dataframe(df=full_df, ncols=2, rotate_labels=True)

    return plot


def get_binary_models_():
    models = [
        LogisticRegression(solver="saga", penalty="l2"),
        # LogisticRegression(solver="liblinear"),
        # GradientBoostingClassifier(),
        RandomForestClassifier(n_jobs=-1),
    ]

    param_search_spaces = [
        {
            "C": Real(1e-3, 1e3),
        },  # LR
        # {"penalty": Categorical(["l2", "l1"]), "C": Real(1e-3, 1e6)},
        # {
        #     "learning_rate": Real(1e-4, 1e-1),
        #     "n_estimators": Integer(1e0, 1e3),
        #     "max_depth": Integer(1, 10),
        # },  # GB
        {
            "n_estimators": Integer(1e0, 1e3),
            "max_depth": Integer(1, 25),
        },  # RF
    ]
    return models, param_search_spaces


def get_multiclass_models_():

    models = [
        LogisticRegression(multi_class="multinomial", solver="saga", penalty="l2"),
        # GradientBoostingClassifier(),
        RandomForestClassifier(n_jobs=-1),
    ]

    param_search_spaces = [
        {
            "C": Real(1e-3, 1e3),
        },  # LR
        # {
        #     "learning_rate": Real(1e-4, 1e-1),
        #     "n_estimators": Integer(1e0, 1e3),
        #     "max_depth": Integer(1, 10),
        # },  # GB
        {
            "n_estimators": Integer(1e0, 1e3),
            "max_depth": Integer(1, 25),
        },  # RF
    ]

    return models, param_search_spaces


def get_regression_models_():
    # SGD regressor for same type of tuning as in classification
    models = [
        Ridge(solver="saga"),
        # GradientBoostingRegressor()
        RandomForestRegressor(n_jobs=-1),
    ]
    param_search_spaces = [
        {
            "alpha": Real(1e-3, 1e3),
        },
        # {
        #     "learning_rate": Real(1e-4, 1e-1),
        #     "n_estimators": Integer(1e0, 1e3),
        #     "max_depth": Integer(1, 10),
        # },  # GB
        {
            "n_estimators": Integer(1e0, 1e3),
            "max_depth": Integer(1, 25),
        },  # RF
    ]

    return models, param_search_spaces


def get_pred_models_(dt):

    # get list of models
    if dt == "binary":
        models, param_search_spaces = get_binary_models_()
        scoring = make_scorer(matthews_corrcoef)
    elif dt == "multiclass":
        models, param_search_spaces = get_multiclass_models_()
        scoring = make_scorer(matthews_corrcoef)
    elif dt == "continuous":
        models, param_search_spaces = get_regression_models_()
        scoring = "neg_root_mean_squared_error"

    return models, param_search_spaces, scoring


def get_best_model_(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model: any,
    param_space: dict,
    scoring: any,
):
    cv = 3
    n_iter = 32
    n_points = 3
    n_jobs = int(n_points * cv)

    opt = BayesSearchCV(
        model,
        param_space,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=0,
        verbose=0,
        n_jobs=n_jobs,
        n_points=n_points,
    )
    opt.fit(X_train, y_train)
    return opt.best_estimator_, opt.score(X_test, y_test)


def get_attribute_inference_(
    data: pd.DataFrame, config: dict, sd_sets: list = [], real_key: str = "real"
):
    """
    Retrieves attribute inference scores for a dictionary of datasets and formats output. Dictionary of datasets is in the format:
    key:[X_train,X_test,y_train,y_test]. See attribute_inference() function for what attribute inference score means.

    sd_sets: keys of datasets which are synthetic (and thus used to train attribute inference models)
    real_key: key which denotes the real dataset (and thus used to test attribute inference models)

    """
    output = []
    real_data = data[real_key][0]
    sensitive_fields = config["sensitive_fields"]
    for name, (X_tr, X_te, y_tr, y_te) in data.items():
        if name in sd_sets:
            print(f"we are at subdataset: {name}")
            for sensitive_field in sensitive_fields:
                # decode sensitive target to single feature if one hot encoded
                rd = preprocess.decode_categorical_target(
                    data=real_data, target=sensitive_field
                )
                sd = preprocess.decode_categorical_target(
                    data=X_tr, target=sensitive_field
                )

                best_models, best_scores = attribute_inference(
                    real_data=rd,
                    synthetic_data=sd,
                    sensitive_field=sensitive_field,
                )
                names = np.repeat(name, len(best_models))
                fields = np.repeat(sensitive_field, len(best_models))
                res = np.column_stack((names, fields, best_models, best_scores))
                result = pd.DataFrame(
                    res,
                    columns=[
                        "dataset",
                        "sensitive target feature",
                        "best models",
                        "best scores",
                    ],
                )
                if isinstance(output, pd.DataFrame):
                    output = pd.concat([output, result], ignore_index=True)
                else:
                    output = result.copy()
    return output


def get_authenticity_(data: pd.DataFrame, sd_sets: list = [], real_key: str = "real"):
    """
    Retrieves authenticity scores for a dictionary of datasets and formats output. Dictionary of datasets is in the format:
    key:[X_train,X_test,y_train,y_test]. See authenticity() function for what authenticity score means.

    sd_sets: list of keys of datasets which are synthetic (and thus used to train attribute inference models)
    real_key: single key which denotes the real dataset (and thus used to test attribute inference models)

    """

    output = []
    real_data = data[real_key][0]
    for name, (X_tr, _, _, _) in data.items():
        if name in sd_sets:
            print(f"we are at subdataset: {name}")
            score = authenticity(
                real_data=real_data, synthetic_data=X_tr, metric="euclidean"
            )
            result = pd.DataFrame(
                np.array([[name], [score]]).T, index=[0], columns=["dataset", "scores"]
            )
            if isinstance(output, pd.DataFrame):
                output = pd.concat([output, result], ignore_index=True)
            else:
                output = result.copy()
    return output
