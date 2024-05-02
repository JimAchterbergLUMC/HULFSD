from utils import preprocess
from scipy.spatial import distance
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from skopt import BayesSearchCV


def get_utility_(data: dict):
    # loops over datasets and finds best prediction model with corresponding test score for each
    output = []
    for name, (X_tr, X_te, y_tr, y_te) in data.items():
        best_models, best_scores = utility(X_tr, X_te, y_tr, y_te)
        names = np.repeat(name, len(best_models))
        res = np.column_stack((names, best_models, best_scores))
        result = pd.DataFrame(res, columns=["dataset", "best models", "best scores"])
        if isinstance(output, pd.DataFrame):
            output = pd.concat([output, result], ignore_index=True)
        else:
            output = result
        return output


def utility(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
):
    # infer datatype of target
    dt = preprocess.infer_data_type(y_train)
    assert dt in ["binary", "multiclass", "continuous"]

    # get list of models
    if dt == "binary":
        models, param_search_spaces = get_binary_models_()
        scoring = "roc_auc"
    elif dt == "multiclass":
        models, param_search_spaces = get_multiclass_models_()
        scoring = "accuracy"
    elif dt == "continuous":
        models, param_search_spaces = get_regression_models_()
        scoring = "neg_mean_squared_error"

    best_models, best_scores = [], []

    # for the different types of models, find the best hyperparameters and return score on a test set
    for model, param_space in zip(models, param_search_spaces):

        best_model, best_score = get_best_model_(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            model=model,
            param_space=param_space,
            scoring=scoring,
        )
        best_models.append(best_model)
        best_scores.append(best_score)

    return best_models, best_scores


def get_attribute_inference_(data, sd_sets, config):
    output = []
    real_data = data["real"][0]
    sensitive_fields = config["sensitive_fields"]
    for name, (X_tr, X_te, y_tr, y_te) in data.items():
        if name in sd_sets:
            for sensitive_field in sensitive_fields:
                best_models, best_scores = attribute_inference(
                    real_data=real_data,
                    synthetic_data=X_tr,
                    sensitive_field=sensitive_field,
                )
                names = np.repeat(name, len(best_models))
                res = np.column_stack((names, best_models, best_scores))
                result = pd.DataFrame(
                    res, columns=["dataset", "best models", "best scores"]
                )
                if isinstance(output, pd.DataFrame):
                    output = pd.concat([output, result], ignore_index=True)
                else:
                    output = result
    return output


def attribute_inference(
    real_data: pd.DataFrame, synthetic_data: pd.DataFrame, sensitive_field: str
):
    """
    Perform an attribute inference attack using synthetic data, by training an inference model on synthetic data and inferring on a real test set.
    In the real test set, non-sensitive key fields are known for inference.
    Note that this is the same as the utility function, only for sensitive targets and different predictors.

    """

    # find preprocessed names of sensitive fields (first word will always equal original feature name)
    c = []
    for column in real_data.columns:
        if column[: len(sensitive_field)] == sensitive_field:
            c.append(column)
    y_train_syn = synthetic_data[c]
    y_test_real = real_data[c]
    # if more than 1, decode back to single feature (with categories corresponding to column names)
    if len(c) > 1:
        y_train_syn = y_train_syn.idxmax(axis=1)
        y_test_real = y_test_real.idxmax(axis=1)
    y_train_syn, y_test_real = y_train_syn.squeeze(), y_test_real.squeeze()

    # key fields are all fields except for the sensitive fields
    key_fields = [x for x in real_data.columns if x not in c]

    best_models, best_scores = utility(
        X_train=synthetic_data[key_fields],
        X_test=real_data[key_fields],
        y_train=y_train_syn,  # potentially dataframe instead of series
        y_test=y_test_real,
    )
    return best_models, best_scores


def get_authenticity_(data, sd_sets, config):
    output = []
    real_data = data["real"][0]
    for name, (X_tr, X_te, y_tr, y_te) in data.items():
        if name in sd_sets:
            score = authenticity(
                real_data=real_data, synthetic_data=X_tr, metric="euclidean"
            )
            result = pd.DataFrame([name, score], columns=["dataset", "scores"])
            if isinstance(output, pd.DataFrame):
                output = pd.concat([output, result], ignore_index=True)
            else:
                output = result
        return output


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


def get_binary_models_():
    models = [
        # GaussianNB(),
        LogisticRegression(solver="liblinear"),
        # SVC(),
        # GradientBoostingClassifier(),
    ]

    param_search_spaces = [
        # {"var_smoothing": [1e-9, 1e-6]},  # NB
        {"penalty": ["l2", "l1"], "C": (1e-3, 1e6)},  # LR
        # {
        #     "kernel": [
        #         "linear",
        #         "poly",
        #         "rbf",
        #         "sigmoid",
        #     ],
        #     "C": (1e-6, 1e6),
        #     "degree": (1, 10),
        # },  # SVM
        # {
        #     "learning_rate": (1e-6, 1e-1),
        #     "n_estimators": (1e0, 1e5),
        #     "max_depth": (1e0, 1e2),
        # },  # GB
    ]
    return models, param_search_spaces


def get_multiclass_models_():

    models = [LogisticRegression(multi_class="multinomial", solver="saga")]
    param_search_spaces = [{"penalty": ["l2", "l1"], "C": (1e-3, 1e6)}]

    return models, param_search_spaces


def get_regression_models_():

    models = [Lasso(max_iter=1000)]
    param_search_spaces = [{"alpha": (1e-3, 1e6)}]

    return models, param_search_spaces


def get_best_model_(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model: any,
    param_space: dict,
    scoring: str,
    cv: int = 3,
):
    opt = BayesSearchCV(
        model,
        param_space,
        n_iter=3,
        cv=cv,
        scoring=scoring,
        random_state=0,
        verbose=0,
        n_jobs=cv,
        iid=False,
    )
    opt.fit(X_train, y_train)
    return opt.best_estimator_, opt.score(X_test, y_test)


def get_column_plots(
    real_data: pd.DataFrame,
    regular_synthetic: pd.DataFrame,
    decoded_synthetic: pd.DataFrame,
    config,
):
    """
    Get plots of real data, regular synthetic data, and decoded synthetic projections.
    All columns are plotted in a single plot.
    """

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

    plot_dataframe(df=full_df, ncols=2, rotate_labels=True)

    return plt


def plot_dataframe(df: pd.DataFrame, ncols: int, rotate_labels: bool):
    """
    Loop over columns and plot in a single figure. Histogram for numericals, barplot else.
    """

    nrows = int(np.ceil(len(df.columns) / ncols))
    fig = plt.figure(figsize=(16, 16))

    for i, col in enumerate([x for x in df.columns if x != "dataset"]):
        ax = plt.subplot(nrows, ncols, i + 1)
        dt = preprocess.infer_data_type(df[col])
        if dt == "continuous":
            sns.histplot(data=df, x=col, hue="dataset", alpha=0.5, ax=ax)
            # sns.histplot(data=df, x=col, hue="dataset", alpha=0.5)
        else:
            sns.countplot(data=df, x=col, hue="dataset", stat="proportion", ax=ax)
            # sns.countplot(data=df, x=col, hue="dataset", stat="proportion")

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


def tsne_projections(real, synthetic):
    X = pd.concat([real, synthetic], axis=0)
    X = X.reset_index(drop=True)
    perplexity = 35
    tsne = TSNE(
        n_components=2,
        learning_rate="auto",
        init="pca",
        perplexity=perplexity,
        verbose=10,
    ).fit_transform(X)
    emb = pd.DataFrame(tsne, index=np.arange(tsne.shape[0]), columns=["comp1", "comp2"])

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
    print(emb)
    print(labels)
    emb = pd.concat([emb, labels], axis=1)
    emb.columns = ["comp1", "comp2", "labels"]

    print(emb)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=emb, x="comp1", y="comp2", hue="labels", alpha=0.1)
    plt.title(f"tSNE plot ({perplexity} perplexity)", fontsize=11)
    plt.tight_layout()
    return plt
