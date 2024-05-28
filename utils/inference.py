from utils import preprocess, sd
from scipy.spatial import distance
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.metrics import roc_auc_score, mean_squared_error, f1_score, make_scorer
import keras

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import label_binarize


def utility(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    models: list,
    param_spaces: dict,
):
    """
    Finds utility score (prediction task) for a list of models and parameter spaces.
    For keras models these are training arguments (epochs and batch size). For sklearn these are tunable hyperparameters,
    which are tuned internally through CV.


    Utility score is defined as the prediction performance on a test set.
    The unit of this score depends on the datatype of the target, i.e. binary, categorical, continuous, and is automatically inferred.
    returns: lists of best models and corresponding best scores.

    """

    # infer datatype of target
    dt = preprocess.infer_data_type(y_train)
    assert dt in ["binary", "multiclass", "continuous"]
    # create numeric labels if not numeric
    y_train, y_test = y_train.copy(), y_test.copy()
    if not pd.api.types.is_numeric_dtype(y_train):
        y_train, _ = pd.factorize(y_train)
    if not pd.api.types.is_numeric_dtype(y_test):
        y_test, _ = pd.factorize(y_test)

    output = {}
    for model, param_space in zip(models, param_spaces):
        if isinstance(model, keras.models.Model):
            if dt == "binary":
                score = roc_auc_score
                kwargs = {}
                loss = "binary_crossentropy"
                metric = "AUC"
            elif dt == "multi_class":
                score = roc_auc_score
                kwargs = {"multi_class": "ovr", "average": "weighted"}
                loss = "categorical_crossentropy"
                metric = "accuracy"
            elif dt == "continuous":
                score = mean_squared_error
                kwargs = {"squared": False}
                loss = "mean_squared_error"
                metric = "mean_squared_error"

            model.compile(optimizer="adam", loss=loss, metrics=[metric])
            sd.fit_model(
                model=model,
                X=X_train,
                y=y_train,
                model_args=param_space,
                monitor="val_" + metric,
            )
            preds = model.predict(X_test)
            output[str(model)] = score(y_test, preds, **kwargs)
        else:
            if dt == "binary":
                score = "roc_auc"
            elif dt == "multiclass":
                score = make_scorer(multiclass_roc_auc, needs_proba=True)
            elif dt == "continuous":
                score = "neg_root_mean_squared_error"
            best_model, best_score = get_best_model_(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                model=model,
                param_space=param_space,
                scoring=score,
            )
            output[str(best_model)[:3]] = best_score

    return output


def multiclass_roc_auc(y_true, y_score, multi_class="ovr", average="weighted"):
    """
    multiclass ROCAUC of sklearn cannot handle when some classes are not predicted.
    With this function we first binarize the true and predicted labels, ensuring they are of the same size.
    """
    classes = np.unique(y_true)
    n_classes = len(classes)

    # Ensure y_score has the correct shape
    if y_score.shape[1] != n_classes:
        y_score_adjusted = np.zeros((y_score.shape[0], n_classes))
        for i, cls in enumerate(classes):
            if cls < y_score.shape[1]:
                y_score_adjusted[:, cls] = y_score[:, cls]
    else:
        y_score_adjusted = y_score

    # Binarize y_true for multiclass roc_auc_score
    y_true_binarized = label_binarize(y_true, classes=classes)
    return roc_auc_score(
        y_true_binarized, y_score_adjusted, multi_class=multi_class, average=average
    )


def attribute_inference(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    sensitive_field: str,
    models: list,
    param_spaces: list,
):
    """
    Perform an attribute inference attack using synthetic data, by training an inference model on synthetic data and inferring on a real test set.
    In the real test set, non-sensitive key fields are known for inference.
    Note that this is the same as the utility function, only for sensitive targets and different predictors.

    """

    # key fields are all fields except for the sensitive field
    key_fields = [x for x in real_data.columns if x != sensitive_field]

    output = utility(
        X_train=synthetic_data[key_fields],
        X_test=real_data[key_fields],
        y_train=synthetic_data[sensitive_field],
        y_test=real_data[sensitive_field],
        models=models,
        param_spaces=param_spaces,
    )
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
    legend = plt.legend(fontsize=11)
    for legend_handle in legend.legend_handles:
        legend_handle.set_alpha(1)
    plt.tight_layout()
    return plt


class RunningStats:

    def __init__(self, df=None):
        self.df = df
        self.n = 0
        if df is not None:
            self.mean = df
            self.mean = self.mean * 0
            self.M2 = df
            self.M2 = self.M2 * 0
        else:
            self.mean = 0
            self.M2 = 0

    def update(self, x):
        # x is a pandas df/series or scalar
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self):
        if self.n < 2:
            # return NaNs if n less than two
            if self.df is not None:
                out = np.empty(shape=(self.df.shape))
                out[:] = np.nan
                if isinstance(self.df, pd.DataFrame):
                    out = pd.DataFrame(out)
                elif isinstance(self.df, pd.Series):
                    out = pd.Series(out)
                else:
                    raise Exception(
                        "only pandas DF, pandas Series or None (scalar) supported"
                    )
            else:
                out = np.nan
            return out
        return self.M2 / (self.n - 1)

    @property
    def standard_deviation(self):
        return np.sqrt(self.variance)

    @property
    def mean_value(self):
        return self.mean


def get_column_plots_(
    data: dict,
    config: dict,
):
    """
    Get plots of real data, regular synthetic data, and decoded synthetic projections.
    All columns are plotted in a single plot.
    """

    # collapse one hot encoded data back to single columns
    full_df = []
    for name, (X_tr, _, _, _) in data.items():
        full_df.append(
            (preprocess.collapse_onehot(X_tr, config["cat_features"])).assign(
                dataset=name
            )
        )
    full_df = pd.concat(full_df)

    # cast categoricals as strings (so all non numericals) to ensure binaries also get plotted as category
    cats = [
        x for x in full_df.columns if x not in (config["num_features"] + ["dataset"])
    ]
    full_df[cats] = full_df[cats].astype(str)
    # put ordering as numericals first and categoricals later for nice aesthetic
    full_df = full_df[config["num_features"] + cats + ["dataset"]]

    plot = plot_dataframe(df=full_df, ncols=2, rotate_labels=True)

    return plot


def get_binary_models_():
    models = [LogisticRegression(solver="saga", penalty="l2"), XGBClassifier(n_jobs=1)]

    param_search_spaces = [
        {
            "C": Real(1e-3, 1e3),
        },  # LR
        {
            "n_estimators": Integer(1, 300),
            "max_depth": Integer(1, 10),
            "subsample": Real(0.5, 1),
            "lambda": Real(1, 100),
        },  # GB
    ]
    return models, param_search_spaces


def get_multiclass_models_():
    models = [
        LogisticRegression(multi_class="multinomial", solver="saga", penalty="l2"),
        XGBClassifier(n_jobs=1, objective="multi:softmax"),
    ]

    param_search_spaces = [
        {
            "C": Real(1e-3, 1e3),
        },  # LR
        {
            "n_estimators": Integer(1, 300),
            "max_depth": Integer(1, 10),
            "subsample": Real(0.5, 1),
            "lambda": Real(1, 100),
        },  # GB
    ]

    return models, param_search_spaces


def get_regression_models_():

    models = [Ridge(solver="saga"), XGBRegressor(n_jobs=1)]

    param_search_spaces = [
        {
            "alpha": Real(1e-3, 1e3),
        },  # LR
        {
            "n_estimators": Integer(1, 300),
            "max_depth": Integer(1, 10),
            "subsample": Real(0.5, 1),
            "lambda": Real(1, 100),
        },  # GB
    ]
    return models, param_search_spaces


def get_pred_models_(dt):

    # get list of models
    if dt == "binary":
        models, param_search_spaces = get_binary_models_()
    elif dt == "multiclass":
        models, param_search_spaces = get_multiclass_models_()
    elif dt == "continuous":
        models, param_search_spaces = get_regression_models_()

    return models, param_search_spaces


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

    # opt = GridSearchCV(
    #     model,
    #     param_space,
    #     cv=cv,
    #     scoring=scoring,
    #     verbose=0,
    #     n_jobs=-1,
    # )

    opt.fit(X_train, y_train)
    return opt.best_estimator_, opt.score(X_test, y_test)


def get_utility_(
    data: dict, encoder: bool = False, encoder_args: dict = {}, sklearn: bool = False
):
    """
    Retrieves utility scores for a dictionary of datasets.

    Will use encoder model and/or sklearn models if specified.

    Dictionary of datasets is in the format name:[X_train,X_test,y_train,y_test].
    See utility() function for what utility score means.
    """
    output = {}
    for name, (X_tr, X_te, y_tr, y_te) in data.items():
        models = []
        param_search_spaces = []
        dt = preprocess.infer_data_type(y_tr)
        if sklearn:
            sklearn_models, sklearn_param_search_spaces = get_pred_models_(dt=dt)
            models.extend(sklearn_models)
            param_search_spaces.extend(sklearn_param_search_spaces)
        if encoder:
            models.append(
                sd.EncoderModel(
                    latent_dim=X_tr.shape[1],
                    input_dim=X_tr.shape[1],
                    compression_layers=[],
                )
            )
            param_search_spaces.append(encoder_args)

        output[name] = utility(
            X_tr, X_te, y_tr, y_te, models=models, param_spaces=param_search_spaces
        )

    output = pd.DataFrame(output)

    return output


def get_attribute_inference_(
    data: pd.DataFrame, config: dict, sd_sets: list = [], real_key: str = "Real"
):
    """
    Retrieves attribute inference scores for a dictionary of datasets and formats output. Dictionary of datasets is in the format:
    key:[X_train,X_test,y_train,y_test]. See attribute_inference() function for what attribute inference score means.

    sd_sets: keys of datasets which are synthetic (and thus used to train attribute inference models)
    real_key: key which denotes the real dataset (and thus used to test attribute inference models)

    """
    output = {}
    sensitive_fields = config["sensitive_fields"]

    for name, (X_tr, _, _, _) in data.items():
        if name in sd_sets:
            att = {}
            for sensitive_field in sensitive_fields:
                rd = data[real_key][0].copy()
                sd = X_tr.copy()

                if sensitive_field in config["bin_features"]:
                    # remove suffix for binary features
                    for col in sd.columns:
                        if col.split("_")[0] == sensitive_field:
                            sd = sd.rename({col: sensitive_field}, axis=1)
                            rd = rd.rename({col: sensitive_field}, axis=1)
                elif sensitive_field in config["cat_features"]:
                    # collapse to single feature for categorical non binary features
                    rd = preprocess.collapse_onehot(
                        df=rd, categorical_features=[sensitive_field]
                    )
                    sd = preprocess.collapse_onehot(
                        df=sd, categorical_features=[sensitive_field]
                    )

                # retrieve prediction models and parameter spaces
                dt = preprocess.infer_data_type(rd[sensitive_field])
                models, param_spaces = get_pred_models_(dt=dt)
                # perform attribute inference
                att[sensitive_field] = attribute_inference(
                    real_data=rd,
                    synthetic_data=sd,
                    sensitive_field=sensitive_field,
                    models=models,
                    param_spaces=param_spaces,
                )
            output[name] = att

    # flatten the three-layer dictionary (dataset,feature,model)
    output = {key: preprocess.flatten_dict(value) for key, value in output.items()}
    output = pd.DataFrame(output)
    return output


def get_authenticity_(data: pd.DataFrame, sd_sets: list = [], real_key: str = "Real"):
    """
    Retrieves authenticity scores for a dictionary of datasets and formats output. Dictionary of datasets is in the format:
    key:[X_train,X_test,y_train,y_test]. See authenticity() function for what authenticity score means.

    sd_sets: list of keys of datasets which are synthetic (and thus used to train attribute inference models)
    real_key: single key which denotes the real dataset (and thus used to test attribute inference models)

    """

    output = {}
    for name, (X_tr, _, _, _) in data.items():
        if name in sd_sets:
            output[name] = authenticity(
                real_data=data[real_key][0].copy(),
                synthetic_data=X_tr.copy(),
                metric="euclidean",
            )
    output = pd.DataFrame(output, index=[0])
    return output
