from utils import pred_models, preprocess
from scipy.spatial import distance
import pandas as pd

# includes scripts for running inference on the target feature and sensitive features


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
        models, param_search_spaces = pred_models.get_binary_models_()
        scoring = "roc_auc"
    elif dt == "multiclass":
        models, param_search_spaces = pred_models.get_multiclass_models_()
        scoring = "accuracy"
    elif dt == "continuous":
        models, param_search_spaces = pred_models.get_regression_models_()
        scoring = "neg_mean_squared_error"

    best_models, best_scores = [], []

    # for the different types of models, find the best hyperparameters and return score on a test set
    for model, param_space in zip(models, param_search_spaces):

        best_model, best_score = pred_models.get_best_model_(
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

    return labels
