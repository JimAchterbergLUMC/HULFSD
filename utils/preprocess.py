import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# includes scripts for general and dataset specific preprocessing

# this sets the random state to be similar across train test splits, so we get the same indices


def sklearn_preprocessor(processor: str, data: pd.DataFrame, features: list):
    """
    Take a scikit learn preprocesser and use it to replace features in the dataframe with processed version.
    """
    assert processor in ["one-hot", "normalize"]
    if processor == "one-hot":
        processor = OneHotEncoder(drop=None)
    elif processor == "normalize":
        processor = MinMaxScaler((0, 1))

    # reset index of splitted data
    data = data.copy().reset_index(drop=True)
    # get transformed data as np array
    fitted_processor = processor.fit(data[features])

    df = fitted_processor.transform(data[features])
    if not isinstance(df, np.ndarray):
        df = df.toarray()
    # get transformed data in dataframe with correct feature names
    df = pd.DataFrame(df, columns=fitted_processor.get_feature_names_out(features))
    # replace features with preprocessed features
    data = data.drop(features, axis=1)
    data = pd.concat([data, df], axis=1)
    return data


def sd_pre_preprocess(X: pd.DataFrame, y: pd.Series, config: dict, random_state: int):
    # regular synthetic data requires only encoding, dropping, and train test split preprocessing (no one hot/normalizing)

    ds_name = config["name"]

    X = X.copy()
    y = y.copy()

    # dataset specific encoding
    if ds_name == "adult":
        X, y = preprocess_adult(X=X, y=y)

    # drop features
    X = X.drop(config["drop_features"], axis=1, errors="ignore")

    # turn numericals into floats
    X[config["num_features"]] = X[config["num_features"]].astype(float)

    # split train test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def sd_preprocess(
    syn_X: pd.DataFrame,
    real_X: pd.DataFrame,
    syn_y: pd.Series,
    config: dict,
    random_state: int,
):
    ds_name = config["name"]
    syn_X = syn_X.copy()
    real_X = real_X.copy()
    syn_y = syn_y.copy()

    # one hot encode synthetic data together with real data for correct vocab
    c_X = pd.concat([syn_X, real_X])
    c_X = sklearn_preprocessor(
        processor="one-hot", data=c_X, features=config["cat_features"]
    )

    # select only synthetic data for splitting and returning
    syn_X = c_X[: syn_X.shape[0]]

    # split train test
    X_train, X_test, y_train, y_test = train_test_split(
        syn_X, syn_y, stratify=syn_y, test_size=0.3, random_state=random_state
    )

    # normalize numericals
    X_train = sklearn_preprocessor(
        processor="normalize", data=X_train, features=config["num_features"]
    )
    X_test = sklearn_preprocessor(
        processor="normalize", data=X_test, features=config["num_features"]
    )

    return X_train, X_test, y_train, y_test


def preprocess(X: pd.DataFrame, y: pd.Series, config: dict, random_state: int):
    ds_name = config["name"]

    X = X.copy()
    y = y.copy()

    # dataset specific encoding
    if ds_name == "adult":
        X, y = preprocess_adult(X=X, y=y)

    # drop features
    X = X.drop(config["drop_features"], axis=1, errors="ignore")

    # turn numericals into floats
    X[config["num_features"]] = X[config["num_features"]].astype(float)

    # one hot encode
    X = sklearn_preprocessor(
        processor="one-hot", data=X, features=config["cat_features"]
    )

    # split train test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=random_state
    )

    # normalize numericals
    X_train = sklearn_preprocessor(
        processor="normalize", data=X_train, features=config["num_features"]
    )
    X_test = sklearn_preprocessor(
        processor="normalize", data=X_test, features=config["num_features"]
    )

    return X_train, X_test, y_train, y_test


def infer_data_type(series: pd.Series):
    """
    Infers data type of a pandas series.
    """
    # if only two values, it is binary
    p = series.nunique(dropna=False)
    if p == 2:
        return "binary"
    # if more than two values and not floats, it is multiclass
    elif not (pd.api.types.is_float_dtype(series.dtype)):
        return "multiclass"
    else:
        # else it is continuous (if it is numeric at least)
        if pd.api.types.is_numeric_dtype(series):
            return "continuous"
        else:
            return "unknown"


def decoding_onehot(df: pd.DataFrame, categorical_features: list):
    """
    Decode onehot features if the onehot feature names start with the original feature names, provided in the list
    """

    # ensure we are not changing the original dataframe
    df = df.copy()

    # store list of preprocessed names per categorical feature in a dictionary
    c = {}
    for cat in categorical_features:
        c_ = []
        for column in df.columns:
            if column[: len(cat)] == cat:
                c_.append(column)
            c[cat] = c_

    # remove one hot features and replace with decoded features
    for cat, names in c.items():
        new = pd.DataFrame(df[names].idxmax(axis=1), columns=[cat])
        df = df.drop(names, axis=1)
        df = pd.concat([df, new], axis=1)

    return df


def postprocess_projections(data, config):
    """
    Postprocess projections such that categories are integers instead of continuous.
    Multiclass features get the highest column as feature instance, binary columns are rounded.
    """
    # [0,1] scale all data, then use a threshold for categoricals
    data = data.copy()

    data = sklearn_preprocessor(
        processor="normalize",
        data=data,
        features=data.columns,
    )

    # store list of preprocessed names per categorical feature in a dictionary
    cat_features = config["cat_features"]
    c = {}
    for cat in cat_features:
        c_ = []
        for column in data.columns:
            if column[: len(cat)] == cat:
                c_.append(column)
            c[cat] = c_

    for cat, names in c.items():
        if len(names) > 1:
            # for multiclass categories, set the maximum index as the category instance
            # get list of max probability per row for this category

            data[names] = pd.DataFrame(
                np.eye(len(names))[np.argmax(data[names].values, axis=1)],
                columns=names,
            )

        else:
            # for binaries we simply round
            data[names] = data[names].round(0)

        data[names] = data[names].astype(int)

    return data


def preprocess_adult(X, y):
    """
    Preprocessing specific to Adult census dataset
    """
    # encode target income
    if isinstance(y, pd.DataFrame):
        y = y.income.copy()
    else:
        y = y.copy()
    y = y.replace({"<=50K": 0, "<=50K.": 0, ">50K": 1, ">50K.": 1})

    # encode native-country as: US, Mexico, Other
    X.loc[~X["native-country"].isin(["United-States", "Mexico"]), "native-country"] = (
        "Other"
    )

    return X, y


def decode_categorical_target(data: pd.DataFrame, target: str):
    """
    If a target feature is categorical and one hot encoded, decode the target back so we can do regular categorical prediction.
    If no target feature found, raise exception. If single target feature found, do nothing. If more than 1, decode to single feature and
    replace original one hot encoded features.
    """

    data = data.copy()

    real_target = []
    for col in data.columns:
        if col.split("_")[0] == target:
            real_target.append(col)

    if len(real_target) == 0:
        raise Exception("No target feature found in columns")
    elif len(real_target) > 1:
        y = data[real_target]
        y = y.idxmax(axis=1)
        y = y.squeeze()
        y.name = target
        data = data.drop(real_target, axis=1)
        data = pd.concat([data, y], axis=1)

    return data
