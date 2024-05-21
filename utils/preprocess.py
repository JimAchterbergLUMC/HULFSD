import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import numpy as np


def preprocess(X, y, config):
    X = X.copy()
    y = y.copy()

    # drop features
    X = X.drop(config["drop_features"], axis=1, errors="ignore")

    # turn numericals into floats
    X[config["num_features"]] = X[config["num_features"]].astype(float)

    # encode
    if config["name"] == "adult":
        X, y = preprocess_adult(X=X, y=y)

    return X, y


def sklearn_preprocessor(processor: str, data: pd.DataFrame, features: list):
    """
    Take a scikit learn preprocesser and use it to replace features in the dataframe with processed version.
    """
    assert processor in ["one-hot", "normalize"]
    if processor == "one-hot":
        processor = OneHotEncoder(drop=None)
    elif processor == "normalize":
        processor = MinMaxScaler((0, 1))

    # reset index if data is splitted
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


def collapse_onehot(df: pd.DataFrame, categorical_features: list):
    """
    collapse onehot features if the onehot feature names start with the original feature names, provided in the list

    should only include the category name not the entire feature column name!
    """

    # ensure we are not changing the original dataframe
    df = df.copy()

    # store list of preprocessed names per categorical feature in a dictionary
    c = {}
    for cat in categorical_features:
        c_ = []
        for column in df.columns:
            if column.split("_")[0] == cat:
                c_.append(column)
            c[cat] = c_

    # remove one hot features and replace with decoded features
    for cat, names in c.items():

        new = pd.DataFrame(df[names].idxmax(axis=1), columns=[cat])
        # now all thats left is to remove prefix
        new = new.map(remove_prefix)

        df = df.drop(names, axis=1)
        df = pd.concat([df, new], axis=1)

    return df


def remove_prefix(s):
    return s.split("_", 1)[1] if "_" in s else s


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
        y = y.income
    y = y.copy()

    y = y.replace({"<=50K": 0, "<=50K.": 0, ">50K": 1, ">50K.": 1})

    # encode native-country as: US, Mexico, Other
    X.loc[~X["native-country"].isin(["United-States", "Mexico"]), "native-country"] = (
        "Other"
    )

    return X, y


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
