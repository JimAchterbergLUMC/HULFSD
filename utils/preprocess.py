import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# includes scripts for general and dataset specific preprocessing

# this sets the random state to be similar across train test splits, so we get the same indices
random_state = 999


def reshuffle(ori_X, prepr_X):
    """
    Reshuffle a preprocessed dataset according to the original dataframe column order.
    Only works if column names of preprocessed columns start with the full original column name.
    """
    reindex_cols = []
    for col in ori_X:
        # in a list store all column names starting with col
        for col_ in prepr_X:
            if col_.startswith(col):
                reindex_cols.append(col_)
    prepr_X = prepr_X[reindex_cols]
    return prepr_X


def get_num_classes_(ori_X):
    classes = []
    for col in ori_X.columns:
        dt = infer_data_type(ori_X[col])
        if dt == "multiclass":
            n_class = ori_X[col].nunique(dropna=False)
        elif dt == "binary":
            n_class = 2
        elif dt == "continuous":
            n_class = 1
        classes.append(n_class)
    return classes


def sklearn_preprocessor(processor: str, data: pd.DataFrame, features: list):
    """
    Take a scikit learn preprocesser and use it to replace features in the dataframe with processed version.
    """
    assert processor in ["one-hot", "normalize"]
    if processor == "one-hot":
        processor = OneHotEncoder(drop="if_binary")
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


def preprocess(X: pd.DataFrame, y: pd.Series, cat_features: list, num_features: list):
    """
    Preprocesses predictors and targets to train and test sets ready for validation. Performs one hot encoding on categoricals and normalization for numericals independently in train and test sets.
    """
    # one hot encode all categoricals (also binaries, is handled by encoder)
    X = sklearn_preprocessor(processor="one-hot", data=X, features=cat_features)

    # setup train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=random_state
    )

    # scale numericals after splitting (to avoid information leakage)
    X_train = sklearn_preprocessor(
        processor="normalize", data=X_train, features=num_features
    )
    X_test = sklearn_preprocessor(
        processor="normalize", data=X_test, features=num_features
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


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


def decode_datatypes(data, cat_features):
    """
    Use some threshold (i.e. 0.5) to round, while ensuring multiclass features only get 1 feature instance.
    """
    # [0,1] scale all data, then use a threshold for categoricals
    data = sklearn_preprocessor(
        processor="normalize",
        data=data,
        features=data.columns,
    )

    data = data.copy()

    # store list of preprocessed names per categorical feature in a dictionary
    c = {}
    for cat in cat_features:
        c_ = []
        for column in data.columns:
            if column[: len(cat)] == cat:
                c_.append(column)
            c[cat] = c_

    for cat, names in c.items():
        print(f"decoding for category {cat}")
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


def preprocess_adult(X: pd.DataFrame, y: pd.Series, num_features, drop_features):
    """
    Preprocessing specific to Adult census dataset
    """
    # encode target income
    y = y.income.copy()
    y = y.replace({"<=50K": 0, "<=50K.": 0, ">50K": 1, ">50K.": 1})

    # encode native-country as: US, Mexico, Other
    X.loc[~X["native-country"].isin(["United-States", "Mexico"]), "native-country"] = (
        "Other"
    )

    X[num_features] = X[num_features].astype(float)

    X = X.drop(drop_features, axis=1)

    return X, y


def traintest_split(X: pd.DataFrame, y: pd.Series):
    # setup train test split with same random state across splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


# TBD:
# should more categoricals be merged?
# should numericals receive other transformations?
