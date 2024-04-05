import pandas as pd
from sklearn.preprocessing import OneHotEncoder, Normalizer
from sklearn.model_selection import train_test_split
import numpy as np

# this sets the random state to be similar across train test splits, so we get the same indicess
random_state = 999


def sklearn_preprocessor(processor: any, data: pd.DataFrame, features: list):
    """
    Take a scikit learn preprocesser and use it to replace features in the dataframe with processed version.
    """
    # reset index of splitted data
    data = data.copy().reset_index(drop=True)
    # get transformed data as np array
    df = processor.fit_transform(data[features])
    if not isinstance(df, np.ndarray):
        df = df.toarray()
    # get transformed data in dataframe with correct feature names
    df = pd.DataFrame(df, columns=processor.get_feature_names_out(features))
    # replace features with preprocessed features
    data = data.drop(features, axis=1)
    data = pd.concat([data, df], axis=1)
    return data


def preprocess(X: pd.DataFrame, y: pd.Series, cat_features: list, num_features: list):
    """
    Preprocesses predictors and targets to train and test sets ready for validation. Performs one hot encoding on categoricals and normalization for numericals independently in train and test sets.
    """
    # one hot encode all categoricals (also binaries, is handled by encoder)
    X = sklearn_preprocessor(
        processor=OneHotEncoder(drop="if_binary"), data=X, features=cat_features
    )

    # setup train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=random_state
    )

    # scale numericals after splitting (to avoid information leakage)
    X_train = sklearn_preprocessor(
        processor=Normalizer(norm="l2"), data=X_train, features=num_features
    )
    X_test = sklearn_preprocessor(
        processor=Normalizer(norm="l2"), data=X_test, features=num_features
    )

    return X_train, X_test, y_train, y_test


def preprocess_adult(X: pd.DataFrame, y: pd.Series):
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

    return X, y


def traintest_split(X, y):
    # setup train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


# TBD:
# should more categoricals be merged?
# should numericals receive other transformations?
