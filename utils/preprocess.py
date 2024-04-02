import pandas as pd
from sklearn.preprocessing import OneHotEncoder, Normalizer
from sklearn.model_selection import train_test_split


def preprocess(X, y, cat_features, num_features, drop_features):
    """
    Preprocesses predictors and targets to train and test sets ready for validation. Performs one hot encoding on categoricals and normalization for numericals.
    """

    # drop unnecessary features
    X = X.drop(drop_features, axis=1)

    # one hot encode all categoricals (also binaries, is handled by encoder)
    ohe = OneHotEncoder(drop="if_binary")
    ohe_df = ohe.fit_transform(X[cat_features]).toarray()
    feature_names = ohe.get_feature_names_out(cat_features)
    ohe_df = pd.DataFrame(ohe_df, columns=feature_names)
    X = X.drop(cat_features, axis=1)
    X = pd.concat([X, ohe_df], axis=1)

    # setup train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=0
    )

    # scale numericals after splitting (to avoid information leakage)
    norm = Normalizer(norm="l2")
    X_train[num_features] = norm.transform(X_train[num_features])
    X_test[num_features] = norm.transform(X_test[num_features])

    return X_train, X_test, y_train, y_test


def preprocess_adult(X, y):
    """
    Preprocessing specific to Adult census dataset
    """
    # encode target income
    y = y.income.replace({"<=50K": 0, "<=50K.": 0, ">50K": 1, ">50K.": 1})
    # encode native-country as: US, Mexico, Other
    X["native-country"] = X["native-country"].mask(
        ~X["native-country"].isin(["United-States", "Mexico"]), "Other"
    )
    return X, y


# TBD:
# should more categoricals be merged?
# should numericals receive other transformations?
