from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from skopt import BayesSearchCV
import pandas as pd


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

    models = [Lasso(max_iter=None)]
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
):
    cv = 3
    opt = BayesSearchCV(
        model,
        param_space,
        n_iter=32,
        cv=cv,
        scoring=scoring,
        random_state=0,
        verbose=10,
        n_jobs=cv,
        iid=False,
    )
    opt.fit(X_train, y_train)
    return opt.best_estimator_, opt.score(X_test, y_test)
