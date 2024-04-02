from ucimlrepo import fetch_ucirepo
from utils import preprocess
import json

# open configuration of datasets
with open("config.json", "r") as f:
    config = json.load(f)

# set which dataset we are using (and assert it is one of the options)
ds = "adult"
assert ds in ["adult"]

# fetch dataset
dataset = fetch_ucirepo(id=config[ds]["id"])
# load features and target
X = dataset.data.features
y = dataset.data.targets

if ds == "adult":
    X, y = preprocess.preprocess_adult(X, y)
else:
    pass


X_train, X_test, y_train, y_test = preprocess.preprocess(
    X,
    y,
    cat_features=config[ds]["cat_features"],
    num_features=config[ds]["num_features"],
    drop_features=config[ds]["drop_features"],
)

from sklearn.naive_bayes import GaussianNB  # issues with hyperparameter tuning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from skopt import BayesSearchCV

models = [GaussianNB(), LogisticRegression(solver="saga"), GradientBoostingClassifier()]
param_spaces = [
    {"var_smoothing": (1e-9, 1e-6)},  # NB
    {"penalty": ["l2", "l1"], "C": (1e-6, 1e6)},  # LR
    {
        "learning_rate": (1e-6, 1e-1),
        "n_estimators": (1e0, 1e5),
        "max_depth": (1e0, 1e2),
    },  # GB
]

for model, param_space in zip(models, param_spaces):
    opt = BayesSearchCV(
        model,
        param_space,
        n_iter=32,
        cv=3,
        scoring="roc_auc",
        random_state=0,
    )
    opt.fit(X_train, y_train)
    print(f"Best test score: {opt.score(X_test,y_test)}")
    print(f"Using model: {opt.best_estimator_} \n")


# models = [
#     GaussianNB(),
#     LogisticRegression(penalty="l1", C=0.5, solver="saga"),
#     GradientBoostingClassifier(learning_rate=0.1, n_estimators=200, max_depth=3),
# ]
# for model in models:
#     model.fit(X_train, y_train)
#     preds = model.predict(X_test)
#     auc = roc_auc_score(y_test, preds)
#     print(f"Model: {model} achieves AUC: {auc}")


# classifiers to use:
# LASSO logistic regression
# gradient boosting
# and possibly a lot more...
