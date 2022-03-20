from gc import callbacks
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import pickle

from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from skopt.callbacks import VerboseCallback

from util import lithology_key


params = {
    "seed": 123,
    "n_folds": 5,
    "n_iters": 15,
    "n_jobs": 12,
    "base_estimator": "RF"
}

def report_performance(optimizer: BayesSearchCV, X, y, callbacks):
    optimizer.fit(X, y, callback=callbacks)

    return optimizer.best_params_


def rdf_exhaust(X, y, groups):
    kfold_generator = GroupKFold(n_splits=params["n_folds"])

    model = RandomForestClassifier(random_state=params["seed"], n_jobs=4)
    model = GradientBoostingClassifier(random_state=params["seed"], n_jobs=4)
    
    distribution = {
        "min_samples_split": Integer(2, 100, "uniform"),
        "n_estimators": Integer(100, 500, "uniform")
    }
    optimizer = BayesSearchCV(model,
                              distribution,
                              cv=kfold_generator,
                              scoring="f1_micro",
                              n_iter=params["n_iters"],
                              n_jobs=params["n_jobs"],
                              return_train_score=True,
                              optimizer_kwargs={
                                  "base_estimator": params["base_estimator"]
                              },
                              refit=True)
    optimizer.fit(X, y, callback=[VerboseCallback(params["n_iters"])], groups=groups)
    print(optimizer.best_params_)
    print()

    best_model = RandomForestClassifier(**optimizer.best_params_)
    best_model.fit(X, y)
    y_hat = best_model.predict(X)
    print(f1_score(y, y_hat, average="micro"))
    print()

    cv_results = pd.DataFrame(optimizer.cv_results_)
    cv_results.sort_values(by=["mean_test_score"], inplace=True)
    print(cv_results[["mean_test_score", "mean_train_score"]])
    print()

    print(optimizer.cv_results_)


def ada_exhaust(X, y, groups):
    kfold_generator = GroupKFold(n_splits=params["n_folds"])

    model = AdaBoostClassifier(random_state=params["seed"])

    distribution = {
        "learning_rate": Real(1e-1, 10, "log-uniform"),
        "n_estimators": Integer(50, 100, "uniform")
    }
    optimizer = BayesSearchCV(model,
                              distribution,
                              cv=kfold_generator,
                              scoring="f1_micro",
                              n_iter=params["n_iters"],
                              n_jobs=params["n_jobs"],
                              return_train_score=True,
                              optimizer_kwargs={
                                  "base_estimator": params["base_estimator"]
                              },
                              refit=True)
    optimizer.fit(X, y, callback=[VerboseCallback(params["n_iters"])], groups=groups)
    print(optimizer.best_params_)
    print()

    best_model = AdaBoostClassifier(**optimizer.best_params_)
    best_model.fit(X, y)
    y_hat = best_model.predict(X)
    print(f1_score(y, y_hat, average="micro"))
    print()

    cv_results = pd.DataFrame(optimizer.cv_results_)
    cv_results.sort_values(by=["mean_test_score"], inplace=True)
    print(cv_results[["mean_test_score", "mean_train_score"]])
    print()

    print(optimizer.cv_results_)


def xgboost_exhaust(X, y, groups):
    kfold_generator = GroupKFold(n_splits=params["n_folds"])

    model = GradientBoostingClassifier(learning_rate=0.1, random_state=params["seed"])

    distribution = {
        #"n_estimators": Real(50, 300, "uniform"),
        "learning_rate": Real(1e-2, 1, "log-uniform"),
        "subsample": Real(0.5, 1.0, "uniform"),
        "min_samples_split": Integer(2, 100, "uniform"),
    }
    optimizer = BayesSearchCV(model,
                              distribution,
                              cv=kfold_generator,
                              scoring="f1_micro",
                              n_iter=params["n_iters"],
                              n_jobs=params["n_jobs"],
                              return_train_score=True,
                              optimizer_kwargs={
                                  "base_estimator": params["base_estimator"]
                              },
                              refit=True)
    optimizer.fit(X, y, callback=[VerboseCallback(params["n_iters"])], groups=groups)
    print(optimizer.best_params_)
    print()

    best_model = GradientBoostingClassifier(**optimizer.best_params_)
    best_model.fit(X, y)
    y_hat = best_model.predict(X)
    print(f1_score(y, y_hat, average="micro"))
    print()

    cv_results = pd.DataFrame(optimizer.cv_results_)
    cv_results.sort_values(by=["mean_test_score"], inplace=True)
    print(cv_results[["mean_test_score", "mean_train_score"]])
    print()

    print(optimizer.cv_results_)


def mlp_exhaust(X, y, groups):
    kfold_generator = GroupKFold(n_splits=params["n_folds"])
    num_neurons = 512

    model = MLPClassifier(hidden_layer_sizes=(num_neurons, ), random_state=params["seed"], max_iter=500)

    distribution = {
        "learning_rate_init": Real(1e-3, 1e-1, "log-uniform"),
    }
    optimizer = BayesSearchCV(model,
                              distribution,
                              cv=kfold_generator,
                              scoring="f1_micro",
                              n_iter=params["n_iters"],
                              n_jobs=params["n_jobs"],
                              return_train_score=True,
                              optimizer_kwargs={
                                  "base_estimator": params["base_estimator"]
                              },
                              refit=True)
    optimizer.fit(X, y, callback=[VerboseCallback(params["n_iters"])], groups=groups)
    print(optimizer.best_params_)
    print()

    best_model = MLPClassifier(hidden_layer_sizes=(num_neurons), **optimizer.best_params_)
    best_model.fit(X, y)
    y_hat = best_model.predict(X)
    print(f1_score(y, y_hat, average="micro"))
    print()

    cv_results = pd.DataFrame(optimizer.cv_results_)
    cv_results.sort_values(by=["mean_test_score"], inplace=True)
    print(cv_results[["mean_test_score", "mean_train_score"]])
    print()

    print(optimizer.cv_results_)


if __name__ == "__main__":
    train_dataset = pd.read_csv('.\\data\\Train-dataset.csv')
    train_dataset.rename(columns={"DEPOSITIONAL_ENVIRONMENT": "DENV"}, inplace=True)
    train_dataset["DENV"] = LabelEncoder().fit_transform(train_dataset["DENV"])

    train_dataset = train_dataset.loc[train_dataset["WELL"].isin(["Well-6", "Well-7", "Well-8", "Well-10", "Well-11"])]

    X = train_dataset[["MD", "GR", "RT", "DEN", "CN", "DENV"]]
    #X = train_dataset[["MD", "GR", "RT", "DEN", "CN"]]
    X = preprocessing.StandardScaler().fit_transform(X)
    groups = train_dataset["WELL"]

    y_encoder = LabelEncoder().fit(train_dataset["LITH_NAME"])
    y = y_encoder.transform(train_dataset["LITH_NAME"])

    # rdf_exhaust(X, y, groups)
    # ada_exhaust(X, y, groups)
    xgboost_exhaust(X, y, groups)
    # mlp_exhaust(X, y, groups)
