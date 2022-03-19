from gc import callbacks
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import pickle

from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from skopt.callbacks import VerboseCallback

from util import lithology_key


params = {
    "seed": 123,
    "n_folds": 5,
    "n_iters": 10,
    "n_jobs": 12,
    "base_estimator": "RF"
}

def report_performance(optimizer: BayesSearchCV, X, y, callbacks):
    optimizer.fit(X, y, callback=callbacks)

    return optimizer.best_params_


if __name__ == "__main__":
    train_dataset = pd.read_csv('.\\data\\Train-dataset.csv')
    train_dataset.rename(columns={"DEPOSITIONAL_ENVIRONMENT": "DENV"}, inplace=True)
    train_dataset["DENV"] = LabelEncoder().fit_transform(train_dataset["DENV"])

    X = train_dataset[["MD", "GR", "RT", "DEN", "CN", "DENV"]]
    # X = train_dataset[["MD", "GR", "RT", "DEN", "CN"]]
    X = preprocessing.StandardScaler().fit_transform(X)

    y_encoder = LabelEncoder().fit(train_dataset["LITH_NAME"])
    y = y_encoder.transform(train_dataset["LITH_NAME"])

    kfold_generator = KFold(n_splits=params["n_folds"], shuffle=True, random_state=params["seed"])

    model = RandomForestClassifier(criterion="gini", random_state=params["seed"])
    # model.fit(X, y)

    distribution = {"min_samples_split": Integer(2, 10)}
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
    best_params = report_performance(optimizer, X, y, callbacks=[VerboseCallback(params["n_iters"])])
    print(best_params)
    print()

    cv_results = pd.DataFrame(optimizer.cv_results_)
    cv_results.sort_values(by=["mean_test_score"], inplace=True)
    print(cv_results[["mean_test_score", "mean_train_score"]])

