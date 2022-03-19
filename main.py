import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import pickle

from util import lithology_key

SEED = 31415
N_FOLDS = 5

train_dataset = pd.read_csv('data/Train-dataset.csv')

# Feature Extraction
MAPPING = {
    'Continental': 1,
    'Transitional': 2,
    'Marine': 3,
}

train_dataset['D_Env']=train_dataset['DEPOSITIONAL_ENVIRONMENT'].apply(lambda x: MAPPING[x])

X = train_dataset[['MD','GR', 'RT', 'DEN', 'CN','D_Env']]
X = preprocessing.StandardScaler().fit(X).transform(X)

y = train_dataset['LITH_CODE']

# Splitting
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
kf.get_n_splits(X)

# Training
acc = np.zeros((N_FOLDS))
f1 = np.zeros((N_FOLDS))
RDF_models = []

for i, (train_index, val_index) in enumerate(kf.split(X)):
    
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    model = RandomForestClassifier(random_state=SEED)
    model.fit(X_train, y_train)

    yhat = model.predict(X_val)
    
    acc[i] = metrics.accuracy_score(y_val, yhat)
    f1[i] = metrics.f1_score(y_val, yhat, average='micro')

    print(acc[i])
    print(f1[i])
    print()

