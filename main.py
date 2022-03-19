import os

import numpy as np
import pandas as pd

from collections import Counter
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.model_selection import KFold, GroupKFold
import pickle

from util import fix_dataset

SEED = 31415
N_FOLDS = 8

LOAD_MODEL = False

Pkl_path = 'checkpoints/RDF1.pkl' 

train_dataset = pd.read_csv('data/Train-dataset.csv')
test_dataset = pd.read_csv('data/Test-dataset.csv')

# Feature Extraction
MAPPING = {
    'Continental': 1,
    'Transitional': 2,
    'Marine': 3,
}

train_dataset = pd.read_csv('data/Train-dataset.csv')
# train_dataset = fix_dataset(train_dataset)

X = train_dataset[['MD','GR', 'RT', 'DEN', 'CN','DEPOSITIONAL_ENVIRONMENT']]
ct = ColumnTransformer([
        ('some_name', preprocessing.StandardScaler(), ['MD','GR', 'RT', 'DEN', 'CN'])], remainder='passthrough')
X = pd.DataFrame(ct.fit_transform(X), columns=X.columns)

# Mapping encorder
X['DEPOSITIONAL_ENVIRONMENT']=X['DEPOSITIONAL_ENVIRONMENT'].apply(lambda x: MAPPING[x])
# One hot encoder
# X = pd.get_dummies(X, columns=['DEPOSITIONAL_ENVIRONMENT'])

X = X.to_numpy()

y = train_dataset['LITH_CODE']

models = []

# Load model if it exists
if LOAD_MODEL:
    with open(Pkl_path, 'rb') as file: 
        models = pickle.load(file)
    print("Loaded model")

else:
    # Training
    kf = GroupKFold(n_splits=N_FOLDS)
    groups = train_dataset['WELL']
    groups = groups.replace(['Well-1', 'Well-2', 'Well-3'], 'Well-4')
    # groups = groups.replace(['Well-01', 'Well-02', 'Well-03'], 'Well-04')
    # group_list = ['Well-04', 'Well-05', 'Well-06', 'Well-07', 'Well-08', 'Well-09', 'Well-10', 'Well-11']
    # print(groups)

    f1 = np.zeros((N_FOLDS))
    val_well = []
    for i, (train_index, val_index) in enumerate(kf.split(X, groups=groups)):
        
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        #print(train_index)

        model = ExtraTreesClassifier(n_estimators=100, min_samples_split=10, random_state=SEED, max_features='log2')
        #model = AdaBoostClassifier(n_estimators=50, learning_rate=0.05, random_state=SEED)

        model.fit(X_train, y_train)
        yhat = model.predict(X_val)

        models.append(model)
        f1[i] = metrics.f1_score(y_val, yhat, average='micro')

        val_well.append(list(train_dataset.iloc[val_index]['WELL'].unique()))

    print(f1)
    print(val_well)
    # Save the model to file in the current working directory
    with open(Pkl_path, 'wb') as file:  
        pickle.dump(models, file)

# Inferrence
# X_test = test_dataset[['MD','GR', 'RT', 'DEN', 'CN','DEPOSITIONAL_ENVIRONMENT']]
# X_test = preprocessing.StandardScaler().fit(X_test).transform(X_test)
# y_test = test_dataset['LITH_CODE']

X_test = X
y_test = y
yhats = np.zeros((len(X_test), N_FOLDS))

for i, model in enumerate(models):
    yhat = model.predict(X_test)
    yhats[:, i] = yhat

final_yhat = []
for i in range(yhats.shape[0]):
    counts = Counter(yhats[:][i])
    final_yhat.append(counts.most_common(1)[0][0])

f1 = metrics.f1_score(y_test, final_yhat, average='micro')
print(f1)
