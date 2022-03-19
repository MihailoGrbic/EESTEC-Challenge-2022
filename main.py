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
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold, GroupKFold
import pickle

from util import fix_dataset

SEED = 31415
N_FOLDS = 5

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

train_dataset = train_dataset[train_dataset['WELL'] != 'Well-1']
train_dataset = train_dataset[train_dataset['WELL'] != 'Well-2']
train_dataset = train_dataset[train_dataset['WELL'] != 'Well-3']
train_dataset = train_dataset[train_dataset['WELL'] != 'Well-4']
train_dataset = train_dataset[train_dataset['WELL'] != 'Well-5']
train_dataset = train_dataset[train_dataset['WELL'] != 'Well-9']
# print(train_dataset['WELL'].unique())
train_dataset = train_dataset.reset_index()


X = train_dataset[['MD','GR','RT','DEN','CN','DEPOSITIONAL_ENVIRONMENT']]
X_test = test_dataset[['MD','GR','RT','DEN','CN','DEPOSITIONAL_ENVIRONMENT']]
ct = ColumnTransformer([
        ('some_name', preprocessing.StandardScaler(), ['MD','GR', 'RT', 'DEN', 'CN'])], remainder='passthrough')
X = pd.DataFrame(ct.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(ct.fit_transform(X_test), columns=X_test.columns)

# Mapping encorder
X['DEPOSITIONAL_ENVIRONMENT']=X['DEPOSITIONAL_ENVIRONMENT'].apply(lambda x: MAPPING[x])
X_test['DEPOSITIONAL_ENVIRONMENT']=X_test['DEPOSITIONAL_ENVIRONMENT'].apply(lambda x: MAPPING[x])
X = X.to_numpy()
X_test = X_test.to_numpy()
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

    f1_train = np.zeros((N_FOLDS))
    f1_val = np.zeros((N_FOLDS))
    val_well = []
    for i, (train_index, val_index) in enumerate(kf.split(X, groups=groups)):
        
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        #model_for_ada = DecisionTreeClassifier(max_depth=3)
        #model = AdaBoostClassifier(base_estimator=model_for_ada, n_estimators=200, learning_rate=0.1, random_state=SEED)
        model = RandomForestClassifier(n_estimators=102, min_samples_split=57, random_state=SEED)
        # model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, subsample = 1.0, min_samples_split=5, random_state=SEED)
        # model = MLPClassifier(hidden_layer_sizes=(100), activation='logistic', alpha=0.0001, learning_rate='adaptive', 
        #                         learning_rate_init=0.001, random_state=SEED)

        model.fit(X_train, y_train)
        
        yhat_train = model.predict(X_train)
        yhat_val = model.predict(X_val)

        models.append(model)
        f1_train[i] = metrics.f1_score(y_train, yhat_train, average='micro')
        f1_val[i] = metrics.f1_score(y_val, yhat_val, average='micro')

        val_well.append(list(train_dataset.iloc[val_index]['WELL'].unique()))

    print(f1_train)
    print(f1_val)
    print(val_well)
    print()
    # Save the model to file in the current working directory
    with open(Pkl_path, 'wb') as file:  
        pickle.dump(models, file)

# # Testing
# yhats = np.zeros((len(X), N_FOLDS))

# for i, model in enumerate(models):
#     yhat = model.predict(X)
#     yhats[:, i] = yhat

# final_yhat = []
# for i in range(yhats.shape[0]):
#     counts = Counter(yhats[:][i])
#     final_yhat.append(counts.most_common(1)[0][0])

# f1 = metrics.f1_score(y, final_yhat, average='micro')
# print(f1)


# Inferrence
yhats_test = np.zeros((len(X_test), N_FOLDS))
for i, model in enumerate(models):
    yhat = model.predict(X_test)
    yhats_test[:, i] = yhat

final_test_yhat = []
for i in range(yhats_test.shape[0]):
    counts = Counter(yhats_test[:][i])
    final_test_yhat.append(int(counts.most_common(1)[0][0]))

test_dataset['LITH_CODE'] = final_test_yhat
submission = test_dataset[['Id','LITH_CODE']]
submission.to_csv('submission.csv', index=False)