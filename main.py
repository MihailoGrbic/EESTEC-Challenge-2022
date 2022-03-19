import os

import numpy as np
import pandas as pd

from collections import Counter
import matplotlib.pyplot as plt
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

Pkl_path = 'checkpoints/RDF1.pkl' 

train_dataset = pd.read_csv('data/Train-dataset.csv')
test_dataset = pd.read_csv('data/Test-dataset.csv')


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


models = []

# Load model if it exists
if os.path.exists(Pkl_path):
    with open(Pkl_path, 'rb') as file: 
        models = pickle.load(file)
    print("Loaded model")
    
else:
    # Training
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    f1 = np.zeros((N_FOLDS))
    for i, (train_index, val_index) in enumerate(kf.split(X)):
        
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model = RandomForestClassifier(random_state=SEED)
        model.fit(X_train, y_train)
        yhat = model.predict(X_val)

        models.append(model)
        f1[i] = metrics.f1_score(y_val, yhat, average='micro')

    # Save the model to file in the current working directory
    with open(Pkl_path, 'wb') as file:  
        pickle.dump(models, file)

# Inferrence
yhats = []
for model in models:
    yhat = model.predict(X)
    yhats.append(yhats)

yhats = np.array(yhats)
for i in range(yhats.size[1]):
    counts = Counter(yhats[:,i])
    print(counts.most_common(1))
    final_yhat = [counts.most_common(1)] 

# f1[i] = metrics.f1_score(y, yhat, average='micro')

