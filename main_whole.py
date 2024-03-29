
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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
import pickle

from util import fix_dataset, show_conf_matrix, remove_classes, remove_wells, add_features, lithology_key

SEED = 31415

LOAD_MODEL = True

Pkl_path = 'checkpoints/FINAL.pkl'

train_dataset = pd.read_csv('data/Train-dataset.csv')
test_dataset = pd.read_csv('data/Test-dataset.csv')

# Feature Extraction
MAPPING = {
    'Continental': 1,
    'Transitional': 2,
    'Marine': 3,
}

train_dataset = remove_wells(train_dataset)
train_dataset = remove_classes(train_dataset)

train_dataset = train_dataset.reset_index()

# train_dataset = add_features(train_dataset)
# test_dataset = add_features(test_dataset)

# print(train_dataset['WELL'].unique())
# print(train_dataset['LITH_NAME'].unique())

train_dataset = train_dataset.reset_index()

# X = train_dataset[['MD','GR','RT','DEN','CN','GR_+1', 'GR_-1', 'CN_+1', 'CN_-1', 'DEPOSITIONAL_ENVIRONMENT']]
# X_test = test_dataset[['MD','GR','RT','DEN','CN','GR_+1', 'GR_-1', 'CN_+1', 'CN_-1', 'DEPOSITIONAL_ENVIRONMENT']]
X = train_dataset[['MD','GR','RT','DEN','CN', 'DEPOSITIONAL_ENVIRONMENT']]
X_test = test_dataset[['MD','GR','RT','DEN','CN', 'DEPOSITIONAL_ENVIRONMENT']]
# ct = ColumnTransformer([
#         ('some_name', preprocessing.StandardScaler(), ['MD','GR', 'RT', 'DEN', 'CN', 'GR_+1', 'GR_-1', 'CN_+1', 'CN_-1'])], 
#         remainder='passthrough')
ct = ColumnTransformer([
        ('some_name', preprocessing.StandardScaler(), ['MD','GR', 'RT', 'DEN', 'CN'])], 
        remainder='passthrough')

ct.fit(X)
X = pd.DataFrame(ct.transform(X), columns=X.columns)
X_test = pd.DataFrame(ct.transform(X_test), columns=X_test.columns)

# Mapping encorder
X['DEPOSITIONAL_ENVIRONMENT']=X['DEPOSITIONAL_ENVIRONMENT'].apply(lambda x: MAPPING[x])
X_test['DEPOSITIONAL_ENVIRONMENT']=X_test['DEPOSITIONAL_ENVIRONMENT'].apply(lambda x: MAPPING[x])
X = X.to_numpy()
X_test = X_test.to_numpy()
y = train_dataset['LITH_CODE'].to_numpy()

rdf_models = []
gb_models = []
models = []

# Load model if it exists
if LOAD_MODEL:
    with open(Pkl_path, 'rb') as file: 
        models = pickle.load(file)

else:
    # Training
    f1_train = np.zeros((1))
    f1_val = np.zeros((1))
    val_well = []

    X_train = X
    y_train = y
        

    # rdf = RandomForestClassifier(n_estimators=102, min_samples_split=57, random_state=SEED)
    # gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.3, subsample = 0.9, min_samples_split=50, random_state=SEED)
    # xgb = XGBClassifier(n_estimators=100, learning_rate=0.3, max_depth=6, reg_lambda=3, random_state=SEED, 
    #                 n_jobs=4, gpu_id=0, verbosity=0)
    # mlp = MLPClassifier(hidden_layer_sizes=(100, 200, 100), max_iter=1000, alpha=0.0001, learning_rate='adaptive', 
    #                     learning_rate_init=0.001, early_stopping=True, random_state=SEED)
    # model = VotingClassifier(estimators=[('rdf', rdf), ('gb', gb)], n_jobs=4)
    # model = StackingClassifier(estimators=[('rdf', rdf), ('gb', gb), ('mlp', mlp)], 
    #                         passthrough=True, n_jobs=4)

    # tree_for_ada = DecisionTreeClassifier(max_depth=5)
    # model = AdaBoostClassifier(base_estimator=tree_for_ada, n_estimators=400, learning_rate=0.1, random_state=SEED)
    # model = RandomForestClassifier(n_estimators=102, min_samples_split=57, random_state=SEED)
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.3, subsample = 0.9, min_samples_split=50, 
                                        random_state=SEED)
    # model = XGBClassifier(n_estimators=50, learning_rate=0.1, max_depth=6, reg_lambda=10, random_state=SEED, 
    #                         n_jobs=4, gpu_id=0, verbosity=0)
    # model = MLPClassifier(hidden_layer_sizes=(100, 200, 100), max_iter=1000, alpha=0.0001, learning_rate='adaptive', 
    #                         learning_rate_init=0.001, early_stopping=True, random_state=SEED)
    # model = LogisticRegression(C=0.9, random_state=SEED, n_jobs=4)
    # model = DecisionTreeClassifier(min_samples_split=25, random_state=SEED)

    model.fit(X_train, y_train)
    
    yhat_train = model.predict(X_train)

    # show_conf_matrix(y_val, yhat_val)

    models.append(model)
    f1_train[0] = metrics.f1_score(y_train, yhat_train, average='micro')

    print(f1_train)
    print()
    # Save the model to file in the current working directory
    with open(Pkl_path, 'wb') as file:  
        pickle.dump(models, file)
        

model = models[0]
# Testing
final_yhat = model.predict(X)
f1 = metrics.f1_score(y, final_yhat, average='micro')
print(f1)


# Inferrence
final_test_yhat = model.predict(X_test)

test_dataset['LITH_CODE'] = final_test_yhat
submission = test_dataset[['Id','LITH_CODE']]
submission.to_csv('submission.csv', index=False)