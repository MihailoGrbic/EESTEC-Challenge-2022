
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
N_FOLDS = 5

LOAD_MODEL = False

Pkl_path = 'checkpoints/VC1.pkl'

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
    kf = GroupKFold(n_splits=N_FOLDS)
    groups = train_dataset['WELL']
    groups = groups.replace(['Well-1', 'Well-2', 'Well-3', 'Well-4'], 'Well-5')

    f1_train = np.zeros((N_FOLDS))
    f1_val = np.zeros((N_FOLDS))
    val_well = []

    for i, (train_index, val_index) in enumerate(kf.split(X, groups=groups)):
        
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # rdf = RandomForestClassifier(n_estimators=102, min_samples_split=57, random_state=SEED)
        # gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.3, subsample = 0.9, min_samples_split=50, random_state=SEED)
        # xgb = XGBClassifier(n_estimators=100, learning_rate=0.3, max_depth=6, reg_lambda=3, random_state=SEED, 
        #                 n_jobs=4, gpu_id=0, verbosity=0)
        # mlp = MLPClassifier(hidden_layer_sizes=(100, 200, 100), max_iter=1000, alpha=0.0001, learning_rate='adaptive', 
        #                     learning_rate_init=0.001, early_stopping=True, random_state=SEED)
        # # model = VotingClassifier(estimators=[('rdf', rdf), ('gb', gb)], n_jobs=4)
        # final_estimator=GradientBoostingClassifier(n_estimators=100, learning_rate=0.3, subsample = 0.9, min_samples_split=50, random_state=SEED)
        # model = StackingClassifier(estimators=[('rdf', rdf), ('gb', gb), ('mlp', mlp)], 
        #                         final_estimator=final_estimator,passthrough=True, n_jobs=4)

        # model = AdaBoostClassifier(n_estimators=100,  learning_rate=0.1, random_state=SEED)
        # model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=5), n_estimators=100, 
        # learning_rate=0.1, random_state=SEED)
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
        yhat_val = model.predict(X_val)
        # print(yhat_val)

        # show_conf_matrix(y_val, yhat_val)

        models.append(model)
        f1_train[i] = metrics.f1_score(y_train, yhat_train, average='micro')
        f1_val[i] = metrics.f1_score(y_val, yhat_val, average='micro')
        print(f1_train[i])

        val_well.append(list(train_dataset.iloc[val_index]['WELL'].unique()))

    print(f1_train)
    print(f1_val)
    print(val_well)
    print()
    # Save the model to file in the current working directory
    with open(Pkl_path, 'wb') as file:  
        pickle.dump(models, file)
        

# Testing
yhats = np.zeros((len(X), N_FOLDS))

for i, model in enumerate(models):
    yhat = model.predict(X)
    yhats[:, i] = yhat

final_yhat = []
for i in range(yhats.shape[0]):
    counts = Counter(yhats[:][i])
    final_yhat.append(counts.most_common(1)[0][0])

f1 = metrics.f1_score(y, final_yhat, average='micro')
print(f1)


# # Inferrence
# yhats_test = np.zeros((len(X_test), N_FOLDS))
# for i, model in enumerate(models):
#     yhat = model.predict(X_test)
#     yhats_test[:, i] = yhat

# final_test_yhat = []
# for i in range(yhats_test.shape[0]):
#     counts = Counter(yhats_test[:][i])
#     final_test_yhat.append(int(counts.most_common(1)[0][0]))

# test_dataset['LITH_CODE'] = final_test_yhat
# submission = test_dataset[['Id','LITH_CODE']]
# submission.to_csv('submission.csv', index=False)