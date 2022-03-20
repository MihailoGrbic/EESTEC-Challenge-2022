import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def show_conf_matrix(y_test, y_pred):
    # Calculate confusion matrix
    conf = confusion_matrix(y_test, y_pred)

    target_lithologys = []
    labels = np.sort(y_test.unique())

    for l_code in labels:
        lithology = lithology_key[l_code]
        target_lithologys.append(lithology)
    print(y_test.value_counts())
    classes = target_lithologys

    plt.figure(figsize=(12,12))
    sns.set(font_scale=1)
    sns.heatmap(conf, annot=True, annot_kws={"size": 16}, fmt="d", linewidths=.5, cmap="YlGnBu", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted value')
    plt.ylabel('True value')
    
    plt.show()

def fix_dataset(dataset):
    dataset = dataset.replace('Well-1', 'Well-01')
    dataset = dataset.replace('Well-2', 'Well-02')
    dataset = dataset.replace('Well-3', 'Well-03')
    dataset = dataset.replace('Well-4', 'Well-04')
    dataset = dataset.replace('Well-5', 'Well-05')
    dataset = dataset.replace('Well-6', 'Well-06')
    dataset = dataset.replace('Well-7', 'Well-07')
    dataset = dataset.replace('Well-8', 'Well-08')
    dataset = dataset.replace('Well-9', 'Well-09')
    dataset = dataset.sort_values('WELL')
    print(dataset)
    return dataset

def remove_wells(dataset):
    dataset = dataset[dataset['WELL'] != 'Well-1']
    dataset = dataset[dataset['WELL'] != 'Well-2']
    dataset = dataset[dataset['WELL'] != 'Well-3']
    dataset = dataset[dataset['WELL'] != 'Well-4']
    dataset = dataset[dataset['WELL'] != 'Well-5']
    dataset = dataset[dataset['WELL'] != 'Well-9']
    return dataset

def remove_classes(dataset):
    dataset = dataset[dataset['LITH_CODE'] != 200] # Siltstone/Loess
    dataset = dataset[dataset['LITH_CODE'] != 1400] # Marl clay
    dataset = dataset[dataset['LITH_CODE'] != 1500] # Siltstone clay
    dataset = dataset[dataset['LITH_CODE'] != 1100] # Coal clay
    dataset = dataset[dataset['LITH_CODE'] != 800] # Tight
    dataset = dataset[dataset['LITH_CODE'] != 1200] # Marly sandstone
    dataset = dataset[dataset['LITH_CODE'] != 300] # Marl
    return dataset

def add_features(dataset):
    i = 5

    # GR
    new = dataset['GR'][i:]
    new.index -= i
    dataset['GR_+1'] = new
    for j in range(i): dataset.loc[:,('GR_+1')].iloc[-j-1] = dataset['GR'].iloc[-j-1]

    new = dataset['GR'][:-i]
    new.index += i
    dataset['GR_-1'] = new
    for j in range(i): dataset.loc[:,('GR_-1')].iloc[j] = dataset['GR'].iloc[j]

    # CN
    new = dataset['CN'][i:]
    new.index -= i
    dataset['CN_+1'] = new
    for j in range(i): dataset.loc[:,('CN_+1')].iloc[-j-1] = dataset['CN'].iloc[-j-1]

    new = dataset['CN'][:-i]
    new.index += i
    dataset['CN_-1'] = new
    for j in range(i): dataset.loc[:,('CN_-1')].iloc[j] = dataset['CN'].iloc[j]

    # print(dataset.isnull().values.any())
    # print(dataset['GR_-1'])
    # print(dataset['GR_+1'])
    return dataset

lithology_key = {100: 'Clay',
                 200: 'Siltstone/Loess',
                 300: 'Marl',
                 400: 'Clay marl',
                 500: 'Clay sandstone',
                 600: 'Sandstone',
                 700: 'Limestone',
                 800: 'Tight',
                 900: 'Dolomite',
                 1000: 'Coal',
                 1100: 'Coal clay',
                 1200: 'Marly sandstone',
                 1300: 'Sandy marl',
                 1400: 'Marl clay',
                 1500: 'Siltstone clay'
                }
