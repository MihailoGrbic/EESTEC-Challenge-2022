import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def show_conf_matrix(y_test, y_pred, classes):
    
    # Calculate confusion matrix
    conf = confusion_matrix(y_test, y_pred)

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