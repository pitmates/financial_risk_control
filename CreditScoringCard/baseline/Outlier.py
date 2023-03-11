import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


base_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_file = os.path.join(base_folder, 'data/cs-training.csv')

def data_visualization(data):
    numb30 = data['NumberOfTime30-59DaysPastDueNotWorse']
    numb90 = data['NumberOfTimes90DaysLate']
    numb60 = data['NumberOfTime60-89DaysPastDueNotWorse']
    labels = 'NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfTimes90DaysLate'
    plt.boxplot([numb30, numb60, numb90], labels)
    plt.show()

def remove_outliers(data):
    # remove outlier
    data = data[data['age'] > 0]
    data = data[data['NumberOfTime30-59DaysPastDueNotWorse'] < 90]
    # data = data[data['NumberOfTime60-89DaysPastDueNotWorse'] < 90]
    # data = data[data['NumberOfTimes90DaysLate'] < 90]
    # get reverse for feature SeriousDlqin2yrs
    data['SeriousDlqin2yrs'] = 1 - data['SeriousDlqin2yrs']
    return data

def data_split(data):
    Y = data['SeriousDlqin2yrs']
    X = data.iloc[:, 1:]
    # test data for 30%
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    train = pd.concat([Y_train, X_train], axis=1)
    train = train.T.drop_duplicates().T
    test = pd.concat([Y_test, X_test], axis=1)
    test = test.T.drop_duplicates().T

    classTest = test.groupby('SeriousDlqin2yrs')['SeriousDlqin2yrs'].count()
    print('test class count: ', classTest)
    train_data_file = os.path.join(base_folder, 'data/TrainData.csv')
    test_data_file = os.path.join(base_folder, 'data/TestData.csv')
    train.to_csv(train_data_file, index=False)
    test.to_csv(test_data_file, index=False)

if __name__ == '__main__':
    missing_file = os.path.join(base_folder, 'data/MissingData.csv')
    data = pd.read_csv(missing_file)
    # data_visualization(data)

    data = remove_outliers(data)

    data_split(data)

