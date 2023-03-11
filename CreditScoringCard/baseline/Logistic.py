import os

import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

base_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def logistic_model(data, test):
    Y = data['SeriousDlqin2yrs']
    X = data.drop(['SeriousDlqin2yrs', 'NumberOfTimes90DaysLate', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfOpenCreditLinesAndLoans', 'NumberRealEstateLoansOrLines', 'NumberOfDependents'], axis=1)
    X1 = sm.add_constant(X)
    logit = sm.Logit(Y, X1)
    result = logit.fit()
    print(result.summary())

    Y_test = test['SeriousDlqin2yrs']

    X_test = test.drop(['SeriousDlqin2yrs', 'NumberOfTimes90DaysLate', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfOpenCreditLinesAndLoans', 'NumberRealEstateLoansOrLines', 'NumberOfDependents'], axis=1)
    X3 = sm.add_constant(X_test)
    resu = result.predict(X3)
    fpr, tpr, threshold = roc_curve(Y_test, resu)
    rocauc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label='AUC - %.2f' % rocauc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.show()

if __name__ == '__main__':
    woe_file = os.path.join(base_folder, 'data/WoeData.csv')
    data = pd.read_csv(woe_file)
    test_file = os.path.join(base_folder, 'data/TestData.csv')
    test = pd.read_csv(test_file)
    logistic_model(data, test)


    
