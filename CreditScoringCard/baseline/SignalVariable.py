import os
import math

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

base_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_file = os.path.join(base_folder, 'data/cs-training.csv')

# auto split box
def mono_bin(Y, X, n=20):
    r = 0
    good = Y.sum()
    bad = Y.count() - good
    while np.abs(r) < 1:
        d1 = pd.DataFrame({'X': X, 'Y': Y, 'Bucket': pd.qcut(X, n, duplicates='drop')})
        d2 = d1.groupby('Bucket', as_index=True)
        r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
        n = n - 1
    d3 = pd.DataFrame(d2.X.min(), columns=['min'])
    d3['min'] = d2.min().X
    d3['max'] = d2.max().X
    d3['sum'] = d2.sum().Y
    d3['total'] = d2.count().Y
    d3['rate'] = d2.mean().Y
    d3['woe'] = np.log((d3['rate']/(1-d3['rate']))/(good/bad))
    d3['goodattribute'] = d3['sum'] / good
    d3['badattribute'] = (d3['total'] - d3['sum']) / bad
    iv = ((d3['goodattribute'] - d3['badattribute']) * d3['woe']).sum()
    d4 = (d3.sort_values(by='min')).reset_index(drop=True)
    # print('+'*55)
    # print(d4)
    cut = []
    cut.append(float('-inf'))
    for i in range(1, n+1):
        qua = X.quantile(i/(1+n))
        cut.append(round(qua, 4))
    cut.append(float('inf'))
    woe = list(d4['woe'].round(3))
    return d4, iv, cut, woe

# descret value split
def self_bin(Y, X, cut):
    good = Y.sum()
    bad = Y.count() - good
    d1 = pd.DataFrame({'X': X, 'Y': Y, 'Bucket': pd.qcut(X, len(cut), duplicates='drop')})
    d2 = d1.groupby('Bucket', as_index=True)
    d3 = pd.DataFrame(d2.X.min(), columns=['min'])
    d3['min'] = d2.min().X
    d3['max'] = d2.max().X
    d3['sum'] = d2.sum().Y
    d3['total'] = d2.count().Y
    d3['rate'] = d2.mean().Y
    d3['woe'] = np.log((d3['rate']/(1-d3['rate']))/(good/bad))
    d3['goodattribute'] = d3['sum'] / good
    d3['badattribute'] = (d3['total'] - d3['sum']) / bad
    iv = ((d3['goodattribute'] - d3['badattribute']) * d3['woe']).sum()
    d4 = (d3.sort_values(by='min')).reset_index(drop=True)
    woe = list(d4['woe'].round(3))
    return d4, iv, woe


def replace_woe(series, cut, woe):
    list = []
    i = 0
    while i < len(series):
        value = series[i]
        j = len(cut) - 2
        m = len(cut) - 2
        while j > 0:
            if value >= cut[j]:
                j -= 1
            else:
                j -= 1
                m -= 1
        if m >= len(woe):
            m = len(woe) - 1
        list.append(woe[m])
        i += 1
    return list

# calc score
def get_score(coe, woe, factor):
    scores = []
    for w in woe:
        score = round(coe * w * factor, 0)
        scores.append(score)
    return scores

# get score by param
def compute_score(series, cut, score):
    list = []
    i = 0
    while i < len(series):
        value = series[i]
        j = len(cut) - 2
        m = len(cut) - 2
        while j >= 0:
            if value >= cut[j]:
                j -= 1
            else:
                j -= 1
                m -= 1
        list.append(score[m])
        i += 1
    return list

def correlation_analysis(data):
    corr = data.corr()
    xticks = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']
    yticks = list(corr.index)
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    sns.heatmap(corr, annot=True, cmap='rainbow', ax=ax1, annot_kws={'size': 9, 'weight': 'bold', 'color': 'blue'})
    ax1.set_xticklabels(xticks, rotation=0, fontsize=10)
    ax1.set_yticklabels(yticks, rotation=0, fontsize=10)
    plt.show()

def iv_analysis(ivlist):
    index = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)
    x = np.arange(len(index)) + 1
    ax1.bar(x, ivlist, width=0.4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(index, rotation=0, fontsize=12)
    ax1.set_ylabel('IV', fontsize=12)
    # add label in graph
    for a, b in zip(x, ivlist):
        plt.text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=12)
    plt.show()


if __name__ == '__main__':
    train_data_file = os.path.join(base_folder, 'data/TrainData.csv')
    data = pd.read_csv(train_data_file)
    pinf = float('inf')
    ninf = float('-inf')
    Y = data['SeriousDlqin2yrs']
    # print(data.columns)
    dfx1, ivx1, cutx1, woex1 = mono_bin(Y, data['RevolvingUtilizationOfUnsecuredLines'], n=10)
    dfx2, ivx2, cutx2, woex2 = mono_bin(Y, data['age'], n=10)
    dfx4, ivx4, cutx4, woex4 = mono_bin(Y, data['DebtRatio'], n=20)
    dfx5, ivx5, cutx5, woex5 = mono_bin(Y, data['MonthlyIncome'], n=10)
    # decret continuous params
    cutx3 = [ninf, 0, 1, 3, 5, pinf]
    cutx6 = [ninf, 1, 2, 3, 5, pinf]
    cutx7 = [ninf, 0, 1, 3, 5, pinf]
    cutx8 = [ninf, 0, 1, 2, 3, pinf]
    cutx9 = [ninf, 0, 1, 3, pinf]
    cutx10 = [ninf, 0, 1, 2, 3, 5, pinf]
    df3, ivx3, woex3 = self_bin(Y, data['NumberOfTime30-59DaysPastDueNotWorse'], cutx3)
    df6, ivx6, woex6 = self_bin(Y, data['NumberOfOpenCreditLinesAndLoans'], cutx6)
    df7, ivx7, woex7 = self_bin(Y, data['NumberOfTimes90DaysLate'], cutx7)
    df8, ivx8, woex8 = self_bin(Y, data['NumberRealEstateLoansOrLines'], cutx8)
    df9, ivx9, woex9 = self_bin(Y, data['NumberOfTime60-89DaysPastDueNotWorse'], cutx9)
    df10, ivx10, woex10 = self_bin(Y, data['NumberOfDependents'], cutx10)

    ivlist = [ivx1, ivx2, ivx3, ivx4, ivx5, ivx6, ivx7, ivx8, ivx9, ivx10]
    iv_analysis(ivlist)

    # data -> woe
    data['RevolvingUtilizationOfUnsecuredLines'] = pd.Series(replace_woe(data['RevolvingUtilizationOfUnsecuredLines'], cutx1, woex1))
    data['age'] = pd.Series(replace_woe(data['age'], cutx2, woex2))
    data['NumberOfTime30-59DaysPastDueNotWorse'] = pd.Series(replace_woe(data['NumberOfTime30-59DaysPastDueNotWorse'], cutx3, woex3))
    data['DebtRatio'] = pd.Series(replace_woe(data['DebtRatio'], cutx4, woex4))
    data['MonthlyIncome'] = pd.Series(replace_woe(data['MonthlyIncome'], cutx5, woex5))
    data['NumberOfOpenCreditLinesAndLoans'] = pd.Series(replace_woe(data['NumberOfOpenCreditLinesAndLoans'], cutx6, woex6))
    data['NumberOfTimes90DaysLate'] = pd.Series(replace_woe(data['NumberOfTimes90DaysLate'], cutx7, woex7))
    data['NumberRealEstateLoansOrLines'] = pd.Series(replace_woe(data['NumberRealEstateLoansOrLines'], cutx8, woex8))
    data['NumberOfTime60-89DaysPastDueNotWorse'] = pd.Series(replace_woe(data['NumberOfTime60-89DaysPastDueNotWorse'], cutx9, woex9))
    data['NumberOfDependents'] = pd.Series(replace_woe(data['NumberOfDependents'], cutx10, woex10))
    woe_data_file = os.path.join(base_folder, 'data/WoeData.csv')
    data.to_csv(woe_data_file, index=False)

    # test -> woe
    test_data_file = os.path.join(base_folder, 'data/TestData.csv')
    test = pd.read_csv(test_data_file)
    test['RevolvingUtilizationOfUnsecuredLines'] = pd.Series(replace_woe(test['RevolvingUtilizationOfUnsecuredLines'], cutx1, woex1))
    test['age'] = pd.Series(replace_woe(test['age'], cutx2, woex2))
    test['NumberOfTime30-59DaysPastDueNotWorse'] = pd.Series(replace_woe(test['NumberOfTime30-59DaysPastDueNotWorse'], cutx3, woex3))
    test['DebtRatio'] = pd.Series(replace_woe(test['DebtRatio'], cutx4, woex4))
    test['MonthlyIncome'] = pd.Series(replace_woe(test['MonthlyIncome'], cutx5, woex5))
    test['NumberOfOpenCreditLinesAndLoans'] = pd.Series(replace_woe(test['NumberOfOpenCreditLinesAndLoans'], cutx6, woex6))
    test['NumberOfTimes90DaysLate'] = pd.Series(replace_woe(test['NumberOfTimes90DaysLate'], cutx7, woex7))
    test['NumberRealEstateLoansOrLines'] = pd.Series(replace_woe(test['NumberRealEstateLoansOrLines'], cutx8, woex8))
    test['NumberOfTime60-89DaysPastDueNotWorse'] = pd.Series(replace_woe(test['NumberOfTime60-89DaysPastDueNotWorse'], cutx9, woex9))
    test['NumberOfDependents'] = pd.Series(replace_woe(test['NumberOfDependents'], cutx10, woex10))
    woe_test_file = os.path.join(base_folder, 'data/TestWoeData.csv')
    data.to_csv(woe_test_file, index=False)

    # calc score
    # coe: logistic co-efficient
    coe = [9.738849, 0.638002, 0.505995, 1.032246, 1.790041, 1.131956]
    # base score = 600, PDO = 20, g/b = 20
    p = 20 / math.log(2)
    q = 600 - 20 * math.log(20) / math.log(2)
    basescore = round(q + p * coe[0], 0)
    x1 = get_score(coe[1], woex1, p)
    x2 = get_score(coe[2], woex2, p)
    x3 = get_score(coe[3], woex3, p)
    x7 = get_score(coe[4], woex7, p)
    x9 = get_score(coe[5], woex9, p)
    print(basescore, x1, x2, x3, x7, x9)
    test1 = pd.read_csv(test_data_file)
    test1['BaseScores'] = pd.Series(np.zeros(len(test1))) + basescore
    test1['x1'] = pd.Series(compute_score(test1['RevolvingUtilizationOfUnsecuredLines'], cutx1, x1))
    test1['x2'] = pd.Series(compute_score(test1['age'], cutx2, x2))
    test1['x3'] = pd.Series(compute_score(test1['NumberOfTime30-59DaysPastDueNotWorse'], cutx3, x3))
    test1['x7'] = pd.Series(compute_score(test1['NumberOfTimes90DaysLate'], cutx7, x7))
    test1['x9'] = pd.Series(compute_score(test1['NumberOfTime60-89DaysPastDueNotWorse'], cutx9, x9))
    test1['Score'] = test1['x1'] + test1['x2'] + test1['x3'] + test1['x7'] + test1['x9'] + basescore
    score_file = os.path.join(base_folder, 'data/ScoreData.csv')
    test1.to_csv(score_file, index=False)
    