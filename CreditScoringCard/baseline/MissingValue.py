import os

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

base_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_file = os.path.join(base_folder, 'data/cs-training.csv')

# RF predict missing value
def set_missing(df):
    # get value of feature
    process_df = df.iloc[:, :10]
    cols = list(process_df)
    monthlyIncome = df['MonthlyIncome']
    process_df.drop(labels=['MonthlyIncome'], axis=1, inplace=True)
    process_df.insert(0, 'MonthlyIncome', monthlyIncome)
    
    # split to unknow and know part
    known = process_df[~(np.isnan(process_df['MonthlyIncome']))].values
    unknown = process_df[np.isnan(process_df['MonthlyIncome'])].values
    # X as the feature value
    X_raw = known[:, 1:]
    # Y as the label
    y = known[:, 0]

    # create imputer to replace missing data values with mean
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp = imp.fit(X_raw)
    # impute data
    X = imp.transform(X_raw)

    # fit them to RandomForestRegressor
    rfr = RandomForestRegressor(random_state=0,
        n_estimators=200, max_depth=3, n_jobs=-1)
    rfr.fit(X, y)
    # predict missing value by RF model
    predicted = rfr.predict(unknown[:, 1:]).round(0)
    print('predict:', len(predicted))
    # fill the missing value by predicted value
    df.loc[(df.MonthlyIncome.isnull()), 'MonthlyIncome'] = predicted
    return df

if __name__ == '__main__':
    data = pd.read_csv(train_file)
    # data.describe().to_csv(os.path.join(base_folder, 'data/DataDescribe.csv'))
    missing_file = os.path.join(base_folder, 'data/MissingData.csv')
    df = pd.DataFrame(data)
    data = set_missing(df)
    data = data.dropna()
    data = data.drop_duplicates()
    data.to_csv(missing_file, index=False)
