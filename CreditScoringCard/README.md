# Credit Scoring Card
Base on Logistic regression model.
# Data
Unzip the 'GiveMeSomeCredit.zip' from datasets folder and put the files to ./CreditScoringCard/data </p>
```
.
|-- CreditScoringCard
    |-- data
    |   |-- cs-test.csv
    |   |-- cs-training.csv
    |   |-- Data Dictionary.xls
    |   |-- sampleEntry.csv
    |-- ...
```
# baseline
run the scripts in order:
```
python /file/to/path/MissingValue.py
python /file/to/path/Outlier.py
python /file/to/path/MissingValue.py
python /file/to/path/Logistic.py
```