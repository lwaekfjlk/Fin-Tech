import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# some usable model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import Normalizer
from sklearn.impute import SimpleImputer as Imputer
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

def data_visualize(data):
    yes_count = data[data['y']=='yes'].count()['y']
    no_count = data[data['y']=='no'].count()['y']
    print(yes_count)
    print(no_count)
    ratio = [yes_count/(yes_count+no_count),no_count/(yes_count+no_count)]
    labels = ['yes','no']
    plt.figure(figsize=(8, 4)) 
    plt.title('Dataset')
    plt.pie(ratio,labels=labels)
    plt.show()

    job = data[data['y']=='yes'].groupby('job').count()['y'].index
    job_count_yes = data[data['y']=='yes'].groupby('job').count()['y']
    job_count_no = data[data['y']=='no'].groupby('job').count()['y']
    width = 0.5
    plt.figure(figsize=(8, 4)) 
    plt.subplot(121)
    plt.bar(job,job_count_yes, width, color='yellow', label='Yes')
    plt.bar(job,job_count_no, width, bottom = job_count_yes, color='green', label='No')
    plt.xticks(rotation=90)
    plt.legend()
    plt.subplot(122)
    plt.bar(job,job_count_yes/job_count_no, width, color='red', label='Yes/No')
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()

    marital = data[data['y']=='yes'].groupby('marital').count()['y'].index
    marital_count_yes = data[data['y']=='yes'].groupby('marital').count()['y']
    marital_count_no = data[data['y']=='no'].groupby('marital').count()['y']
    plt.figure(figsize=(8, 4)) 
    plt.subplot(121)
    plt.title('Yes')
    plt.pie(marital_count_yes.values,labels = marital)
    plt.subplot(122)
    plt.title('No')
    plt.pie(marital_count_no.values,labels = marital)
    plt.show()

    housing = data[data['y']=='yes'].groupby('housing').count()['y'].index
    housing_count_yes = data[data['y']=='yes'].groupby('housing').count()['y']
    housing_count_no = data[data['y']=='no'].groupby('housing').count()['y']
    plt.figure(figsize=(8, 4)) 
    plt.subplot(121)
    plt.title('Yes')
    plt.pie(housing_count_yes.values,labels = housing)
    plt.subplot(122)
    plt.title('No')
    plt.pie(housing_count_no.values,labels = housing)
    plt.show()

    loan = data[data['y']=='yes'].groupby('loan').count()['y'].index
    loan_count_yes = data[data['y']=='yes'].groupby('loan').count()['y']
    loan_count_no = data[data['y']=='no'].groupby('loan').count()['y']
    plt.figure(figsize=(8, 4)) 
    plt.subplot(121)
    plt.title('Yes')
    plt.pie(loan_count_yes.values,labels = loan)
    plt.subplot(122)
    plt.title('No')
    plt.pie(loan_count_no.values,labels = loan)
    plt.show()

def data_preprocess(data):
    # your code here
    # example:
    label = LabelEncoder()
    label_count = 0

    for col in data:
        if data[col].dtype == 'object':
            if len(list(data[col].unique())) <= 2:
                # Train on data
                label.fit(data[col])
                # Transform data
                data[col] = label.transform(data[col])
                label_count += 1
    x = pd.get_dummies(data)

    scaler = Normalizer()
    imputer = Imputer(strategy = 'median')
    imputer.fit(x)
    x = imputer.transform(x)
    scaler.fit(x)
    x = scaler.transform(x)

    # your code end
    return x

def predict_KNeighbors(x_train, x_test, y_train):

    # your code here begin
    # train your model on 'x_train' and 'x_test'
    # predict on 'y_train' and get 'y_pred'
    model = KNeighborsClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # your code here end

    return y_pred

def predict_AdaBoost(x_train, x_test, y_train):

    # your code here begin
    # train your model on 'x_train' and 'x_test'
    # predict on 'y_train' and get 'y_pred'
    model = SVC()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # your code here end

    return y_pred

def predict_LR(x_train, x_test, y_train):

    # your code here begin
    # train your model on 'x_train' and 'x_test'
    # predict on 'y_train' and get 'y_pred'
    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # your code here end

    return y_pred

def predict_RandomForest(x_train, x_test, y_train):

    # your code here begin
    # train your model on 'x_train' and 'x_test'
    # predict on 'y_train' and get 'y_pred'
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # your code here end

    return y_pred

def predict_GradientBoosting(x_train, x_test, y_train):

    # your code here begin
    # train your model on 'x_train' and 'x_test'
    # predict on 'y_train' and get 'y_pred'
    model = GradientBoostingClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # your code here end

    return y_pred

def predict_Bagging(x_train, x_test, y_train):

    # your code here begin
    # train your model on 'x_train' and 'x_test'
    # predict on 'y_train' and get 'y_pred'
    model = BaggingClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # your code here end

    return y_pred

def predict_AdaBoost(x_train, x_test, y_train):

    # your code here begin
    # train your model on 'x_train' and 'x_test'
    # predict on 'y_train' and get 'y_pred'
    model = AdaBoostClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # your code here end

    return y_pred

def predict_MLP(x_train, x_test, y_train):

    # your code here begin
    # train your model on 'x_train' and 'x_test'
    # predict on 'y_train' and get 'y_pred'
    model = MLPClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # your code here end

    return y_pred

def predict_DecisionTree(x_train, x_test, y_train):

    # your code here begin
    # train your model on 'x_train' and 'x_test'
    # predict on 'y_train' and get 'y_pred'
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # your code here end

    return y_pred

def split_data(data):
    y = data.y
    x = data.loc[:, data.columns != 'y']
    x = data_preprocess(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    return x_train, x_test, y_train, y_test

def print_result(y_test, y_pred):
    report = confusion_matrix(y_test, y_pred)
    precision = report[1][1] / (report[:, 1].sum())
    recall = report[1][1] / (report[1].sum())
    print('model precision:' + str(precision)[:4] + ' recall:' + str(recall)[:4])

if __name__ == '__main__':
    data = pd.read_csv('bank-additional-full.csv', sep=';')
    data_visualize(data)
    
    x_train, x_test, y_train, y_test = split_data(data)
    y_pred = predict_DecisionTree(x_train, x_test, y_train)
    print_result(y_test, y_pred)
    y_pred = predict_MLP(x_train, x_test, y_train)
    print_result(y_test, y_pred)
    y_pred = predict_Bagging(x_train, x_test, y_train)
    print_result(y_test, y_pred)
    y_pred = predict_GradientBoosting(x_train, x_test, y_train)
    print_result(y_test, y_pred)
    y_pred = predict_RandomForest(x_train, x_test, y_train)
    print_result(y_test, y_pred)
    y_pred = predict_LR(x_train, x_test, y_train)
    print_result(y_test, y_pred)
    y_pred = predict_AdaBoost(x_train, x_test, y_train)
    print_result(y_test, y_pred)
    y_pred = predict_KNeighbors(x_train, x_test, y_train)
    print_result(y_test, y_pred)
    

