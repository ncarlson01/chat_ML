# -*- coding: utf-8 -*-
"""
@author: Nick
"""

import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

#designate input file
input_file = "chatdata.csv"

#pandas read input csv
dataset = pd.read_csv(input_file, header = 0,  sep=',')

#select data
X = dataset.iloc[:, 2:]  #select columns 2 through end, predictors
y = dataset.iloc[:, 1]   #select column 1, target

#shuffle the data and split the sample into training and test data
X_train, X_test, y_train, y_test = train_test_split( X, y, train_size=433, test_size=100, shuffle=True)

#standarize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = tree.DecisionTreeClassifier(criterion='entropy')

#train model
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)

#print classification report
clf = clf.predict(X_test)
report = classification_report(y_test, clf)
print(report)

print("Test score with L1 penalty: %.4f" % score)
