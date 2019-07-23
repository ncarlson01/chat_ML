import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
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

#kneighbors classifier object
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, 
                           p=2, metric='minkowski', metric_params=None, n_jobs=None)

#fit model
knn.fit(X_train, y_train)

#response prediction
pred = knn.predict(X_test)

#accuracy
knn.score(X_test, y_test)

#print classification report
knn = knn.predict(X_test)
report = classification_report(y_test, knn)
print(report)

#evaluate accuracy
print("Accuracy on test set: %0.3f%%"%(accuracy_score(y_test, pred)*100))
