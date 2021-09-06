# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
#from sklearn import datasets
import numpy as np
import pandas as pd
df=pd.read_csv('D:/uiuc/fa21/IE517/DS1.csv')

X=df.iloc[:,[2,3]].values
y=df.iloc[:,11].values

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=1,stratify=y)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

k_range=np.arange(80,110)
test_accuracy=np.empty(len(k_range))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
for i, k in enumerate (k_range):
    knn=KNeighborsClassifier(k)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    test_accuracy[i]=accuracy_score(y_test,y_pred)   
plt.plot(k_range, test_accuracy, label = 'Testing Accuracy')
print(max(test_accuracy))
plt.title('Varying Number of Neighbors')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

print("My name is Yuanqing Guo")
print("My NetID is: yg8")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")