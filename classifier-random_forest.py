import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import median

# democrat = 0, republican = 1

dataset = pd.read_csv('Parliment-1984.csv')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

for i in range(0,434):
    if y[i] == 'democrat':
        y[i] = 0
    elif y[i] == 'republican':
        y[i] = 1
y = y.astype(int)

for a in range(0, 434):
    for b in range(0,16):
        if ('y' in X[a][b]):
            X[a][b] = 1
        elif ('n' in X[a][b]):
            X[a][b] = 0
            
medians = []
for x in range(0, 16):
    acceptable = []
    for z in range(0,434):
        if((X[z][x] == 1) or (X[z][x] == 0)):
            acceptable.append(X[z][x])
    med = median(acceptable)
    medians.append(int(med))
    
for c in range(0, 434):
    for d in range(0,16):
        if ((X[c][d] != 1) and (X[c][d] != 0)):
            X[c][d] = medians[d]
                        
X = X.astype(int)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import f1_score
f1_score(y_test, y_pred, average='binary')  
# F1 score is approximately 0.92