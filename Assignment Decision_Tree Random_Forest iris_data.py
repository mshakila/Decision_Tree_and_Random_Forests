########### ASSIGNMENT DECISION TREE and RANDOM FOREST on IRIS dataset

import  pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from collections import Counter

data = pd.read_csv("E:/Datasets/iris.csv")
data.head()
data.tail()
data.columns
# 'Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width', 'Species'
data.Species.value_counts()
data.Species.unique()

#### Splitting the datasets
col = data.columns
X =data[col[0:4]]
y = data[col[4]]
train, test = train_test_split(data,stratify=y ,test_size=0.3, random_state=123)
# by using stratify=y, we are maintaining the proportion of categories of 
# dependant variable in train and test sets. i.e., original dataset and train and
# test data, all have 33% of each category of species

train.Species.value_counts()
test.Species.value_counts()
Counter(train.Species)

##### Model building using Decision Tree

model = DecisionTreeClassifier(criterion="entropy")
model.fit(train[col[0:4]],train.Species)

## Finding training accuracy
pred_train = model.predict(train[col[0:4]])
type(pred_train)
# pd.Series(pred_train).value_counts()
pd.crosstab(train.Species, pred_train)
metrics.accuracy_score(train.Species, pred_train)  # 100% accuracy

## Finding testing accuracy
pred_test = model.predict(test[col[0:4]])
pd.crosstab(test.Species, pred_test)
np.mean(test.Species == pred_test) # 0.933
print(metrics.classification_report(test.Species, pred_test))

# training accuracy is 100% 
# But testing accuracy is 93%. Recall rates for setosa and virginica Species is
# 100% whereas for versicolor it is 80%

############## Random Forest
# from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(n_estimators=100, criterion='entropy', 
                                  n_jobs=2, oob_score=True, random_state=123)
model_rf.fit(train[col[0:4]], train.Species)

pred_train = model_rf.predict(train[col[0:4]])
pd.crosstab(train.Species, pred_train)
metrics.accuracy_score(train.Species, pred_train)  # 100% accuracy

pred_test = model_rf.predict(test[col[0:4]])
pd.crosstab(test.Species, pred_test)
print(metrics.classification_report(test.Species, pred_test))
# training accuracy is 100% and testing accuracy is 96%. Recall rates for 
# setosa and virginica Species is 100% whereas for versicolor it is 87%

# Both Decision tree and Random Forest classifiers, have not been able to
# correctly predict versicolor 

############# Naive Bayes

from sklearn.naive_bayes import GaussianNB
model_gnb = GaussianNB()
# using GaussianNB because predictors are all continuous variables

pred_gnb = model_gnb.fit(train[col[0:4]], train.Species).predict(test[col[0:4]])

np.mean(test.Species == pred_gnb) # 0.9777
metrics.confusion_matrix(test.Species, pred_gnb)
print(metrics.classification_report(test.Species, pred_gnb))

# testing accuracy is 98% (which is highly impossible in real business scenarios)
# versicolor is 93% correctly predicted

############# Logistic Regression

from sklearn.linear_model import  LogisticRegression

model_logreg = LogisticRegression()
model_logreg.fit(train[col[0:4]], train.Species)

pred_logreg = model_logreg.predict(test[col[0:4]])

metrics.accuracy_score(test.Species, pred_logreg) # 0.977
metrics.confusion_matrix(test.Species, pred_logreg)
print(metrics.classification_report(test.Species, pred_logreg))

# the testing accuracy results are similar to Naive Bayes algorithm

'''
CONCLUSIONS

We have used Decision Tree and Random Forest classifiers to predict the 
Species of iris dataset. 

We have also modelled using Naive Bayes and Logistic Regression to compare
the results.


'''

