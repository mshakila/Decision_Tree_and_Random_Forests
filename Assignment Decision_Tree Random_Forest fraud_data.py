########### ASSIGNMENT DECISION TREE and RANDOM FOREST on FRAUD dataset

# PROBLEM STATEMENT : Developing a model using DECISION TREE for classifying
# individuals as 'risky' or 'good'

import pandas as pd
import  numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

fraud_raw = pd.read_csv("E:/Decision Trees/Fraud_check.csv")
fraud_raw.columns
fraud_raw.dtypes

fraud_raw.head()
fraud_raw.tail()
fraud_raw.describe()
# treating those who have taxable_income <= 30000 as "Risky" and others as "Good"
#adding a new column category by modifying Taxable.Income group. 

category = pd.cut(fraud_raw['Taxable.Income'], bins=[0,30000,100000],
                  labels=['Risky','Good'])

fraud = fraud_raw.copy()
fraud.insert(6,'Tax_group',category)
fraud.dtypes

fraud.drop(fraud.columns[[2]],axis=1, inplace=True)
fraud.head()
fraud.columns
# 'Undergrad', 'Marital.Status', 'City.Population', 'Work.Experience',
#      'Urban', 'Tax_group'

fraud.Tax_group.value_counts()
fraud.Tax_group.value_counts(normalize=True) # this gives proportion
'''
Good     0.793333
Risky    0.206667  '''

################## encoding the variable text classes to numbered-values

string_columns=['Undergrad', 'Marital.Status', "Urban","Tax_group"]

le = LabelEncoder()
for i in string_columns:
    fraud[i] = le.fit_transform(fraud[i])

### splitting the data

col = fraud.columns
X = fraud[col[0:5]]
y = fraud[col[5]]

train, test =train_test_split(fraud,stratify=y,test_size=0.25, random_state=111)

train.Tax_group.value_counts()
# Actual counts of target in train group
# 0    357
# 1     93
train.Tax_group.value_counts(normalize=True)

test.Tax_group.value_counts()
# Actual counts of target in test group
# 0    119
# 1     31
test.Tax_group.value_counts(normalize=True)

fraud.Tax_group.value_counts(normalize=True)
# similar proportion of Tax_group (y) in original, train and test data

##################### Model building

############# MODEL1

model1 = DecisionTreeClassifier(criterion="entropy")
model1.fit(train[col[0:5]], train.Tax_group)

pred_train = model1.predict(train[col[0:5]])
pd.crosstab(train.Tax_group, pred_train)
'''
col_0        0   1
Tax_group         
0          357   0
1            0  93 '''
np.mean(train.Tax_group ==pred_train) # 100% training accuracy

pred_test = model1.predict(test[col[0:5]])
pd.crosstab(test.Tax_group, pred_test)
np.mean(test.Tax_group == pred_test) # 66% testing accuracy
print(metrics.classification_report(test.Tax_group, pred_test))

'''
Here 0-class indicates Good and 1-class indicates Risky Tax-payers
Let us consider 0-class as Negative class and 1-class as Positive class
'''
# we find that the training accuracy is 100% but testing accuracy is just
# 66%. This indicates that our model is overfitting. 
#  Also, 76% of good tax-payers are correctly predicted but only 26% of
# Risky tax-payers are correctly predicted.

################# MODEL 2 

# Let us change the random-state=123 during splitting the data. Previously the 
# seed was 111, now if we change to 123, the results are as given below:
'''
The overall training accuracy is 100%. Also, the True-Negative and True-Positive 
rates are 100% each. 
The overall testing accuracy is 60.67%. But 74% of Good are
correctly predicted and only 10% of Risky are correctly predicted. 

when splitting is done using seed as 111, TP rate (recall) is higher. Hence 
for further modelling we will use random-state=111.
'''

############# MODEL 3 
# Let us change the Decision Tree criterion to 'gini' .

'''
The overall training accuracy is 100%. Also, the True-Negative and True-Positive 
rates are 100% each. 
The overall testing accuracy is 60.67%. TN rate is 71% and TP rate is 19%

Model-1 is better than other two models. 
We will be using entropy and seed-111 for our further models
''''
###############

# All the above deciison tree were built completely without pruning 
# (default max_depth=None). 
# Now let us prune the trees to overcome the problem of overfitting. 

############# OTHER DECISION TREE MODELS after pruning

model_prune = DecisionTreeClassifier(criterion='entropy',max_depth=30,min_samples_leaf=5)
model_prune.fit(train[col[0:5]] , train[col[5]])

pred_train = model_prune.predict(train[col[0:5]])
pd.crosstab(train.Tax_group, pred_train)
metrics.accuracy_score(train.Tax_group, pred_train)
print(metrics.classification_report(train.Tax_group, pred_train))

pred_test = model_prune.predict(test[col[0:5]])
pd.crosstab(test.Tax_group, pred_test)
metrics.accuracy_score(test.Tax_group, pred_test)
print(metrics.classification_report(test.Tax_group, pred_test))


'''
When we use max_depth=1,2 then the results:-
The overall training accuracy is 79.33%. TN rate is 100% and TP rate is 0%
The overall testing accuracy is 79.33%. TN rate is 100% and TP rate is 0%

When we use max_depth=3 then the results:-
The overall training accuracy is 79.78%. TN rate is 99% and TP rate is 5%
The overall testing accuracy is 76.67%. TN rate is 97% and TP rate is 0%

When we use max_depth=4 then the results:-
The overall training accuracy is 80.67%. TN rate is 97% and TP rate is 19%
The overall testing accuracy is 74%. TN rate is 92% and TP rate is 3%

When we use max_depth=4 then the results:-
The overall training accuracy is 80.67%. TN rate is 97% and TP rate is 19%
The overall testing accuracy is 74%. TN rate is 92% and TP rate is 3%


Models	   trainAcc testAcc   TN(0)	TP(1)  seed	criterion max_depth	
MODEL-01	100.00%	66.00%	76.00%	26.00%	111	entropy	default=None	
MODEL-02	100.00%	60.67%	74.00%	10.00%	123	entropy	default=None	
MODEL-03	100.00%	59.33%	71.00%	16.00%	111	gini	default=None	
MODEL-04	79.33%	79.33%	100.00%	0.00%	111	entropy	    1	
MODEL-05	79.33%	79.33%	100.00%	0.00%	111	entropy	    2	
MODEL-06	79.78%	76.67%	97.00%	0.00%	111	entropy	    3	
MODEL-07	80.67%	74.00%	92.00%	3.00%	111	entropy	    4	
MODEL-08	81.11%	76.00%	94.00%	6.00%	111	entropy	    5	
MODEL-09	81.56%	74.00%	92.00%	6.00%	111	entropy	   6,7
MODEL-10	82.22%	74.00%	92.00%	6.00%	111	entropy	    8	
MODEL-11	82.89%	71.33%	87.00%	13.00%	111	entropy	  15,20	

Note: more details in excel sheet

# we find that the best training and testing accuracy is obtained when we
# prune trees to 15 or 20 depth. Overall training accuracy is 83%.
Overall testing accuracy is 71%, but TP rate has increased to 13% .
'''

############### RANDOM FOREST CLASSIFIER

from sklearn.ensemble import  RandomForestClassifier
help(RandomForestClassifier)

model_rf = RandomForestClassifier(n_jobs=4, oob_score=True, n_estimators=500, criterion='entropy',random_state=100)
# model_rf = RandomForestClassifier(n_jobs=4, oob_score=True, n_estimators=500, criterion='entropy',random_state=100,max_features=3)
model_rf.fit(train[col[0:5]], train.Tax_group)

model_rf.oob_score_  # 74.67%

pred_train = model_rf.predict(train[col[0:5]])
pd.crosstab(train.Tax_group, pred_train)
metrics.accuracy_score(train.Tax_group, pred_train)
print(metrics.classification_report(train.Tax_group, pred_train))

pred_test = model_rf.predict(test[col[0:5]])
pd.crosstab(test.Tax_group, pred_test)
metrics.accuracy_score(test.Tax_group, pred_test)
print(metrics.classification_report(test.Tax_group, pred_test))

'''
When we use n_estimators=100 then the results:-
The overall training accuracy is 100%. TN rate is 100% and TP rate is 100%
The overall testing accuracy is 72.67%. TN rate is 91% and TP rate is 3%

When we use n_estimators=500, seed=100 then the results:-
The overall training accuracy is 100%. TN rate is 100% and TP rate is 100%
The overall testing accuracy is 71.33%. TN rate is 89% and TP rate is 3%

When we use criterion='gini' , n_estimators=500, seed=100 then the results:-
The overall training accuracy is 100%. TN rate is 100% and TP rate is 100%
The overall testing accuracy is 72%. TN rate is 89% and TP rate is 6%

When we change max_depth, max_features, random_state, we get different results,
but all have lower TP rate

##########################

CONCLUSIONS:

We have used Decision tree and Random forest classifier to predict Tax-payers
as Good or Risky. 

we have used Gini index and entropy criterion to choose nodes for branching.

We have pruned the trees, made changes to the functions in order to improve
model performance.

Accuracy of correct predictions is very low for this dataset even when we have
employed various  combinations. Even when we used different 
machine learning alogorithms, differen app (R-Studio) the accuracy was very low.

To increase the prediction accuracy, we have to add more relevant features, 
add more samples to this dataset. 

'''
