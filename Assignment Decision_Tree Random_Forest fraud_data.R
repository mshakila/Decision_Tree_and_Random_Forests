########## ASSIGNMENT DECISION TREE and RANDOM FOREST using FRAUD dataset

# PROBLEM STATEMENT : Developing a model using DECISION TREE for classifying
# individuals as 'risky' or 'good'

'''
Data Description :
  
Undergrad : person is under graduated or not
Marital.Status : marital status of a person
Taxable.Income : amount an individual owes to the government 
Work Experience : Work experience of an individual person
Urban : Whether that person belongs to urban area or not
'''

library(C50)
library(caTools)
library(SamplingStrata) # to convert continuous vars to bins

fraud_raw <- read.csv("E:\\Decision Trees\\Fraud_check.csv")
names(fraud_raw)
# "Undergrad"       "Marital.Status"  "Taxable.Income"  "City.Population"
# "Work.Experience" "Urban" 

head(fraud_raw)
tail(fraud_raw)
str(fraud_raw)
summary(fraud_raw)

''' 
variables Undergrad, Marital.Status and Urban are categorical variables
variables Taxable.Income, City.Population and Work.Experience are continuous ones

We need to convert Taxable.Income into 2 classes: risky and good, this will be
our target variable. 
'''
fraud <- fraud_raw
fraud['Tax_group'] <- ifelse(fraud$Taxable.Income <=30000, 'Risky', 'Good')

fraud <- fraud[,-3]
fraud$Tax_group <- as.factor(fraud$Tax_group)
str(fraud)

head(fraud)

table(fraud$Tax_group)
prop.table(table(fraud$Tax_group))
#      Good     Risky 
#  0.7933333 0.2066667 

barplot(table(fraud$Tax_group),main="Barplot of Risky and Good Tax payers",
        col='blue', border = 'red', density = 50)

table(fraud$Tax_group,fraud$Undergrad)

barplot(table(fraud$Tax_group, fraud$Undergrad),main="Graduation of Tax group",
        col=c("red",'green'), beside = TRUE)
legend("topright", c('Good','Risky'),fill = c('red','green'))
# the level of risk of a taxpayer is the same whether he is graduated or not


#################### splitting the dataset into train and test 
# also maintain the prop of risky and good types

library(caTools)
set.seed(seed = NULL)

fraud$spl <- sample.split(fraud$Tax_group, SplitRatio = 0.7)
head(fraud,3)
train1 <- subset(fraud, fraud$spl==TRUE, select=-(spl)) # 420 obs
test1 <- subset(fraud, fraud$spl==FALSE, select = -(spl)) # 180 obs

prop.table(table(train1$Tax_group))
prop.table(table(test1$Tax_group))
# the proportion of risky and good tax-payers is same in original,train and test data

############ Model building on training data
names(train1)
model1 <- C5.0(train1[,-6], train1$Tax_group, trails=40)
model1 <- C5.0(train1[,c(1,5)], train1$Tax_group,trails=40)
summary(model1)
plot(model1)

pred_train1 <- predict(model1, train1[,-6])
table(pred_train1, train1$Tax_group)
# pred_train1 Good Risky
#      Good   333    87
#      Risky    0     0

mean(pred_train1 == train1$Tax_group) # 0.7928571

pred_test1 <- predict(model1, newdata =test1[,-6])

table(pred_test1, test1$Tax_group)
mean(pred_test1 == test1$Tax_group) # 0.7944444

# all good tax-payers can be correctly predicted but then risky tax-players
# have 0% correct prediction

################################################################################

################ Data Manipulation for Building MODEL2
# let us convert all continuous var into categorical vars

library(SamplingStrata)

fraud$City.Population <- var.bin(fraud$City.Population, bins=5)
fraud$Work.Experience <- var.bin(fraud$Work.Experience, bins=5)

fraud$City.Population <- as.factor(fraud$City.Population)
fraud$Work.Experience <- as.factor(fraud$Work.Experience)
str(fraud)
summary(fraud)
# we have almost similar number of observations in all 5 bins of both
# variables City.Population, Work.Experience

# data splitting
fraud$spl <- sample.split(fraud$Tax_group, SplitRatio = 0.7)
train2 <- subset(fraud, fraud$spl==TRUE, select=-(7)) # 420 obs
test2 <- subset(fraud, fraud$spl==FALSE, select = -(7)) # 180 obs

#### Model building on training data
names(train2)
model2 <- C5.0(train2[,-6], train2$Tax_group, trails=40)

summary(model2)
plot(model2)

## Prediction on train and test data
pred_train2 <- predict(model2, train2[,-6])
mean(pred_train2 == train2$Tax_group) # 0.7928571
table(pred_train2, train2$Tax_group)
# Total accuracy is 79% (0.7928571), but all records are predicted as 'Good' .
# Risky class correct prediction is 0% and Good class 100% correctly predicted

pred_test2 <- predict(model2, test2[,-6])
mean(pred_test2 == test2$Tax_group) # 0.7944444
table(pred_test2, test2$Tax_group)
# even here the error rate of Risky class is 100%

###########################################################################
###################### MPDEL3 - let us drop the 2 continuous var
# data splitting
head(fraud)
fraud$spl <- sample.split(fraud$Tax_group, SplitRatio = 0.9)
train3 <- subset(fraud, fraud$spl==TRUE, select=-c(3,4,7))
test3 <- subset(fraud, fraud$spl==FALSE, select=-c(3,4,7))
head(train3)
head(test3)

model3 <- C5.0(train3[,-4], train3[,4], trails=20)
model3
summary(model3)
plot(model3)
?C5.0
pred_train3 <- predict(model3, train3[,-4])
table(pred_train3, train3$Tax_group)
# here also all records are predicted as Good.

############### Modelling using Naive BAyes
library(e1071)
head(train1)
model_nb <- naiveBayes(train1$Tax_group~., data=train1[,-6], laplace=1) 
summary(model_nb)
pred_train_nb <- predict(model_nb, train1[,-6])
table(pred_train_nb, train1$Tax_group)
mean(pred_train_nb == train1$Tax_group) #0.7928571
# same results as model1, all are predicted as Good

###########################################################################


#################### Decision Tree with Taxable.Income as continuous variable
library(rpart)
head(fraud_raw)
cor(fraud_raw[,c(3,4,5)])  
# cor is very low btw target and predictors, still let us run the model

boxplot(fraud_raw)
summary(fraud_raw)
str(fraud_raw)

# Building a regression tree using rpart 
# Simple model
model_cart <- rpart(fraud_raw$Taxable.Income~.,data=fraud_raw,method="anova")
plot(model_cart)
text(model_cart)
summary(model_cart)

pred_cart <- predict(model_cart,fraud_raw)
rmse_fraud <- sqrt(mean((pred_cart-fraud_raw$Taxable.Income)^2))
rmse_fraud # 25466.8

Adjusted_RSqred <- function(pred, obs, formula = "corr", na.rm = FALSE) {
  n <- sum(complete.cases(pred))
  switch(formula,
         corr = cor(obs, pred, use = ifelse(na.rm, "complete.obs", "everything"))^2,
         traditional = 1 - (sum((obs-pred)^2, na.rm = na.rm)/((n-1)*var(obs, na.rm = na.rm))))
}

Adjusted_RSqred(pred_cart,fraud_raw$Taxable.Income) # 0.05395793, since low, a poor model

plot(pred_cart,fraud_raw$Taxable.Income)
cor(pred_cart,fraud_raw$Taxable.Income) # 0.2322, since low not a good model


##############################################################################

############# Modelling using Logistic Regression
library()
model_logreg <- glm(train1$Tax_group~., data=train1[,-6], family="binomial")
summary(model_logreg)

logreg_prob <- predict(model_logreg, train1[,-6], type='response')

confusion <- table(logreg_prob>0.5, train1$Tax_group,dnn = c('predicted','actual'))
confusion

sum(diag(confusion)/sum(confusion)) # 0.7928571

# # same results as model1, all are predicted as Good

#########################################################################

###################### Building a random forest model 
library(randomForest)
set.seed(123)
fit.forest <- randomForest(Tax_group~.,data=train1, na.action=na.roughfix,
                           ntree=500, importance=TRUE)
fit.forest
'''
when ntree=500 and set.seed(123)
Confusion matrix:
      Good Risky class.error
Good   326     7  0.02102102
Risky   86     1  0.98850575

when ntree=1000 and set.seed(123)
      Good Risky class.error
Good   327     6  0.01801802
Risky   86     1  0.98850575   '''

varImpPlot(fit.forest)

pred_train_RF <- predict(fit.forest, train1[,-6])
table(pred_train_RF, train1[,6])
mean(pred_train_RF == train1[,6]) 
library(caret)
confusionMatrix(pred_train_RF, train1[,6])
''' 
pred_train_RF Good Risky
        Good   333    37
        Risky    0    50 
overall training accuracy is  0.9119048 . But if we look at individual class
accuracy, we find that Good-Tax-payers have 100% correct prediction and
Risly-tax payers are correctly predicted only 57% of the times. '''

pred_test_RF <-  predict(fit.forest, test1[,-6])
confusionMatrix(pred_test_RF, test1[,6])
'''
pred_test_RF Good Risky
       Good   141    36
       Risky    2     1
Overall testing accuracy is 78%. Good-Tax-payers accuracy is 98% and
Risky-tax-payers accuracy is only 2% . 
'''

#############################################################################

################# Finding proportion of variable classes w.r.t Tax-group
names(fraud)

table(fraud$Tax_group,fraud$Undergrad)
prop.table(table(fraud$Tax_group,fraud$Undergrad))
'''
                NO        YES
  Good  0.38333333 0.41000000
  Risky 0.09666667 0.11000000
'''

table(fraud$Tax_group,fraud$Marital.Status)
prop.table(table(fraud$Tax_group,fraud$Marital.Status))
'''
          Divorced    Married     Single
  Good  0.25500000 0.24833333 0.29000000
  Risky 0.06000000 0.07500000 0.07166667
'''

table(fraud$Tax_group,fraud$Urban)
prop.table(table(fraud$Tax_group,fraud$Urban))
'''
               NO       YES
  Good  0.3950000 0.3983333
  Risky 0.1016667 0.1050000
'''
'''
if we look at the proportion of each class w.r.t Tax-groups, we find similar
distribution. 

First, let us look at Urban:
Under Good tax-payers, 40% are non-urban and 40% stay in Urban locations
Under Risky tax-payers, 10% stay in non-Urban places and 10% stay in Urban places


First, let us look at Under-graduates:
Under Good tax-payers, 38% are graduates and 41% are not graduates
Under Risky tax-payers, 10% are graduates and 11% are not graduates

First, let us look at Marital-status:
Under Good tax-payers, divorced and married are each 25% and single are 29%
Under Risky tax-payers, each class is around 7% of the population

Since each class of predictor (category) variables has almost equal distribution w.r.t
tax-payer groups, the machine learning algorithms have not been successful
in predicting the right Tax-group
'''

###########################################################################
'''
CONCLUSIONS

The business problem is to predict Good Tax-payers from Risky ones. We
have used fraud dataset. Decision Tree classifier is used here as target
is a categorical variable. 

We have done data manipulation. First converted continuous variable Taxable-income
to categorical variable Tax-group ( with 2 classes, Good and Risky) and have run
the model. We have also converted predictors City-population and Work-experience 
to categorical variables ( 5 bins) and ran another model. Next we dropped these
2 continuous variables and ran the model.

We have also used Decision Tree Regression to predict continuous variable Taxable_income
We have also used Niave Bayes and Logistic Regression models.

The predictions have been one-sided. For all the models (except Random Forest), 
all records were predicted as belonging to "Good" class. Though the data had about
70% Good and about 30% Risky Tax-payers, predictions showed all as "Good" only.

Random Forest model had better training accuracy, but testing accuracy is very low.

Each class of predictor (category) variables has almost equal distribution w.r.t
tax-payer groups. This might be one of the reasons that the machine learning 
algorithms have not been successful in predicting the right Tax-group.

Since all models had similar results, the problem might lie with the dataset 
itself. We have to increase the records and bring in more relevant variables
to improve model performance

'''


