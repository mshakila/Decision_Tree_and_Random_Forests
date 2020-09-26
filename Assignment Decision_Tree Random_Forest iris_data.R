########## ASSIGNMENT DECISION TREE and RANDOM FOREST using IRIS dataset

# importing libraries
library(C50)

#### loading the data
data("iris")

##### inspecting the data type, levels,
head(iris)
tail(iris)

names(iris)
str(iris)

summary(iris)

table(iris$Species)
prop.table(table(iris$Species))
#     setosa versicolor  virginica 
#       50         50         50    # number of records in each level
#   0.3333333  0.3333333  0.3333333 # proportion of records in each level

#### Splitting the dataset 
set.seed(123)
library(caTools)

iris$spl <- sample.split(iris$Species, SplitRatio = 0.7)
head(iris,3)
train1 <- subset(iris, iris$spl==TRUE, select = -(spl))
test1 <- subset(iris, iris$spl==FALSE, select = -(spl))
head(test1)

prop.table(table(train1$Species))
prop.table(table(test1$Species))
# the proportion of Species in each level is same for original, train and test data

####### building model on training data
model1 <- C5.0(train1[,-5], train1$Species)
# model2 <- C5.0(train1[,-5], train1$Species, trails=100) # gives same result

plot(model1)

summary(model1)
'''
from the plot we can say that,
When petal length is <=1.9, then the iris plant belongs to setosa species
when petal length is >1.9 and sepal length is <=4.7, then it belongs to versicolor
'''

# training accuracy
pred_train1 <- predict(model1, train1[,-5])
table(pred_train1, train1$Species)
mean(pred_train1 == train1$Species) # 0.9809524

# library(caret)
# confusionMatrix(pred_train1, train$Species)

# testing accuracy
pred_test1 <- predict(model1, newdata =test1[,-5])

library(gmodels)
CrossTable(pred_test1, test1$Species,prop.c=FALSE, prop.t=FALSE,prop.chisq=FALSE)

table(pred_test1, test1$Species)
mean(pred_test1 == test1$Species) # 0.8888889

############ MODEL 2
model2 <- C5.0(train1$Species ~. , data = train1, trails=100)
plot(model2)
summary(model2)
pred_train2 <- predict(model2, train1[,-5])
mean(pred_train2 == train1$Species) # 0.9809524
pred_test2 <- predict(model2, newdata = test1[,-5])
mean(pred_test2 == test1$Species) # 0.8888889

# the output and accuracy are same as model1

##############    Bagging technique
set.seed(seed = NULL)
acc<-c()
for(i in 1:100)
{
  print(i)
  iris$spl <- sample.split(iris$Species, SplitRatio = 0.7)
  train2 <- subset(iris, iris$spl==TRUE, select = -(spl))
  test2 <- subset(iris, iris$spl==TRUE, select = -(spl))
  
  fittree<-C5.0(train2$Species~.,data=train2)
  pred<-predict.C5.0(fittree,test2[,-5])
  a<-table(test2$Species,pred)
  
  acc<-c(acc,sum(diag(a))/sum(a))
  
}
acc
summary(acc)

# this is an ensemble technique for predictions, we have created 100 models
# here test accuracy is ranging from 95.24% to 100%. The mean test 
# accuracy is 0.9785

################### Model3
# let us use a different training and testing sample by changing the seed
set.seed(111)
iris$spl <- sample.split(iris$Species, SplitRatio = 0.7)
train3 <- subset(iris, iris$spl==TRUE, select=-(spl))
test3 <- subset(iris, iris$spl==FALSE, select = -(spl))

model3 <- C5.0(train3$Species ~., data=train3, trails=40)
summary(model3)

plot(model3)
'''
from the plot we can say that,
When petal length is <=1.9, then the iris plant belongs to setosa species
When petal width is >1.6, then the iris plant belongs to virginica species
when sepal length is <=4.9, then it belongs to versicolor
'''

pred_train3 <- predict(model3, train3[,-5])
table(pred_train3 , train3$Species)
mean(pred_train3 == train3$Species) # 0.9904762 is training accuracy

pred_test3 <- predict(model3, test3[,-5])
table(pred_test3 , test3$Species)
mean(pred_test3 == test3$Species) # 0.9555556 is testing accuracy

##########################################################################

################################ Using tree function (old one)
library(tree)
# Building a model on training data 
model4 <- tree(Species~.,data=train1)
windows()
plot(model4)
text(model4,pretty = 0)
summary(model4)

pred_train4 <- as.data.frame(predict(model4, train1[,-5]))
pred_train4['final'] <- 'NULL'
pred_train4_df <- predict(model4, newdata = train1[,-5])

pred_train4$final <- colnames(pred_train4_df)[apply(pred_train4_df,1,which.max)]

table(pred_train4$final , train1$Species)
mean(pred_train4$final == train1$Species) # 0.9809524 is training accuracy

pred_test4 <- as.data.frame(predict(model4, test1[,-5]))
pred_test4['final'] <- 'NULL'
pred_test4_df = predict(model4, newdata = test1[,-5])
pred_test4$final <- colnames(pred_test4_df)[apply(pred_test4_df,1,which.max)]

table(pred_test4$final, test1$Species)
mean(pred_test4$final == test1$Species) # 0.8888889

##########################################################################

########################### Using Random Forest

library(randomForest)
names(iris)
set.seed(111)

model_rf <- randomForest(Species ~., data=train1, na.action=na.roughfix,
                            ntree=1000, importance=TRUE)
model_rf

varImpPlot(model_rf)
'''
petal.width here is the top most variable both under Mean-decrease=accuracy
and mean-decrease-gini. This indicates that this variable gives the best 
prediction and conributes most to the model. '''

pred_train_rf <- predict(model_rf, train1[,-5])
table(pred_train_rf, train1[,5])
mean(pred_train_rf == train1[,5]) # 100%

pred_test_rf <- predict(model_rf, test1[,-5])
table(pred_test_rf, test1[,5])
mean(pred_test_rf == test1[,5]) # 0.9111
#######################################################################

########################### MODEL COMPARISON
'''
MODEL TRAIN-ACC  TEST-ACC 	
model1	 0.9809	    0.8888	set.seed(123)
model2	 0.9809	    0.8888	including trails=1000
Bagging		          0.9785	mean of 100 models
model3	 0.99047    0.9555	set.seed(111)
model4	 0.98095    0.8888 	tree function
model5	 1.00000    0.9111	Random forest '''

#############################################################################

'''
CONCLUSIONS

Decision tree is a graph to represent choices and their results in form of a tree.
R has packages to create as well as visualize the decision tree.

It can be used for both classifying and regressing problems.
We have taken the popular iris dataset and used Decision tree classifier to
predict the Species. We have used C5.0 and tree functions to classify. Also
we have used different train-test samples to predict.

We have also used Random Forest algorithm.

The testing accuracy was improved by using Bagging and Random-Forest models
'''