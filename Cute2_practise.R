# Clearing the R environment
rm(list=ls(all=T))
setwd("C:\\Users\\Sai\\Desktop\\Insofe\\exams\\Cute2\\Batch34_CSE7302c_CUTe\\Batch34_CSE7302c_CUTe")

#loading the required libraries
library(dplyr)
library(DMwR)
library(car)
library(MASS)
library(vegan)
library(dummies)
library(infotheo)
library(caTools)
library(caret)
library(glmnet)
library(ROCR)

#Reading and analysing the data
data <- read.csv("Census_Dataset.csv",header = T,sep = ",")
dim(data)
head(data)
data=data[,-c(4)]
str(data)
data$education_num=as.factor(data$education_num)
str(data)
summary(data)
sum(is.na(data))

#Data Splitting
set.seed(786)
split=0.70
trainIndex <- createDataPartition(data$income, p=split, list=F)
data_train <- data[ trainIndex,]
data_test <- data[-trainIndex,]
dim(data_train)
dim(data_test)

# Running Logistic Regression on Train Data
log_reg=glm(income ~ ., data = data_train, family = "binomial")
summary(log_reg)
#AUC is 91%

#But there are so many insignificant coefficients shown in the summary of regression
#lets remove some coefficients in this model
#Step AIC takes lots of effort and time beacause of huge data
#country and education_num looks insignificant, so remove and build a model

log_reg1 <- glm(income ~ .-education_num-country, data = data_train, family = "binomial")
summary(log_reg1)

# ROC & AUC
prob_train <- predict(log_reg1, type = "response")

pred <- prediction(prob_train, data_train$income) 

perf <- performance(pred, measure="tpr", x.measure="fpr")

plot(perf, col=rainbow(10), colorize=T, print.cutoffs.at=seq(0,1,0.05))

perf_auc <- performance(pred, measure="auc")

auc <- perf_auc@y.values[[1]]

print(auc)
#AUC is 89%

cutoffs <- data.frame(cut= perf@alpha.values[[1]], fpr= perf@x.values[[1]], 
                      tpr=perf@y.values[[1]]) 

cutoffs <- cutoffs[order(cutoffs$tpr, decreasing=TRUE),]


plot(perf, colorize = TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7))

View(cutoffs)

#After taking Cutoff as 0.05
pred_class <- ifelse(prob_train> 0.274, ">50K", "<=50K")
table(data_train$income,pred_class)


prob_test <- predict(log_reg1, data_test, type = "response")
preds_test <- ifelse(prob_test > 0.274, ">50K","<=50K")


test_data_labs <- data_test$income
conf_matrix <- table(test_data_labs, preds_test)
print(conf_matrix)
confusionMatrix(preds_test, data_test$income, positive = '<=50K')


#lets try to remove few more insignificant coefficients by regularization
#Regularization

#data
data1=centralImputation(data)
set.seed(786)
split=0.70
trainIndex <- createDataPartition(data1$income, p=split, list=F)
train_data2 <- data[ trainIndex,]
test_data2 <- data[-trainIndex,]
# PreProcess the data to standadize the numeric attributes
preProc<-preProcess(train_data2[,setdiff(names(train_data2),"income")],method = c("center", "scale"))
Rtrain1<-predict(preProc,train_data2)
Rtest1<-predict(preProc,test_data2)

###create dummies for factor varibales using a new function called "dummyVars"

dummies1 = dummyVars(as.factor(income)~., data = Rtrain1)

x.train=predict(dummies1, newdata = Rtrain1)
y.train=Rtrain1$income
x.test = predict(dummies, newdata = Rtest1)
y.test = Rtest1$income



### Elastic Regression
# 1. Let us build a simple  ridge regression
# 2. Lets do a cross validation with  ridge regression
# 3. Also Will tune the Model for perfect "lamda value"

fit.lasso <- glmnet(x.train, y.train, family="gaussian", alpha=0.5)

fit.lasso.cv <- cv.glmnet(x.train, y.train, type.measure="mse", alpha=1, 
                          family="gaussian",nfolds=10,parallel=TRUE)

plot(fit.lasso, xvar="lambda")
plot(fit.lasso.cv)

coef(fit.lasso.cv,s = fit.lasso.cv$lambda.min)

pred.lasso.cv.train <- predict(fit.lasso.cv,x.train,s = fit.lasso.cv$lambda.min)
pred.lasso.cv.test <- predict(fit.lasso.cv,x.test,s = fit.lasso.cv$lambda.min)

regr.eval(y.train,pred.lasso.cv.train)
regr.eval(y.test,pred.lasso.cv.test)
