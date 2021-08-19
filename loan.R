# 2
library(readr)
# Importing the dataset
Loan <- read.csv("C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\Multinomial Regression\\loan.csv")
summary(Loan)
table(Loan$loan_status) 
sum(is.na(Loan))
# there are many Na values in our dataset we are selecting the columns which influence our output the most
colnames(Loan)
loan1<-Loan[,c("loan_amnt","funded_amnt","funded_amnt_inv","installment","home_ownership",
               "annual_inc","verification_status","loan_status","purpose","total_pymnt",
               "total_pymnt_inv","total_rec_prncp","total_rec_int","last_pymnt_amnt")]
summary(loan1)
loan1$home_ownership<-as.factor(loan1$home_ownership)
loan1$verification_status<-as.factor(loan1$verification_status)
loan1$purpose<-as.factor(loan1$purpose)
loan1$loan_status<-as.factor(loan1$loan_status)
attach(loan1)
unique(purpose)
# Data Partitioning
n <-  nrow(loan1)
n1 <-  n * 0.8
n2 <-  n - n1
train_index <- sample(1:n, n1)
train <- loan1[train_index, ]
test <-  loan1[-train_index, ]
colnames(loan1)
library(nnet)
commute <- multinom(loan_status ~., data = train)
summary(commute)
# Significance of Regression Coefficients
z <- summary(commute)$coefficients / summary(commute)$standard.errors
p_value <- (1 - pnorm(abs(z), 0, 1)) * 2
summary(commute)$coefficients
p_value
# odds ratio 
exp(coef(commute))
# check for fitted values on training data
prob <- fitted(commute)
# Predicted on test data
pred_test <- predict(commute, newdata =  test, type = "probs") # type="probs" is to calculate probabilities
pred_test
# Find the accuracy of the model
class(pred_test)
pred_test <- as.data.frame(pred_test)
View(pred_test)
pred_test["prediction"] <- NULL
# Custom function that returns the predicted value based on probability
get_names <- function(i){
  return (names(which.max(i)))
}
predtest_name <- as.data.frame(apply(pred_test,1 , get_names))
colnames(predtest_name)
names(predtest_name)[1] <- "predicted"
View(predtest_name)
str(predtest_name)
# Confusion matrix
table(predtest_name$predicted, test$loan_status)
# confusion matrix visualization
barplot(table(predtest_name$predicted, test$loan_status), beside = T, col =c("red", "lightgreen", "blue"), legend = c("Charged.Off", "Current", "Fully.Paid "), main = "Predicted(X-axis) - Legends(Actual)", ylab ="count")
barplot(table(predtest_name$predicted, test$loan_status), beside = T, col =c("red", "lightgreen", "blue"), main = "Predicted(X-axis) - Legends(Actual)", ylab ="count")
# we are converting the data from character to factor 
predtest_name$predicted <- as.factor(predtest_name$predicted)
str(predtest_name)
levels(predtest_name$predicted)
# Accuracy on test data
mean(predtest_name$predicted == test$loan_status)
# Training Data
pred_train <- predict(commute, newdata =  train, type="probs") # type="probs" is to calculate probabilities
pred_train
# Find the accuracy of the model
class(pred_train)
pred_train <- as.data.frame(pred_train)
View(pred_train)
pred_train["prediction"] <- NULL
predtrain_name <- as.data.frame(apply(pred_train, 1, get_names))
colnames(predtrain_name)
names(predtrain_name)[1] <- "predictedtrain"
View(pred_train)
# Confusion matrix
table(predtrain_name$predictedtrain, train$loan_status)
# confusion matrix visualization
barplot(table(predtrain_name$predictedtrain, train$loan_status), beside = T, col =c("red", "lightgreen", "blue"), legend = c("Charged.Off", "Current", "Fully.Paid "), main = "Predicted(X-axis) - Legends(Actual)", ylab ="count")
barplot(table(predtrain_name$predictedtrain, train$loan_status), beside = T, col =c("red", "lightgreen", "blue"), main = "Predicted(X-axis) - Legends(Actual)", ylab ="count")
# Accuracy 
mean(predtrain_name$predictedtrain == train$loan_status)

# We are having right fight here, there is no huge difference between test and train data.


