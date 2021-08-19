# 1
library(readr)
# Importing the dataset
students <- read.csv("C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\Multinomial Regression\\mdata.csv")
summary(students)
table(students$prog) 
# we are dropping id and index column
students<-students[c(-1,-2)]
# Data Partitioning
n <-  nrow(students)
n1 <-  n * 0.8
n2 <-  n - n1
train_index <- sample(1:n, n1)
train <- students[train_index, ]
test <-  students[-train_index, ]
colnames(students)
library(nnet)
commute <- multinom(prog ~  female + ses + schtyp + read + write + math+ science + honors, data = train)
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
pred_test <- data.frame(pred_test)
View(pred_test)
pred_test["prediction"] <- NULL
# Custom function that returns the predicted value based on probability
get_names <- function(i){
  return (names(which.max(i)))
}
predtest_name <- apply(pred_test, 1, get_names)
pred_test$prediction <- predtest_name
View(pred_test)
# Confusion matrix
table(predtest_name, test$prog)
# confusion matrix visualization
barplot(table(predtest_name, test$prog), beside = T, col =c("red", "lightgreen", "blue"), legend = c("academic", "general ", "vocation "), main = "Predicted(X-axis) - Legends(Actual)", ylab ="count")
barplot(table(predtest_name, test$prog), beside = T, col =c("red", "lightgreen", "blue"), main = "Predicted(X-axis) - Legends(Actual)", ylab ="count")
# Accuracy on test data
mean(predtest_name == test$prog)
# Training Data
pred_train <- predict(commute, newdata =  train, type="probs") # type="probs" is to calculate probabilities
pred_train
# Find the accuracy of the model
class(pred_train)
pred_train <- data.frame(pred_train)
View(pred_train)
pred_train["prediction"] <- NULL
predtrain_name <- apply(pred_train, 1, get_names)
pred_train$prediction <- predtrain_name
View(pred_train)
# Confusion matrix
table(predtrain_name, train$prog)
# confusion matrix visualization
barplot(table(predtrain_name, train$prog), beside = T, col =c("red", "lightgreen", "blue"), legend = c("academic", "general ", "vocation "), main = "Predicted(X-axis) - Legends(Actual)", ylab ="count")
barplot(table(predtrain_name, train$prog), beside = T, col =c("red", "lightgreen", "blue"), main = "Predicted(X-axis) - Legends(Actual)", ylab ="count")
# Accuracy 
mean(predtrain_name == train$prog)

# We are having right fight here, there is no huge difference between test and train data.

