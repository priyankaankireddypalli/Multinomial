# 1
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Importing the dataset
students = pd.read_csv("C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\Multinomial Regression\\mdata.csv")
students.head(10)
students.columns
students.describe()
students.prog.value_counts()
students.info()
# We are dropping id and index column
students.drop(['Unnamed: 0','id'],axis = 1,inplace=True)
# Boxplot of independent variable distribution for each category of choice 
sns.boxplot(x = "prog", y = "read", data = students)
sns.boxplot(x = "prog", y = "write", data = students)
sns.boxplot(x = "prog", y = "math", data = students)
sns.boxplot(x = "prog", y = "science", data = students)
# Scatter plot for each categorical choice of car
sns.stripplot(x = "prog", y = "read", jitter = True, data = students)
sns.stripplot(x = "prog", y = "write", jitter = True, data = students)
sns.stripplot(x = "prog", y = "math", jitter = True, data = students)
sns.stripplot(x = "prog", y = "science", jitter = True, data = students)
# Scatter plot between each possible pair of independent variable and also histogram for each independent variable 
sns.pairplot(students) # Normal
sns.pairplot(students, hue = "prog") # With showing the category of each car choice in the scatter plot
# Correlation values between each independent features
students.corr()
# we are creating dummy variable creation for all the object data except the output data
studentsD=pd.get_dummies(students,columns=['female','ses','schtyp','honors'],drop_first=True)
train, test = train_test_split(studentsD, test_size = 0.2)
# ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers
model = LogisticRegression(multi_class = "multinomial", solver = "newton-cg").fit(train.iloc[:, 1:], train.iloc[:, 0])
test_predict = model.predict(test.iloc[:, 1:]) # Test predictions
# Test accuracy 
accuracy_score(test.iloc[:,0], test_predict)
train_predict = model.predict(train.iloc[:, 1:]) # Train predictions 
# Train accuracy 
accuracy_score(train.iloc[:,0], train_predict) 

# We are having right fight here, there is no huge difference between test and train data.

