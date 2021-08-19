# 2
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Importing the dataset
Loan = pd.read_csv("C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\Multinomial Regression\\loan.csv")
Loan.columns
Loan.describe()
Loan.loan_status.value_counts()
Loan.info()
Loan.isna().sum()
# there are many Na values in our dataset we are selecting the columns which influence our output the most
# Data-Pre_processing 
Loan1 = Loan.dropna(axis=1) 
Loan1 = Loan1.drop(columns=['id', 'member_id',"url"])
Loan1 = Loan1.drop(columns=["zip_code"])
Loan1 = Loan1.drop(columns=["earliest_cr_line"])
# Label Encoding
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
Loan1["term"] = LE.fit_transform(Loan1["term"])
Loan1["int_rate"] = LE.fit_transform(Loan1["int_rate"])
Loan1["grade"] = LE.fit_transform(Loan1["grade"])
Loan1["sub_grade"] = LE.fit_transform(Loan1["sub_grade"])
Loan1["annual_inc"] = LE.fit_transform(Loan1["annual_inc"])
Loan1["home_ownership"] = LE.fit_transform(Loan1["home_ownership"])
Loan1["verification_status"] = LE.fit_transform(Loan1["verification_status"])
Loan1["home_ownership"] = LE.fit_transform(Loan1["home_ownership"])
Loan1["loan_status"] = LE.fit_transform(Loan1["loan_status"])
Loan1["pymnt_plan"] = LE.fit_transform(Loan1["pymnt_plan"])
Loan1["purpose"] = LE.fit_transform(Loan1["purpose"])
Loan1["addr_state"] = LE.fit_transform(Loan1["addr_state"])
Loan1["application_type"] = LE.fit_transform(Loan1["application_type"])
Loan1["initial_list_status"] = LE.fit_transform(Loan1["initial_list_status"])
Loan1["issue_d"] = LE.fit_transform(Loan1["issue_d"])
Loan1.info()
Loan1.columns
# Boxplot of independent variable distribution for each category of choice 
sns.boxplot(x = "loan_status", y = "loan_amnt", data = Loan1)
sns.boxplot(x = "loan_status", y = "funded_amnt", data = Loan1)
sns.boxplot(x = "loan_status", y = "funded_amnt_inv", data = Loan1)
sns.boxplot(x = "loan_status", y = "installment", data = Loan1)
sns.boxplot(x = "loan_status", y = "annual_inc", data = Loan1)
sns.boxplot(x = "loan_status", y = "total_pymnt", data = Loan1)
sns.boxplot(x = "loan_status", y = "total_pymnt_inv", data = Loan1)
sns.boxplot(x = "loan_status", y = "total_rec_prncp", data = Loan1)
sns.boxplot(x = "loan_status", y = "total_rec_int", data = Loan1)
sns.boxplot(x = "loan_status", y = "last_pymnt_amnt", data = Loan1)
# Scatter plot for each categorical choice of car
sns.stripplot(x = "loan_status", y = "loan_amnt", jitter = True, data = Loan1)
sns.stripplot(x = "loan_status", y = "funded_amnt", jitter = True, data = Loan1)
sns.stripplot(x = "loan_status", y = "funded_amnt_inv", jitter = True, data = Loan1)
sns.stripplot(x = "loan_status", y = "installment", jitter = True, data = Loan1)
sns.stripplot(x = "loan_status", y = "annual_inc", jitter = True, data = Loan1)
sns.stripplot(x = "loan_status", y = "total_pymnt", jitter = True, data = Loan1)
sns.stripplot(x = "loan_status", y = "total_pymnt_inv", jitter = True, data = Loan1)
sns.stripplot(x = "loan_status", y = "total_rec_prncp", jitter = True, data = Loan1)
sns.stripplot(x = "loan_status", y = "total_rec_int", jitter = True, data = Loan1)
sns.stripplot(x = "loan_status", y = "last_pymnt_amnt", jitter = True, data = Loan1)
# Correlation values between each independent features
Loan1.corr()
# Assigning "X" inputs and "y" output from the dataset
X=Loan1.drop(columns=['loan_status'],axis=1)
y=Loan1.loc[:,['loan_status']]
# Splitting the data into Train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.30, random_state=42)
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
clf = LogisticRegressionCV(multi_class='multinomial',solver='newton-cg')
clf_1 = MultinomialNB()
model = clf.fit(X_train,y_train)
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# Test data Prediction
y_pred = model.predict(X_test)
acc_test_pred = accuracy_score(y_test,y_pred)
acc_test_pred 
conf_mat_test = confusion_matrix(y_test,y_pred)
conf_mat_test
# Training data Prediction
x_pred = model.predict(X_train)
accu_train_pred = accuracy_score(y_train,x_pred)
accu_train_pred 
conf_mat_train = confusion_matrix(y_train,x_pred)
conf_mat_train
# We are having right fight here, there is no huge difference between test and train data.
