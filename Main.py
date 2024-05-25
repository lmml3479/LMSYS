import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
#Splitting data into training and testing sets as a means to measure overfitting (or ideally, a lack of)
from sklearn.preprocessing import StandardScalar, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
#Creates multiple decision trees + combines their output for final decision
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#Ways of meausring model performance
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
print(train_data.head())
#Printing the first five lines of the data shows that it consists of information of survival, sex, ticket, age, cabin, fare, etc.
