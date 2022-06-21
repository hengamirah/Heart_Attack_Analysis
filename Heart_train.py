# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 09:22:54 2022

1) Develop a model using only one machine learning approach (knn, random forest, regression, etc).
2) Prepare two scripts. One for model training, the other script will be used for model deployment and app development.
3) Perform necessary steps during EDA and justify the taken steps. You may write your justification as comments in your script.
4) Achieve model with validation accuracy of more than 70%.
5) Deploy the model and preform prediction on a web app developed using Streamlit
    
age - age in years
sex - sex (1 = male; 0 = female)
cp - chest pain type (1 = typical angina; 2 = atypical angina; 3 = non-anginal pain; 4 = asymptomatic)
trestbps - resting blood pressure (in mm Hg on admission to the hospital)
chol - serum cholestoral in mg/dlfbs - fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
restecg - resting electrocardiographic results (0 = normal; 1 = having ST-T; 2 = hypertrophy)
thalach - maximum heart rate achieved
exang - exercise induced angina (1 = yes; 0 = no)
oldpeak - ST depression induced by exercise relative to rest
slope - the slope of the peak exercise ST segment (1 = upsloping; 2 = flat; 3 = downsloping)
ca - number of major vessels (0-3) colored by flourosopy
thal - 3 = normal; 6 = fixed defect; 7 = reversable defectnum - 
the predicted attribute - diagnosis of heart disease (angiographic disease status) (Value 0 = < 50% diameter narrowing; Value 1 = > 50% diameter narrowing)

    
@author: Amirah Heng
"""

def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import scipy.stats as ss
import numpy as np
import pandas as pd
import seaborn as sns
import pickle 
import os

#EDA
#%% Statics
BEST_MODEL_PATH = os.path.join(os.getcwd(),'model','best_model.pkl')
BEST_PIPE_PATH = os.path.join(os.getcwd(), 'model','best_pipe.pkl') 
CSV_PATH = os.path.join(os.getcwd(),'dataset','heart.csv')
#%% Step 1) Data Loading
df = pd.read_csv(CSV_PATH)

#%% Step 2) Data Inspection
df.head()
df.info() #check for NaN values
df.describe().T

plt.figure(figsize=(10,6))
df.boxplot()
plt.show()

column_names =df.columns
#plot continuous  data
cont_column = ['age','trtbps','chol','thalachh','oldpeak']
for con in cont_column:
    plt.figure()
    sns.distplot(df[con])
    plt.show()

#plot categorical  data
cat_column = ['sex','cp','fbs','restecg','exng','slp','caa','thall','output']
for cat in  cat_column:
    plt.figure()
    sns.countplot(df[cat])
    plt.show()

df.duplicated().sum() #Check for duplicated data #sum=1
df[df.duplicated()] #sex is 1, so can be remove to make data distribution even

df.isna().any() #No Nan Values
#Check for 0 value in 'thall = Null'
df[df['thall'] == 0]  #there are two data here

#%% Step 3) Data Cleaning
#remove duplicate
df = df.drop_duplicates()
df.duplicated().sum() #no duplicated data anymore

#replace 0 with NaN
df["thall"] = df["thall"].replace(0, np.nan) 
df['thall'].isna().sum() #2NaN values

# Data Imputation with Iterative Imputer
iterative_imputer = IterativeImputer()
imputed_data = iterative_imputer.fit_transform(df)
df = pd.DataFrame(imputed_data)
df.columns=column_names
df.info()

df.iloc[:,-2]= np.floor(df.iloc[:,-2]).astype('int') #roundof data to integer value
df.columns=column_names #to copy column names from df to dfII

df['thall'].isna().sum() #no more NaN values
print(df.describe().T)

#%% Step 4) Features Selection

#target is output
target= df['output']

# for continuous vs categorical data use Logistic Regression
for con in cont_column:
    logreg = LogisticRegression()
    logreg.fit(np.expand_dims(df[con],axis=-1),target)
    print(con,':',logreg.score(np.expand_dims(df[con],axis=-1),target))
# age : 0.6192052980132451
# trtbps : 0.5794701986754967
# chol : 0.5331125827814569
# thalachh : 0.7019867549668874
# oldpeak : 0.6854304635761589

#for categorical vs categorical data use Cramer's V Corrected stat
for cat in cat_column:
    confussion_mat = pd.crosstab(df[cat], target).to_numpy()   
    print(cat,':',cramers_corrected_stat(confussion_mat))
 # sex : 0.2708470833804965
 # cp : 0.508955204032273
 # fbs : 0.0
 # restecg : 0.1601818218998346
 # exng : 0.42533348943620414
 # slp : 0.38615287747239485
 # caa : 0.48113005767539546
 # thall : 0.5206731262866439
 # output : 0.9933057162691583   

#Age, trtbps, thalach,chol and oldpeak,cp, thall has the highest correlation >0.5 
#%% Step 5) Data Pre-Processing
X = df.loc[:,['age','trtbps','chol','thalachh','oldpeak','cp','thall']]
y= df['output']
#Split dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split( X, y, 
                                                    test_size=0.3, 
                                                    random_state=123)

#%% Step 6) Model Development
# 1) Determine whether MMS or SS is better in this case
# 2) Determine which classifier works the best in this case
#     a) Random Forest
#     b) Decision Tree
#     c) Logistic regression
#     d) KNN
#     e) SVC
#Pipeline

# a) Random Forest
step_mms_rf = Pipeline([('MinMaxScaler', MinMaxScaler()),
            ('RFClassifier', RandomForestClassifier())])
step_ss_rf = Pipeline([('StandardScaler', StandardScaler()),
            ('RFClassifier', RandomForestClassifier())])

# b) Logistic Regression
step_mms_lr =Pipeline([('MinMaxScaler', MinMaxScaler()),
            ('LogisticClassifier', LogisticRegression())])
step_ss_lr = Pipeline([('StandardScaler', StandardScaler()),
            ('LogisticClassifier', LogisticRegression())])

# c) Decision Tree
step_mms_tree = Pipeline([('MinMaxScaler', MinMaxScaler()),
            ('DTClassifier', DecisionTreeClassifier())])
step_ss_tree = Pipeline([('StandardScaler', StandardScaler()),
            ('DTClassifier', DecisionTreeClassifier())])

# d) knn 
step_mms_knn = Pipeline([('MinMaxScaler', MinMaxScaler()),
            ('KNNClassifier', KNeighborsClassifier())])
step_ss_knn = Pipeline([('StandardScaler', StandardScaler()),
            ('KNNClassifier', KNeighborsClassifier())])

# e)SVC pipeline
step_mms_svc = Pipeline([('MinMaxScaler', MinMaxScaler()),
                        ('SVClassifier', SVC())])
step_ss_svc = Pipeline([('StandardScaler', StandardScaler()),
                       ('SVClassifier', SVC())])

pipelines = [step_mms_lr,step_ss_lr,step_mms_rf,step_ss_rf,step_mms_tree,
             step_ss_tree,step_mms_knn,step_ss_knn,step_mms_svc, step_ss_svc]

for pipe in pipelines:
    pipe.fit(X_train,y_train)

#%% Pipeline Analysis

best_accuracy = 0
model_scored = []
for i, model in enumerate(pipelines):
    # print(model.score(X_test,y_test))
    model_scored.append(model.score(X_test,y_test))
    # if model.score(X_test,y_test) > best_accuracy:
    #     best_accuracy = model.score(X_test,y_test)
    #     best_pipeline = model

best_model = pipelines[np.argmax(model_scored)]
best_accuracy = model_scored[np.argmax(model_scored)]
print('The best pipeline for this cardio dataset is',str(best_model),  
      'with accuracy of',str(best_accuracy))

# The best pipeline for this cardio dataset is Pipeline(step_mms_lr) 
# with accuracy of 0.7692307692307693
#%% Fine Tuning - Best Pipeline
step_mms_lr =Pipeline([('MinMaxScaler', MinMaxScaler()),
                       ('LogisticClassifier', LogisticRegression())])
best_pipeline=step_mms_lr.fit(X_train,y_train)
#%% GridSearchCV
#search for parameters to train in RF SKlearn
# grid_param

grid_values = [{ 'LogisticClassifier__random_state' : [10,100,1000, None],
                 'LogisticClassifier__C':[0.01,0.09,1.0,5,25]}]

grid_search= GridSearchCV(step_mms_lr, param_grid = grid_values,verbose=1,
                            n_jobs=-1)

gs_model = grid_search.fit(X_train,y_train)
gs_accuracy = grid_search.score(X_test,y_test)
print("This grid search model accuracy is ",gs_model.score(X_test,y_test))
#%% Step 7) Model Evaluation & Analysis
print('The best pipeline for this cardio dataset is',str(best_model),  
      'with accuracy of',str(best_accuracy))

y_true= y_test
y_pred= best_model.predict(X_test)

cm = confusion_matrix(y_true, y_pred) #Model confusion Matrix
cr = classification_report(y_true, y_pred) #Model f1 Score
ac= accuracy_score(y_true, y_pred)
print('Confussion Matrix :\n {} \n Classification Report :\n  {}\n Accuracy_score:\n {}\n'.format(cm,cr, ac))

#since no change in accuracy, initial model is chosen
#%% Step 8) Model saving
with open(BEST_MODEL_PATH, 'wb') as file:
    pickle.dump(best_pipeline,file)

with open(BEST_PIPE_PATH, 'wb') as file:
    pickle.dump(step_mms_lr,file)
    

#%% Step 9) Discussion
# This model accuracy is 77% which is more than 70% 
# The best pipeline chosen is 'MinMaxScaler', MinMaxScaler()), 
# ('LogisticClassifier', LogisticRegression())])
# The dataset shows high correlation between 'age','trtbps',
# 'chol','thalachh','oldpeak','cp','thall' vs probability of getting heart attack 
# The accuracy can be improved by having more data into the training













