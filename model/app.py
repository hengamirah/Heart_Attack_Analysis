# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 13:23:06 2022

age - age in years
sex - sex (1 = male; 0 = female)
cp - chest pain type (1 = typical angina; 2 = atypical angina; 3 = non-anginal pain; 4 = asymptomatic)
trestbps - resting blood pressure (in mm Hg on admission to the hospital)
chol - serum cholestoral in mg/dl
fbs - fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
restecg - resting electrocardiographic results (0 = normal; 1 = having ST-T; 2 = hypertrophy)
thalach - maximum heart rate achieved
exng - exercise induced angina (1 = yes; 0 = no)
oldpeak - ST depression induced by exercise relative to rest
slope - the slope of the peak exercise ST segment (1 = upsloping; 2 = flat; 3 = downsloping)
ca - number of major vessels (0-3) colored by flourosopy
thal - 0 = null, 1 = fixed defect, 2 = normal,3 = reversable defect
the predicted attribute - diagnosis of heart disease (angiographic disease status) 
(Value 0 = < 50% diameter narrowing; Value 1 = > 50% diameter narrowing)

    
@author: Amirah Heng
"""

import os
import pickle
import numpy as np
import streamlit as st

# Step 1) Load Data
MODEL_PATH = os.path.join(os.getcwd(),'best_model.pkl')
with open(MODEL_PATH,'rb') as file:
    model =pickle.load(file)
    
X_new= [60.0,125.0,258.0,141.0,2.8,0.0,3]

temp = np.expand_dims(X_new,axis=0)
outcome = model.predict(temp) # in here ,predict requires matrix of 1 row (1,y), so axis=0
print(outcome)

with st.form("patient_form"):
    st.header("Do you have risk of getting Heart Attack? Find out here!")
   
    age = st.number_input('Age:')
    str_age = st.radio("Gender:",("Female", "Male"))
    if str_age == "Female" :
        age = 0
    elif str_age == "Male":
        age = 1
        
    str_cp = st.radio("Chest pain type:",("1 = typical angina", "2 = atypical angina", "3 = non-anginal pain", "4 = asymptomatic"))
    if str_cp == "1 = typical angina" :
        cp = 0
    elif str_cp == "2 = atypical angina":
        cp = 1
    elif str_cp == "3 = non-anginal pain":
        cp = 3
    elif str_cp == "4 = asymptomatic":
        cp = 4
        
    trtbps = st.number_input('Resting Blood Pressure (in mm Hg )')
    chol = st.number_input('Cholestoral in mg/dl')
    
    str_fbs = st.radio("Fasting Blood Sugar > 120 mg/dl",("True","False"))
    if str_fbs == "True" :
        fbs = 1
    elif str_fbs == "False":
        fbs = 0
        
    str_restecg = st.radio("Resting Electrocardiographic results",("0 = normal", "1 = having ST-T","2 = hypertrophy"))
    
    if str_restecg == "0 = normal" :
        restecg = 0
    elif str_restecg == "1 = having ST-T":
        restecg = 1
    elif str_restecg == "2 = hypertrophy":
        restecg = 2
        
    thalachh = st.number_input('Maximum heart rate achieved')
    
    str_exng = st.radio("Exercise induced angina ",("Yes","No"))
    if str_exng == "Yes" :
        exng = 0
    elif str_exng == "No":
        exng = 1
    
    oldpeak = st.number_input('ST depression induced by exercise relative to rest')
    
    str_slp = st.radio("the slope of the peak exercise ST segment",("1 = upsloping", "2 = flat","3 = downsloping"))
    if str_slp == "1 = upsloping" :
        slp = 1
    elif str_slp == "2 = flat":
        slp = 2
    elif str_slp == "3 = downsloping":
        slp = 3
    
    str_caa = st.radio("number of major vessels (0-3) colored by flourosopy",("0","1","2","3"))
    if str_caa == "0" :
        caa = 0
    elif str_caa == "1":
        caa = 1
    elif str_caa == "2":
        caa = 2     
    elif str_caa == "3":
        caa = 3
        
    str_thall = st.radio("Thalassemia :",("0 = null","1 = fixed defect","2 = normal","3 = reversable defect"))
    if str_thall == "0 = null" :
        thall = 0
    elif str_thall == "1 = fixed defect":
        thall = 1
    elif str_thall == "2 = normal":
        thall = 2     
    elif str_thall == "3 = reversable defect":
        thall = 3
        
    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
    
        X_new = [age,trtbps,chol,thalachh,oldpeak,cp,thall]
        temp = np.expand_dims(np.array(X_new),axis=0)
        outcome = model.predict(temp) # in here ,predict requires matrix of 1 row (1,y), so axis=0
        # outcome_dict = [{ 0:'0: < 50% diameter narrowing. less chance of heart disease',
        #                   1:'1: > 50% diameter narrowing. more chance of heart disease'}]
        
        st.write(outcome)
        if outcome == 0:
            st.write(" 0: < 50% diameter narrowing. You have lesser chance of heart disease")
            st.balloons()
        
        else:
            st.write("Hi! You better take care of your health as > 50% diameter narrowing. You have more chance of getting heart disease!")
    

# X_newtest= [56.0,0.0,1.0,140.0,294.0,0.0,0.0,153.0,0.0,1.3,1.0,0.0,2] #output 1.0


