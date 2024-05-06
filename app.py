# from html.entities import html5
# from pydoc import html
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import joblib

from code import interact
import streamlit as st
st.set_page_config(page_title="Parivartan", layout="wide")

with st.container():
    st.title("PragatiSanket : A Dropout Predictive Model")
    st.markdown("##### 📚🌍 A significant and intriguing issue around the world is the high number of school dropouts among pupils each year. This has adverse effects on their social and economic development, leading to a decline in the nation's literacy rate and hindering innovation. 📉 In India, this issue is particularly noteworthy.📊 According to a survey by the National Statistical Office (NSO) in India, approximately 17.5% of upper elementary pupils and 19.8% of secondary students, along with 12.6% of kids overall, have discontinued their studies. 🚫 A dropout in this context refers to a person who was enrolled in education at some point but did not complete the most recent level they were registered for and is currently not attending any educational institution.🤖 To address this problem, a machine learning model has been developed to identify the factors that contribute to student dropouts. The model utilizes seven classification algorithms from the sklearn library to predict whether a student is likely to drop out of education or not. 🧠 By leveraging these algorithms, educators can intervene early and provide necessary support to at-risk students, potentially reducing dropout rates. 📚✨💻🎯 The ultimate goal is to increase retention rates and create a more inclusive and innovative educational environment for the students. 📈🌟🎓.")
    st.write(' ')
    st.markdown("##### Enter the following basic details to begin : ")

with open('app.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
 
model = joblib.load('model.pkl')
le = joblib.load('encoder.pkl')
st.text_input("Enter Student Name :")
st.radio("Select Student's Gender : ",["Male", "Female", "Other" ])
# st.date_input("Enter Student's DOB : ")
st.slider("Enter Age : ",0,70)
st.selectbox("Select Address Type : ",["Rural", "Urban"])
st.radio("Whether Fee Paid ? ",["Yes","No"])

input_names = {'school': 'Select School type', 'sex': 'Gender', 'age': "Age", 'address': "Address",
               'famsize': "Family Size", 'Pstatus': "Pstatus", 'Medu': "Mother Education (0 - Uneducated, 1 - Till HighSchool, 2 - Till Intermediate, 3 - UnderGraduate, 4- Postgraduate/Doctorate) : ", 'Fedu': "Father's Education (0 - Uneducated,  1 - Till HighSchool,  2 - Till Intermediate,  3 - UnderGraduate,  4- Postgraduate/Doctorate) : ",
               'Mjob': "Mother's Occupation : ", 'Fjob': "Father's Occupation : ", 'reason': "Reason of Dropout : ", 'guardian': "Guardian",
               'traveltime': "Travel Time(in hrs)", 'studytime': "Study Time(in hrs)", 'failures': "Any Backlogs/Failures ? ",
               'schoolsup': "School Support", 'famsup': "Family Support", 'paid': "Fee paid", 'activities': "Activities",
               'nursery': "Nursery", 'higher': "Higher Education??", 'internet': "Internet", 'romantic': "Romantic",
               'famrel': "Family relatives", 'freetime': "Free Time(in hrs)", 'goout': "Vacation/Go out Time(hrs)",
               'Dalc': "Dalc", 'Walc': "Walc", 'health': "Enter Health (on the scale of 1-4, 1 being very Unhealthy to attend)", 'absences': "Number of Days Absent"}

input_type = {'school': ['', "MS", "GP"], 'sex': ['', "M", "F"], 'address': ['', "R", "U"],
              'famsize': ['', "GT3", "LE3"], 'Pstatus': ['', "T", "A"], 'Medu': ['', 0, 1, 2, 3, 4],
              'Fedu': ['', 0, 1, 2, 3, 4], 'Mjob': ['', "Teacher", "at home", "health", "services", "other"],
              'Fjob': ['', "Teacher", "at home", "health", "services", "other"], 'reason': ['', "Reputation", "Course","Home", "other"],
              'guardian': ['', "Father", "Mother", "Other"], 'traveltime': ['', 1, 2, 3, 4,'More than above'], 'studytime': ['', 1, 2, 3, 4, 'More than above'],
              'failures': ['', 0, 1, 2, 3, "More than above"], 'schoolsup': ['', "Yes", "No"], 'famsup': ['', "Yes", "No"],
              'paid': ['', "Yes", "No"], 'activities': ['', "Yes", "No"], 'nursery': ['', "Yes", "No"],
              'higher': ['', "Yes", "No"], 'internet': ['', "Yes", "No"], 'romantic': ['', "Yes", "No"],
              'famrel': ['', 1, 2, 3, 4], 'freetime': ['', 1, 2, 3, 4, "I'm always Free ;)"], 'goout': ['', 1, 2, 3, 4], 'Dalc': ['', 1, 2, 3, 4],
              'Walc': ['', 1, 2, 3, 4], 'health': ['', 1, 2, 3, 4]}

input_lst = []
st.markdown("##### Now enter the following necessary details(as per FeatureSelection) to predict : ")
# with st.form(key="my_form", clear_on_submit=True):
#     selected_features_list = model.feature_names
#     col1, col2, col3 = st.columns(3)  # Adjust the number of columns as needed
#     for i in range(len(selected_features_list)):
#         col = col1 if i < len(selected_features_list) / 3 else col2 if i < 2 * len(selected_features_list) / 3 else col3
#         if selected_features_list[i] == 'age':
#             ele = col.slider("Age", 15, 22)
#         elif selected_features_list[i] != 'age' and selected_features_list[i] != 'absences':
#             ele = col.selectbox(input_names[selected_features_list[i]], input_type[selected_features_list[i]])
#         elif selected_features_list[i] == "absences":
#             ele = col.slider("Days absent", 0, 100)
#         input_lst.append(ele)
with st.form(key="my_form", clear_on_submit=True):
    selected_features_list = model.feature_names
    for i in selected_features_list:
        if i=='age':
            ele = st.slider("Age",15,22)
        elif i!='age' and i!='absences':
            ele = st.selectbox(input_names[i], input_type[i])
        elif i=="absences":
            ele = st.slider("Days absent",0,100)
        input_lst.append(ele)
    submitted = st.form_submit_button("Test")
reload_btn = st.button('Test another')
# submitted = st.form_submit_button("Test")
if submitted:
    X_test_input_cols = list(model.feature_names)
    default_dict = {}
    for i in range(len(X_test_input_cols)):
        default_dict[X_test_input_cols[i]] = input_lst[i]

    X_input_test = pd.DataFrame(default_dict, index=[0])

    for name in X_test_input_cols:
        if X_input_test[name].dtype == 'object':
            X_input_test[name] = le.fit_transform(X_input_test[name])

    y_input_pred = model.predict(X_input_test)
    if input_lst.count('') > 0:
        st.error('Some inputs are missing')
    elif y_input_pred[0] == 0:
        st.success('The Student will NOT Dropout 😆😆😆')
        st.balloons()
    else:
        st.error('The Student will Dropout 😭😭😭')

st.write('')
st.write('')
st.write('Made with 🫶 by Tanveer and Swarnima!')

