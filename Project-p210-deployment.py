
import pandas as pd
import streamlit as st 
from sklearn.linear_model import LogisticRegression

st.title('Fake Bills Classification: Logistic Regression')

st.sidebar.header('User Input Parameters')

def user_input_features():
    diagonal = st.sidebar.number_input('enter diagonal value', value = 171.44)
    height_left = st.sidebar.number_input('enter height_left value', value = 103.96)
    height_right = st.sidebar.number_input('enter height_right value', value = 103.92)
    margin_low = st.sidebar.number_input("enter margin_low value", value = 3.68)
    margin_up = st.sidebar.number_input("enter margin_up value", value = 2.89)
    length = st.sidebar.number_input("enter length value", value = 113.21)
    data = {'diagonal':diagonal,
            'height_left':height_left,
            'height_right':height_right,
            'margin_low':margin_low,
            'margin_up':margin_up,
            'length':length}
    features = pd.DataFrame(data,index = [0])
    return features 
    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

fakebills_nona = pd.read_csv("fake_bills.csv", ';')
fakebills_nona = fakebills_nona.dropna()

X= fakebills_nona.iloc[:, 1:]
Y=fakebills_nona.iloc[:,0]
clf = LogisticRegression()
clf.fit(X,Y)

pred = clf.predict(df)
pred_proba = clf.predict_proba(df)

st.subheader('Predicted Result')
st.write('Genuine Bill' if pred_proba[0][1] > 0.5 else 'Fake Bill')

st.subheader('Prediction Probability')
st.write(pred_proba)
