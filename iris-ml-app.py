import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from PIL import Image

st.write("""
# Ji Hen First Stremlit App
# Simple Iris Flower Prediction App

This app predicts the **Iris flower** type!
(This is part of AirAsia Academy Data Analytics Course)
""")

all_iris = Image.open("Iris Photo.png")
st.image(all_iris, 
         caption='Three Iris Species <source: https://morioh.com/p/eafb28ccf4e3>')

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)

iris = pd.read_csv("iris.csv")

@st.cache
def convert_df(iris):
   return iris.to_csv().encode('utf-8')


csv = convert_df(iris)

st.download_button(
   "Download Iris Data",
   csv,
   "iris.csv",
   "text/csv",
   key='download-csv'
)
