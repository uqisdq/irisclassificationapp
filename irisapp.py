import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
sns.set()

# st.markdown(
#     """
#     <style>
#     .reportview-container {
#         background: url("url_goes_here")
#     }
#    .sidebar .sidebar-content {
#         background: url("url_goes_here")
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

st.write("""
# Simple Iris Flower Classification and Prediction App

***This app classify and predict the Iris Flower Species***
""")

st.sidebar.header('User Input Parameters')
st.sidebar.subheader('Slide the parameters below according to your liking or data')
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

iris = datasets.load_iris()
X = iris.data
Y = iris.target

scaler = StandardScaler().fit(X)

df = user_input_features()
dfscaled = scaler.transform(df)
dfscaledarr = pd.DataFrame(dfscaled)
dfscaledarr.columns = df.columns

XScaled = scaler.transform(X)
clf = RFC()
clf.fit(XScaled, Y)

st.subheader("""***1. Petal & Sepal Classification Graph***""")
st.write("""these are the matrix of Iris Classification based on sklearn's data""")
Xarr = pd.DataFrame(X)
Xarr.columns =df.columns
Xarr['species']= Y
Xarr.loc[Xarr["species"] == 0, "species"] = "setosa"
Xarr.loc[Xarr["species"] == 1, "species"] = "versicolor"
Xarr.loc[Xarr["species"] == 2, "species"] = "virginica"
fig = px.scatter_matrix(Xarr,
    dimensions=["sepal_width", "sepal_length", "petal_width", "petal_length"],
    color="species", symbol="species",
    title="Scatter of Iris Classification data set",
    labels={col:col.replace('_', ' ') for col in df.columns}) # remove underscore
fig.update_traces(diagonal_visible=False)

st.plotly_chart(fig, use_container_width=True)
st.write("""
If you are confuse to read the graph above,\n
just match the row and column of the graph that you want to see, \n
it tells you which parameters are used
""")

prediction = clf.predict(dfscaled)
prediction_proba = clf.predict_proba(dfscaled)

st.subheader("""
***2. The name of species available to predict and its corresponding index in our database***
""")
irisarr = pd.DataFrame(({'species':iris.target_names}))
st.write(irisarr)

st.subheader("""
***3. Your Input Parameters :***
""")
st.write(df)
st.subheader("""
***4. Your Scaled Input Parameters :***
""")
st.write('All the data are scaled to make prediction better')
st.write(dfscaledarr)

st.subheader("""
***5. Prediction Result***
""")
predarr = pd.DataFrame({'species':iris.target_names[prediction]})
st.write('The species of your Iris Flower is :  ' + str(predarr.loc[0][0]))

st.subheader("""
***6. Prediction Probability***
""")
st.write('Probability range from 0 to 1, just like 0% to 100%')
probarr = pd.DataFrame(prediction_proba)
probarr.columns=[irisarr.species]
st.write(probarr)
