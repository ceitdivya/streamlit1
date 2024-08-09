import streamlit as st
from sklearn import metrics as mat
from fruit_eda import data
from sklearn.model_selection import train_test_split as tts,GridSearchCV as gscv
from sklearn import metrics as mat
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import plotly.express as px

#st.set_page_config(page_title="Fruits", page_icon="	:strawberry:",layout="wide")

st.title(":grapes::melon::watermelon: Fruit Label Prediction :lemon::banana::pineapple:")

x,y=data()
xtrain,xtest,ytrain,ytest=tts(x,y,test_size=0.2,random_state=0)
c1,c2=st.columns(2)
c1.subheader('Training data size')
c1.write(xtrain.shape)
c1.write(ytrain.shape)

c2.subheader('Testing data size')
c2.write(xtest.shape)
c2.write(ytest.shape)

knn1=KNeighborsClassifier()
param={'n_neighbors':np.arange(1,10)}
knn_gscv=gscv(knn1,param,cv=5)

knn_gscv.fit(xtrain,ytrain)

c3,c4=st.columns(2)
c3.subheader('Best neighbors')
n1=knn_gscv.best_params_['n_neighbors']
c3.write(knn_gscv.best_params_['n_neighbors'])

c4.subheader('Best score')
c4.write(knn_gscv.best_score_*100)

knnmodel=KNeighborsClassifier(n_neighbors=n1)
knnmodel.fit(xtrain,ytrain)

ypred=knnmodel.predict(xtest)

st.subheader('Classification Report')
st.write(mat.classification_report(ytest,ypred,output_dict=True))

st.subheader("Confusion Matrix of KNN")
cm1=mat.confusion_matrix(ytest,ypred,labels=[1,2,3,4])
fig1=px.imshow(cm1,text_auto=True,labels=dict(x="Predicted Values",y="Actual Values"),x=['Apple','Orange','Mandarin','Lemon'],y=['Apple','Orange','Mandarin','Lemon'])
st.plotly_chart(fig1)

st.header(":grapes: Prediction :pineapple:")
n1=int(st.number_input("Enter mass value"))
n2=int(st.number_input("Enter width value"))
n3=int(st.number_input("Enter height value"))
n4=int(st.number_input("Enter color score value"))
sample=[[n1,n2,n3,n4]]
if st.button("Predict the fruit label"):
    t=knnmodel.predict(sample)
    st.write(t)
    if t==1:
        st.write(":apple: Apple :apple:")
    elif t==2:
        st.write("	:tangerine: Mandarin 	:tangerine:")
    elif t==3:
        st.write(":large_orange_circle: Orange :large_orange_circle:")
    elif t==4:
        st.write(":lemon: Lemon :lemon:")
    else:
        st.write("Fruit not listed")


