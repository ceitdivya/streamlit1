import streamlit as st
import seaborn as sns
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Fruits", page_icon="	:strawberry:",layout="wide")

st.title(":grapes::melon::watermelon: Fruit Data Analysis:lemon::banana::pineapple:")

fdf=pd.read_csv('fruits.csv')
st.header(':tangerine: Sample data :apple:')
st.table(fdf.head())

st.header(':green_apple: Statistical summary of dataset :pear:')
st.table(fdf.describe())

st.header(':peach: Fruit labels :cherries:')
st.table(fdf['fruit_label'].unique())
st.table(fdf['fruit_name'].unique())

st.header(':strawberry: Univariate Analysis :strawberry:')

fig1=px.box(fdf,x='mass',y='fruit_name',title='Mass for fruits',color='fruit_name')
st.plotly_chart(fig1,use_container_width=True)

fig2=px.box(fdf,x='width',y='fruit_name',title='Width for fruits',color='fruit_name')
st.plotly_chart(fig2,use_container_width=True)

fig3=px.box(fdf,x='height',y='fruit_name',title='Heights for fruits',color='fruit_name')
st.plotly_chart(fig3,use_container_width=True)

fig4=px.box(fdf,x='color_score',y='fruit_name',title='Color for fruits',color='fruit_name')
st.plotly_chart(fig4,use_container_width=True)

def data():
    fdf=pd.read_csv('fruits.csv')
    x=fdf.iloc[:,3:7]
    y=fdf[['fruit_label']]
    return x,y
