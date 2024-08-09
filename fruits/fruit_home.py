import streamlit as st

pg=st.navigation([st.Page("fruit_eda.py",title="Fruit data EDA"),
st.Page("fruit_pred.py",title="Fruit Label Prediction")])

pg.run()