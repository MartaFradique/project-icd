import streamlit as st
import pandas as pd

data = pd.read_csv('./icd_marta_ana_scopus_edited.csv');
st.title('Uber pickups in NYC')
chart_data = pd.DataFrame (data, columns = ['Correspondence Address'])
filtered_sum = chart_data["Correspondence Address"].value_counts()
st.bar_chart(filtered_sum)