import streamlit as st
import pandas as pd
import plost
import matplotlib.pyplot as plt
import pydeck as pdk
import plotly.express as px
from wordcloud import WordCloud

from streamlit.components.v1 import components as components
import folium
import numpy as np



st.set_page_config(layout='wide', initial_sidebar_state='expanded')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    


st.title("Ana e Marta Dashboard")

############################

c1, c2, c3, c4 = st.columns((1, 5, 3, 1))
with c1:
     st.write("") 

# Data for countries and their information
with c2:
    
    countries = ['Italy', 'Spain', 'France', 'Germany', 'China', 'India', 'Mexico']
    info1 = [20, 30, 25, 35, 28, 22, 38]  # Data for the first type of information
    info2 = [40, 35, 30, 45, 32, 27, 42]  # Data for the second type of information

    # Set width of bar
    bar_width = 0.35

    # Set position of bars on X axis
    r = np.arange(len(countries))

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(10, 6))  # Set figure size
    pastel_colors = plt.cm.get_cmap('Pastel1', len(countries))

    ax.bar(r - bar_width/2, info1, color=pastel_colors(0), width=bar_width, edgecolor='grey', label='Info 1')
    ax.bar(r + bar_width/2, info2, color=pastel_colors(1), width=bar_width, edgecolor='grey', label='Info 2')

    # Display the values on the bars
    for i, (v1, v2) in enumerate(zip(info1, info2)):
        ax.text(r[i] - bar_width/2, v1 + 1, str(v1), ha='center', va='bottom', color='black')
        ax.text(r[i] + bar_width/2, v2 + 1, str(v2), ha='center', va='bottom', color='black')

    # Customize appearance
    ax.set_xlabel('Countries', fontweight='bold')
    ax.set_ylabel('Values', fontweight='bold')
    ax.set_xticks(r)
    ax.set_xticklabels(countries)
    ax.set_title('Information Comparison Between Countries')
    ax.legend()

    # Hide spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Display the chart using Streamlit
    st.pyplot(fig)

    #############################

with c3:
    # Sample data for horizontal bar chart
    bar_data = pd.DataFrame({
        'Category': ['Andorra', 'Belgica', 'Canada', 'Dinamarca', 'Equador', 'França', 'Portugal'],
        'Values': [23, 45, 56, 78, 12, 34, 65]
    })

    # Create a horizontal bar chart with thin bars and reduced spacing
    fig, ax = plt.subplots(figsize=(8, 4))  # Set figure size to make it smaller
    pastel_colors = plt.cm.get_cmap('Pastel1', len(bar_data))
    bars = ax.barh(bar_data['Category'], bar_data['Values'], color=pastel_colors.colors, height=0.2)  # Adjust bar height for thin bars

    # Display the values on top of each bar
    for i, val in enumerate(bar_data['Values']):
        ax.text(val, i, str(val), ha='left', va='center', fontsize=10)  # Display values on bars

    # Customize appearance: Remove background gridlines and spines
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)  # Hide the y-axis line

    # Hide x-axis line and labels
    ax.xaxis.set_visible(False)

    # Set y-axis label and title
    ax.set_title('Horizontal Bar Chart', fontsize=14)

    # Display the bar chart using Streamlit
    st.pyplot(fig)
with c4:
     st.write("") 


##################################################
# Function to generate word cloud



# first row
c1, c2, c3 = st.columns((1, 8, 1))
with c1:
     st.write("") 
    # # Creating a large white space using an empty placeholder with custom CSS
    # placeholder = st.empty()
    # placeholder.markdown(
    #     '<style>div.css-1l02zno {height: 75px;}</style>',
    #     unsafe_allow_html=True
    # )

with c2:
    st.title('Word Cloud Generator from Dataset')
    def generate_wordcloud(data):
        text = ' '.join(data)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)


    # Streamlit app
    
    # Dropdown menu to select file
    selected_file = st.selectbox("Select file", ["Italy", "China", "France", "Gemany", "Spain", "India", "Mexico"])

    file_mapping = {
        "Italy": "icd_marta_ana_scopus_edited.csv",
        "China": "scopus3.csv",
        "France": "icd_marta_ana_scopus_edited.csv",
        "Germany": "icd_marta_ana_scopus_edited.csv",
        "Spain": "icd_marta_ana_scopus_edited.csv",
        "India": "icd_marta_ana_scopus_edited.csv",
        "Mexico": "icd_marta_ana_scopus_edited.csv"
    }

    if selected_file in file_mapping:
        file_path = file_mapping[selected_file]
    else:
        st.error("Please select a valid file")


    data = pd.read_csv(file_path)
    # st.write(data)  # Display the uploaded data

    text_column = "Title"  # Replace 'YourColumnName' with the actual column containing text data
    text_data = data[text_column].dropna().tolist()
    generate_wordcloud(text_data)

with c3:
     st.write("") 
    # # Creating a large white space using an empty placeholder with custom CSS
    # placeholder = st.empty()
    # placeholder.markdown(
    #     '<style>div.css-1l02zno {height: 75px;}</style>',
    #     unsafe_allow_html=True
    # )

##################################################
# Row A
st.markdown('### Metrics')
col1, col2, col3 = st.columns(3)
col1.metric("Temperature", "70 °F", "1.2 °F")
col2.metric("Wind", "9 mph", "-8%")
col3.metric("Humidity", "86%", "4%")



#############################################################################################################################
    
# Sample data for filtered_df
    

np.random.seed(42)  # Set seed for reproducibility

regions = np.random.choice(['North', 'South', 'East', 'West'], size=100)
categories = np.random.choice(['Electronics', 'Clothing', 'Furniture'], size=100)
sub_categories = np.random.choice(['Phones', 'Laptops', 'Shirts', 'Chairs'], size=100)
sales = np.random.randint(100, 1000, size=100)

filtered_df = pd.DataFrame({
    'Region': regions,
    'Category': categories,
    'Sub-Category': sub_categories,
    'Sales': sales
})


st.subheader("Hierarchical view of Sales using TreeMap")
fig3 = px.treemap(filtered_df, path = ["Region","Category","Sub-Category"], values = "Sales",hover_data = ["Sales"],
                  color = "Sub-Category")
fig3.update_layout(width = 800, height = 650)
st.plotly_chart(fig3, use_container_width=True)
#############################################################################################################################


##first row 
c1, c2 = st.columns((5,5))
with c1:
   # Sample data for horizontal bar chart
    bar_data = pd.DataFrame({
        'Category': ['Andorra', 'Belgica', 'Canada', 'Dinamarca', 'Equador', 'França', 'Portugal'],
        'Values': [23, 45, 56, 78, 12, 34, 65]
    })

    # Create a horizontal bar chart with thin bars and reduced spacing
    fig, ax = plt.subplots(figsize=(8, 4))  # Set figure size to make it smaller
    pastel_colors = plt.cm.get_cmap('Pastel1', len(bar_data))
    bars = ax.barh(bar_data['Category'], bar_data['Values'], color=pastel_colors.colors, height=0.2)  # Adjust bar height for thin bars

    # Display the values on top of each bar
    for i, val in enumerate(bar_data['Values']):
        ax.text(val, i, str(val), ha='left', va='center', fontsize=10)  # Display values on bars

    # Customize appearance: Remove background gridlines and spines
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)  # Hide the y-axis line

    # Hide x-axis line and labels
    ax.xaxis.set_visible(False)

    # Set y-axis label and title
    ax.set_title('Horizontal Bar Chart', fontsize=14)

    # Display the bar chart using Streamlit
    st.pyplot(fig)

 
with c2:
     # Sample data for horizontal bar chart
    bar_data = pd.DataFrame({
        'Category': ['Andorra', 'Belgica', 'Canada', 'Dinamarca', 'Equador', 'França', 'Portugal'],
        'Values': [23, 45, 56, 78, 12, 34, 65]
    })

    # Create a horizontal bar chart with thin bars and reduced spacing
    fig, ax = plt.subplots(figsize=(8, 4))  # Set figure size to make it smaller
    pastel_colors = plt.cm.get_cmap('Pastel1', len(bar_data))
    bars = ax.barh(bar_data['Category'], bar_data['Values'], color=pastel_colors.colors, height=0.2)  # Adjust bar height for thin bars

    # Display the values on top of each bar
    for i, val in enumerate(bar_data['Values']):
        ax.text(val, i, str(val), ha='left', va='center', fontsize=10)  # Display values on bars

    # Customize appearance: Remove background gridlines and spines
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)  # Hide the y-axis line

    # Hide x-axis line and labels
    ax.xaxis.set_visible(False)

    # Set y-axis label and title
    ax.set_title('Horizontal Bar Chart', fontsize=14)

    # Display the bar chart using Streamlit
    st.pyplot(fig)