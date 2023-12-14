import streamlit as st
import pandas as pd
import plost
import matplotlib.pyplot as plt
import pydeck as pdk
import plotly.express as px
from wordcloud import WordCloud

from streamlit.components.v1 import components as components
import folium



st.set_page_config(layout='wide', initial_sidebar_state='expanded')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    

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

##################################################
# Function to generate word cloud



# first row
c1, c2 = st.columns((3, 7))
with c1:
    seattle_weather = pd.read_csv('https://raw.githubusercontent.com/tvst/plost/master/data/seattle-weather.csv', parse_dates=['date'])
    stocks = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/stocks_toy.csv')
    donut_theta = st.selectbox('Select data', ('q2', 'q3'))
    st.markdown('### Donut chart')
    plost.donut_chart(
        data=stocks,
        theta=donut_theta,
        color='company',
        legend='bottom', 
        use_container_width=True)

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


##################################################
# Row A
st.markdown('### Metrics')
col1, col2, col3 = st.columns(3)
col1.metric("Temperature", "70 °F", "1.2 °F")
col2.metric("Wind", "9 mph", "-8%")
col3.metric("Humidity", "86%", "4%")

# Row B
seattle_weather = pd.read_csv('https://raw.githubusercontent.com/tvst/plost/master/data/seattle-weather.csv', parse_dates=['date'])
stocks = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/stocks_toy.csv')


c1, c2 = st.columns((7,3))
with c1:
    time_hist_color = st.selectbox('Color by', ('temp_min', 'temp_max'))
    st.markdown('### Heatmap')
    plost.time_hist(
    data=seattle_weather,
    date='date',
    x_unit='week',
    y_unit='day',
    color=time_hist_color,
    aggregate='median',
    legend=None,
    height=345,
    use_container_width=True)
with c2:
    donut_theta = st.selectbox('Select data', ('q2', 'q3'))
    st.markdown('### Donut chart')
    plost.donut_chart(
        data=stocks,
        theta=donut_theta,
        color='company',
        legend='bottom', 
        use_container_width=True)




# Sample data for countries and their information
country_data = {
    "USA": "Information about USA",
    "Canada": "Information about Canada",
    "Mexico": "Information about Mexico",
    # Add more countries and their information here
}

# Create a map using Folium
m = folium.Map(location=[20, 0], zoom_start=2)
for country, info in country_data.items():
    folium.Marker(
        location=[0, 0],
        popup=f"<b>{country}</b><br>{info}",
        tooltip=country
    ).add_to(m)

# Display the map in Streamlit
folium_static = folium.Map(location=[20, 0], zoom_start=2)
st.write(folium_static._repr_html_(), unsafe_allow_html=True)