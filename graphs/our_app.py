import streamlit as st
import pandas as pd
import plost
import matplotlib.pyplot as plt
import pydeck as pdk
import plotly.express as px

from streamlit.components.v1 import components as components
import folium



st.set_page_config(layout='wide', initial_sidebar_state='expanded')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    

##first row 
c1, c2 = st.columns((7,3))
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
    # Sample data for vertical bar chart
    bar_data = pd.DataFrame({
        'Category': ['A', 'B', 'C', 'D'],
        'Values': [23, 45, 56, 78]
    })

    # Create a smaller vertical bar chart using matplotlib with pastel colors
    fig, ax = plt.subplots(figsize=(4, 4))  # Set figure size to make it smaller
    pastel_colors = plt.cm.get_cmap('Pastel1', len(bar_data))
    bars = ax.bar(bar_data['Category'], bar_data['Values'], color=pastel_colors.colors, width=0.5)  # Adjust bar width

    # Customize appearance: Remove background gridlines and spines
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set labels and title
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Values', fontsize=12)
    ax.set_title('Vertical Bar Chart', fontsize=14)

    # Adjust tick label size
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Display the bar chart using Streamlit
    st.pyplot(fig)



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