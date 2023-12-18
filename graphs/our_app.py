import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pydeck as pdk
import plotly.express as px
from wordcloud import WordCloud
import nltk
from main2 import unique_countries, lda_unique_country
from streamlit.components.v1 import components as components
import numpy as np
unique_countries_lst = unique_countries()
data_scopus = pd.read_csv('../first-sprint/icd_marta_ana_scopus_edited.csv')
data_unesco = pd.read_csv('../first-sprint/unesco_heritage_by_country.csv')
data_unesco_sorted = data_unesco.sort_values(by="Properties inscribed", ascending=False)
top_10_data_unesco = data_unesco_sorted.head(10)
chart_data = pd.DataFrame(data_scopus, columns=['Country'])
filtered_sum = chart_data["Country"].value_counts().reset_index()
filtered_sum.columns = ['Country', 'Document Count']
top_10_data = filtered_sum.head(10)
merged_data = pd.merge(data_unesco_sorted, filtered_sum, on="Country", suffixes=('_file1', '_file2'))
top_10_merged_data = merged_data.head(10)


st.set_page_config(layout='wide', initial_sidebar_state='expanded')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("Recommender Systems in Cultural Heritage Sites")

c1, c2 = st.columns((7, 3))

with c1:
        countries = top_10_merged_data['Country']
        info1 = top_10_merged_data['Properties inscribed']  # Data for the first type of information
        info2 =  top_10_merged_data['Document Count']  # Data for the second type of information

        # Set width of bar
        bar_width = 0.32

        # Set position of bars on X axis
        r = np.arange(len(countries))

        # Create the bar plot
        fig, ax = plt.subplots(figsize=(10, 5))  # Set figure size
        pastel_colors = plt.cm.get_cmap('Pastel1', len(countries))

        ax.bar(r - bar_width/2, info1, color=pastel_colors(0), width=bar_width, edgecolor='grey', label="UNESCO's inscribed properties")
        ax.bar(r + bar_width/2, info2, color=pastel_colors(1), width=bar_width, edgecolor='grey', label='Scientific Papers Count')

        # Display the values on the bars
        for i, (v1, v2) in enumerate(zip(info1, info2)):
            ax.text(r[i] - bar_width/2, v1 + 1, str(v1), ha='center', va='bottom', color='black')
            ax.text(r[i] + bar_width/2, v2 + 1, str(v2), ha='center', va='bottom', color='black')

        # Customize appearance
        # ax.set_xlabel('Countries', fontweight='bold')
        # ax.set_ylabel('Count', fontweight='bold')
        ax.set_xticks(r)
        ax.set_xticklabels(countries)
        ax.set_title("Dsitribution of scientific papers and UNESCO's inscribed properties by country")
        ax.legend()

        # Hide spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Display the chart using Streamlit
        st.pyplot(fig)

       
with c2:
     # Selector for choosing the country
    selected_country = st.selectbox('Select a country:', top_10_merged_data['Country'].unique())

    # Filter data based on selected country
    selected_data = top_10_merged_data[top_10_merged_data['Country'] == selected_country]

    # Calculate the ratio (Documents per Site)
    ratio = round(selected_data['Document Count'] / selected_data['Properties inscribed'],2)

    # Display the ratio
    # st.markdown(f"Documents per inscribed UNESCO property ratio ({selected_country}): **{ratio.values[0]}**")
    st.markdown(f"##### **{ratio.values[0]}** scientific papers per inscribed UNESCO property")




c1, c2 = st.columns((7, 3))
with c1:
    fig3 = px.treemap(filtered_sum, path = ["Country"], values = "Document Count",hover_data = ["Document Count"],color = "Country")
    fig3.update_layout(width = 800, height = 650)
    fig3.update_traces(textinfo='label+text+value', selector=dict(type='treemap'),textfont=dict(size=15))
    # Add " documents" text after the document count
    fig3.update_traces(texttemplate='<b>%{label}</b>'+ '<br><b>%{value}</b> papers', selector=dict(type='treemap'))
    fig3.update_traces(marker=dict(cornerradius=5))
    st.plotly_chart(fig3, use_container_width=True)     


# first row
c1, c2 = st.columns(( 7,3 ))


with c1:
    def generate_wordcloud(topic_words, topic_index):
        # Set up a 2x2 grid for subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Flatten the axs array for easy iteration
        axs = axs.flatten()

        # Generate a wordcloud for each of the first four topics
        for i, ax in enumerate(axs):
            if i < len(topics):
                topic_words = [word for word, _ in lda_model.show_topic(topics[i][0])]
                text = ' '.join(topic_words)
                wordcloud = WordCloud(width=400, height=200, background_color='white').generate(text)
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.set_title(f"Topic {i + 1}")
            ax.axis('off')
        
        # Adjust layout for better spacing
        plt.tight_layout()
        st.pyplot(fig)


    # Streamlit app
    
    # Streamlit app
    selected_file = st.selectbox("Select country", unique_countries_lst)
    lda_model = lda_unique_country(selected_file)
    topics = lda_model.show_topics(formatted=False)  # Adjust the number of words as needed
    
    # Generate and display wordclouds
    generate_wordcloud(lda_model, topics)



with c2:
     st.write("") 



#############################################################################################################################


# Sample data: Country names and some dummy data
data = {
    'country': ['USA', 'Canada', 'Brazil', 'UK', 'Germany', 'China', 'India'],
    'info': ['Info about USA', 'Info about Canada', 'Info about Brazil',
             'Info about UK', 'Info about Germany', 'Info about China', 'Info about India']
}

df = pd.DataFrame(data)

# Streamlit layout with columns
c1, c2 = st.columns((1, 9))

with c1:
    st.write("")  # Empty column
with c2:
    # Creating the world map
    fig = px.choropleth(df, locations="country", locationmode="country names",
                        hover_name="country", hover_data=["info"],
                        color="country",  # Assigns a unique color to each country
                        projection="natural earth")
    # Increase the size of the figure
    fig.update_layout(width=1200, height=900)
    
    # Increase font size for subtitles and annotations
    fig.update_layout(
        title_font_size=20,
        font=dict(size=18)
    )

    st.plotly_chart(fig, use_container_width=False)  # Set use_container_width to False
