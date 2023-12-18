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
data_scopus['Cited by'] = pd.to_numeric(data_scopus['Cited by'], errors='coerce')
# Group by 'Country' and sum the 'Citations' for each group
citations_country = data_scopus.groupby('Country')['Cited by'].sum().reset_index(name='TotalCitations')



st.set_page_config(layout='wide', initial_sidebar_state='expanded')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("Recommender Systems in Cultural Heritage Sites")

c1, c2 = st.columns((8, 2))

with c1:
        countries = top_10_merged_data['Country']
        info1 = top_10_merged_data['Properties inscribed']  # Data for the first type of information
        info2 =  top_10_merged_data['Document Count']  # Data for the second type of information

        # Set width of bar
        bar_width = 0.32

        # Set position of bars on X axis
        r = np.arange(len(countries))

        # Create the bar plot
        fig1, ax = plt.subplots(figsize=(10, 5))  # Set figure size
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
        ax.set_title("Distribution of scientific papers and UNESCO's inscribed properties by country")
        ax.legend()

        # Hide spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Display the chart using Streamlit
        st.pyplot(fig1)

       
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




c1, c2 = st.columns((8, 2))
with c2:
    on = st.toggle('Alternate view')   
with c1:
    if on:
        fig3 = px.treemap(filtered_sum, path = ["Country"], values = "Document Count",hover_data = ["Document Count"],color = "Country")
        fig3.update_layout(width = 800, height = 650)
        fig3.update_traces(textinfo='label+text+value', selector=dict(type='treemap'),textfont=dict(size=15))
        # Add " documents" text after the document count
        fig3.update_traces(texttemplate='<b>%{label}</b>'+ '<br><b>%{value}</b> papers', selector=dict(type='treemap'))
        fig3.update_traces(marker=dict(cornerradius=5))
        st.plotly_chart(fig3, use_container_width=True) 
    else:
        fig2 = px.choropleth(filtered_sum, locations="Country", locationmode="country names",
                        hover_name="Country", hover_data=["Document Count"],
                        labels={'Document Count':'Number of Papers'},
                        color="Document Count",  # Assigns a unique color to each country
                        color_continuous_scale="Reds",
                        projection="natural earth")
        # Increase the size of the figure
        fig2.update_layout(width=1000, height=700)
        
        # Increase font size for subtitles and annotations
        fig2.update_layout(
            font=dict(size=18)
        )
        st.plotly_chart(fig2, use_container_width=False)  # Set use_container_width to False

         

# first row
c1, c2 = st.columns(( 8,2 ))


with c1:
    def generate_wordcloud(data):
        text = ' '.join(data)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)


    # Streamlit app
    
    # Dropdown menu to select file
    selected_file = st.selectbox("Select country", unique_countries_lst)
    lda_model = lda_unique_country(selected_file)
    topics = lda_model.show_topics()  # Adjust the number of words as needed
    topic_words = []
    for topic in topics:
        words = [word for word, _ in lda_model.show_topic(topic[0])]  # Extract words for the current topic
        topic_words.extend(words)

    

    generate_wordcloud(topic_words)



with c2:
     st.write("") 



#############################################################################################################################

# Streamlit layout with columns
# c1, c2 = st.columns((1, 9))

# with c1:
#     st.write("")  # Empty column
# with c2:
#     st.write("")  # Empty column

    # Creating the world map
    # fig = px.choropleth(citations_country, locations="Country", locationmode="country names",
    #                     hover_name="Country", hover_data=["TotalCitations"],
    #                     color="TotalCitations",  # Assigns a unique color to each country
    #                     color_continuous_scale="Reds",
    #                     projection="natural earth")
    # # Increase the size of the figure
    # fig.update_layout(width=1000, height=800)
    
    # # Increase font size for subtitles and annotations
    # fig.update_layout(
    #     font=dict(size=18)
    # )

    # st.plotly_chart(fig, use_container_width=False)  # Set use_container_width to False
    
