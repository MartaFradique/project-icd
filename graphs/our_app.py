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

# with c3:
#     # Sample data for horizontal bar chart
#     bar_data = pd.DataFrame({
#         'Category': ['Andorra', 'Belgica', 'Canada', 'Dinamarca', 'Equador', 'França', 'Portugal'],
#         'Values': [23, 45, 56, 78, 12, 34, 65]
#     })

#     # Create a horizontal bar chart with thin bars and reduced spacing
#     fig, ax = plt.subplots(figsize=(8, 6))  # Set figure size to make it smaller and match height
#     pastel_colors = plt.cm.get_cmap('Pastel1', len(bar_data))
#     bars = ax.barh(bar_data['Category'], bar_data['Values'], color=pastel_colors.colors, height=0.2)  # Adjust bar height for thin bars

#     # Display the values on top of each bar
#     for i, val in enumerate(bar_data['Values']):
#         ax.text(val, i, str(val), ha='left', va='center', fontsize=10)  # Display values on bars

#     # Customize appearance: Remove background gridlines and spines
#     ax.grid(False)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['left'].set_visible(False)  # Hide the y-axis line

#     # Hide x-axis line and labels
#     ax.xaxis.set_visible(False)

#     # Set y-axis label and title
#     ax.set_title('Horizontal Bar Chart', fontsize=14)

#     # Display the bar chart using Streamlit
#     st.pyplot(fig)
# with c4:
#      st.write("") 
#     # Calculate the ratio (Documents per Site)
#     ratio = round(selected_data['Document Count'] / selected_data['Properties inscribed'],2)

#     # Display the ratio
#     # st.markdown(f"Documents per inscribed UNESCO property ratio ({selected_country}): **{ratio.values[0]}**")
#     st.markdown(f"##### **{ratio.values[0]}** scientific papersz     per inscribed UNESCO property")

#     #############################


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
    # file_mapping = {
    #     "Italy": "icd_marta_ana_scopus_edited.csv",
    #     "China": "scopus3.csv",
    #     "France": "icd_marta_ana_scopus_edited.csv",
    #     "Germany": "icd_marta_ana_scopus_edited.csv",
    #     "Spain": "icd_marta_ana_scopus_edited.csv",
    #     "India": "icd_marta_ana_scopus_edited.csv",
    #     "Mexico": "icd_marta_ana_scopus_edited.csv"
    # }

    # if selected_file in file_mapping:
    #     file_path = file_mapping[selected_file]
    # else:
    #     st.error("Please select a valid file")


    # data = pd.read_csv(file_path)
    # # st.write(data)  # Display the uploaded data

    # text_column = "Title"  # Replace 'YourColumnName' with the actual column containing text data
    # text_data = data[text_column].dropna().tolist()
    # Assuming lda_model is your trained LDA model
    topics = lda_model.show_topics()  # Adjust the number of words as needed
    topic_words = []
    for topic in topics:
        words = [word for word, _ in lda_model.show_topic(topic[0])]  # Extract words for the current topic
        topic_words.extend(words)

    

    generate_wordcloud(topic_words)



with c2:
     st.write("") 
    # # Creating a large white space using an empty placeholder with custom CSS
    # placeholder = st.empty()
    # placeholder.markdown(
    #     '<style>div.css-1l02zno {height: 75px;}</style>',
    #     unsafe_allow_html=True
    # )



#############################################################################################################################


##first row 
# c1, c2 = st.columns((5,5))
# with c1:
#    # Sample data for horizontal bar chart
#     bar_data = pd.DataFrame({
#         'Category': ['Andorra', 'Belgica', 'Canada', 'Dinamarca', 'Equador', 'França', 'Portugal'],
#         'Values': [23, 45, 56, 78, 12, 34, 65]
#     })

#     # Display the bar chart using Streamlit
#     st.pyplot(fig)

 
# with c2:
#      # Sample data for horizontal bar chart
#     bar_data = pd.DataFrame({
#         'Category': ['Andorra', 'Belgica', 'Canada', 'Dinamarca', 'Equador', 'França', 'Portugal'],
#         'Values': [23, 45, 56, 78, 12, 34, 65]
#     })

#     # Create a horizontal bar chart with thin bars and reduced spacing
#     fig, ax = plt.subplots(figsize=(8, 4))  # Set figure size to make it smaller
#     pastel_colors = plt.cm.get_cmap('Pastel1', len(bar_data))
#     bars = ax.barh(bar_data['Category'], bar_data['Values'], color=pastel_colors.colors, height=0.2)  # Adjust bar height for thin bars

#     # Display the values on top of each bar
#     for i, val in enumerate(bar_data['Values']):
#         ax.text(val, i, str(val), ha='left', va='center', fontsize=10)  # Display values on bars

#     # Customize appearance: Remove background gridlines and spines
#     ax.grid(False)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['left'].set_visible(False)  # Hide the y-axis line

#     # Hide x-axis line and labels
#     ax.xaxis.set_visible(False)

#     # Set y-axis label and title
#     ax.set_title('Horizontal Bar Chart', fontsize=14)

#     # Display the bar chart using Streamlit
#     st.pyplot(fig)