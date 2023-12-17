import streamlit as st
import altair as alt
import pandas as pd

data_scopus = pd.read_csv('./icd_marta_ana_scopus_edited.csv');
data_unesco = pd.read_csv('./unesco_heritage_by_country.csv');
data_unesco_sorted = data_unesco.sort_values(by="Properties inscribed", ascending=False)
top_10_data_unesco = data_unesco_sorted.head(10)


st.title('documents per country')
# Create a DataFrame with the count of documents per country
chart_data = pd.DataFrame(data_scopus, columns=['Country'])
filtered_sum = chart_data["Country"].value_counts().reset_index()

filtered_sum.columns = ['Country', 'Document Count']
top_10_data = filtered_sum.head(10)
print(top_10_data)

merged_data = pd.merge(top_10_data, top_10_data_unesco, on="Country", suffixes=('_file1', '_file2'))
print(merged_data)


# Create an Altair bar chart
chart = alt.Chart(top_10_data).mark_bar().encode(
    x=alt.X('Country:N', sort='-y'),  # Sort in descending order
    y='Document Count',
    tooltip=['Country', 'Document Count']
)

# Display the Altair chart using Streamlit
st.altair_chart(chart, use_container_width=True)

# Display the title
st.title('UNESCO Data')

# Create an Altair bar chart
chart_unesco = alt.Chart(top_10_data_unesco).mark_bar().encode(
    x=alt.X('Country', sort='-y'),  # Sort in descending order
    y='Properties inscribed',
    tooltip=['Country', 'Properties inscribed']
)

# Display the Altair chart using Streamlit
st.altair_chart(chart_unesco, use_container_width=True)

