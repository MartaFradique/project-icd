import pandas as pd
import altair as alt
import altair_viewer

data = pd.read_csv('./icd_marta_ana_scopus_edited.csv')

year_counts = data['Year'].value_counts().reset_index()
year_counts.columns = ['year', 'count']
chart = alt.Chart(year_counts).mark_bar().encode(
    x='year',
    y='count:Q'
)
chart.save('docs_per_year.html')

country_counts = data['Correspondence Address'].value_counts().reset_index()
country_counts.columns = ['country', 'number of articles']
chart = alt.Chart(country_counts).mark_bar().encode(
    y=alt.Y('country:N', sort='-x'),
    x='number of articles:Q'

).transform_window(
    rank='rank(number of articles)',
    sort=[alt.SortField('number of articles', order='descending')]
).transform_filter(
    (alt.datum.rank <= 5)
)
chart.save('docs_per_country.html') 

