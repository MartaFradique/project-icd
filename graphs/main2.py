# %% [markdown]
# # ICD Project

# %% [markdown]
# ## Importing libraries

# %%
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
##from wordcloud import WordCloud
import contractions
import nltk
from collections import Counter
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')




# %% [markdown]
# ## Load Dataset

# %%
data = pd.read_csv('./icd_marta_ana_scopus_edited.csv')

# %%
def unique_countries():
    return data['Country'].unique()

# %% [markdown]
# ## **1.Exploratory data analysis**
# 

# %% [markdown]
# - Exploratory Data Analysis (EDA) typically involves examining and visualizing various aspects of your dataset to gain insights into its structure and characteristics.
# - Data exploration is a crucial step in any NLP project. It helps us understand the characteristics of the text data we are working with. 

# %%
# Display the first few rows of the DataFrame
data.head()

# %% [markdown]
# ### Checking for duplicate values

# %%
data.duplicated().sum()

# %% [markdown]
# since the value of duplicate values(rows) is zero is not necessary to do anything

# %% [markdown]
#  
# ### Words Frequency Distribution
# 
# The code snippet counts the frequency of each word in the given sample text and plots the top 10 most frequent words
# 

# %%

# Tokenize the text into words for each row in the 'text_column'
data['tokenized_text'] = data['Title'].apply(lambda x: word_tokenize(str(x).lower()))

# Flatten the list of tokenized words for all rows
all_words = [word for tokens in data['tokenized_text'] for word in tokens]

# Count the frequency of each word
word_freq = Counter(all_words)

# Prepare data for plotting
labels, values = zip(*word_freq.items())

# Sort the values in descending order
indSort = sorted(range(len(values)), key=lambda k: values[k], reverse=True)

# Rearrange the data
labels = [labels[i] for i in indSort]
values = [values[i] for i in indSort]

# Create the plot
plt.figure(figsize=(10, 5))
plt.bar(labels[:10], values[:10])
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 10 Most Frequent Words')
plt.xticks(rotation=45)
plt.show()



# %% [markdown]
# Since the top 10 most frequent words are stop words, and we can't retrieve valuable information from it they must be deleted.

# %%
# Concatenate all text data from all columns
all_text = ' '.join(data.astype(str).values.flatten())

# %% [markdown]
# ### Character Frequency Distribution
# 
# Visualizing the data can provide additional insights. Let's create a simple word frequency distribution

# %%

# Calculate the average word length
total_characters = sum(len(word) for word in all_text.split())
total_words = len(all_text.split())

# Count the frequency of each character
char_freq = Counter(all_text)

# Prepare data for plotting
char_labels, char_values = zip(*char_freq.items())

# Create the plot with a specified font
plt.figure(figsize=(25, 5))
plt.bar(char_labels, char_values)
plt.xlabel('Characters')
plt.ylabel('Frequency')
plt.title('Character Frequency Distribution')
plt.show()



# %% [markdown]
# With this graph we verified the existence of special characters and pontuation

# %% [markdown]
# ### Number of Stopwords
# The code snippet calculates the number of stopwords in the given sample text. Stopwords are commonly used words that are generally ignored in text data analysis.
# 

# %%


# Get stopwords for the English language
stop_words = set(stopwords.words('english'))

# Count the number of stopwords in the entire dataset
num_stopwords = len([word for word in all_text.lower().split() if word in stop_words])

# Print or use the result as needed
print(f"Number of stopwords in the entire dataset: {num_stopwords}")

# %% [markdown]
# ### Number of Special Characters
# The code snippet calculates the number of special characters (like punctuation marks) in the given sample text.

# %%
import string


# Count the number of special characters in the entire dataset
num_special_characters = len([char for char in all_text if char in string.punctuation])

# Print or use the result as needed
print(f"Number of special characters in the entire dataset: {num_special_characters}")

# %% [markdown]
# ### Number of Uppercase Words
# The code snippet calculates the number of words that are entirely in uppercase in the given sample text.

# %%

# Count the number of uppercase words in the entire dataset
num_uppercase_words = len([word for word in all_text.split() if word.isupper()])

# Print or use the result as needed
print(f"Number of uppercase words in the entire dataset: {num_uppercase_words}")

# %% [markdown]
# ### Average Word Length
# The code snippet calculates the average length of words in the given sample text.
# 

# %%

# Function to calculate average word length for a given column
def calculate_avg_word_length(column_data):
    total_characters = sum(len(word) for text in column_data.astype(str) for word in text.split())
    total_words = sum(len(text.split()) for text in column_data.astype(str))
    return total_characters / total_words if total_words != 0 else 0

# Calculate average word length for each column
avg_word_lengths = {}

for column in data.columns:
    if data[column].dtype == 'O':  # Check if the column contains object (text) data
        avg_word_lengths[column] = calculate_avg_word_length(data[column])

# Print or use the results as needed
for column, avg_length in avg_word_lengths.items():
    print(f"Average word length for {column}: {avg_length}")

# %% [markdown]
# ### Word cloud 

# %% [markdown]
# 

# %%
from wordcloud import WordCloud

# Concatenate all text data from all columns
all_text = ' '.join(data.astype(str).values.flatten())

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud')
plt.show()

# %% [markdown]
# In here we can see the that there are a lot of null numbers, and it's also possible to see the most frequent words

# %% [markdown]
# ### Cheking for null values
# The code snippet counts the frequency of each word in the given sample text and plots the top 10 most frequent words

# %%
# Checking for missing values
data.isna().sum()

# %% [markdown]
# ## **2.Noise Cleaning** 

# %% [markdown]
# ### Remove useless columns in the context and columns with NaN values
# Deleted the columns that don't have quality information for the analysis, and only bring in noise

# %%
columns_to_drop = ['Author full names', 'Volume', 'Issue', 'Art. No.', 'Page start', 'Page end', 'Page count', 'Cited by', 'DOI', 'Link', 'Affiliations', 'Authors with affiliations', 'Sponsors', 'Molecular Sequence Numbers', 'Chemicals/CAS', 'Tradenames', 'Manufacturers', 'ISBN', 'CODEN', 'PubMed ID']

data = data.drop(columns=columns_to_drop)
data.head(4)

# %% [markdown]
# ## **2.1. Preprocess the dataset**
# 

# %%


# Function to perform text preprocessing
def preprocess_text(text):
    # Check if the input is a string
    if isinstance(text, str):
        # 1. Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = re.sub(r'º', '', text)  # Remove the "º" character
        text = ' '.join(text.split())  # Remove extra spaces

        # 2. Convert the text to lowercase
        text = text.lower()
        
        # 4. Tokenization (dividing the text to a list of words ex: 'a ana e linda ' -> ['a', 'ana', 'e' ,'linda'] )
        words = word_tokenize(text)

        
        # 5. Stopword Removal (remove very common words like 'the', 'a' etc.)
        stop_words = set(stopwords.words('english'))
        # In here he is removing the stop words from the text 
        words = [word for word in words if word.lower() not in stop_words]

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        

        
        # 5. Stemming(reduse the words to their root form ex 'running' -> 'run')
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
        
        # Join the processed words back into a sentence
        processed_text = ' '.join(words)
        
        return processed_text
    else:
        # Return an empty string for non-string values
        return ''
    

#create a list with all the text cloumns 
text_columns = data.select_dtypes(include='object').columns.tolist()

# Apply text preprocessing to specified text columns
for col in text_columns:
    data[col] = data[col].apply(preprocess_text)
# Drop duplicate rows based on the entire row
data = data.drop_duplicates()

# Display the updated DataFrame
data.head()


# %% [markdown]
# ## download the file after preprocessing

# %%
import csv

file_path = 'preprocessed.csv'

with open(file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(data)

print(f'Data has been saved to {file_path}')

# %% [markdown]
# ## Word cloud 2

# %%
from wordcloud import WordCloud

# Concatenate all text data from all columns
all_text = ' '.join(data.astype(str).values.flatten())

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud')
plt.show()

# %% [markdown]
# ### Top 10 most frequent words after cleaning the data

# %%
# Tokenize the text into words for each row in the 'text_column'
data['tokenized_text'] = data['Title'].apply(lambda x: word_tokenize(str(x).lower()))

# Flatten the list of tokenized words for all rows
all_words = [word for tokens in data['tokenized_text'] for word in tokens]

# Count the frequency of each word
word_freq = Counter(all_words)

# Prepare data for plotting
labels, values = zip(*word_freq.items())

# Sort the values in descending order
indSort = sorted(range(len(values)), key=lambda k: values[k], reverse=True)

# Rearrange the data
labels = [labels[i] for i in indSort]
values = [values[i] for i in indSort]

# Create the plot
plt.figure(figsize=(10, 5))
plt.bar(labels[:10], values[:10])
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 10 Most Frequent Words')
plt.xticks(rotation=45)
plt.show()


# %% [markdown]
# ## **4.  Representative text**

# %% [markdown]
# ## Bag of Words
# 

# %%
column_to_vectorize = 'Title'

# Extract text data from the specified column
text_data = data[column_to_vectorize].tolist()

# Initialize the CountVectorizer with 1-grams
vectorizer = CountVectorizer(ngram_range=(1, 1))

# Fit and transform the text data
X = vectorizer.fit_transform(text_data)

# Convert the result to a DataFrame for better visualization
vectorized_data = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
print("Bag of Words:", X.toarray())
# Display the vectorized data
print(vectorized_data.head(4))




# %%
# Generate the heatmap
#plt.figure(figsize=(20, 1))  # Adjust the figure size
#sns.heatmap(vectorized_data, annot=True, cmap="YlGnBu", cbar=False)
#
#plt.show()

# %% [markdown]
# ## Term Frequency 

# %% [markdown]
# Term Frequency measures how often a word appears in a document.

# %%
from collections import Counter
import pandas as pd

# Assuming 'Title' is the column name in your dataset
title_column = data['Title'].astype(str)

# Tokenize each document in the 'Title' column
tokenized_documents = [doc.split() for doc in title_column]

# Tokenize the first document in the 'Title' column
text_tokens = title_column.iloc[0].split()

# Count the frequency of each word
word_counts = Counter(text_tokens)

# Calculate the total number of words
total_words = len(text_tokens)

# Calculate Term Frequency (TF) for each word
term_frequency = {word: count / total_words for word, count in word_counts.items()}

print(f"Term Frequency: {term_frequency}")



# %% [markdown]
# ## Inverse Document Frequency (IDF)

# %% [markdown]
# ### Inverse Document Frequency (IDF)
# IDF measures how important a term is across multiple documents. It's often used in tandem with TF.

# %%
import math

# Assuming 'Title' is the column name in your dataset
documents = data['Title'].astype(str).tolist()

# Convert all documents to lowercase
documents = [doc.lower() for doc in documents]

# Tokenize and count words for each document
tf_values = [Counter(doc.split()) for doc in documents]

# Calculate IDF values
all_words = set(word for doc in tf_values for word in doc.keys())
idf_values = {}
N = len(documents)

for word in all_words:
    df = sum(word in doc for doc in tf_values)
    idf_values[word] = math.log(N / df)

# Convert to DataFrame for better visualization
idf_df = pd.DataFrame(list(idf_values.items()), columns=['Word', 'IDF Value'])

# Sort the DataFrame by IDF Value in descending order for better visualization
idf_df = idf_df.sort_values('IDF Value', ascending=False)

# Display the table
print(idf_df)

# %%
import math
from collections import Counter
import pandas as pd

# Assuming 'Title' is the column name in your dataset
documents = data['Title'].astype(str).tolist()

# Convert all documents to lowercase
documents = [doc.lower() for doc in documents]

# Tokenize and count words for each document
tf_values = [Counter(doc.split()) for doc in documents]

# Calculate IDF values
all_words = set(word for doc in tf_values for word in doc.keys())
idf_values = {}
N = len(documents)

for word in all_words:
    df = sum(word in doc for doc in tf_values)
    idf_values[word] = math.log(N / df)

# Calculate TF-IDF values
tf_idf_values = []

for tf in tf_values:
    tf_idf = {}
    for word, count in tf.items():
        tf_idf[word] = count * idf_values[word]
    tf_idf_values.append(tf_idf)

# Convert to DataFrame for better visualization
tf_idf_df = pd.DataFrame(tf_idf_values)

# Replace NaN with 0
tf_idf_df.fillna(0, inplace=True)

print(tf_idf_df)


# %% [markdown]
# **IDF Frequency Distribution**
# 
# The code snippet plots the frequency of each word in the given sample text

# %%
# Using the DataFrame from the previous example (tf_idf_df)
#plt.figure(figsize=(12, 8))
#sns.heatmap(tf_idf_df, annot=True, cmap="YlGnBu", cbar=True)
#plt.title('TF-IDF Heatmap')
#plt.xlabel('Words')
#plt.ylabel('Documents')
#plt.show()

# %% [markdown]
# ### 4. Term Frequency-Inverse Document Frequency (TF-IDF)
# TF-IDF is a combination of TF and IDF, often used for text mining tasks.
# 

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Assuming 'Title' is the column name in your dataset
documents = data['Title'].astype(str).tolist()

# Create a TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the documents
tfidf_matrix = vectorizer.fit_transform(documents)

# Convert the TF-IDF matrix to a dense array for better visualization
dense_tfidf_matrix = tfidf_matrix.toarray()

# Convert to DataFrame for better visualization
tfidf_df = pd.DataFrame(dense_tfidf_matrix, columns=vectorizer.get_feature_names_out())

print("TF-IDF Matrix:")
print(tfidf_df)

# %% [markdown]
# Visualization of the TF-IDF

# %%
#from sklearn.feature_extraction.text import TfidfVectorizer
#
## Assume 'documents' is your list of text documents
#vectorizer = TfidfVectorizer()
#tfidf_matrix = vectorizer.fit_transform(documents)
#
## Convert to DataFrame
#df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
#
## Generate the heatmap
#plt.figure(figsize=(20, 5))  # Adjust the figure size
#sns.heatmap(df_tfidf, annot=True, cmap="YlGnBu", cbar=True)
#
#plt.show()



# %% [markdown]
# ### 6. Sentiment Analysis
# Sentiment Analysis aims to determine the attitude or emotion of the writer.

# %%
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd

# Assuming 'Title' is the column name in your dataset
documents = data['Title'].astype(str).tolist()

# Download the VADER lexicon file
nltk.download('vader_lexicon')

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Initialize lists to store sentiment components
compound_scores = []
positive_scores = []
neutral_scores = []
negative_scores = []

# Analyze sentiment for each document in the list
for doc in documents:
    sentiment = sia.polarity_scores(doc)
    compound_scores.append(sentiment['compound'])
    positive_scores.append(sentiment['pos'])
    neutral_scores.append(sentiment['neu'])
    negative_scores.append(sentiment['neg'])

# Create the plot
labels = [f"Doc {i+1}" for i in range(len(documents))]
x = range(len(labels))

plt.figure(figsize=(15, 7))

plt.bar(x, compound_scores, width=0.2, label='Compound', color='b', align='center')
plt.bar(x, positive_scores, width=0.2, label='Positive', color='g', align='edge')
plt.bar(x, neutral_scores, width=0.2, label='Neutral', color='y', align='edge')
plt.bar(x, negative_scores, width=0.2, label='Negative', color='r', align='edge')

plt.xlabel('Documents')
plt.ylabel('Scores')
plt.title('Sentiment Analysis Scores for Each Document')
plt.xticks(x, labels, rotation='vertical')
plt.legend()
plt.show()


# %% [markdown]
# ### 7. Word Embedding
# Word Embeddings are a type of word representation that captures semantic meanings of words.

# %%
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Assuming 'Title' is the column name in your dataset
documents = data['Title'].astype(str).tolist()

# Tokenize and lowercase each document in the documents list
tokenized_documents = [word_tokenize(doc.lower()) for doc in documents]

# Train a Word2Vec model
model = Word2Vec(tokenized_documents, min_count=1)

# Access the word embedding for the word 'dados'
if 'dados' in model.wv:
    print("Word Embedding for 'dados':", model.wv['dados'])
else:
    print("The word 'dados' is not in the vocabulary.")


# %% [markdown]
# Visualization of the word Embeddings

# %%
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Assuming 'Title' is the column name in your dataset
documents = data['Title'].astype(str).tolist()

# Tokenize and lowercase each document in the documents list
tokenized_documents = [word_tokenize(doc.lower()) for doc in documents]

# Train a Word2Vec model
model = Word2Vec(tokenized_documents, min_count=1)

# Get all the keys (words) from the vocabulary
words = list(model.wv.index_to_key)

# Get the corresponding vectors for each word
word_vectors = [model.wv[word] for word in words]

# Perform PCA to reduce dimensions
pca = PCA(n_components=2)
result = pca.fit_transform(word_vectors)

# Create a scatter plot
plt.figure(figsize=(12, 12))
plt.scatter(result[:, 0], result[:, 1])

# Annotate each point with its corresponding word
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))

plt.title('Word2Vec Embeddings Visualization with PCA')
plt.show()


# %% [markdown]
# ### 8. Part-of-Speech Tagging
# Understanding the distribution of different parts-of-speech (nouns, verbs, adjectives, etc.) can provide insights into the grammatical structure of the text.

# %%
# Perform part-of-speech tagging using nltk
nltk.download('averaged_perceptron_tagger')
pos_tags = nltk.pos_tag(words)

# Count the frequency of each part-of-speech tag
pos_freq = Counter(tag for word, tag in pos_tags)
pos_freq

# %%
# Create the plot
plt.figure(figsize=(10, 5))
plt.bar(pos_freq.keys(), pos_freq.values())
plt.xlabel('Part-of-Speech Tags')
plt.ylabel('Frequency')
plt.title('Part-of-Speech Tag Distribution')
plt.show()

# %% [markdown]
# ## **5.Modelação de Tópicos**

# %% [markdown]
# ### 5.1 Latent Dirichlet Allocation (LDA)

# %%
document = data['Title']
document

# %%
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess

def preprocess_data(df, country):
    
    
    documents = df[df['Country'] == country]['Title']
    texts = [[word for word in simple_preprocess(str(doc))] for doc in documents]
    return texts

from gensim import corpora
from gensim.models.ldamodel import LdaModel
def lda_unique_country(country):
    # Preprocess data for the current country
    processed_texts = preprocess_data(data, country)

    print(f"Number of documents for {country}: {len(processed_texts)}")

    if not processed_texts:
        print(f"No documents for {country}. Skipping.")
        
    else:

        # Create Dictionary
        id2word = corpora.Dictionary(processed_texts)

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in processed_texts]

        # Set number of topics
        num_topics = 10
        lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, passes=50, alpha=1.0, eta=0.001, per_word_topics=True)


        

        # Coherence Score
        #coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_texts, dictionary=id2word, coherence='c_v')
        #coherence_lda = coherence_model_lda.get_coherence()
        

        #vis = gensimvis.prepare(lda_model, corpus, id2word)

        return lda_model

# %%
# Print the keywords for each topic


# %%


# # %%
# from gensim.models.coherencemodel import CoherenceModel
# coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_texts, dictionary=id2word, coherence='c_v')
# coherence_lda = coherence_model_lda.get_coherence()
# print('Coherence Score:', coherence_lda)






