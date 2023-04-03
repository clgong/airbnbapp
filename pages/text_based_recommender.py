


"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# set pages
st.set_page_config(
    page_title = 'Multippage App',
    page_icon='👋'
)

st.title('AirBnb Rentals in Seattle')
st.write('Hello from team-spirit :)')
st.write('hello world from Xinqian again')

# header
st.header('Try the customized recommender in UI')

#make a input box
defult_input = "I want a private room close to uw campus with parking and coffee shop."
input_query = st.text_input('Descripe the rental you like here',defult_input)
submit = st.button('Submit')

# TODO: run the code below after click the submit button instead of running the code automatically
################################################################
# code to get top 5 recommendations

# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore',category=DeprecationWarning)
pd.set_option('display.max_columns', 100)
RANDOM_STATE= 42

from collections import Counter
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk import word_tokenize, pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re

# get data
@st.cache_data
def get_data():
    # load dataset
    raw_df = pd.read_pickle('data/cleaned_v2/cleaned_listing_and_review_with_polarity.zip')
    # st.write(raw_df.shape)

    # fill non comments with empty
    df = raw_df.copy()
    df['comments'].fillna(' ', inplace=True)

    # select cols
    listing_cols = ['listing_id','listing_url','review_scores_rating', 'polarity','comments'] 
    content_cols = ['listing_name', 'description', 
                    'host_name', 'host_location', 'host_about',
                    'host_response_time', 'host_neighbourhood',
                    'host_verifications', 'neighbourhood_cleansed',
                    'neighbourhood_group_cleansed','neighborhood_overview',
                    'property_type', 'room_type','amenities',
    #                'comments',
                ]
    # combine all text together
    df['content'] = df.loc[:,content_cols].apply(lambda x: ' '.join(x), axis=1)

    # get final df
    df = df.loc[:,[*listing_cols, *content_cols,'content']]

    # use full data to run
    df_rec = df.copy()

    return df_rec

df_rec = get_data()
# st.write(df_rec.head(2))


#preprocess data  # take a long time to run
added_stopwords = ["can't",'t', 'us', 'say','would', 'also','within','stay', 'since']
nltk_STOPWORDS = set(stopwords.words("english"))
nltk_STOPWORDS.update(added_stopwords)
@st.cache_data
def preprocess_text(text, stopwords = nltk_STOPWORDS, stem=False, lemma=False):
    # clean the text
    text = text.lower()
    # remove html and all other sybols
    text = re.sub("(<.*?>)|([^0-9A-Za-z \t])","",text)
    text = re.sub("(br)", '', text)
    # tokenize the text
    text = word_tokenize(text)
    # remove stopwords and non alpha words
    text = [word for word in text if word not in stopwords]
    # get the root of word
    if stem == True:
        stemmer = PorterStemmer()
        text = [stemmer.stem(word) for word in text]
    # normalize the word
    if lemma == True:
        lemmatizer = WordNetLemmatizer()
        text = [lemmatizer.lemmatize(word) for word in text]
    # list to string
    text = ' '.join(text)

    return text

# get cleaned content
df_rec['cleaned_content'] = df_rec['content'].apply(lambda x: preprocess_text(x,stopwords = nltk_STOPWORDS,stem=False, lemma=True))
# st.write(df_rec.head(2))


# Vectorize the data¶
def vectorize_data(corpus):
    # TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(
                                    ngram_range = (1,2),
                                        min_df=2,
                                        max_df=0.9,
                                    stop_words='english')
    # update: use todense() and np.asarray to avoid error in streamlit app
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus).todense()
    tfidf_matrix = np.asarray(tfidf_matrix)

    return tfidf_vectorizer, tfidf_matrix

# get corpus
corpus = df_rec['content'].values
tfidf_vectorizer, tfidf_matrix = vectorize_data(corpus)
# st.write(tfidf_matrix.shape)

# Build content based recommender system
@st.cache_data
def extract_best_indices(similarity, top_n, mask=None):
    """
    Use sum of the cosine distance over all tokens ans return best mathes.
    m (np.array): cos matrix of shape (nb_in_tokens, nb_dict_tokens)
    topk (int): number of indices to return (from high to lowest in order)
    """
    # return the sum on all tokens of consine for the input query
    if len(similarity.shape) > 1:
        cos_sim = np.mean(similarity, axis=0)
    else:
        cos_sim = similarity
    index = np.argsort(cos_sim)[::-1]

    mask = np.ones(len(cos_sim))
    mask = np.logical_or(cos_sim[index] != 0, mask) #eliminate 0 cosine distance
    best_index = index[mask][:top_n]
    return best_index

@st.cache_data
def get_recommendations(df, input_query, tfidf_matrix, n=5):
    # embed input query
    tokens = preprocess_text(input_query,stopwords = nltk_STOPWORDS, stem=False, lemma=True).split()
    query_vector = tfidf_vectorizer.transform(tokens)

    # get similarity
    similarity = cosine_similarity(query_vector, tfidf_matrix)

    # best cosine distance for each token independantly
    best_index = extract_best_indices(similarity, top_n=n)

    # return the top n similar listing ids and raw comments
    result_df = df.loc[best_index, ['listing_id', 'listing_url', 'listing_name', 'description',
                                    'room_type','property_type',
                                 'neighborhood_overview', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed',
                                 'host_about', 'amenities', 'comments', 'review_scores_rating']]
    return result_df

# Try the recommender system
recomended_listings = get_recommendations(df_rec, input_query, tfidf_matrix, n=5)
st.write(recomended_listings)




################################################################
##### add review sentiment plot for the recommended listings #####


import altair as alt
@st.cache_data
def get_review_data():
    # directly read the saved cleaned_review_with_polarity dataset
    review_df = pd.read_pickle('data/cleaned_v2/cleaned_review_with_polarity.zip')
    # print(review_df.shape)
    # review_df.head(2)
    return review_df
review_df = get_review_data()

# make plot
# notice: altair can only take <=5000 rows, so cannot show all listings at once
@st.cache_data
def plot_listing_sentiment_over_time(df,listing_id = None):
    sub_df = df[df['listing_id'].isin(listing_id)]

    plot = alt.Chart(sub_df, width=500).mark_line().encode(
                x='year(date):T',
                y='mean(polarity)',
                color=alt.Color('listing_id:O', scale=alt.Scale(scheme= 'dark2'))
            ).interactive()
    return plot

# plot the sentiment changes over time by year for the recommended listings
rec_listing_ids = recomended_listings['listing_id'].values

sentiment_plot = plot_listing_sentiment_over_time(review_df, rec_listing_ids)
# print(recs_listing_ids)

# write a note
st.write('Review sentiment trends')


# plot the figure 
st.altair_chart(sentiment_plot, use_container_width=True)



################################################################
##### add wordcloud for the recommended listings #####
#### below only use the first recomendation as example###



# # generate wordcloud for a recommended listing (has comments)
# top_1_recommended_listing = recomended_listings['listing_id'].values[0]
# wordcloud_STOPWORDS = STOPWORDS
# make_wordcloud(df_rec,'cleaned_content', top_1_recommended_listing, wordcloud_STOPWORDS, mask=None)


@st.cache_data
def make_wordcloud(df, col, listing_id, stop_words, mask=None):
    
    if listing_id in df['listing_id'].values:
        text = df[df['listing_id'] == listing_id][col].values[0]
        wordcloud = WordCloud(width = 100, 
                              height = 100, 
                              stopwords=stop_words, 
                              scale=10, 
                              colormap = 'PuRd', 
                              background_color ='black',
#                               mask = None,
                            #   max_words=100,
                             ).generate(text)

        fig, ax = plt.subplots(figsize=(4,4))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        plt.show()
        st.pyplot(fig)
    else:
        print('Oops, this listing currently has no comments.') 
        st.write('Oops, this listing currently has no comments.')

# # generate wordcloud for a recommended listing (has comments)
top_1_recommended_listing = recomended_listings['listing_id'].values[0]
wordcloud_STOPWORDS = STOPWORDS
# make_wordcloud(df_model,'comments', top_1_recommended_listing, wordcloud_STOPWORDS, mask=None)

ok = st.button("Make Wordcloud for the listing description")
if ok:   
    with st.spinner('Making Wordcloud...'):
        make_wordcloud(df_rec,'cleaned_content', top_1_recommended_listing, wordcloud_STOPWORDS, mask=None)
        # make_wordcloud(text, stopwords_list, "picture/wine_image.jpg")
    st.success('Done!')


# issue
# the wordcould showed fewer words in streamlit than in the deepnote recommender_system_text_content_based
st.write(len(df_rec[df_rec['listing_id'] == top_1_recommended_listing]['cleaned_content'].values[0].split()))
