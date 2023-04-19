#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# %load combine_num_text_recommender.py

"""
# My first app
Here's our first attempt at using data to create a table:
"""


# import libraries
import streamlit as st
import json
import math
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import sparse
from scipy.stats import kurtosis, skew
import altair as alt
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore',category=DeprecationWarning)
pd.set_option('display.max_columns', 100)
RANDOM_STATE= 42

from collections import Counter
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from kneed import KneeLocator
from pickle import dump, load

import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

from nltk import word_tokenize, pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer


########################################################################################################
# set pages
st.set_page_config(
    page_title = 'Multippage App',
    page_icon='👋'
)

st.title('AirBnb Rentals in Seattle')

# header
st.header('Try the recommender which combines the text and numeric features')

########################################################################################################
##### filtering listing data
@st.cache_data
def get_data(price_range,num_of_beds,num_of_bedrooms,num_of_bathrooms):

    df = pd.read_pickle('data/data_cleaned/cleaned_listing_finalized_for_streamlit.zip')
    df_cluster = pd.read_pickle('data/data_cleaned/update_cluster.zip')
    df['cluster'] = df_cluster['cluster']


    if len(df.loc[(df['price']>price_range[0])&(df['price']<=price_range[1])])!=0:
        df_filter = df.loc[(df['price']>=price_range[0])&(df['price']<=price_range[1])]
        st.write("Your expected price range:", price_range)

    else:
        df_filter = df
        st.write('There are no listings within your preferred price range.\nYou can try a new price range, or ignore prices and query on other conditions.')

    if len(df_filter.loc[df_filter['beds']==num_of_beds])!=0:
        df_filter = df_filter.loc[df_filter['beds']==num_of_beds]
        st.write("Your expected number of beds:", num_of_beds)

    else:
        df_filter = df_filter
        st.write('There are no listings with {} beds.\nYou can try a new number of beds, or ignore the number of beds and query on other conditions.'.format(num_of_beds))


    if len(df_filter.loc[df_filter['bedrooms']==num_of_bedrooms])!=0:
        df_filter = df_filter.loc[df_filter['bedrooms']==num_of_bedrooms]
        st.write("Your expected number of bedrooms:", num_of_bedrooms)

    else:
        df_filter = df_filter
        st.write('There are no listings with {} bedrooms.\nYou can try a new number of bedrooms, or ignore the number of bedrooms and query on other conditions.'.format(num_of_bedrooms))


    if len(df_filter.loc[df_filter['bathrooms_count']==num_of_bathrooms])!=0:
        df_filter = df_filter.loc[df_filter['bathrooms_count']==num_of_bathrooms]
        st.write("Your expected number of bathrooms:", num_of_bathrooms)

    else:
        df_filter = df_filter
        st.write('There are no listings with {} bathrooms.\nYou can try a new number of bathrooms, or ignore the number of bathrooms and query on other conditions.'.format(num_of_bathrooms))

    return df_filter

# make a price query slider
st.subheader(":green[your query on prices]")
price_range = st.slider("Please choose your preferred price range",
                        value = [50,5000])

# make a num of beds slider
st.subheader(":green[your query on the number of beds]")
st.write("(Note: If you choose 0, it means that you ignore this query.)")
bed_range = range(0,16)
num_of_beds = st.select_slider("Choose your preferred number of beds:",
                               options = bed_range, value = 1)

# make a num of bedrooms slider
st.subheader(":green[your query on the number of bedrooms]")
st.write("(Note: If you choose 0, it means that you ignore this query.)")
room_range = range(0,16)
num_of_bedrooms = st.select_slider("Choose your preferred number of bedrooms:",
                                   options = room_range, value = 1)

# make a num of bathrooms slider
st.subheader(":green[your query on the number of bathrooms]")
st.write("(Note: If you choose -1, it means that you ignore this query.)")
bath_range = range(-1,16)
num_of_bathrooms = st.select_slider("Choose your preferred number of bathrooms:",
                                   options = bath_range, value = 1)

# make an input box
st.subheader(":green[more queries by describing what else you are looking for]")
defult_input = ""
input_query = st.text_input("Please describe the rental you're looking for here ",defult_input)
submit = st.button('Submit')


## dataframe with satisfying the filter queries
filter_df = get_data(price_range,num_of_beds,num_of_bedrooms,num_of_bathrooms)


########################################################################################################
##### Cosine similarity

major_cluster = filter_df['cluster'].value_counts().sort_values(ascending=False).index[0]
cosine_similarity_col = ['host_response_rate', 'host_acceptance_rate',
       'host_is_superhost', 'host_has_profile_pic', 'host_identity_verified',
       'accommodates', 'bedrooms', 'beds', 'price', 'minimum_nights',
       'maximum_nights', 'has_availability', 'number_of_reviews',
       'review_scores_rating', 'review_scores_accuracy',
       'review_scores_cleanliness', 'review_scores_checkin',
       'review_scores_communication', 'review_scores_location',
       'review_scores_value', 'has_license', 'instant_bookable',
       'calculated_host_listings_count',
       'calculated_host_listings_count_entire_homes',
       'calculated_host_listings_count_private_rooms',
       'calculated_host_listings_count_shared_rooms', 'reviews_per_month',
       'bathrooms_count', 'amenities_count', 'host_operate_years', 'polarity']

similarity_df = filter_df.loc[filter_df['cluster']==major_cluster][cosine_similarity_col]
similarity_df = similarity_df.fillna(0)
num_similarity = cosine_similarity(similarity_df)


########################################################################################################
##### build up num-based model

model_columns_all = list(filter_df.columns.values)

ui_display_columns = ['cluster',
                      'listing_id',
                      'listing_url',
                      'listing_name',
                      'price',
                      'beds',
                      'bedrooms',
                      'bathrooms_count',
                      'description',
                      'room_type',
                      'property_type',
                      'neighborhood_overview',
                      'neighbourhood_cleansed',
                      'neighbourhood_group_cleansed',
                      'amenities',
                      'number_of_reviews','review_scores_rating','host_about']

iloc_cols = [model_columns_all.index(x) for x in ui_display_columns]

@st.cache_data
def get_num_recommendations(df, similarity, n, listing_id=None, listing_url=None, query_element=None):

    # convert query into and a similarity matrix row index
    item_index = None
    try:
        if listing_id is not None:
            item_index = df['listing_id'].tolist().index(listing_id)
        elif listing_url is not None:
            item_index = df['listing_url'].tolist().index(listing_url)
        elif query_element is not None:
            item_index = query_element
    except ValueError as error:
        print(error)

    if len(df)>=n:
        # get the top n similar items
        top_idx = np.argsort(similarity[item_index])[::-1][:n]
        result_df = df.iloc[top_idx, iloc_cols]
        # add in similarity score as a column
        top_scores = [similarity[item_index][x] for x in top_idx]
        result_df.insert(loc=2, column='similarity', value=top_scores)

    else:
        # get the top n similar items
        top_idx = np.argsort(similarity[item_index])[::-1]
        result_df = df.iloc[top_idx, iloc_cols]
        # add in similarity score as a column
        top_scores = [similarity[item_index][x] for x in top_idx]
        result_df.insert(loc=0, column='similarity', value=top_scores)

    result_df = result_df.reset_index().iloc[:,1:]
    result_df.index = np.arange(1,len(result_df)+1)

    return result_df



########################################################################################################
##### build up text-based model


##### prepare stopword set
added_stopwords = ["can't",'t', 'us', 'say','would', 'also','within','stay', 'since']
nltk_STOPWORDS = set(stopwords.words("english"))
nltk_STOPWORDS.update(added_stopwords)


##### preprocess input query
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

##### Vectorize data
@st.cache_data
def vectorize_data(corpus):
    # TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(
                                    ngram_range = (1,2),
                                    stop_words='english')
    # update: use todense() and np.asarray to avoid error in streamlit app
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    return tfidf_vectorizer, tfidf_matrix

corpus = filter_df['content'].values
tfidf_vectorizer, tmatrix = vectorize_data(corpus)

##### get similarity
@st.cache_data
def extract_best_indices(similarity, top_n, mask=None):
    """
    Use sum of the cosine distance over all tokens and return best mathes.
    m (np.array): cos matrix of shape (nb_in_tokens, nb_dict_tokens)
    topk (int): number of indices to return (from high to lowest in order)
    """
    # return the sum on all tokens of consine for the input query
    if len(similarity.shape) > 1:
        cos_sim = np.mean(similarity, axis=0)
    else:
        cos_sim = similarity
    index = np.argsort(cos_sim)[::-1]
    if mask is not None:
        assert mask.shape == m.shape
        mask = mask[index]
    else:
        mask = np.ones(len(cos_sim))
    mask = np.logical_or(cos_sim[index] != 0, mask) #eliminate 0 cosine distance
    best_index = index[mask][:top_n]
    return best_index


##### get recommendations
@st.cache_data
def get_text_recommendations(df, input_query, _tfidf_matrix, n=5):

    # embed input query
    tokens = preprocess_text(input_query,stopwords = nltk_STOPWORDS, stem=False, lemma=True).split()
    query_vector = tfidf_vectorizer.transform(tokens)

    # get similarity
    similarity = cosine_similarity(query_vector, _tfidf_matrix)

    # best cosine distance for each token independantly
    best_index = extract_best_indices(similarity, top_n=n)

    # return the top n similar listing ids and raw comments
    result_df = df.loc[best_index,:]
    result_df = result_df.loc[:, ['cluster',
                                  'listing_id',
                                  'listing_url',
                                  'listing_name',
                                  'price',
                                  'beds',
                                  'bedrooms',
                                  'bathrooms_count',
                                  'description',
                                  'room_type',
                                  'property_type',
                                  'neighborhood_overview',
                                  'neighbourhood_cleansed',
                                  'neighbourhood_group_cleansed',
                                  'amenities',
                                  'number_of_reviews','review_scores_rating','host_about']]
    result_df = result_df.reset_index().iloc[:,1:]
    result_df.index = np.arange(1,len(result_df)+1)

    return result_df


########################################################################################################
##### build up the combined model

def get_recommendation(df,input_query,_tfidf_matrix, n):
    if input_query == "":
        rec_df = df.loc[df['cluster']==major_cluster]
        select_listing_id_ = st.selectbox("Choose listing id:", rec_df['listing_id'])
        index = rec_df['listing_id'].tolist().index(select_listing_id_)
        recomended_listings = get_num_recommendations(rec_df, num_similarity, n, listing_id=select_listing_id_)
    else:
        # get corpus
        df = df.reset_index()
        recomended_listings = get_text_recommendations(df, input_query, _tfidf_matrix, n)

    return recomended_listings

st.header(":blue[Top recommendations]")
recomended_listings_update = get_recommendation(filter_df,input_query,tmatrix, 5)
st.write(recomended_listings_update)


########################################################################################################
# add rental review sentiment trends plot for the recommended listing #

@st.cache_data
def get_review_data():
    # directly read the saved cleaned_review_with_polarity_and_topic dataset
    review_df = pd.read_pickle('data/data_cleaned/cleaned_review_with_polarity_and_topic.zip')
    return review_df
review_df = get_review_data()

# make plot
# notice: altair can only take <=5000 rows, so cannot show all listings at once
@st.cache_data
def plot_listing_sentiment_over_time(df,listing_id = None):
    # prepare dataframe
    sub_df = df[df['listing_id'].isin(listing_id)]
    # map recommended rentals to listing_id
    dict_map = dict(zip(listing_id,['1st_rec','2nd_rec','3rd_rec','4th_rec','5th_rec'])) 
    sub_df['recommended_listings'] = sub_df['listing_id'].map(dict_map)

    if sub_df.shape[0] != 0:
        base = alt.Chart(sub_df).encode(
            x='year(date):T',
            y='mean(polarity)',
            color=alt.Color('recommended_listings:O', scale=alt.Scale(scheme= 'dark2')))

        plot = alt.layer(base.mark_line() + base.mark_circle(size=100)
                        ).properties(
                            width = 500,
                            height= 350,
                         ).interactive()

        st.altair_chart(plot, use_container_width=True)
    else:
        st.info("Oops, there is no relevant review data (before Dec 24, 2022), then no graph to show here.")


# write a note
st.header(':blue[Rental review sentiment trends]')
st.caption("(Note: No graph to show if the rental doesn't have any comments.)")

# plot the figure
rec_listing_ids = recomended_listings_update['listing_id'].values
plot_listing_sentiment_over_time(review_df, rec_listing_ids)


########################################################################################################
# add rental description wordcloud #

@st.cache_data
def make_wordcloud(df, col, listing_id, stop_words, mask=None):

    if listing_id in df['listing_id'].values:
        text = df[df['listing_id'] == listing_id][col].values[0]
        if type(text) == str:
            wordcloud = WordCloud(width = 100,
                        height = 100,
                        stopwords=stop_words,
                        scale=10,
                        colormap = 'PuRd',
                        background_color ='black',
#                       mask = None,
                        max_words=100,
                        ).generate(text)

            fig, ax = plt.subplots(figsize=(4,4))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            plt.show()
            st.pyplot(fig)
        else:
            if 'comment' in col:
                st.info('Oops, this listing currently has no comments.')
            else:
                st.info('Oops, this listing currently has no descriptions.')

# generate wordcloud for a recommended listing
st.header(':blue[Rental description WordCloud]')
st.subheader(':green[Pick a top n recommendation for more info]')
st.caption("(Note: the first listing id is the first recommended listing (the most relevant), and the second listing id is the second recommended rental, and so on.)")

# get a top n listing id from the user
selected_listing_id = st.selectbox("Choose wanted listing id:", recomended_listings_update['listing_id'])
index = recomended_listings_update['listing_id'].tolist().index(selected_listing_id)
link = recomended_listings_update.listing_url.tolist()[index]

# Show name and URL of selected property
st.write("\"{}\" - [{}]({})".format(recomended_listings_update.listing_name.tolist()[index],link,link))

# Draw the word cloud
wordcloud_STOPWORDS = STOPWORDS
make_wordcloud(filter_df,'cleaned_content', selected_listing_id, wordcloud_STOPWORDS, mask=None)

#########################################################################################################
# add rental review wordcloud #

# generate review wordcloud
st.header(':blue[Rental review WordCloud]')

# Show name and URL of selected property
st.write("\"{}\" - [{}]({})".format(recomended_listings_update.listing_name.tolist()[index],link,link))

# Draw the word cloud
make_wordcloud(filter_df,'comments_nouns_adjs', selected_listing_id, wordcloud_STOPWORDS, mask=None)


########################################################################################################
##### add rental review topic bar chart #####

# make review topic bar chart
@st.cache_data
def plot_listing_review_topics(df, col, listing_id):

    if listing_id in df['listing_id'].values:
        sub_df = df[df['listing_id'] == listing_id]
        plot = alt.Chart(sub_df).mark_bar().encode(
                        y=alt.Y('review_topic_interpreted:O', sort='-x', axis=alt.Axis(labelLimit=200,labelFontSize=14)),
                        x='count():Q',
                        color=alt.Color('listing_id:O', scale=alt.Scale(scheme= 'dark2'))
                    ).properties(
                        width=400,
                        height=300
                    ).interactive()
        st.altair_chart(plot, use_container_width=True)
    else:
        st.info("Oops, this listing currently has no comments.")

# generate review report for a recommended listing (has comments)
st.header(':blue[Rental review topics BarChart]')

# Show name and URL of selected property
st.write("\"{}\" - [{}]({})".format(recomended_listings_update.listing_name.tolist()[index],link,link))

# Draw the topic bar chart
plot_listing_review_topics(review_df,'review_topic_interpreted', selected_listing_id)


########################################################################################################
# add rental review report (negative review sentences) #

@st.cache_data
def get_review_sentiment_report(df,col,listing_id):
    sorted_neg_sentences = np.nan
    if listing_id in df['listing_id'].values:
        comments = df[df['listing_id'] == listing_id]['comments'].values[0]
        if len(comments) <=1:
            st.info('Oops, this listing currently has no comments.')
        else:
            # segement all comments into sentences for the given listing
            review_sentences = df[df['listing_id'] == listing_id]['comments'].apply(lambda x: re.sub("(<.*?>)|([\t\r])","",x)).str.split('[.?!]').values.tolist()[0]
            review_sentences = [sentence for sentence in review_sentences if sentence]
            num_review_sentences = len(review_sentences)

            # get polarity score for the positives sentences in all the comments
            neg_sentences = []
            for i, text in enumerate(review_sentences):
                sid_obj = SentimentIntensityAnalyzer().polarity_scores(text)
                if (sid_obj['neg'] > 0.2) & (sid_obj['compound'] < -0.1):
                    neg_sentences.append((sid_obj['compound'],sid_obj['neg'],review_sentences[i]))
                else:
                    pass

            neg_percent = (len(neg_sentences)/num_review_sentences)
            non_neg_percent = 1-neg_percent
            # sort the negative sentences first by 'compund' score (ascending order) then by 'neg' score (descending order)
            sorted_neg_sentences = [comment for score1, score2, comment in sorted(neg_sentences, key=lambda x: (x[0], -x[1]))]

            st.subheader(":green[Overall:]")
            st.write("{}% of all the reviews sentences ({}/{}) on Airbnb for this listing are not negative!".format(round(non_neg_percent*100,2), num_review_sentences-len(neg_sentences),num_review_sentences))
            st.write("{}% of them ({}/{}) are estimated to be negative review sentences.".format(round(neg_percent*100,2),len(neg_sentences),num_review_sentences))

            st.subheader(":green[Potentially helpful negative review sentences:]")
            st.caption("(Note: some of the negative review sentences might not sound negative to you due to the prediction algorithm used in our model. However, those review sentences are already the most negative ones in all the comments about this rental.)")

            if len(sorted_neg_sentences) >0:
                for i, sentence in enumerate(sorted_neg_sentences):
                    st.write("{}: {}".format(i+1, sentence))
            else:
                st.write("Wow, this listing currently doesn't have any estimated negative sentences!")
        return sorted_neg_sentences


# generate review report for a recommended listing (has comments)
st.header(':blue[Rental review report]')

# Show name and URL of selected property
st.write("\"{}\" - [{}]({})".format(recomended_listings_update.listing_name.tolist()[index],link,link))

# Return the report
sorted_neg_sentences = get_review_sentiment_report(filter_df,'comments',selected_listing_id)
