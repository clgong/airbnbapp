


"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore',category=DeprecationWarning)

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

RANDOM_STATE= 42

from sklearn.metrics.pairwise import cosine_similarity


st.title('AirBnb Rentals in Seattle')
st.write('Hello from team-spirit :)')
st.write('v0.02')
st.write('hello world from Xinqian again')

# 1. Prepare data

# load dataset that has already been cleaned
raw_df = pd.read_pickle('./data/cleaned_v2/cleaned_listing_and_review_with_polarity.zip')

# Note this will print to the terminal
#print(raw_df.shape)

# Note this will print to the web app
# raw_df.shape


# Continue with rest of notebook...

#raw_df.columns

# Some columns we decided to remove with the reason in comments

feature_to_remove = ['host_total_listings_count','host_listings_count', # same description with different values, use calculated_host_listings_count instead
                    'minimum_minimum_nights', 'maximum_minimum_nights', # those num are from calender
                    'minimum_maximum_nights', 'maximum_maximum_nights', # they're constantly changing
                    'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', # so do not make much sense
                    'has_availability','availability_30',
                    'availability_60', 'availability_90','availability_365',
                    #'reviewer_count', #TODO: remove it, already has it # REMOVED
                    #'host_number_of_year', # TODO: rename it as host_operating_years #RENAMEED
                    ]

df_model = raw_df.drop(columns=feature_to_remove)
# df_model.shape

#df_model.isna().sum()

# check if any columns have Nan...
df_model.columns[df_model.isna().any()].tolist()

# remove na polarity rows, polarity is numeric, goal is to get all numeric columns
df_model = df_model.dropna()
# print(df_model.columns[df_model.isna().any()].tolist(), df_model.shape)
#df_model.head(5)

# Reset index so easier to debug later

df_model.reset_index(inplace=True, drop=True)
# df_model.head(5)

# TODO: Add in categorical values using one hot encoding
# TODO: Should we also add in datetime types too since they are kind of numeric?
# TODO: Should we also add in clustering results as a column?
#   If we decide to add in the cluster id, would be great if we had a
#   cleaned data set that had the cluster id once the clustering research
#   is complete

# Take all the numerical columns as features for our model
#df_model.shape
#df_model.dtypes

number_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df_model_num = df_model.select_dtypes(include=number_types)
#df_model_num.shape

#df_model_num.dtypes


# 2. Find cosine similarity between all rental properties

# Build similarity matrix

#df_model_num.head(5)

st.write('Similarity matrix')
@st.cache_data
def get_sim(df):
    return cosine_similarity(df)

similarity = get_sim(df_model_num)
similarity

#similarity[500]



# 3. Recommend top n similar properties

model_columns_all = list(df_model.columns.values)
#model_columns_all[:10]

ui_display_columns = ['listing_id', 'listing_url', 'polarity',
       'neighbourhood_cleansed', 'neighbourhood_group_cleansed', 'latitude',
       'longitude', 'property_type', 'room_type', 'accommodates', 'bedrooms',
       'beds', 'amenities', 'price', 'review_scores_rating',
       'bathrooms_count', 'amenities_count', 'host_response_time_encoded',
       'host_operate_years']

# Since we're using iloc in our recommender to retrieve the rows,
#   we need to also use the numeric index for the columns.

iloc_cols = [model_columns_all.index(x) for x in ui_display_columns]
#iloc_cols


# Get the recommendations

def get_recommendations(df, similarity, n, listing_id=None, listing_url=None, query_element=None):

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

    # get the top n similar items
    top_idx = np.argsort(similarity[item_index])[::-1][:n]
    print(top_idx)
    result_df = df.iloc[top_idx, iloc_cols]

    # add in similarity score as a column
    top_scores = [similarity[item_index][x] for x in top_idx]
    result_df.insert(loc=2, column='similarity', value=top_scores)

    return result_df


# Try the recommender system

df_recs = get_recommendations(df_model, similarity, 5, query_element=500)

# The similarities seem so close! Too good to be true?

# with pd.option_context('display.float_format', '{:0.10f}'.format):
#     display(df_recs)
#df_recs


st.subheader('Try the recommender in UI')

# select a listing from sd_merged
selected_listing = st.selectbox("Choose a listing url:", df_model.listing_url)
df_recs = get_recommendations(df_model, similarity, 5, listing_url=selected_listing)

st.write('Top 5 recommendations')
st.write(df_recs.style.format({"similarity": "{:0.10f}"}))


# You can access the value of a text input like this
# st.text_input("Your name", key="name")
# st.session_state.name



# Search for the simular properties using listing_id

# df_recs = get_recommendations(df_model, similarity, 5, listing_id=26258898)
# with pd.option_context('display.float_format', '{:0.10f}'.format):
#     display(df_recs)


# Search for simular properties by listing_url

# df_recs = get_recommendations(df_model, similarity, 5,
#                               listing_url='https://www.airbnb.com/rooms/26258898')
# with pd.option_context('display.float_format', '{:0.10f}'.format):
#     display(df_recs)



#####################################################################################
#######################################################
##### test code from Xinqian #####
##### add review sentiment plot for the recommended listings #####


import altair as alt

# directly read the saved cleaned_review_with_polarity dataset
review_df = pd.read_pickle('data/cleaned_v2/cleaned_review_with_polarity.zip')
# print(review_df.shape)
# review_df.head(2)

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
recs_listing_ids = df_recs['listing_id'].values
sentiment_plot = plot_listing_sentiment_over_time(review_df, recs_listing_ids)
# print(recs_listing_ids)

# write a note
st.write('Review sentiment trends')


# plot the figure 
st.altair_chart(sentiment_plot, use_container_width=True)



##### add review sentiment report for a recommended listings #####


import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def get_review_sentiment_report(df,col,listing_id):
    
    if listing_id in df['listing_id'].values:
        # segement all comments into sentences for the given listing
        review_sentences = df[df['listing_id'] == listing_id]['comments'].apply(lambda x: re.sub("(<.*?>)|([\t\r])","",x)).str.split('.').values.tolist()[0]
        num_review_sentences = len(review_sentences)
        
        # get polarity score of both the positives and negatives for each sentence in all the comments 
        neg_sentences = []
        pos_sentences = []
        neu_sentences = []
        # nutrual_comment = []
        for i, text in enumerate(review_sentences):
            score = SentimentIntensityAnalyzer().polarity_scores(text)['compound']
            if score < 0:
                neg_sentences.append((score,review_sentences[i]))
            elif score > 0:
                pos_sentences.append((score,review_sentences[i]))
            else:
                neu_sentences.append((score,review_sentences[i]))
        
        neg_percent = round(len(neg_sentences)/num_review_sentences*100,2)
        pos_percent = round(len(pos_sentences)/num_review_sentences*100,2)   
        neu_percent = round(len(neu_sentences)/num_review_sentences*100,2)
        
        sorted_neg_sentences = [comment for score, comment in sorted(neg_sentences, key=lambda x: x[0])]
        sorted_pos_sentences = [comment for score, comment in sorted(pos_sentences, key=lambda x: x[0])]
        
        st.write("{}% of all the reviews sentences ({}/{}) on Airbnb for this listing are positive!".format(pos_percent, len(pos_sentences),num_review_sentences))
        st.write("{}% of them ({}/{}) are negative, and {}% of them ({}/{}) are neutral.".format(neg_percent,len(neg_sentences),num_review_sentences, neu_percent,len(neu_sentences),num_review_sentences))
        st.write('---------------')
        st.write("The negative sentences that are helpful for later improvement are as follows:")
        print(num_review_sentences, len(pos_sentences), len(neg_sentences))
        for i, sentence in enumerate(sorted_neg_sentences):    
            st.write("{}: {}".format(i+1, sentence)) # need to yield every 3 items from a list

    else:
        st.write('Oops, this listing currently has no comments.')
        
    return sorted_neg_sentences, sorted_pos_sentences


# check the result
top_1_recommended_listing = df_recs['listing_id'].values[0]
# print("{} listing review sentiment report: ".format(top_1_recommended_listing))
st.write("{} listing review sentiment report: ".format(top_1_recommended_listing))
st.write('---------------')
sorted_neg_sentences, sorted_pos_sentences = get_review_sentiment_report(df_model,'comments',top_1_recommended_listing)




#######################################################
##### test code from Xinqian #####
##### add wordcloud for a recommended listings #####
#### below only use the first recomendation as example###

from wordcloud import WordCloud, STOPWORDS
@st.cache_data
def make_wordcloud(df, col, listing_id, stop_words, mask=None):
    
    if listing_id in df['listing_id'].values:
        text = df[df['listing_id'] == listing_id][col].values[0]
        wordcloud = WordCloud(width = 100, 
                              height = 100, 
                              stopwords=stop_words, 
                              scale=10, 
                              colormap = 'PuRd', 
                              background_color ='white',
#                               mask = None,
                              max_words=100,
                             ).generate(text)

        # plt.figure(figsize=(8,8))
        fig, ax = plt.subplots(figsize=(4,4))
        ax.imshow(wordcloud, interpolation="bilinear")
        # plt.imshow(wordcloud, interpolation="bilinear")
        # plt.axis("off")
        ax.axis("off")
        plt.show()
        st.pyplot(fig)
    else:
        print('Oops, this listing currently has no comments.') 
        st.write('Oops, this listing currently has no comments.')

# # generate wordcloud for a recommended listing (has comments)
top_1_recommended_listing = df_recs['listing_id'].values[1]
wordcloud_STOPWORDS = STOPWORDS
# make_wordcloud(df_model,'comments', top_1_recommended_listing, wordcloud_STOPWORDS, mask=None)

ok = st.button("Make Wordcloud")
if ok:   
    with st.spinner('Making Wordcloud...'):
        make_wordcloud(df_model,'comments', top_1_recommended_listing, wordcloud_STOPWORDS, mask=None)
        # make_wordcloud(text, stopwords_list, "picture/wine_image.jpg")
    st.success('Done!')

print((df_model[df_model['listing_id']== top_1_recommended_listing]['comments']))

print(top_1_recommended_listing)