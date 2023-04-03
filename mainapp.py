


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


# set pages
st.set_page_config(
    page_title = 'Multippage App',
    page_icon='👋'
)

st.title('AirBnb Rentals in Seattle')
st.write('Hello from team-spirit :)')


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

