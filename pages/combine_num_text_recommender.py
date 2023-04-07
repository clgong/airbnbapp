
"""
# My first app
Here's our first attempt at using data to create a table:
"""

## add some comment here for test 2

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

from nltk import word_tokenize, pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords


# set pages
st.set_page_config(
    page_title = 'Multippage App',
    page_icon='👋'
)

st.title('AirBnb Rentals in Seattle')
st.write('Hello from team-spirit :)')


# header
st.header('Try the customized recommender in UI')


#make a price query slider
price_range = st.slider("Please choose your preferred price range",
                        value = [50,5000])
st.write("Your preferred price range:", price_range)
submit_price = st.button('Confirm')


#make an input box
defult_input = "I want a private room close to uw campus with parking and coffee shop."
input_query = st.text_input("Please describe the rental you're looking for here ",defult_input)
submit = st.button('Submit')


# TODO: run the code below after click the submit button instead of running the code automatically
################################################################
# code to get top 5 recommendations

##### prepare stopword set
added_stopwords = ["can't",'t', 'us', 'say','would', 'also','within','stay', 'since']
nltk_STOPWORDS = set(stopwords.words("english"))
nltk_STOPWORDS.update(added_stopwords)

##### get data
@st.cache_data
def get_data():
    # directly load the saved dataset
    df = pd.read_pickle('data/cleaned_v2/cleaned_listing_and_review_with_polarity_and_text_content.zip')
    # get date cols
    date_col = df.select_dtypes('datetime64[ns]').columns.to_list()
    # remove the following cols
    removed_col = ['host_id',
                    'host_listings_count',
                    'host_total_listings_count',
                    'latitude',
                    'longitude',
                    'minimum_minimum_nights',
                    'maximum_minimum_nights',
                    'minimum_maximum_nights',
                    'maximum_maximum_nights',
                    'minimum_nights_avg_ntm',
                    'maximum_nights_avg_ntm',
                    'availability_30',
                    'availability_60',
                    'availability_90',
                    'availability_365',
                    'number_of_reviews_ltm',
                    'number_of_reviews_l30d',
                    'host_response_time_encoded'] + date_col + ['description',
                                                                'neighborhood_overview',
                                                                'picture_url',
                                                                'host_url',
                                                                'host_name',
                                                                'host_location',
                                                                'host_about',
                                                                'host_picture_url',
                                                                'host_neighbourhood',
                                                                'comments',
                                                                'host_verifications',
                                                                'amenities',
                                                                'listing_url',
                                                                'listing_name',
                                                                'content',
                                                                'cleaned_content']
    # get final df
    df_listing = df.loc[:,~df.columns.isin(removed_col)]
    return df_listing

listing_df = get_data()
# st.write(listing_df.shape)
# st.write(listing_df.head(2))


##### preprocess listing data for clustering
@st.cache_data
def get_transformed_data(df):
    # set listing id as the index
    df = df.set_index('listing_id')
    # categorical columns in the dataframe
    cat_col = df.select_dtypes('object').columns
    # convert host response time to categorical dtype
    df['host_response_time'] = df['host_response_time'].astype('category')
    # define order of the ordinal features
    response_time_list = ['within an hour',
                          'within a few hours',
                          'within a day',
                          'a few days or more',
                          'no response']
    # define nominal and ordinal features in the categorical columns
    nom_cols = ['property_type','room_type','neighbourhood_cleansed','neighbourhood_group_cleansed']
    ordinal_cols = df.select_dtypes(['category']).columns

    # define numeric transformation pipeline that scales the numbers
    numeric_pipeline = Pipeline([('numnorm', StandardScaler())])

    # define an ordinal transformation pipeline that ordinal encodes the cats
    ordinal_pipeline = Pipeline([('ordinalenc', OrdinalEncoder(categories = [response_time_list]))])

    # define a nominal transformation pipeline that OHE the cats
    nominal_pipeline = Pipeline([('onehotenc', OneHotEncoder(categories= "auto",
                                                             sparse = False,
                                                             handle_unknown = 'ignore'))])

    # construct column transformer for the selected columns with pipelines
    ct = ColumnTransformer(transformers = [("nominalpipe", nominal_pipeline, nom_cols),
                                           ("ordinalpipe", ordinal_pipeline, ['host_response_time']),
                                           ("numericpipe", numeric_pipeline,
                                           df.select_dtypes(['int', 'float']).columns)])
    # dataframe after data transformation
    df_trans = pd.DataFrame(ct.fit_transform(df))

    # get nominal values
    nominal_features = list(nominal_pipeline.named_steps['onehotenc'].fit(df[nom_cols]).get_feature_names_out())

    # get ordinal values
    ordinal_features = list(ordinal_cols)

    # get numeric values
    numeric_features = ['host_response_rate', 'host_acceptance_rate', 'host_is_superhost',
                         'host_has_profile_pic', 'host_identity_verified', 'has_license',
                         'instant_bookable','accommodates','bedrooms', 'beds',
                         'bathrooms_count', 'amenities_count', 'price',
                         'minimum_nights', 'maximum_nights','has_availability',
                         'number_of_reviews', 'review_scores_rating',
                         'review_scores_accuracy', 'review_scores_cleanliness',
                         'review_scores_checkin', 'review_scores_communication',
                         'review_scores_location', 'review_scores_value',
                         'calculated_host_listings_count',
                         'calculated_host_listings_count_entire_homes',
                         'calculated_host_listings_count_private_rooms',
                         'calculated_host_listings_count_shared_rooms', 'reviews_per_month',
                         'host_operate_years', 'polarity']
    # naming the columns of the transformed dataframe
    df_trans.columns = nominal_features + ordinal_features + numeric_features

    # handling the missing/infinity values
    df_trans = df_trans.fillna(0)

    return df_trans

listing_trans = get_transformed_data(listing_df)


##### get the cluster
@st.cache_data
# initialize and compute PCA
def pca_component(dataframe):
    pca = PCA()
    pca.fit_transform(dataframe)

    components = pca.components_

    return components

# PCA factor loadings
df_c = pd.DataFrame(pca_component(listing_trans), columns=list(listing_trans.columns)).T

component_n = []
for i in range(len(df_c)):
    if df_c.iloc[:,0].sort_values(ascending=False)[:i].sum() > 0.9:
        component_n.append(i)
principle_component_lst = list(df_c.iloc[:,0].sort_values(ascending=False)[:component_n[0]].index)

pca_df = listing_trans[principle_component_lst]


@st.cache_data
# initialize Kmeans and find clusters
def kmeans_cluster(df):
    kmeans_kwargs = {"init": "random",
                     "n_init": 10,
                     "max_iter": 300,
                     "random_state": 42}

    # A list holds the SSE values for each k
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters = k, **kmeans_kwargs)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)

    kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")

    # we will use this result as the best number of clusters
    K = kl.elbow

    # initiate kmeans
    kmeans = KMeans(n_clusters = K,
                    init='k-means++',
                    verbose=0,
                    n_init=10,
                    max_iter=300,
                    random_state=42,
                    algorithm='lloyd')

    # predict the clusters
    predict_cluster = kmeans.fit_predict(df)

    return predict_cluster

# generate clusters
pca_df['cluster'] = list(kmeans_cluster(pca_df))
listing_df['cluster'] = list(pca_df['cluster'])


##### get data for recommendation
raw_df = pd.read_pickle('data/cleaned_v2/cleaned_listing_and_review_with_polarity_and_text_content.zip')
new_col = ['listing_url', 'listing_name',
           'description','neighborhood_overview',
           'host_about', 'amenities', 'comments','cleaned_content','content']
listing_df[new_col] = raw_df[new_col].values
listing_trans['listing_id'] = listing_df['listing_id']
listing_trans['cluster'] = listing_df['cluster']

numeric_features = ['host_response_rate', 'host_acceptance_rate', 'host_is_superhost',
                     'host_has_profile_pic', 'host_identity_verified', 'has_license',
                     'instant_bookable','accommodates','bedrooms', 'beds',
                     'bathrooms_count', 'amenities_count', 'price',
                     'minimum_nights', 'maximum_nights','has_availability',
                     'number_of_reviews', 'review_scores_rating',
                     'review_scores_accuracy', 'review_scores_cleanliness',
                     'review_scores_checkin', 'review_scores_communication',
                     'review_scores_location', 'review_scores_value',
                     'calculated_host_listings_count',
                     'calculated_host_listings_count_entire_homes',
                     'calculated_host_listings_count_private_rooms',
                     'calculated_host_listings_count_shared_rooms', 'reviews_per_month',
                     'host_operate_years', 'polarity','cluster']
similarity_df = listing_df[numeric_features].fillna(0)
num_similarity = cosine_similarity(similarity_df)
listing_df.insert(loc=1,column='similarity',value=num_similarity[0])

df_filter = listing_df.loc[(listing_df['price'] < price_range[1]) &(listing_df['price'] > price_range[0])]
df_filter_std = listing_trans.loc[listing_trans['listing_id'].isin(df_filter['listing_id'])]
df_filter = df_filter.reset_index()


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

# get corpus
corpus = df_filter['cleaned_content'].values
tfidf_vectorizer, tfidf_matrix = vectorize_data(corpus)
# st.write(tfidf_matrix.shape)

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

similarity = get_similarity(input_query, tfidf_matrix)
# st.write(similarity.shape)

##### get recommendations
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
    result_df = df.loc[best_index,:]
    result_df['recommendations'] = ['recommendation_'+ str(i) for i in range(1,len(result_df)+1)]
    result_df = result_df.loc[:, ['recommendations','cluster',
                                      'similarity', 'price',
                                      'listing_id',
                                      'listing_url',
                                      'listing_name',
                                      'description',
                                      'room_type',
                                      'property_type',
                                      'neighborhood_overview',
                                      'neighbourhood_cleansed',
                                      'neighbourhood_group_cleansed',
                                      'host_about',
                                      'amenities',
                                      'comments',
                                      'review_scores_rating']].sort_values('similarity',ascending=False)

    return result_df


# Try the recommender system
recomended_listings = get_recommendations(df_filter, input_query, tfidf_matrix, n=5)

def update_recommend_listing(recomended_list, filtered_std_df, original_df, n):

    if len(recomended_list['cluster'].value_counts()) > 1:
        if len(recomended_list['cluster'].mode()) >= 1:
            cluster_label = recomended_list['cluster'].mode().iloc[0]
            listing_id = list(recomended_list.loc[recomended_list['cluster']==cluster_label]['listing_id'])
            if len(filtered_std_df) >= n:
                df_std_new = filtered_std_df.loc[(filtered_std_df['cluster']==cluster_label)&(listing_id not in filtered_std_df['listing_id'].to_list())]
                df_d = pd.DataFrame(get_pca_df(df_std_new.iloc[:,:-3],threshold=0.9)[0], columns=list(df_std_new.iloc[:,:-3].columns)).T
                new_id = df_std_new.sort_values(df_d.iloc[:,0].sort_values(ascending=False)[:1].index[0],
                           ascending=False).head(n-len(listing_id))['listing_id']

                updated_list = listing_id + list(new_id)

                df_recommend = original_df.loc[original_df['listing_id']\
                                           .isin(updated_list)]
                df_recommend = df_recommend.sort_values('similarity',ascending=False)
                df_recommend['recommendations'] = ['recommendation_'+ str(i) for i in range(1,len(df_recommend)+1)]
                df_recommend = df_recommend[['recommendations', 'cluster',
                                                       'similarity',
                                                       'price',
                                                       'listing_id',
                                                       'listing_url',
                                                       'listing_name',
                                                       'description',
                                                       'room_type',
                                                       'property_type',
                                                       'neighborhood_overview',
                                                       'neighbourhood_cleansed',
                                                       'neighbourhood_group_cleansed',
                                                       'host_about',
                                                       'amenities',
                                                       'comments',
                                                       'review_scores_rating']]
            else:
                df_recommend = original_df.loc[(original_df['cluster']==cluster_label)]
                df_recommend = df_recommend.sort_values('similarity',ascending=False)
                df_recommend['recommendations'] = ['recommendation_'+ str(i) for i in range(1,len(df_recommend)+1)]
                df_recommend = df_recommend[['recommendations', 'cluster',
                                                                       'similarity',
                                                                       'price',
                                                                       'listing_id',
                                                                       'listing_url',
                                                                       'listing_name',
                                                                       'description',
                                                                       'room_type',
                                                                       'property_type',
                                                                       'neighborhood_overview',
                                                                       'neighbourhood_cleansed',
                                                                       'neighbourhood_group_cleansed',
                                                                       'host_about',
                                                                       'amenities',
                                                                       'comments',
                                                                       'review_scores_rating']]

    if len(recomended_list['cluster'].value_counts()) == 1:
        df_recommend = recomended_list


    return df_recommend

# Try the combined recommender system
recomended_listings_update = update_recommend_listing(recomended_listings, df_filter_std, df_filter, n=5)
st.write(recomended_listings_update)


################################################################
##### add review sentiment plot for the recommended listings #####

#### ISSUES TODO:  the wordcloud code only use the first recomendation as example
# 1. change the trends legend labels from listing_id to recommendation_n for users!!


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
rec_listing_ids = recomended_listings_update['listing_id'].values
sentiment_plot = plot_listing_sentiment_over_time(review_df, rec_listing_ids)
# print(recs_listing_ids)

# write a note
st.write('Review sentiment trends')

# plot the figure
st.altair_chart(sentiment_plot, use_container_width=True)



# ################################################################
# ##### add wordcloud for the recommended listings #####
# #### below only use the first recomendation as example###

#### ISSUES TODO:  the wordcloud code only use the first recomendation as example
# 1. make a selection box/doropdown for users to choose which recommendation to generate!!
# 2. how to run the code section one by one istead of starting from begaining when choosing different recommendation listing

# top_n_reco mmended_listing = recomended_listings['recommendations'].values.tolist()
# option = st.selectbox('select an option',top_n_recommended_listing)
# st.write('you selected', option)


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

#
# generate wordcloud for a recommended listing (has comments)
#

st.subheader('Pick a top n recommendation for more info.')
wordcloud_STOPWORDS = STOPWORDS

# get a top n listing id from the user
selected_listing_id = st.selectbox("Choose listing id:", recomended_listings_update['listing_id'])
index = recomended_listings_update['listing_id'].tolist().index(selected_listing_id)
link = recomended_listings_update.listing_url.tolist()[index]

# Show name and URL of selected property
st.write("\"{}\" - [{}]({})".format(recomended_listings_update.listing_name.tolist()[index],link,link))

# Draw the word cloud
make_wordcloud(df_filter,'cleaned_content', selected_listing_id, wordcloud_STOPWORDS, mask=None)




# generate wordcloud using a button
# ok = st.button("Make Wordcloud for the listing description")
# if ok:
#     with st.spinner('Making Wordcloud...'):
#         make_wordcloud(df_rec,'cleaned_content', top_1_recommended_listing, wordcloud_STOPWORDS, mask=None)
#         # make_wordcloud(text, stopwords_list, "picture/wine_image.jpg")
#     st.success('Done!')


# # issue
# # the wordcould showed fewer words in streamlit than in the deepnote recommender_system_text_content_based
# st.write(len(df_rec[df_rec['listing_id'] == top_1_recommended_listing]['cleaned_content'].values[0].split()))
