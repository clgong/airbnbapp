


"""
 UPDATE Apr 6: (add/delete dataset)
 1.use the finalized dataset cleaned_listing_finalized_for_streamlit
 2.removed picke lib, use pandas to load the pickled file
 3. added price column and edited some words

"""

## add some comment here for test 2

# import libraries
import streamlit as st
import pandas as pd
import numpy as np
from scipy import sparse
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt

import warnings
warnings.filterwarnings('ignore',category=DeprecationWarning)
pd.set_option('display.max_columns', 100)
RANDOM_STATE= 42

from collections import Counter
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
# nltk.download('vader_lexicon') 

from nltk import word_tokenize, pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# set pages
st.set_page_config(
    page_title = 'Multippage App',
    page_icon='👋'
)

st.title('AirBnb Rentals in Seattle')
st.write('Hello from team-spirit :)')
st.write('Streamlit version: '+st.__version__)

# header
st.header('Try the customized recommender in UI')

#make an input box
defult_input = "I want a private room close to uw campus with parking and coffee shop."
input_query = st.text_input("Please describe the rental you're looking for here ",defult_input)
submit = st.button('Submit')  ## submit buttion 


# TODO: run the code below after click the submit button instead of running the code automatically
########################################################################################################
# code to get top 5 recommendations #

##### prepare stopword set
added_stopwords = ["can't",'t', 'us', 'say','would', 'also','within','stay', 'since']
nltk_STOPWORDS = set(stopwords.words("english"))
nltk_STOPWORDS.update(added_stopwords)

##### get data
@st.cache_data
def get_data():
    # directly load the saved dataset
    df = pd.read_pickle('data/cleaned_v2/cleaned_listing_finalized_for_streamlit.zip')
    # select cols
    listing_cols = ['listing_id','listing_url','price',''
                    'review_scores_rating', 'polarity',
                    'comments', 'comments_nouns_adjs']
    # get content cols
    content_cols = ['listing_name', 'description',
                    'host_name', 'host_location', 'host_about',
                    'host_response_time', 'host_neighbourhood',
                    'host_verifications', 'neighbourhood_cleansed',
                    'neighbourhood_group_cleansed','neighborhood_overview',
                    'property_type', 'room_type','amenities',
                    'content','cleaned_content'
                   ]
    # get final df
    df_rec = df.loc[:,[*listing_cols, *content_cols]]
    return df_rec

df_rec = get_data()
# st.write(df_rec.shape)
# st.write(df_rec.head(2))

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
    # load tfidf vectorizer and do the transformation
    tfidf_vectorizer = pd.read_pickle(("data/cleaned_v2/tfidf_vectorizer.pk"))
    tfidf_matrix = tfidf_vectorizer.transform(corpus).todense()
    tfidf_matrix = np.asarray(tfidf_matrix)

    return tfidf_vectorizer, tfidf_matrix


# get corpus
corpus = df_rec['content'].values
tfidf_vectorizer, tfidf_matrix = vectorize_data(corpus)
# st.write(tfidf_matrix.shape)

##### get similarity
@st.cache_data
def get_similarity(input_query, tfidf_matrix):
    # embed input query
    tokens = preprocess_text(input_query,stopwords = nltk_STOPWORDS, stem=False, lemma=True).split()
    query_vector = tfidf_vectorizer.transform(tokens)
    # get similarity
    tfidf_matrix_sparse = sparse.csr_matrix(tfidf_matrix)
    similarity = cosine_similarity(query_vector, tfidf_matrix_sparse)

    return similarity

similarity = get_similarity(input_query, tfidf_matrix)
# st.write(similarity.shape)

##### get recommendations
@st.cache_data
def get_recommendations(df,similarity, n=5):

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

    # best cosine distance for each token independantly
    best_index = extract_best_indices(similarity, top_n=n)

    # return the top n similar listings
    result_df = df.loc[best_index,:]
    result_df = result_df.loc[:, ['listing_id', 'listing_url','listing_name', 'description',
                                 'price','room_type','property_type',
                                 'neighborhood_overview', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed',
                                 'host_about', 'amenities', 'review_scores_rating']]
    result_df.reset_index(drop=True, inplace=True)
    return result_df

# Try the recommender system
recomended_listings = get_recommendations(df_rec, similarity, n=5)

st.header('Top 5 recommended rentals')
st.write(recomended_listings)


################################################################
##### add review sentiment plot for the recommended listings #####

#### ISSUES TODO:  the wordcloud code only use the first recomendation as example
# 1. change the trends legend labels from listing_id to recommendation_n for users!!


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

# write a note
st.header('Rental review sentiment trends')
# plot the figure
st.altair_chart(sentiment_plot, use_container_width=True)


################################################################
# ##### add rental description wordcloud #####

#### TODO:
# how to run the code section one by one istead of starting from begainning when choosing different recommendation listing


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
                              max_words=100,
                             ).generate(text)

        fig, ax = plt.subplots(figsize=(4,4))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        plt.show()
        st.pyplot(fig)
    else:
        print('Oops, this listing currently has no comments.')
        st.write('Oops, this listing currently has no comments.')

# generate wordcloud for a recommended listing (has comments)
st.header('Word cloud for rental description')
st.subheader('Pick a top n recommendation for more info')
wordcloud_STOPWORDS = STOPWORDS

# get a top n listing id from the user
selected_listing_id = st.selectbox("Choose listing id:", recomended_listings['listing_id'])
index = recomended_listings['listing_id'].tolist().index(selected_listing_id)
link = recomended_listings.listing_url.tolist()[index]

# Show name and URL of selected property
st.write("\"{}\" - [{}]({})".format(recomended_listings.listing_name.tolist()[index],link,link))

# Draw the word cloud
make_wordcloud(df_rec,'cleaned_content', selected_listing_id, wordcloud_STOPWORDS, mask=None)

# # generate wordcloud using a button
# ok = st.button("Make Wordcloud for the rental description")
# if ok:
#     with st.spinner('Making Wordcloud...'):
#         make_wordcloud(df_rec,'cleaned_content', selected_listing_id, wordcloud_STOPWORDS, mask=None)
#     st.success('Done!')


###############################################################
##### add rental review wordcloud #####

# generate review wordcloud 
st.header('Word cloud for rental reviews')

# Show name and URL of selected property
st.write("\"{}\" - [{}]({})".format(recomended_listings.listing_name.tolist()[index],link,link))

# Draw the word cloud
make_wordcloud(df_rec,'comments_nouns_adjs', selected_listing_id, wordcloud_STOPWORDS, mask=None)


################################################################
##### add rental review report #####

def get_review_sentiment_report(df,col,listing_id):

    if listing_id in df['listing_id'].values:
        # segement all comments into sentences for the given listing
        review_sentences = df[df['listing_id'] == listing_id]['comments'].apply(lambda x: re.sub("(<.*?>)|([\t\r])","",x)).str.split('.').values.tolist()[0]
        num_review_sentences = len(review_sentences)

        # get polarity score of both the positives and negatives for each sentence in all the comments
        neg_sentences = []
        pos_sentences = []
        # nutrual_comment = []
        for i, text in enumerate(review_sentences):
            score = SentimentIntensityAnalyzer().polarity_scores(text)['compound']
            if score < 0:
                neg_sentences.append((score,review_sentences[i]))
            elif score > 0:
                pos_sentences.append((score,review_sentences[i]))
            else:
                pass

        neg_percent = round(len(neg_sentences)/num_review_sentences*100,2)
        pos_percent = round(len(pos_sentences)/num_review_sentences*100,2)
        sorted_neg_sentences = [comment for score, comment in sorted(neg_sentences, key=lambda x: x[0])]
        sorted_pos_sentences = [comment for score, comment in sorted(pos_sentences, key=lambda x: x[0])]
        st.subheader("Overall:")
        st.write("{}% of all the reviews sentences ({}/{}) on Airbnb for this listing are positive!".format(pos_percent, len(pos_sentences),num_review_sentences))
        st.write("{}% of them ({}/{}) are negative.".format(neg_percent,len(neg_sentences),num_review_sentences))
        # st.write('---------------')
        st.subheader("Helpful negative sentences: ")

        if len(sorted_neg_sentences) >0:
            for i, sentence in enumerate(sorted_neg_sentences):
                st.write("{}: {}".format(i+1, sentence)) # need to yield every 3 items from a list
        else:
            st.write("Wow, this listing currently doesn't have any negative sentences!)")
    else:
        st.write('Oops, this listing currently has no comments.')

    return sorted_neg_sentences, sorted_pos_sentences


# generate review report for a recommended listing (has comments)
st.header('Rental review report')

# Show name and URL of selected property
st.write("\"{}\" - [{}]({})".format(recomended_listings.listing_name.tolist()[index],link,link))

# Draw the word cloud
sorted_neg_sentences, sorted_pos_sentences = get_review_sentiment_report(df_rec,'comments',selected_listing_id)