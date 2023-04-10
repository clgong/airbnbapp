


"""
Generate the document term matrix used in the text based recommender
and save to a pickle file.

Based on code originally from text_based_recommender.py, this file
is used to generate a pickle file for the tfidf_matrix that will be
used in text_based_recommender.

This helps efficiency and alleiates memory problmes in the streamlit server.

This helps avoiding checking in large files into GitHub which has 2GB limit.

DEVOPS NOTE: This file should be run when setting up a new streamlit
server or when "cleaned_listing_finalized_for_streamlit.zip" is updated.

"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

source_datafile = 'data/cleaned_v2/cleaned_listing_finalized_for_streamlit.zip'
destination_datafile = 'data/cleaned_v2/tfidf_matrix.pk'

#
# get data
#

def get_data():
    # directly load the saved dataset
    df = pd.read_pickle(source_datafile)
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

print("Reading data frame from \'{}\'...".format(source_datafile))

df_rec = get_data()

print("  Read complete. DataFrame shape:",df_rec.shape)

#
# Vectorize data: get corpus and vectorize it
#

def vectorize_data(corpus):
   # load tfidf vectorizer and do the transformation
   tfidf_vectorizer = pd.read_pickle(("data/cleaned_v2/tfidf_vectorizer.pk"))
   tfidf_matrix = tfidf_vectorizer.transform(corpus).todense()
   tfidf_matrix = np.asarray(tfidf_matrix)

   return tfidf_vectorizer, tfidf_matrix

print("Vectorizing data...")

corpus = df_rec['content'].values
tfidf_vectorizer, tfidf_matrix = vectorize_data(corpus)

print("  Vectorize complete, matrix shape:",tfidf_matrix.shape)

#
# Save output matrix to pickle file.
#

print("Saving to file to \'{}\'...".format(destination_datafile))

with open(destination_datafile,'wb') as file:
   pickle.dump(tfidf_matrix,file)

print("DONE!\n")
