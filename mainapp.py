


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


# Prepare data

# load dataset that has already been cleaned
raw_df = pd.read_pickle('./data/cleaned_v2/cleaned_listing_and_review_with_polarity.zip')
st.write(raw_df.shape)
