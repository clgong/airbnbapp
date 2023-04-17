


"""
TODO: Remove this page or make it an intro page!!
  Useful for debugging, but probably should not go live to public?

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
# st.write('Hello from team-spirit :)')
#
# st.write('Streamlit version: '+st.__version__)
#
# import subprocess
# if st.button('Pull from git'):
#     subprocess.run("git pull origin", shell=True)
#     st.experimental_rerun()
#
# result = subprocess.run("git rev-parse HEAD", shell=True, capture_output=True, text=True)
# st.write('Latest github code id running on this server:', result.stdout)


st.subheader(":blue[Welcome to our experimental app!]")
st.markdown("**Motivation**")
st.markdown("Both Airbnb hosts and customers face the daunting task of uncovering how other users truly feel about their Airbnb rental experience. Judging from the qualitative analysis results, customer reviews are influential to customers’ decision-making, and both the sentiment trends and negative reviews are helpful in verifying the rental descriptions and gaining helpful opinions towards the rental. Our interviewees valued negative reviews and even negative sentences in positive reviews up to 50% more than positive reviews.")
st.markdown("However reading through every review in a large number of listings can be very time consuming. So we decided to see if we could make it easier. Our goal was to use data science to see what kind of content-based recommender system we could make by focusing mostly on the text elements of the rental listings.")
st.markdown("**How to use**")
st.markdown("The left side of the screen shows two recommenders.")
st.markdown("- The text based recommender focuses on using just NLP techniques to suggest listings. Simply enter some text and click \"Confirm\".")
st.markdown("- The combined recommender uses the same NLP techniques, but also uses some numeric features to see if the results are different. Move the sliders for some numeric values. Then type some text and click \"Confirm\". (If you don't type any text it simply shows a cosine similarity table between the listings).")
st.markdown("After a search result listing is returned, both recommenders allow the user select a specific rental and see more details about that listing including sentiment analysis as well a review report which extricates just the negative sentences from all reviews, positive or negative.")
st.markdown("Please try it out and see which recommender returns better results.")
st.markdown("TODO: UPDATE LINKS: A [full report](https://www.google.com) and [GitHub repo](https://www.yahoo.com) accompany this app. Thanks for trying it out!")
