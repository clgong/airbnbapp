


"""
 UPDATE Apr 6: (changed the dataset)
 1.use the finalized dataset cleaned_listing_finalized_for_streamlit.zip file
  instead of the cleaned_listing_and_review_with_polarity.zip file
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

st.write('DEBUG, branch from xinqian_branch_2 to cliff_awsdebug_2 v0.3')

import subprocess
if st.button('Pull from git'):
    subprocess.run("git pull origin", shell=True)
    st.experimental_rerun()
