


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

st.write('DEBUG, branch from kunma_branch to cliff_awsdebug_from_kunma_branch v0.1')

import subprocess
if st.button('Pull from git'):
    subprocess.run("git pull origin", shell=True)
    st.experimental_rerun()
