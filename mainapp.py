


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
st.write('Hello from team-spirit :)')

st.write('Streamlit version: '+st.__version__)

import subprocess
if st.button('Pull from git'):
    subprocess.run("git pull origin", shell=True)
    st.experimental_rerun()

result = subprocess.run("git rev-parse HEAD", shell=True, capture_output=True, text=True)
st.write('Latest github code id running on this server:', result.stdout)
