import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.header("Debug page, for pulling from git")

st.write('Hello from team-spirit :)')

st.write('Streamlit version: '+st.__version__)

import subprocess
if st.button('Pull from git'):
    subprocess.run("git pull origin", shell=True)
    st.experimental_rerun()

result = subprocess.run("git rev-parse HEAD", shell=True, capture_output=True, text=True)
st.write('Latest github code id running on this server:', result.stdout)
