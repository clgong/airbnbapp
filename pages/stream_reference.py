


"""
# streamlit reference
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


st.set_page_config(
    page_title = 'Multippage App',
    page_icon='👋'
)

st.title('Main Page')
st.sidebar.success("Select a page above.")

# title
st.title('This is a title')
# header
st.header('this is a heading')
# subheader
st.subheader('this is a subheading')
st.text('hello! welcom to coding')

# success, info
st.success('executed successfully')
st.info('this is information')
st.warning('this is warning')
st.error('this is an error')

#write
st.write('im weriting a sample code')
st.write(range(20))
channel='codingwithme'
st.write('subscribe my channel', channel)

#code
code_boday="""for i in range(5):
                print(i)"""
st.code(code_boday, language='python')

#checkbox
if(st.checkbox('I agree')):
    st.text('You agreed the condition')

#radio
val = st.radio('select a language',('pyton','jiava'))
st.write(val,'was selected')


# #image
# img =Image.open('logo.jpeg')
# st.image(img)

#selection box/doropdown
option = st.selectbox('select an option',['a','b','c'])
st.write('you selected', option)

#multiselection
options = st.multiselect('select multiple options',['a','b','c'] )
st.write('you selected,')
for op in options:
    st.write(op)

#text input box
name = st.text_input('enter your name','name')
st.write('your nam is ', name)

#buttion
st.button('click me!')
if(st.button('SUBSCRIBE')):
    st.text('thanks for subscribing')

#slider
sal = st.slider('your salary', 1000, 5000)
st.write('your salary is ', sal)




