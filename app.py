


import pandas as pd
import numpy as np
#from scipy.stats import uniform
#from pycaret.classification import *
#import pickle
import streamlit as st

def main():
    st.write('Hello World')
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)

if __name__=='__main__':
    main()
