import pandas as pd
import streamlit as st
st.title('SIMPLE STREAMLIT APP')
st.text ('type a number in the box below')
n=st.number_input('Number=int')
st.write(f'{n} +1={n+1}')
s=st.text_input('Tpye a namein the box below')
st.write(f'Hello {s}')