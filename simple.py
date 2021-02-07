import pandas as pd
import streamlit as st
st.title('  Welcome to My App')
st.text ('type a number in the box below')
n=st.number_input('Number=int')
st.write(f'{n} +1={n+1}')
s=st.text_input('Tpye a namein the box below')
st.write(f'Hello {s}')
a=st.text_input('Tpye your age the box below')
st.write(f'You are {a} years old')