import pandas as pd
import streamlit as st
st.title('  Welcome to My App')
st.text ('type the first  number in the box below')
n1=st.number_input('Number=step1')
st.text ('type the second  number in the box below')
n2=st.number_input('Number=step2')
st.write(f'{n1} +{n2}1={n1+n2}')
st.write(f'{n1} -{n2}1={n1-n2}')
st.write(f'{n1} *{n2}1={n1*n2}')
s=st.text_input('Tpye your name in the box below')
st.write(f'Hello {s}')
a=st.text_input('Tpye your age the box below')
st.write(f'Happy {a} {s}, God loves you')