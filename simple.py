import pandas as pd
import streamlit as st
st.title('  Welcome to My App')
s=st.text_input('What is your name?')
st.write(f'Hello {s}')
a=st.text_input('What is your age?')
st.write(f'Happy {a} {s}, God loves you')
c=st.text_input('What is your favourite color?')
st.write(f' {c}, wow!')
st.title('This app can add, subtract and multiply two numbers')
n6=st.number_input('Number1')
n7=st.number_input('Number2')
st.write(f'{n6} +{n7}={n6+n7}');
st.write(f'{n6} -{n7}={n6-n7}');
st.write(f'{n6} *{n7}={n6*n7}');
st.title('Calcuate the area of a Rectangle');
n1=st.number_input('Length');
n2=st.number_input('Width');
st.write(f'area of a rectangle ={n1*n2}');
st.title('Calcuate the area of a square');
n3=st.number_input('length');
st.write(f'area of a square={n3*n3}');
st.title('Calculate the area of a circle');
n4=st.number_input('radius');
n5=st.number_input('pi');
st.write(f'area of a circle  = {n5* n4* n4}')

st.write('Thank you for using my app')