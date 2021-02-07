from random import *

import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objects as go
import pandas as pd
from dash.dependencies import Input, Output
import requests
import json


# Initialize the app
password = dash.Dash(__name__)
#app.config.suppress_callback_exceptions = True

user_pass=input("Enter your password")

password =['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p',
           'q','r','s','t','u','v','w','x','y','z','1','2','3','4','5','6','7','8','9','0']
guess=""

while(guess !=user_pass):
    guess=""
    for letter in range(len(user_pass)):
        guess_letter=password[randint(0,27)]
        guess=str(guess_letter)+str(guess)


print(guess)
print("Your password is",guess)





if __name__ == '__main__':
    password.run_server(debug=True,port=600)

