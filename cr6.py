import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

import seaborn as sns

website='https://www.worldometers.info/coronavirus/#countries'
import socket
socket.getaddrinfo('localhost', 8080)

website_url=requests.get(website).text

soup = BeautifulSoup(website_url,'html.parser')

my_table = soup.find('tbody')
import matplotlib.pyplot as plt

table_data = []
for row in my_table.findAll('tr'):
    row_data = []
    for cell in row.findAll('td'):
        row_data.append(cell.text)
    if(len(row_data) > 0):
        data_item = {"Country": row_data[1],
                     "TotalCases": row_data[2],
                     "NewCases": row_data[3],
                     "TotalDeaths": row_data[4],
                     "NewDeaths": row_data[5],
                     "TotalRecovered": row_data[6],
                     "ActiveCases": row_data[7],
                     "CriticalCases": row_data[8],
                     "Totcase1M": row_data[9],
                     "Totdeath1M": row_data[10],
                     "TotalTests": row_data[11],
                     "Tottest1M": row_data[12],
        }
        table_data.append(data_item)

        df = pd.DataFrame(table_data)

        df.to_excel('Covid19_data.xlsx', index=True)

        df.shape

        df.isnull().sum()

        df.info()

        df['TotalCases'] = pd.to_numeric(df['TotalCases'], errors='coerce')
        df['NewCases'] = pd.to_numeric(df['NewCases'], errors='coerce')
        df['TotalDeaths'] = pd.to_numeric(df['TotalDeaths'], errors='coerce')
        df['NewDeaths'] = pd.to_numeric(df['NewDeaths'], errors='coerce')
        df['TotalRecovered'] = pd.to_numeric(df['TotalRecovered'], errors='coerce')
        df['ActiveCases'] = pd.to_numeric(df['ActiveCases'], errors='coerce')
        df['CriticalCases'] = pd.to_numeric(df['CriticalCases'], errors='coerce')
        df['Totcase1M'] = pd.to_numeric(df['Totcase1M'], errors='coerce')
        df['Totdeath1M'] = pd.to_numeric(df['Totdeath1M'], errors='coerce')

        df['Totdeath1M'] = pd.to_numeric(df['Totdeath1M'], errors='coerce')

        df1 = df.drop(columns=['Country', 'Tottest1M', 'Totdeath1M', 'Totcase1M', 'TotalTests'])

        df1.describe()

        df2 = df1.fillna(df.mean())
        df2

        df2.isnull().sum()

        df2['infection_rate'] = df2['NewCases'] / df2['TotalCases'] * 100



        df2.info()

        df2.head()

        # Creating x (all the feature columns)
        x = df2.drop("infection_rate", axis=1)

        # Creating y (the target column)
        y = df2["infection_rate"]

        # Using Pearson Correlation
        plt.figure(figsize=(12, 10))
        cor = df2.corr()
        sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
        plt.show()

        graph = sns.relplot(x="TotalRecovered", y="infection_rate", data=df2)

        # Split the data into training and test sets
        from sklearn.model_selection import train_test_split

        x_train, x_test, y_train, y_test = train_test_split(x, y)

        x_train.shape, x_test.shape, y_train.shape, y_test.shape

        # Random Forest
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier()

        clf.get_params()

        clf.fit(x_train, y_train)

        x_test.head()

        y_preds = clf.predict(x_test)

        # Evaluating the model on the test set
        clf.score(x_test, y_test)

        from sklearn.model_selection import cross_val_score

        #  cross-validation
        np.random.seed(42)
        for i in range(10, 100, 10):
            print(f"Trying model with {i} estimators...")
            model = RandomForestClassifier(n_estimators=i).fit(x_train, y_train)
            print(f"Model accuracy on test set: {model.score(x_test, y_test) * 100}%")
            print(f"Cross-validation score: {np.mean(cross_val_score(model, x, y, cv=5)) * 100}%")
            print("")

            from sklearn.linear_model import Ridge

            np.random.seed(50)

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, train_size=0.75)

            model = Ridge()
            model.fit(x_train, y_train)

            model.score(x_test, y_test)

            from sklearn.ensemble import RandomForestRegressor

            np.random.seed(50)

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,train_size=0.75)

            #
            model_r = RandomForestRegressor()
            model_r.fit(x_train, y_train)

            model_r.score(x_test, y_test)


from xgboost import XGBClassifier
model_xg = XGBClassifier()
result=model_xg.fit(x_train,y_train)
model_xg.predict(x_test)

model_xg.score(x_test, y_test)

import pickle

# Saving
pickle.dump(RandomForestRegressor(), open("RandomForestRegressor_model_1.pkl", "wb"))


# Load a saved model
loaded_pickle_model_r = pickle.load(open("RandomForestRegressor_model_1.pkl", "rb"))