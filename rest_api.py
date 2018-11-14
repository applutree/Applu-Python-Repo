# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 19:02:12 2017

@author: applu
"""

import os
import requests
import pandas as pd
import numpy as np

IMDB_PATH = os.path.join("datasets", "imdb")

def load_imdb_data(imdb_path=IMDB_PATH):
    print(imdb_path)
    csv_path = os.path.join(imdb_path, "IMDB-Movie-Data-indexed.csv")
    return pd.read_csv(csv_path)

imdb_loaded = load_imdb_data()
imdb_ix = pd.DataFrame(imdb_loaded)

imdb = imdb_ix.drop(imdb_ix.Index[10:])
imdb['Country'] = np.nan

#set index to track imdb dataframe
i=0
# Iteratively loop through index of imdb and get pass value to OMDB API using the Title in imdb data
for index, row in imdb.iterrows():
    omdb_resp = requests.get('http://www.omdbapi.com/?apikey=f77b6999&t=' + row['Title'] + '')
    if omdb_resp.status_code != 200:
        # This means something went wrong.
        print("Error!")
    else:
        print("Got it!")
        omdb_json = omdb_resp.json()
        omdb_df = pd.DataFrame(omdb_json)
        print(omdb_df['Title'].iloc[0] + ' ' + omdb_df['Country'].iloc[0])
        # omdb query can contain multiple rows, use iloc[0] to retrieve first element by default
        imdb['Country'].iloc[i] = omdb_df['Country'].iloc[0]
        i += 1
        

print(imdb)
# Write output to excel
writer = pd.ExcelWriter('imdb_trimmed_data_w_country_test.xlsx')
imdb.to_excel(writer, 'Sheet1')
writer.save()
