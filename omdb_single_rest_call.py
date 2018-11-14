# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 11:40:59 2017

@author: applu
"""
import requests
import pandas as pd

omdb_resp = requests.get('http://www.omdbapi.com/?apikey=1a7666bc&i=tt2069830')
omdb_resp.json()

omdb_json = omdb_resp.json()
omdb_df = pd.DataFrame(omdb_json)

if omdb_df.empty:
    print("omdb_df is empty!")
    pass
else:
    print("omdb_df is not empty!")
    print(omdb_df)
    print("Title: " + omdb_df['Year'].iloc[0] + ' - added!')
    
print("bypassed!")