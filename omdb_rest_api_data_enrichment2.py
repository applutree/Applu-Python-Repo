# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 15:45:48 2017

@author: applu
"""

import os
import requests
import pandas as pd
import numpy as np


IMDB_PATH = os.path.join("datasets", "imdb")

def load_imdb_data(imdb_path=IMDB_PATH):
    print(imdb_path)
    read_path = os.path.join(imdb_path, "movies_metadata_cleaned_11-17-17-ix-batch2.xlsx")
    #read_path = os.path.join(imdb_path, "movies_metadata_imdbid_only-ix_100rec.xlsx")
    return pd.read_excel(read_path)

imdb_loaded = load_imdb_data()
imdb_ix = pd.DataFrame(imdb_loaded)

#imdb = imdb_ix.drop(imdb_ix.Index[10:])
imdb = imdb_ix

imdb['omdb_Title'] = np.nan
imdb['omdb_Rated'] = np.nan
imdb['omdb_Actors'] = np.nan
imdb['omdb_Awards'] = np.nan
imdb['omdb_Director'] = np.nan
imdb['omdb_Writer'] = np.nan
imdb['omdb_Metascore'] = np.nan
imdb['omdb_imdbRating'] = np.nan
imdb['omdb_imdbVotes'] = np.nan

#print(imdb)

#set index to track imdb dataframe
i=0
# Iteratively loop through index of imdb and get pass value to OMDB API using the Title in imdb data
for index, row in imdb.iterrows():
    # omdb_resp = requests.get('http://www.omdbapi.com/?apikey=f77b6999&i=' + row['IMDB ID'] + '')
    #if not row['IMDB ID']:      
    omdb_resp = requests.get('http://www.omdbapi.com/?apikey=1a7666bc&i=' + row['IMDB ID'] + '')
    #print(omdb_resp.json())
    if omdb_resp.status_code != 200:
        # This means something went wrong.
        print("Error getting " + row['IMDB ID'] + " -- Title: " + row['Original Title'])
        continue
    else:
        #print("Got it!")
        try:
            omdb_json = omdb_resp.json()
            omdb_df = pd.DataFrame(omdb_json)
            if omdb_df.empty:
                print(row['IMDB ID'] + " bypassed due to empty omdb_df...")
                pass
            else:
                source_imdb_id = row['IMDB ID']
                api_imdb_id = omdb_df['imdbID'].iloc[0]
                #print("Source: " + source_imdb_id)
                #print("Export: " + api_imdb_id)
                if (source_imdb_id is api_imdb_id):
                    imdb['omdb_Title'].iloc[i] = omdb_df['Title'].iloc[0] 
                    imdb['omdb_Rated'].iloc[i] = omdb_df['Rated'].iloc[0]                
                    imdb['omdb_Actors'].iloc[i] = omdb_df['Actors'].iloc[0]
                    imdb['omdb_Awards'].iloc[i] = omdb_df['Awards'].iloc[0]
                    imdb['omdb_Director'].iloc[i] = omdb_df['Director'].iloc[0]
                    imdb['omdb_Writer'].iloc[i] = omdb_df['Writer'].iloc[0]
                    imdb['omdb_Metascore'].iloc[i] = omdb_df['Metascore'].iloc[0]
                    imdb['omdb_imdbRating'].iloc[i] = omdb_df['imdbRating'].iloc[0]
                    imdb['omdb_imdbVotes'].iloc[i] = omdb_df['imdbVotes'].iloc[0]
                    # omdb query can contain multiple rows, use iloc[0] to retrieve first element by default
                    print(str(i) + ". IMDB ID: " + row['IMDB ID'] + " -- Title: " + omdb_df['Title'].iloc[0] + ' - added!')
                else:
                    print("Source imdb does not match omdb imdb! Source: " + row['IMDB ID'] + " omdb_df: " + omdb_df['imdbID'])
                    break
        except (RuntimeError, TypeError, NameError):
            print("Error writing " + row['IMDB ID'] + " -- Title: " + row['Original Title'])
            continue               
    i += 1
     
#print(imdb)
# Write output to excel

writer = pd.ExcelWriter('metadata_enriched.xlsx')
imdb.to_excel(writer, 'Sheet1')
writer.save()

