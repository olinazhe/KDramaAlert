import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd

#change string representation of genres to list
def genre_to_array():
    convert = ['genre','tags','network','main-cast']
    for t in convert: 
        for idx,row in kdramas_df.iterrows():
            genre = row[t]
            if isinstance(genre, list):
                list_of_strings = genre
            else: 
                list_of_strings = genre.split(",")
            new_list = []
            for s in list_of_strings:
                new_list.append(s.strip())
            kdramas_df.at[idx, t] = new_list
    

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'init.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)
    kdramas_df = pd.DataFrame(data)
    genre_to_array()

app = Flask(__name__)
CORS(app)

# Sample search using json with pandas
def json_search(query):
    matches = []
    matches = kdramas_df[kdramas_df['synopsis'].str.lower().str.contains(query.lower())]
    matches_filtered = matches[['name', 'synopsis', 'score']]
    matches_filtered_json = matches_filtered.to_json(orient='records')
    return matches_filtered_json

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/episodes")
def episodes_search():
    text = request.args.get("title")
    return json_search(text)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)


       

