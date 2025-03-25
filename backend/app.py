import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.similarity import get_title_sim, cossim_scores
import pandas as pd
from helpers import preprocessing, similarity
from collections import defaultdict

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

app = Flask(__name__)
CORS(app)

# Sample search using json with pandas
def json_search(query):
    synopsis_matches, title_matches = [], []
    query = query.split(" ")
    for word in query:
        synopsis_matches.append(kdramas_df[kdramas_df['synopsis'].str.lower().str.contains(word.lower())])
        title_matches.append(kdramas_df[kdramas_df['name'].str.lower().str.contains(word.lower())])

    synopsis_matches = pd.concat(synopsis_matches).drop_duplicates(keep="first")
    title_matches = pd.concat(title_matches).drop_duplicates(keep="first")

    matches = pd.concat([title_matches, synopsis_matches]).drop_duplicates(keep="first")

    matches_filtered = matches
    if matches_filtered.empty:
        matches_filtered = kdramas_df
    matches_filtered_json = matches_filtered.to_json(orient='records')
    return matches_filtered_json

def cossim_search(query):
    query_dictionary = similarity.query_word_counts(query)
    inv_idx = preprocessing.build_inverted_index(kdramas_df['synopsis'])
    idf_dict = preprocessing.compute_idf(inv_idx, len(kdramas_df['synopsis']))
    doc_norms = preprocessing.compute_doc_norms(inv_idx, idf_dict, len(kdramas_df['synopsis']))
    scores = similarity.dot_scores(query_dictionary, inv_idx, idf_dict)
    


def search(query):
    synopsis_score = cossim_scores()
    title_score  = get_title_sim(query, kdramas_df["name"])

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/episodes")
def episodes_search():
    text = request.args.get("title")
    return json_search(text)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)