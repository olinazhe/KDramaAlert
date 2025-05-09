import json
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from helpers import preprocessing, similarity
import pandas as pd
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
    kdramas_df = preprocessing.process_data(data)
    vectorizer, synopsis_td_mat, terms = preprocessing.build_td_mat(kdramas_df)
    inv_idx = preprocessing.build_inverted_index(synopsis_td_mat, terms)
    doc_norms = preprocessing.compute_doc_norms(synopsis_td_mat)
    docs_compressed, words_compressed = preprocessing.svd_prepreprocessing(kdramas_df, vectorizer)


app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/episodes")
def episodes_search():
    text = request.args.get("title")
    keywords = request.args.get("exclude")
    include_genres = request.args.get("genre")
    exclude_genres = request.args.get("exclude_genre")
    results = json.loads(similarity.get_sim(text.lower(), kdramas_df, synopsis_td_mat, inv_idx, terms, doc_norms, vectorizer, docs_compressed, words_compressed))
    if include_genres:
        included = set(include_genres.split(","))
        results = [
            drama for drama in results
            if included.issubset(set(drama.get('genres', [])))
        ]
    if exclude_genres:
        excluded = set(exclude_genres.split(","))
        results = [
            drama for drama in results
            if excluded.isdisjoint(set(drama.get('genres', [])))
        ]
    if keywords:
        for keyword in keywords.split(","):
            keyword = keyword.strip().lower()
            if keyword: 
                filtered = []
                for drama in results:
                    if keyword in drama['name'].lower() or keyword in drama.get('synopsis').lower():
                        continue
                    else:
                        filtered.append(drama)
                results = filtered
    return {"results": results,
            "dims": similarity.get_top_latent_dims(text, vectorizer, words_compressed)}

@app.route("/<id>")
def drama(id):
    return render_template('drama.html', drama=id)

@app.route("/drama/<id>")
def drama_details(id):
    return jsonify(similarity.get_drama_details(id, kdramas_df, synopsis_td_mat, docs_compressed, vectorizer, words_compressed))

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=3000)