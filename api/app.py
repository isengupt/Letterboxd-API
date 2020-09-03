import pandas as pd
from flask import Flask, jsonify, request
from tensorflow import keras
from keras.utils import get_file
from keras.layers import Input, Embedding, Dot, Reshape, Dense
from flask_cors import CORS
from keras.models import Model
import pandas as pd
import numpy as np
import pickle
import logging
import json
import collections as OrderDict
import requests


def load_obj(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def getMovieInfo(moviename):
    movieInfo = "no match"
    for movie in movie_info:
        if movie[0] == str(moviename):
            return movie[1]
            break
    
    return movieInfo


def find_similar(name, weights, index_name = 'movie', n = 10, least = False, return_dist = False, plot = False):
    """Find n most similar items (or least) to name based on embeddings. Option to also plot the results"""
    completeMovies = []
    errors = []
    print(name)
    
    if index_name == 'movie':
        index = movie_index
        rindex = index_movie
    elif index_name == 'page':
        index = link_index
        rindex = index_link
    
    
    try:
        
        dists = np.dot(weights, weights[index[name]])
    except KeyError:
        print(f'{name} Not Found.')
        errorString = f'{name} Not Found.'
        errors.append(errorString)
        return completeMovies, errors
    
    
    sorted_dists = np.argsort(dists)
    

  
    
    
    if least:
        
        closest = sorted_dists[:n]
         
        #print(f'{index_name.capitalize()}s furthest from {name}.\n')
        
    
    else:
        
        closest = sorted_dists[-n:]
        
        
        if return_dist:
            return dists, closest
        
    
        #print(f'{index_name.capitalize()}s closest to {name}.\n')
        
    
    if return_dist:
        return dists, closest
    
    
    
    max_width = max([len(rindex[c]) for c in closest])
    
    
    for c in reversed(closest):
        movieComp = {}
   
        movieComp['name'] = rindex[c]
        movieComp['score'] = str(dists[c])
        movieComp['info'] = getMovieInfo(rindex[c])
      
        completeMovies.append(movieComp)
        
    return completeMovies, errors
      

def extract_weights(name, model):
    """Extract weights from a neural network model"""
    
    # Extract weights
    weight_layer = model.get_layer(name)
    weights = weight_layer.get_weights()[0]
    
    
    weights = weights / np.linalg.norm(weights, axis = 1).reshape((-1, 1))
    return weights



app = Flask(__name__)
CORS(app)


model_file = get_file('class_attempt_class.h5', 'https://raw.githubusercontent.com/isengupt/Letterboxd-Recommender/master/data/class_attempt_class.h5')
global model
model = keras.models.load_model(model_file)
print(" * Model Loaded")


x = get_file('index_link.pkl', 'https://raw.githubusercontent.com/isengupt/Letterboxd-Recommender/master/data/index_link.pkl')
global index_link
index_link = load_obj(x)

y = get_file('link_index.pkl', 'https://raw.githubusercontent.com/isengupt/Letterboxd-Recommender/master/data/link_index.pkl')
global link_index 
link_index = load_obj(y)

z = get_file('index_movie.pkl', 'https://raw.githubusercontent.com/isengupt/Letterboxd-Recommender/master/data/index_movie.pkl')
global index_movie 
index_movie = load_obj(z)

a = get_file('movie_index.pkl', 'https://raw.githubusercontent.com/isengupt/Letterboxd-Recommender/master/data/movie_index.pkl')
global movie_index 
movie_index = load_obj(a)

b = get_file('movies_info.pkl', 'https://raw.githubusercontent.com/isengupt/Letterboxd-Recommender/master/data/movies_info.pkl')
global movies_info 
movie_info = load_obj(b)









@app.route('/predict', methods=['POST'])
def predict():
    #print(request.json['movie'])

    
    
    movie_weights_class = extract_weights('movie_embedding', model)
    completeMovies, errors = find_similar(str(request.json['movie']), movie_weights_class, n = 6)
    if (errors):
        #print("errors")
        completeMovies = errors
    
    #print(completeMovies)
    

    return jsonify(completeMovies), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0')