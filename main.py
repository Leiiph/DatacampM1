from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sentencepiece as spm
from transformers import T5Tokenizer

app = Flask(__name__)

def load():
    '''Load the data from the csv'''
    de = pd.read_csv('tcc_ceds_music.csv')
    return de

@app.route('/')
def index():
    '''load the main page'''
    return render_template('home.html')


def recommendation(input):
    '''Find the mood of the input lyrics - main function'''
    # Initialize the tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    genres = {0: 'dating', 1: 'violence', 2: 'world/life', 3: 'night/time', 4: 'shake the audience', 5: 'family/gospel', 6: 'romantic', 7: 'communication', 8: 'obscene', 9: 'music', 10: 'movement/places', 11: 'light/visual perceptions', 12: 'family/spiritual', 13: 'like/girls', 14: 'sadness', 15: 'feelings', 16: 'danceability', 17: 'loudness', 18: 'acousticness', 19: 'instrumentalness', 20: 'valence', 21: 'energy'}
    model_path = "my_model_d372.h5"
    loaded_model = load_model(model_path)
    # Tokenize the text
    sample_text = input
    tokenized_sample = [tokenizer.encode(sample_text, truncation=True, max_length=512)]

    # Pad the sequence
    padded_sample = pad_sequences(tokenized_sample, maxlen=252)

    # Make a prediction
    prediction = loaded_model.predict(padded_sample)

    # Interpret the prediction
    predicted_class = np.argmax(prediction)
    predicted_genre = genres[predicted_class]
    return predicted_genre

def rec_music(de, predicted_genre):
    '''Will recommed music based on the predicted genre'''
    stock = de[de['topic'] == predicted_genre]
    if stock.empty:
        return "Sorry, we don't have any recommendation for this genre yet."
    else:
        stock = stock.sample(n=3)
        text = ""
        for i in range(len(stock)):
            text += str(stock.iloc[i]['artist_name']).title() + " - " + str(stock.iloc[i]['track_name']).title() + " | "
        return text

@app.route('/process_text', methods=['POST'])
def process_text():
    '''Process the text input by the user'''
    de = load()
    user_input = request.form.get('user_input')
    singer_name = request.form.get('singer_name')
    song_name = request.form.get('song_name')
    # Lower all letters (as the database uses only lower case)
    singer_name = singer_name.lower()
    song_name = song_name.lower()

    show_lyric_input = True  # Assume lyric input is shown by default
    lyric_input_invalid = False  # Assume lyric input is valid
    lyrics_exist = False  # Assume lyrics do not exist in the database by default


    if singer_name and song_name:
        song_info = de[(de['artist_name'] == singer_name) & (de['track_name'] == song_name)]

        if not song_info.empty:
            lyrics_exist = True
            show_lyric_input = False 
        else:
            # User enter lyrics
            lyrics_exist = False
            lyrics = user_input
            lyric_input_invalid = True
    else:
        lyrics = user_input


    # The rest of your code to process the lyrics and provide recommendations
    if lyrics_exist == True:
        predicted_genre = song_info['topic'].values[0]
    else:
        predicted_genre = recommendation(lyrics)

    return render_template('result_template.html', predicted_genre=str(predicted_genre).title(), rec_music=rec_music(de, predicted_genre), show_lyric_input=show_lyric_input, lyric_input_invalid=lyric_input_invalid)

if __name__ == '__main__':
    app.run()
