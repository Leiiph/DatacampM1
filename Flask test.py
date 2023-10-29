from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import keras
import transformers
import sentencepiece
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import sentencepiece as spm
from transformers import T5Tokenizer
import sentencepiece as spm
from transformers import T5Tokenizer

app = Flask(__name__)

def load():
    de = pd.read_csv('tcc_ceds_music.csv')
    return de

@app.route('/')
def index():
    return render_template('home.html')


def recommendation(input):
    # Initialize the tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    genres = {0: 'dating', 1: 'violence', 2: 'world/life', 3: 'night/time', 4: 'shake the audience', 5: 'family/gospel', 6: 'romantic', 7: 'communication', 8: 'obscene', 9: 'music', 10: 'movement/places', 11: 'light/visual perceptions', 12: 'family/spiritual', 13: 'like/girls', 14: 'sadness', 15: 'feelings', 16: 'danceability', 17: 'loudness', 18: 'acousticness', 19: 'instrumentalness', 20: 'valence', 21: 'energy'}
    model_path = "my_model_d37.h5"
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
    grouped_de = de.groupby(predicted_genre)
    top_tracks = grouped_de[['artist_name', 'track_name']].head(5)
    return top_tracks

@app.route('/process_text', methods=['POST'])
def process_text():
    de = load()
    user_input = request.form.get('user_input')
    singer_name = request.form.get('singer_name')
    song_name = request.form.get('song_name')

    show_lyric_input = True  # Assume lyric input is shown by default
    lyric_input_invalid = False  # Assume lyric input is valid

    if singer_name and song_name:
        # Check if singer_name and song_name exist in the DataFrame
        song_info = de[(de['artist_name'] == singer_name) & (de['track_name'] == song_name)]

        if not song_info.empty:
            # Use the lyrics from the DataFrame if the song info exists
            lyrics = song_info['lyrics'].values[0]
            show_lyric_input = False  # Don't show lyric input
        else:
            # Prompt the user to enter lyrics
            lyrics = user_input
            lyric_input_invalid = True
    else:
        # If either singer_name or song_name is missing, use the user's input
        lyrics = user_input

    # The rest of your code to process the lyrics and provide recommendations

    predicted_genre = recommendation(lyrics)

    return render_template('result_template.html', user_input=user_input, predicted_genre=predicted_genre, rec_music=rec_music(de, predicted_genre), show_lyric_input=show_lyric_input, lyric_input_invalid=lyric_input_invalid)

if __name__ == '__main__':
 
    # run() method of Flask class runs the application 
    # on the local development server.
    app.run()