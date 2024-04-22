# app.py

import pickle
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

CLIENT_ID = "209a507b5898449d865ba0a819ee42e5"
CLIENT_SECRET = "03d72247e9fb48019df6655f0c42b4d0"

# Initialize the Spotify client
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def get_song_album_cover_url(song_name, artist_name):
    search_query = f"track:{song_name} artist:{artist_name}"
    results = sp.search(q=search_query, type="track")

    if results and results["tracks"]["items"]:
        track = results["tracks"]["items"][0]
        album_cover_url = track["album"]["images"][0]["url"]
        return album_cover_url
    else:
        return "https://i.postimg.cc/0QNxYz4V/social.png"

def recommend(song):
    # Placeholder function for recommendation
    # Example: Using the Naive Bayes model to predict similar songs based on the selected song
    song_index = df[df['song'] == song].index[0]
    song_text = df.loc[song_index, 'text']
    # Transform song text using the TF-IDF vectorizer
    song_vector = tfidf_vectorizer.transform([song_text])

    # Use KNN model to find similar songs
    _, indices = knn_model.kneighbors(song_vector)

    similar_song_names = []
    similar_song_posters = []
    for idx in indices[0]:
        if idx != song_index:
            similar_song_name = df.loc[idx, 'song']
            similar_song_artist = df.loc[idx, 'artist']
            similar_song_names.append(similar_song_name)
            similar_song_posters.append(get_song_album_cover_url(similar_song_name, similar_song_artist))
    
    return similar_song_names, similar_song_posters


# Load serialized objects
df = pickle.load(open('df.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
knn_model = pickle.load(open('knn_model.pkl', 'rb'))
nb_text_classifier = pickle.load(open('nb_text_classifier.pkl', 'rb'))
dt_classifier = pickle.load(open('dt_classifier.pkl', 'rb'))

st.header('Music Recommender System')
music_list = df['song'].values
selected_song = st.selectbox(
    "Type or select a song from the dropdown",
    music_list
)

if st.button('Show Recommendation'):
    recommended_music_names, recommended_music_posters = recommend(selected_song)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_music_names[0])
        st.image(recommended_music_posters[0])
    with col2:
        st.text(recommended_music_names[1])
        st.image(recommended_music_posters[1])
    with col3:
        st.text(recommended_music_names[2])
        st.image(recommended_music_posters[2])
