import streamlit as st
import pandas as pd
import joblib

# Load models and data
tree_model = joblib.load("mood_decision_tree_model.pkl")
kmeans = joblib.load("mood_kmeans_model.pkl")
scaler = joblib.load("mood_scaler.pkl")
df = pd.read_csv("data.csv")

# Mood → cluster mapping
mood_to_cluster = {
    "happy": 2,
    "sad": 0,
    "chill": 3,
    "groovy": 1
}

# Questionnaire
questions = {
    "likes_dancing": "Do you enjoy dancing?",
    "prefers_loud_music": "Do you prefer loud and energetic music?",
    "likes_acoustic": "Do you enjoy calm, acoustic-style music?",
    "feels_energetic": "Are you feeling energetic right now?",
    "likes_relaxing_music": "Would you rather listen to something relaxing?"
}

st.title("🎵 Mood-Based Music Recommender")

# Init session state
if 'start_index' not in st.session_state:
    st.session_state.start_index = 0
if 'predicted_mood' not in st.session_state:
    st.session_state.predicted_mood = None
if 'matching_songs' not in st.session_state:
    st.session_state.matching_songs = pd.DataFrame()

# Questionnaire UI
with st.form("mood_form"):
    st.subheader("🧠 Tell us how you're feeling:")
    responses = []
    for key, q in questions.items():
        res = st.radio(q, ["Yes", "No"], key=key)
        responses.append(1 if res == "Yes" else 0)
    submitted = st.form_submit_button("Predict Mood")

# Prediction and clustering
if submitted:
    mood = tree_model.predict([responses])[0]
    st.session_state.predicted_mood = mood
    st.session_state.start_index = 0  # reset on new prediction

    st.success(f"🎧 Based on your answers, your mood is: **{mood.upper()}**")

    # Filter songs
    cluster = mood_to_cluster[mood]
    features = ['valence', 'energy', 'danceability', 'acousticness',
                'instrumentalness', 'speechiness', 'liveness']
    X_scaled = scaler.transform(df[features])
    df['mood_cluster'] = kmeans.predict(X_scaled)
    matching_songs = df[df['mood_cluster'] == cluster].reset_index(drop=True)
    st.session_state.matching_songs = matching_songs

# Show songs
if not st.session_state.matching_songs.empty:
    st.subheader("🎶 Recommended Songs:")
    songs = st.session_state.matching_songs
    start = st.session_state.start_index
    end = start + 5

    for i in range(start, min(end, len(songs))):
        row = songs.iloc[i]
        artists = ', '.join(eval(row['artists']))
        st.markdown(f"• **{row['name']}** — _{artists}_")

    if end < len(songs):
        if st.button("Show More"):
            st.session_state.start_index += 5
    else:
        st.markdown("✅ You've reached the end of the list.")
