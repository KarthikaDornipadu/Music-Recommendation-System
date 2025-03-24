import streamlit as st
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
from sklearn.metrics.pairwise import cosine_similarity

# Load Data
tracks = pd.read_csv('tracks.csv')
ratings = pd.read_csv('ratings.csv')

# Collaborative Filtering Model
def train_cf_model():
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[['user_id', 'track_id', 'rating']], reader)
    
    trainset, testset = train_test_split(data, test_size=0.2)
    model = SVD()
    model.fit(trainset)
    
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)
    print(f"RMSE: {rmse}")
    
    return model

# Content-Based Filtering Model
def recommend_cb(track_index, top_n=5):
    tfidf_matrix = pd.get_dummies(tracks['genre'])
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    sim_scores = list(enumerate(similarity_matrix[track_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    recommendations = [tracks.iloc[i[0]]['title'] for i in sim_scores]
    return recommendations

# Streamlit UI
st.title("🎵 Music Recommendation System")

# User Input
user_id = st.number_input("Enter User ID", min_value=1, max_value=1000, step=1)
track_index = st.number_input("Enter Track Index", min_value=0, max_value=len(tracks)-1, step=1)

if st.button("Recommend"):
    cf_model = train_cf_model()
    
    # Collaborative Filtering Recommendation
    predictions = cf_model.predict(user_id, track_index)
    st.write(f"🎧 CF Prediction (Rating): {predictions.est:.2f}")
    
    # Content-Based Filtering Recommendation
    cb_recommendations = recommend_cb(track_index)
    st.write("🎼 CB Recommendations:", cb_recommendations)
