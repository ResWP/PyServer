from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
books_df = None
svd_model = None
item_factors = None
isbn_to_idx = None
idx_to_isbn = None
user_demo_data = None

def load_model_components(model_dir="model_data"):
    """Load trained model components from disk"""
    global books_df, svd_model, item_factors, isbn_to_idx, idx_to_isbn, user_demo_data
    
    try:
        # Load SVD model and item factors
        with open(os.path.join(model_dir, "svd_model.pkl"), "rb") as f:
            svd_model = pickle.load(f)
        
        with open(os.path.join(model_dir, "item_factors.pkl"), "rb") as f:
            item_factors = pickle.load(f)
        
        # Load mappings
        with open(os.path.join(model_dir, "isbn_mappings.pkl"), "rb") as f:
            mappings = pickle.load(f)
            isbn_to_idx = mappings["isbn_to_idx"]
            idx_to_isbn = mappings["idx_to_isbn"]
        
        # Load books dataframe
        books_df = pd.read_pickle(os.path.join(model_dir, "books_minimal.pkl"))
        
        # Load user demographic data
        with open(os.path.join(model_dir, "user_demo_data.pkl"), "rb") as f:
            user_demo_data = pickle.load(f)
        
        logger.info(f"Model components loaded from {model_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model components: {str(e)}")
        return False

def demographic_similarity(user_age, user_demo_data, rated_books):
    """Calculate demographic similarity scores for books based on user age"""
    try:
        demo_scores = {}
        
        # Access user age data from pre-loaded demo data
        for isbn in idx_to_isbn.values():
            # Skip books the user has already rated
            if isbn in rated_books:
                continue
                
            # Find users who rated this book
            relevant_users = []
            for user_id, demo_data in user_demo_data.items():
                if 'Age' in demo_data:
                    relevant_users.append((user_id, demo_data['Age']))
            
            if not relevant_users:
                demo_scores[isbn] = 0.5  # Neutral score
                continue
                
            # Calculate age-based similarity
            age_scores = []
            for _, age in relevant_users:
                if age is not None:
                    age_diff = abs(user_age - age)
                    # Age similarity decreases as age difference increases
                    # Max difference of 20 years leads to 0 similarity
                    age_sim = max(0, 1 - (age_diff / 20))
                    age_scores.append(age_sim)
            
            # Use average age similarity as the demographic score
            if age_scores:
                demo_scores[isbn] = sum(age_scores) / len(age_scores)
            else:
                demo_scores[isbn] = 0.5  # Neutral score
                
        return demo_scores
        
    except Exception as e:
        logger.error(f"Error calculating demographic similarity: {str(e)}")
        return {}

def get_recommendations(user_ratings, user_age=None, n_recommendations=10):
    """Generate book recommendations based on user ratings and age"""
    global books_df, item_factors, isbn_to_idx, idx_to_isbn, svd_model, user_demo_data
    
    try:
        # Convert user ratings to float values between 0-1
        user_ratings_norm = {isbn: float(rating)/10.0 for isbn, rating in user_ratings.items()}
        
        # Find which books in the user's ratings exist in our model
        valid_isbns = [isbn for isbn in user_ratings_norm.keys() if isbn in isbn_to_idx]
        
        if not valid_isbns:
            return {"error": "None of the rated books exist in our database"}
        
        # Project user ratings into latent space
        user_vector = np.zeros(svd_model.n_components)
        for isbn in valid_isbns:
            item_idx = isbn_to_idx[isbn]
            rating = user_ratings_norm[isbn]
            user_vector += rating * item_factors[item_idx]
        
        # Normalize user vector
        norm = np.linalg.norm(user_vector)
        if norm > 0:
            user_vector = user_vector / norm
        
        # Calculate similarity scores for all books
        content_scores = {}
        for idx, isbn in idx_to_isbn.items():
            # Skip books the user has already rated
            if isbn in user_ratings_norm:
                continue
                
            item_vec = item_factors[idx]
            item_norm = np.linalg.norm(item_vec)
            
            if item_norm > 0:
                # Cosine similarity
                similarity = np.dot(user_vector, item_vec) / item_norm
                content_scores[isbn] = similarity
            else:
                content_scores[isbn] = 0.0
        
        # Final scores - start with content scores
        final_scores = content_scores.copy()
        
        # Apply demographic filtering if age data is provided
        if user_age is not None:
            # Calculate demographic similarity scores
            demo_scores = demographic_similarity(user_age, user_demo_data, user_ratings_norm)
            
            # Blend content-based and demographic scores (70% content, 30% demographic)
            for isbn in content_scores:
                if isbn in demo_scores:
                    final_scores[isbn] = (0.7 * content_scores[isbn]) + (0.3 * demo_scores[isbn])
        
        # Get top N recommendations
        top_isbns = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        
        # Format recommendations with book details
        recommendations = []
        for isbn, score in top_isbns:
            book_info = books_df[books_df['ISBN'] == isbn].iloc[0]
            recommendations.append({
                'ISBN': isbn,
                'title': book_info['bookTitle'],
                'author': book_info['bookAuthor'],
                'year': book_info['yearOfPublication'],
                'publisher': book_info['publisher'],
                'score': float(score)
            })
        
        return {"recommendations": recommendations}
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return {"error": f"Failed to generate recommendations: {str(e)}"}

@app.route('/api/recommendations', methods=['POST'])
def recommend():
    """API endpoint to get book recommendations"""
    try:
        data = request.get_json()
        
        if not data or 'ratings' not in data:
            return jsonify({"error": "No ratings provided"}), 400
        
        # Get user ratings and demographic info
        user_ratings = data['ratings']
        user_age = data.get('age')
        
        # Validate ratings
        if not isinstance(user_ratings, dict):
            return jsonify({"error": "Ratings must be a dictionary with ISBN as keys and ratings as values"}), 400
        
        for isbn, rating in user_ratings.items():
            if not isinstance(rating, (int, float)) or rating < 0 or rating > 10:
                return jsonify({"error": f"Invalid rating for ISBN {isbn}. Ratings must be between 0 and 10"}), 400
        
        # Check if enough ratings provided
        if len(user_ratings) < 2:
            return jsonify({"error": "Please provide at least 2 book ratings"}), 400
        
        # Get number of recommendations
        num_recommendations = data.get('num_recommendations', 10)
        
        # Generate recommendations
        result = get_recommendations(user_ratings, user_age, num_recommendations)
        
        if "error" in result:
            return jsonify(result), 400
            
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error processing recommendation request: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    if all([books_df is not None, svd_model is not None]):
        return jsonify({
            "status": "healthy", 
            "stats": {
                "books": len(books_df),
                "model_components": svd_model.n_components,
                "demographic_data": "available" if user_demo_data else "unavailable"
            }
        }), 200
    else:
        return jsonify({"status": "unhealthy"}), 500

def initialize(model_dir="model_data"):
    """Initialize the recommendation system on startup"""
    global books_df, svd_model, item_factors, isbn_to_idx, idx_to_isbn, user_demo_data
    
    # Load pre-trained model components
    if not os.path.exists(model_dir):
        logger.error(f"Model directory '{model_dir}' not found. Please run train_model.py first.")
        return False
        
    if load_model_components(model_dir):
        logger.info("Recommendation system initialized successfully")
        return True
    else:
        logger.error("Failed to initialize the recommendation system")
        return False

if __name__ == '__main__':
    # Initialize data and model
    if initialize():
        # Run the Flask app
        app.run(host='0.0.0.0', port=5001)
    else:
        logger.error("Failed to initialize the recommendation system")