from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for models and data
models = {
    'books_df': None,
    'ratings_df': None,
    'svd_model': None,
    'item_factors': None,
    'isbn_to_idx': None,
    'idx_to_isbn': None,
    'city_popularity': None,
    'user_demo_data': None,
    'metadata': None
}

def load_models():
    """Load all model files from current directory"""
    model_files = {
        'books_df': ('books_df_slim.pkl', pd.read_pickle),
        'ratings_df': ('ratings_slim.pkl', pd.read_pickle),
        'svd_model': ('svd_model.pkl', joblib.load),
        'item_factors': ('item_factors.pkl', joblib.load),
        'isbn_to_idx': ('isbn_to_idx.pkl', joblib.load),
        'idx_to_isbn': ('idx_to_isbn.pkl', joblib.load),
        'city_popularity': ('city_popularity.pkl', joblib.load),
        'user_demo_data': ('user_demo_data.pkl', joblib.load),
        'metadata': ('metadata.pkl', joblib.load)
    }
    
    missing_files = []
    
    for model_name, (filename, loader) in model_files.items():
        try:
            if not os.path.exists(filename):
                missing_files.append(filename)
                continue
                
            logger.info(f"Loading {filename}...")
            models[model_name] = loader(filename)
        except Exception as e:
            logger.error(f"Failed to load {filename}: {str(e)}")
            missing_files.append(filename)
    
    if missing_files:
        logger.error(f"Missing or failed to load: {', '.join(missing_files)}")
        return False
        
    # Verify critical components
    critical_components = ['books_df', 'ratings_df', 'svd_model', 'item_factors']
    for component in critical_components:
        if models[component] is None:
            logger.error(f"Critical component {component} is None")
            return False
    
    logger.info(f"Successfully loaded {len(models['books_df'])} books and {len(models['ratings_df'])} ratings")
    return True

def demographic_similarity(user_age, user_city):
    """Calculate demographic similarity scores for books based on user age and city"""
    try:        
        demo_scores = {}
        
        # Normalize user city to lowercase for comparison
        if user_city:
            user_city = user_city.lower()
        
        # Prepare ratings with demographic data
        isbn_groups = {}
        for _, row in models['ratings_df'].iterrows():
            isbn = row['ISBN']
            user_id = row['userID']
            rating = row['bookRating']
            
            # Get user demographic data
            demo_data = models['user_demo_data'].get(user_id, {})
            age = demo_data.get('Age')
            city = demo_data.get('City')
            
            if isbn not in isbn_groups:
                isbn_groups[isbn] = []
                
            isbn_groups[isbn].append((rating, age, city))
        
        # Calculate scores based on age and city
        for isbn, ratings_list in isbn_groups.items():
            demo_scores_list = []
            
            for rating, age, city in ratings_list:
                # Initialize combined similarity score
                combined_sim = 0.0
                sim_factors = 0
                
                # Calculate age similarity if available
                if age is not None and user_age is not None:
                    age_diff = abs(user_age - age)
                    # Age similarity decreases as age difference increases
                    # Max difference of 20 years leads to 0 similarity
                    age_sim = max(0, 1 - (age_diff / 20))
                    combined_sim += age_sim
                    sim_factors += 1
                
                # Calculate city similarity if available
                if city is not None and user_city is not None:
                    # Exact city match gets highest similarity
                    if city.lower() == user_city:
                        city_sim = 1.0
                    else:
                        # Base similarity for different cities (using city popularity as a factor)
                        # More popular cities get a slightly higher similarity score
                        popularity_factor = models['city_popularity'].get(city, 0.01)
                        city_sim = 0.1 + (0.2 * popularity_factor)  # Between 0.1 and 0.3
                    
                    combined_sim += city_sim
                    sim_factors += 1
                
                # Calculate average similarity if we have any factors
                if sim_factors > 0:
                    final_sim = combined_sim / sim_factors
                    demo_scores_list.append((final_sim, rating))
            
            # Calculate weighted scores
            if demo_scores_list:
                sim_sum = sum(sim for sim, _ in demo_scores_list)
                if sim_sum > 0:  # Prevent division by zero
                    weighted_rating = sum(sim * rating for sim, rating in demo_scores_list) / sim_sum
                else:
                    weighted_rating = 5.0  # Neutral rating
            else:
                weighted_rating = 5.0  # Neutral rating
                
            # Final demographic score normalized to 0-1 range
            demo_scores[isbn] = weighted_rating / 10.0
        
        return demo_scores
        
    except Exception as e:
        logger.error(f"Error calculating demographic similarity: {str(e)}")
        return {}

def get_recommendations(user_ratings, user_age=None, user_city=None, n_recommendations=10):
    """Generate book recommendations based on user ratings, age, and city"""
    try:
        # Convert user ratings to float values between 0-1
        user_ratings_norm = {isbn: float(rating)/10.0 for isbn, rating in user_ratings.items()}
        
        # Find which books in the user's ratings exist in our model
        valid_isbns = [isbn for isbn in user_ratings_norm.keys() if isbn in models['isbn_to_idx']]
        
        if not valid_isbns:
            return {"error": "None of the rated books exist in our database"}
        
        # Project user ratings into latent space
        user_vector = np.zeros(models['svd_model'].n_components)
        for isbn in valid_isbns:
            item_idx = models['isbn_to_idx'][isbn]
            rating = user_ratings_norm[isbn]
            user_vector += rating * models['item_factors'][item_idx]
        
        # Normalize user vector
        norm = np.linalg.norm(user_vector)
        if norm > 0:
            user_vector = user_vector / norm
        
        # Calculate similarity scores for all books
        content_scores = {}
        for idx, isbn in models['idx_to_isbn'].items():
            # Skip books the user has already rated
            if isbn in user_ratings_norm:
                continue
                
            item_vec = models['item_factors'][idx]
            item_norm = np.linalg.norm(item_vec)
            
            if item_norm > 0:
                # Cosine similarity
                similarity = np.dot(user_vector, item_vec) / item_norm
                content_scores[isbn] = similarity
            else:
                content_scores[isbn] = 0.0
        
        # Final scores - start with content scores
        final_scores = content_scores.copy()
        
        # Apply demographic filtering if demographic data is provided
        if user_age is not None or user_city is not None:
            # Calculate demographic similarity scores
            demo_scores = demographic_similarity(user_age, user_city)
            
            # Blend content-based and demographic scores (70% content, 30% demographic)
            for isbn in content_scores:
                if isbn in demo_scores:
                    final_scores[isbn] = (0.7 * content_scores[isbn]) + (0.3 * demo_scores[isbn])
        
        # Get top N recommendations
        top_isbns = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        
        # Format recommendations with book details
        recommendations = []
        for isbn, score in top_isbns:
            book_info = models['books_df'][models['books_df']['ISBN'] == isbn]
            if not book_info.empty:
                book_info = book_info.iloc[0]
                recommendations.append({
                    'ISBN': isbn,
                    'title': book_info['bookTitle'],
                    'author': book_info['bookAuthor'],
                    'year': int(book_info['yearOfPublication']) if pd.notna(book_info['yearOfPublication']) else None,
                    'publisher': book_info['publisher'],
                    'score': float(score)
                })
        
        # Include demographic factors used in recommendation
        demographic_info = {
            "age_used": user_age is not None,
            "city_used": user_city is not None
        }
        
        return {
            "recommendations": recommendations,
            "demographic_factors": demographic_info
        }
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return {"error": f"Failed to generate recommendations: {str(e)}"}

@app.route('/recommendations', methods=['POST'])
def recommend():
    """API endpoint to get book recommendations"""
    if not all(models[key] is not None for key in ['books_df', 'ratings_df', 'svd_model']):
        return jsonify({
            "error": "Recommendation system not properly initialized",
            "status": "Service unavailable"
        }), 503
    
    try:
        data = request.get_json()
        
        if not data or 'ratings' not in data:
            return jsonify({"error": "No ratings provided"}), 400
        
        # Get user ratings and demographic info
        user_ratings = data['ratings']
        user_age = data.get('age')
        user_city = data.get('city')
        
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
        
        # Generate recommendations with demographic data
        result = get_recommendations(user_ratings, user_age, user_city, num_recommendations)
        
        if "error" in result:
            return jsonify(result), 400
            
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error processing recommendation request: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/', methods=['GET'])
def health():
    """Health check endpoint with model information"""
    if all(models[key] is not None for key in ['books_df', 'ratings_df', 'svd_model']):
        return jsonify({
            "status": "healthy", 
            "stats": {
                "books": len(models['books_df']),
                "ratings": len(models['ratings_df']),
                "model_components": models['svd_model'].n_components,
                "demographic_data": "available" if models['user_demo_data'] else "unavailable",
                "demographic_factors": ["Age", "City"] if models['user_demo_data'] else [],
                "model_timestamp": models['metadata'].get("timestamp", "unknown") if models['metadata'] else "unknown"
            }
        }), 200
    else:
        # Return which components are missing
        missing_components = [k for k, v in models.items() if v is None]
        return jsonify({
            "status": "unhealthy",
            "error": "Model or data not properly loaded",
            "missing_components": missing_components
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Endpoint to get information about the loaded model"""
    if models['metadata']:
        try:
            return jsonify({
                "model_metadata": models['metadata'],
                "memory_usage": {
                    "books_df_size": f"{models['books_df'].memory_usage(deep=True).sum() / (1024*1024):.2f} MB" 
                        if models['books_df'] is not None else "Not loaded",
                    "ratings_df_size": f"{models['ratings_df'].memory_usage(deep=True).sum() / (1024*1024):.2f} MB" 
                        if models['ratings_df'] is not None else "Not loaded",
                    "item_factors_size": f"{models['item_factors'].nbytes / (1024*1024):.2f} MB" 
                        if models['item_factors'] is not None else "Not loaded"
                },
                "model_files_location": os.path.abspath('.'),
                "available_model_files": [f for f in os.listdir('.') if f.endswith('.pkl') or f.endswith('.joblib')]
            }), 200
        except Exception as e:
            return jsonify({
                "error": f"Error retrieving model info: {str(e)}",
                "model_metadata": models['metadata']
            }), 500
    else:
        return jsonify({"error": "Model metadata not available"}), 404

if __name__ == '__main__':
    # Initialize the recommendation system
    logger.info("Initializing recommendation system...")
    initialization_successful = load_models()
    
    if initialization_successful:
        port = int(os.environ.get('PORT', 5001))
        logger.info(f"Starting recommendation service on port {port}")
        app.run(host='0.0.0.0', port=port)
    else:
        logger.error("Failed to initialize the recommendation system. Exiting.")
        exit(1)