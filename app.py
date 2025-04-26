from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import logging
import os
import gc
import time
import signal
import psutil
from flask_cors import CORS
from functools import lru_cache
from datetime import datetime, timedelta
import threading
from functools import wraps

def timeout(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [TimeoutError("Function timed out")]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)
            
            if thread.is_alive():
                return {"error": "Operation timed out, using fallback recommendations",
                        "recommendations": get_simple_recommendations(*args, **kwargs)}
            
            if isinstance(result[0], Exception):
                raise result[0]
            return result[0]
        
        return wrapper
    return decorator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains on all routes

# Simple in-memory cache for recommendations
recommendation_cache = {}
cache_expiry = {}  # To track when cache entries should expire
CACHE_DURATION = 3600  # Cache duration in seconds (1 hour)

# Initialize for demographic cache
demographic_cache = {}

class TimeoutError(Exception):
    """Custom exception for operation timeouts"""
    pass

def timeout_handler(signum, frame):
    """Signal handler for timeouts"""
    raise TimeoutError("Operation timed out")

# Load models with optimized memory usage
try:
    logger.info("Loading recommendation system models...")
    start_time = time.time()
    
    # Load only essential columns to reduce memory usage
    books_df = pd.read_pickle("models/books_df_slim.pkl")
    
    # Convert string columns to categorical to save memory
    for col in ['bookAuthor', 'publisher']:
        if col in books_df.columns:
            books_df[col] = books_df[col].astype('category')
    
    # Create book lookup dictionary for faster access
    books_dict = books_df.set_index('ISBN').to_dict('index')
    
    # Load ratings with only necessary columns
    ratings_df = pd.read_pickle("models/ratings_slim.pkl")
    
    # Load other model components
    svd_model = joblib.load("models/svd_model.pkl")
    item_factors = joblib.load("models/item_factors.pkl")
    isbn_to_idx = joblib.load("models/isbn_to_idx.pkl")
    idx_to_isbn = joblib.load("models/idx_to_isbn.pkl")
    city_popularity = joblib.load("models/city_popularity.pkl")
    user_demo_data = joblib.load("models/user_demo_data.pkl")
    metadata = joblib.load("models/metadata.pkl")
    
    # Report memory usage
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 * 1024)
    
    load_time = time.time() - start_time
    logger.info(f"Successfully loaded {len(books_df)} books and {len(ratings_df)} ratings in {load_time:.2f} seconds")
    logger.info(f"Current memory usage: {memory_usage:.2f} MB")
    models_loaded = True
    
    # Force garbage collection after loading models
    gc.collect()
    
except FileNotFoundError as e:
    logger.error(f"Error: Model files not found. {str(e)}")
    models_loaded = False
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    models_loaded = False

def precompute_demographic_affinities():
    """Pre-compute demographic affinities for common age/city combinations"""
    try:
        logger.info("Pre-computing demographic affinities for common combinations...")
        start_time = time.time()
        
        # Common age groups
        common_ages = range(10, 80, 10)  # Age buckets: 10, 20, 30, etc.
        
        # Get top 20 cities by popularity
        popular_cities = sorted(city_popularity.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Pre-compute for ages
        for age in common_ages:
            demographic_cache[f"age_{age}"] = demographic_similarity_optimized(age, None)
            
        # Pre-compute for cities
        for city, _ in popular_cities:
            demographic_cache[f"city_{city.lower()}"] = demographic_similarity_optimized(None, city)
            
        # Pre-compute for common age-city combinations
        for age in [20, 30, 40]:  # Most common age groups
            for city, _ in popular_cities[:5]:  # Top 5 cities
                demographic_cache[f"age_{age}_city_{city.lower()}"] = demographic_similarity_optimized(age, city)
        
        compute_time = time.time() - start_time
        logger.info(f"Pre-computed {len(demographic_cache)} demographic affinity combinations in {compute_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error during demographic pre-computation: {str(e)}")

def demographic_similarity_optimized(user_age, user_city):
    """Vectorized implementation of demographic similarity calculation"""
    try:
        # Skip if no demographic data provided
        if user_age is None and user_city is None:
            return {}
            
        # Check if we have a cached result
        cache_key = None
        if user_age is not None and user_city is not None:
            cache_key = f"age_{user_age}_city_{user_city.lower()}"
        elif user_age is not None:
            cache_key = f"age_{user_age}"
        elif user_city is not None:
            cache_key = f"city_{user_city.lower()}"
            
        if cache_key in demographic_cache:
            return demographic_cache[cache_key]
        
        # Initialize empty result
        demo_scores = {}
        
        # Process in batches to reduce memory usage
        batch_size = 50000  # Adjust based on available memory
        num_batches = (len(ratings_df) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(ratings_df))
            
            # Get current batch
            batch_df = ratings_df.iloc[start_idx:end_idx].copy()
            
            # Add user demographic data
            batch_df['user_age'] = batch_df['userID'].map(
                lambda uid: user_demo_data.get(uid, {}).get('Age')
            )
            batch_df['user_city'] = batch_df['userID'].map(
                lambda uid: user_demo_data.get(uid, {}).get('City')
            )
            
            # Calculate similarities only if we have data
            if user_age is not None:
                batch_df['age_diff'] = batch_df['user_age'].apply(
                    lambda x: abs(x - user_age) if pd.notna(x) else None
                )
                batch_df['age_sim'] = batch_df['age_diff'].apply(
                    lambda x: max(0, 1 - (x / 20)) if pd.notna(x) else 0
                )
                batch_df['has_age_sim'] = batch_df['age_sim'] > 0
            else:
                batch_df['age_sim'] = 0
                batch_df['has_age_sim'] = False
                
            if user_city is not None:
                user_city_lower = user_city.lower()
                batch_df['city_sim'] = batch_df['user_city'].apply(
                    lambda x: 1.0 if pd.notna(x) and x.lower() == user_city_lower
                    else 0.1 + (0.2 * city_popularity.get(x, 0.01)) if pd.notna(x)
                    else 0
                )
                batch_df['has_city_sim'] = batch_df['city_sim'] > 0
            else:
                batch_df['city_sim'] = 0
                batch_df['has_city_sim'] = False
                
            # Count factors and calculate similarity
            batch_df['sim_factors'] = batch_df['has_age_sim'].astype(int) + batch_df['has_city_sim'].astype(int)
            batch_df['combined_sim'] = batch_df['age_sim'] + batch_df['city_sim']
            
            # Calculate final similarity and weighted ratings
            batch_df['final_sim'] = batch_df.apply(
                lambda row: row['combined_sim'] / row['sim_factors'] if row['sim_factors'] > 0 else 0,
                axis=1
            )
            batch_df['weighted_rating'] = batch_df['final_sim'] * batch_df['bookRating']
            
            # Group by ISBN and aggregate
            isbn_group = batch_df.groupby('ISBN').agg(
                sim_sum=('final_sim', 'sum'),
                weighted_sum=('weighted_rating', 'sum')
            )
            
            # Update the results
            for isbn, row in isbn_group.iterrows():
                if isbn not in demo_scores:
                    demo_scores[isbn] = {'sim_sum': 0, 'weighted_sum': 0}
                
                demo_scores[isbn]['sim_sum'] += row['sim_sum']
                demo_scores[isbn]['weighted_sum'] += row['weighted_sum']
                
            # Force garbage collection after each batch
            del batch_df
            gc.collect()
        
        # Calculate final scores
        final_scores = {}
        for isbn, data in demo_scores.items():
            if data['sim_sum'] > 0:
                final_scores[isbn] = (data['weighted_sum'] / data['sim_sum']) / 10.0
            else:
                final_scores[isbn] = 0.5  # Neutral score if no similarity
        
        # Cache the result if we have a cache key
        if cache_key is not None:
            demographic_cache[cache_key] = final_scores
            
        return final_scores
        
    except Exception as e:
        logger.error(f"Error calculating demographic similarity: {str(e)}")
        return {}

def demographic_similarity_batched(user_age, user_city):
    """Process demographic similarity in batches to reduce memory usage"""
    try:
        if user_age is None and user_city is None:
            return {}
            
        # Cache key for lookup
        cache_key = None
        if user_age is not None and user_city is not None:
            cache_key = f"age_{user_age}_city_{user_city.lower()}"
        elif user_age is not None:
            cache_key = f"age_{user_age}"
        elif user_city is not None:
            cache_key = f"city_{user_city.lower()}"
            
        if cache_key in demographic_cache:
            return demographic_cache[cache_key]
        
        demo_scores = {}
        user_city = user_city.lower() if user_city else None
        
        # Process in batches to reduce memory usage
        batch_size = 10000
        
        for i in range(0, len(ratings_df), batch_size):
            batch = ratings_df.iloc[i:i+batch_size]
            
            for _, row in batch.iterrows():
                isbn = row['ISBN']
                user_id = row['userID']
                rating = row['bookRating']
                
                # Get user demographic data
                demo_data = user_demo_data.get(user_id, {})
                age = demo_data.get('Age')
                city = demo_data.get('City')
                
                # Initialize scores
                combined_sim = 0.0
                sim_factors = 0
                
                # Age similarity
                if age is not None and user_age is not None:
                    age_diff = abs(user_age - age)
                    age_sim = max(0, 1 - (age_diff / 20))
                    combined_sim += age_sim
                    sim_factors += 1
                
                # City similarity
                if city is not None and user_city is not None:
                    if city.lower() == user_city:
                        city_sim = 1.0
                    else:
                        popularity_factor = city_popularity.get(city, 0.01)
                        city_sim = 0.1 + (0.2 * popularity_factor)
                    
                    combined_sim += city_sim
                    sim_factors += 1
                
                # Calculate final similarity
                if sim_factors > 0:
                    final_sim = combined_sim / sim_factors
                    
                    # Update scores
                    if isbn not in demo_scores:
                        demo_scores[isbn] = {'sim_sum': 0, 'weighted_sum': 0}
                    
                    demo_scores[isbn]['sim_sum'] += final_sim
                    demo_scores[isbn]['weighted_sum'] += final_sim * rating
            
            # Force garbage collection after each batch
            del batch
            gc.collect()
        
        # Calculate final scores
        final_scores = {}
        for isbn, data in demo_scores.items():
            if data['sim_sum'] > 0:
                final_scores[isbn] = (data['weighted_sum'] / data['sim_sum']) / 10.0
            else:
                final_scores[isbn] = 0.5  # Neutral score
        
        # Cache the result
        if cache_key is not None:
            demographic_cache[cache_key] = final_scores
            
        return final_scores
        
    except Exception as e:
        logger.error(f"Error in batched demographic similarity: {str(e)}")
        return {}

def get_recommendations(user_ratings, user_age=None, user_city=None, n_recommendations=10, timeout_seconds=20):
    """Generate book recommendations based on user ratings, age, and city with timeout"""
    try:
        # Set timeout handler
        start_time = time.time()
        
        # Convert user ratings to float values between 0-1
        user_ratings_norm = {isbn: float(rating)/10.0 for isbn, rating in user_ratings.items()}
        
        # Find which books in the user's ratings exist in our model
        valid_isbns = [isbn for isbn in user_ratings_norm.keys() if isbn in isbn_to_idx]
        
        if not valid_isbns:
            signal.alarm(0)  # Cancel the alarm
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
        
        # Process in batches to avoid memory issues
        batch_size = 1000
        num_items = len(idx_to_isbn)
        
        for batch_start in range(0, num_items, batch_size):
            batch_end = min(batch_start + batch_size, num_items)
            
            for idx in range(batch_start, batch_end):
                isbn = idx_to_isbn.get(idx)
                
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
        
        # Check time before proceeding with demographic filtering
        current_time = time.time()
        time_spent = current_time - start_time
        
        demographic_info = {"age_used": False, "city_used": False}
        
        # Apply demographic filtering if there's enough time left and demographic data provided
        if (time_spent < timeout_seconds * 0.7) and (user_age is not None or user_city is not None):
            try:
                # Calculate demographic similarity scores
                demo_scores = demographic_similarity_optimized(user_age, user_city)
                
                # Update demographic info
                demographic_info["age_used"] = user_age is not None
                demographic_info["city_used"] = user_city is not None
                
                # Blend content-based and demographic scores (70% content, 30% demographic)
                for isbn in content_scores:
                    if isbn in demo_scores:
                        final_scores[isbn] = (0.7 * content_scores[isbn]) + (0.3 * demo_scores[isbn])
            except TimeoutError:
                logger.warning("Demographic similarity calculation timed out, using content scores only")
            except Exception as e:
                logger.error(f"Error in demographic calculation: {str(e)}")
        
        # Get top N recommendations
        top_isbns = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        
        # Format recommendations with book details
        recommendations = []
        for isbn, score in top_isbns:
            book_info = books_dict.get(isbn)
            if book_info:
                recommendations.append({
                    'ISBN': isbn,
                    'title': book_info['bookTitle'],
                    'author': book_info['bookAuthor'],
                    'year': int(book_info['yearOfPublication']) if pd.notna(book_info['yearOfPublication']) else None,
                    'publisher': book_info['publisher'],
                    'score': float(score)
                })
        
        # Cancel the alarm
        signal.alarm(0)
        
        # Log performance metrics
        total_time = time.time() - start_time
        logger.info(f"Generated {len(recommendations)} recommendations in {total_time:.2f} seconds")
        
        return {
            "recommendations": recommendations,
            "demographic_factors": demographic_info,
            "processing_time_ms": int(total_time * 1000)
        }
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return {"error": f"Failed to generate recommendations: {str(e)}"}
        
 
def get_simple_recommendations(user_ratings, user_age=None, user_city=None, n_recommendations=10):
    """Simplified recommendation method for fallback"""
    try:
        start_time = time.time()
        
        # Use only basic content similarity - no demographic data
        user_ratings_norm = {isbn: float(rating)/10.0 for isbn, rating in user_ratings.items()}
        valid_isbns = [isbn for isbn in user_ratings_norm.keys() if isbn in books_dict]
        
        if not valid_isbns:
            return {"error": "None of the rated books exist in our database"}
        
        # Find similar books based on genres/authors
        similar_books = []
        
        # Get authors and publishers of rated books
        favorite_authors = set()
        favorite_publishers = set()
        
        for isbn in valid_isbns:
            if isbn in books_dict:
                book = books_dict[isbn]
                if 'bookAuthor' in book and book['bookAuthor']:
                    favorite_authors.add(book['bookAuthor'])
                if 'publisher' in book and book['publisher']:
                    favorite_publishers.add(book['publisher'])
        
        # Score all candidate books
        candidates = []
        
        # Process in batches
        batch_size = 1000
        for start_idx in range(0, len(books_df), batch_size):
            end_idx = min(start_idx + batch_size, len(books_df))
            batch = books_df.iloc[start_idx:end_idx]
            
            for _, book in batch.iterrows():
                isbn = book['ISBN']
                
                # Skip books the user has already rated
                if isbn in valid_isbns:
                    continue
                
                score = 0.0
                
                # Author match
                if book['bookAuthor'] in favorite_authors:
                    score += 0.6
                
                # Publisher match
                if book['publisher'] in favorite_publishers:
                    score += 0.3
                
                # Recent books get a small boost
                try:
                    year = int(book['yearOfPublication']) if pd.notna(book['yearOfPublication']) else 0
                    if year > 2000:
                        score += 0.1
                except:
                    pass
                
                if score > 0:
                    candidates.append((isbn, score))
        
        # Sort candidates by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Get top recommendations
        recommendations = []
        for isbn, score in candidates[:n_recommendations]:
            book = books_dict.get(isbn)
            if book:
                recommendations.append({
                    'ISBN': isbn,
                    'title': book['bookTitle'],
                    'author': book['bookAuthor'],
                    'year': int(book['yearOfPublication']) if pd.notna(book['yearOfPublication']) else None,
                    'publisher': book['publisher'],
                    'score': float(score)
                })
        
        total_time = time.time() - start_time
        logger.info(f"Generated {len(recommendations)} simple recommendations in {total_time:.2f} seconds")
        
        return {
            "recommendations": recommendations,
            "demographic_factors": {"age_used": False, "city_used": False},
            "note": "Used simplified recommendation engine for better performance",
            "processing_time_ms": int(total_time * 1000)
        }
        
    except Exception as e:
        logger.error(f"Error generating simple recommendations: {str(e)}")
        return {"error": f"Failed to generate recommendations: {str(e)}"}

def clean_cache():
    """Remove expired cache entries"""
    now = datetime.now()
    expired_keys = [k for k, v in cache_expiry.items() if now > v]
    
    for key in expired_keys:
        if key in recommendation_cache:
            del recommendation_cache[key]
        del cache_expiry[key]
        
    logger.info(f"Cleaned {len(expired_keys)} expired cache entries. Current cache size: {len(recommendation_cache)}")

@app.route('/recommendations', methods=['POST'])
def recommend():
    """API endpoint to get book recommendations"""
    if not models_loaded:
        return jsonify({
            "error": "Recommendation system not properly initialized",
            "status": "Service unavailable"
        }), 503
    
    try:
        # Clean cache periodically
        if len(cache_expiry) > 100:  # Only clean when cache gets large
            clean_cache()
        
        start_time = time.time()
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
        
        # Build cache key
        cache_key = f"{hash(frozenset(user_ratings.items()))}-{user_age}-{user_city}-{num_recommendations}"
        
        # Check cache first
        if cache_key in recommendation_cache:
            logger.info(f"Cache hit for recommendations - serving cached result")
            result = recommendation_cache[cache_key]
            result["cache_hit"] = True
            return jsonify(result), 200
        
        # Generate recommendations with demographic data and timeout
        result = get_recommendations(user_ratings, user_age, user_city, num_recommendations)
        
        if "error" in result:
            return jsonify(result), 400
        
        # Store in cache
        recommendation_cache[cache_key] = result
        cache_expiry[cache_key] = datetime.now() + timedelta(seconds=CACHE_DURATION)
        
        # Add processing metadata
        total_time = time.time() - start_time
        result["api_processing_time_ms"] = int(total_time * 1000)
        result["cache_hit"] = False
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error processing recommendation request: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500
    
@app.route('/', methods=['GET'])
def health():
    """Health check endpoint with model information"""
    if models_loaded:
        # Get memory usage
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return jsonify({
            "status": "healthy", 
            "stats": {
                "books": len(books_df),
                "ratings": len(ratings_df),
                "model_components": svd_model.n_components,
                "demographic_data": "available" if user_demo_data else "unavailable",
                "demographic_factors": ["Age", "City"] if user_demo_data else [],
                "model_timestamp": metadata.get("timestamp", "unknown") if metadata else "unknown",
                "cache_size": len(recommendation_cache),
                "memory_usage_mb": f"{memory_info.rss / (1024 * 1024):.2f}",
                "uptime_seconds": time.time() - app.start_time if hasattr(app, 'start_time') else 0
            }
        }), 200
    else:
        return jsonify({
            "status": "unhealthy",
            "error": "Model or data not properly loaded"
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Endpoint to get information about the loaded model"""
    if models_loaded and metadata:
        try:
            # Get memory usage details
            memory_stats = {}
            for var_name, var in [
                ("books_df", books_df), 
                ("ratings_df", ratings_df),
                ("item_factors", item_factors)
            ]:
                if isinstance(var, pd.DataFrame):
                    memory_stats[f"{var_name}_size"] = f"{var.memory_usage(deep=True).sum() / (1024*1024):.2f} MB"
                elif isinstance(var, np.ndarray):
                    memory_stats[f"{var_name}_size"] = f"{var.nbytes / (1024*1024):.2f} MB"
            
            # Get cache statistics
            cache_stats = {
                "recommendation_cache_size": len(recommendation_cache),
                "demographic_cache_size": len(demographic_cache),
                "cached_combinations": list(demographic_cache.keys())[:10]  # Show first 10 cached combinations
            }
            
            return jsonify({
                "model_metadata": metadata,
                "memory_usage": memory_stats,
                "cache_stats": cache_stats,
                "model_files_location": os.path.abspath('models'),
                "available_model_files": [f for f in os.listdir('models') if f.endswith('.pkl') or f.endswith('.joblib')]
            }), 200
        except Exception as e:
            return jsonify({
                "error": f"Error retrieving model info: {str(e)}",
                "model_metadata": metadata
            }), 500
    else:
        return jsonify({"error": "Model metadata not available"}), 404

@app.route('/memory', methods=['GET'])
def memory_usage():
    """Endpoint to get current memory usage"""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return jsonify({
            "memory_used_mb": memory_info.rss / (1024 * 1024),
            "memory_percent": process.memory_percent(),
            "virtual_memory": dict(psutil.virtual_memory()._asdict()),
            "swap_memory": dict(psutil.swap_memory()._asdict())
        }), 200
    except Exception as e:
        return jsonify({"error": f"Error getting memory info: {str(e)}"}), 500

@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Endpoint to clear recommendation cache"""
    try:
        # Get authentication token (implement proper auth in production)
        auth_token = request.headers.get('Authorization')
        if not auth_token or auth_token != f"Bearer {os.environ.get('ADMIN_TOKEN', 'admin')}":
            return jsonify({"error": "Unauthorized"}), 401
        
        # Clear caches
        cache_size = len(recommendation_cache)
        demographic_cache_size = len(demographic_cache)
        
        recommendation_cache.clear()
        cache_expiry.clear()
        
        # Don't clear pre-computed demographic data
        
        return jsonify({
            "status": "success",
            "cleared_items": cache_size,
            "demographic_cache_size": demographic_cache_size,
            "message": "Recommendation cache cleared successfully"
        }), 200
    except Exception as e:
        return jsonify({"error": f"Error clearing cache: {str(e)}"}), 500

# Custom error handling
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Record start time for uptime tracking
    app.start_time = time.time()
    
    # Pre-compute demographic affinities
    if models_loaded:
        precompute_demographic_affinities()
    
    # Start the server
    port = int(os.environ.get('PORT', 5001))
    logger.info(f"Starting recommendation service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)