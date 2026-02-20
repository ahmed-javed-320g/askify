# backend/cacheengine.py
import hashlib
import pickle
import os
import json
from datetime import datetime

CACHE_DIR = "cache/"

# Make sure the cache directory exists
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_cache_key(query):
    """Generate unique cache key for query"""
    return hashlib.md5(query.encode()).hexdigest()

def check_cache(query):
    """Check if query result exists in cache"""
    key = get_cache_key(query)
    path = os.path.join(CACHE_DIR, f"{key}.pkl")
    
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                cached_data = pickle.load(f)
            
            # Add timestamp if not present (backward compatibility)
            if 'timestamp' not in cached_data:
                cached_data['timestamp'] = datetime.now().isoformat()
                
            return cached_data
        except Exception as e:
            print(f"Error loading cache: {e}")
            return None
    
    return None

def save_to_cache(query, response):
    """Save query response to cache"""
    key = get_cache_key(query)
    path = os.path.join(CACHE_DIR, f"{key}.pkl")
    
    # Add timestamp to response
    response['timestamp'] = datetime.now().isoformat()
    response['query'] = query
    
    try:
        with open(path, "wb") as f:
            pickle.dump(response, f)
    except Exception as e:
        print(f"Error saving to cache: {e}")

def get_cache_stats():
    """Get statistics about cache usage"""
    if not os.path.exists(CACHE_DIR):
        return {"total_cached": 0, "cache_size_mb": 0}
    
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.pkl')]
    total_size = sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in files)
    
    return {
        "total_cached": len(files),
        "cache_size_mb": round(total_size / (1024 * 1024), 2)
    }

def clear_cache():
    """Clear all cached responses"""
    if os.path.exists(CACHE_DIR):
        for filename in os.listdir(CACHE_DIR):
            if filename.endswith('.pkl'):
                os.remove(os.path.join(CACHE_DIR, filename))
