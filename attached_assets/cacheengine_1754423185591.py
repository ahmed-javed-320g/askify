# chatbot/cache_engine.py
import hashlib
import pickle
import os

CACHE_DIR = "cache/"

# ✅ Make sure the cache directory exists
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


def get_cache_key(query):
    return hashlib.md5(query.encode()).hexdigest()

def check_cache(query):
    key = get_cache_key(query)
    path = os.path.join(CACHE_DIR, f"{key}.pkl")
    return pickle.load(open(path, "rb")) if os.path.exists(path) else None

def save_to_cache(query, response):
    key = get_cache_key(query)
    path = os.path.join(CACHE_DIR, f"{key}.pkl")
    with open(path, "wb") as f:
        pickle.dump(response, f)
