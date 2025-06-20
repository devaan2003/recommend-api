import pandas as pd
import numpy as np
import difflib
import re
from rapidfuzz import fuzz, process

# --- Scoring Function ---
def calculate_scores(df, rating_col, price_col, w_rating=0.7, w_price=0.3):
    df = df.copy()
    df['rating_norm'] = (df[rating_col] - df[rating_col].min()) / (df[rating_col].max() - df[rating_col].min() + 1e-6)
    df['price_norm'] = (df[price_col].max() - df[price_col]) / (df[price_col].max() - df[price_col].min() + 1e-6)
    df['score'] = w_rating * df['rating_norm'] + w_price * df['price_norm']
    return df.sort_values(by='score', ascending=False)

# --- Matching and Utility Functions ---
def match_category(input_category):
    valid_categories = ['headphones', 'laptops', 'mobiles']
    closest_match = difflib.get_close_matches(input_category.lower(), valid_categories, n=1, cutoff=0.4)
    return closest_match[0] if closest_match else None

def improved_fuzzy_match(input_list, model_list):
    matched = []
    for item in input_list:
        result = process.extractOne(item.strip(), model_list, scorer=fuzz.WRatio, score_cutoff=60)
        if result and isinstance(result, (tuple, list)):
            matched.append(result[0])
    return list(set(matched))

# --- MAIN Function ---
def recommend_product(
    salary,
    budget,
    brand_query,
    category_input,
    selected_models,
    verbose=True
):
    # --- Load datasets inside the function ---
    headphones = pd.read_csv("head.csv")
    laptops = pd.read_csv("lap.csv")
    mobiles = pd.read_csv("mob.csv")

    # --- Preprocess Headphones ---
    headphones['Rate'] = headphones['Rate'].astype(str).str.replace(',', '.').astype(float)
    headphones['Actual Price'] = pd.to_numeric(headphones['Actual Price'], errors='coerce')
    headphones['Basic Price'] = pd.to_numeric(headphones['Basic Price'], errors='coerce')
    headphones['Price'] = headphones['Actual Price'].fillna(headphones['Basic Price'])
    headphones_cleaned = headphones.dropna(subset=['Rate', 'Price'])

    # --- Preprocess Laptops ---
    laptops['Prices'] = laptops['Prices'].astype(str).replace(r'[\u20B9,₹]', '', regex=True).astype(float)
    laptops['Ratings'] = pd.to_numeric(laptops['Ratings'], errors='coerce')
    laptops_cleaned = laptops.dropna(subset=['Prices', 'Ratings'])

    # --- Preprocess Mobiles ---
    mobiles['price'] = mobiles['price'].astype(str).replace(r'[\u20B9,₹]', '', regex=True)
    mobiles['price'] = pd.to_numeric(mobiles['price'], errors='coerce')
    mobiles['ratings'] = pd.to_numeric(mobiles['ratings'], errors='coerce')
    mobiles_cleaned = mobiles.dropna(subset=['ratings', 'price'])
    mobiles_cleaned['Brand'] = mobiles_cleaned['name'].apply(lambda x: str(x).split()[0])

    # --- Start Recommendation Logic ---
    category = match_category(category_input)
    if not category:
        return {"error": "Invalid category. Choose from headphones, laptops, mobiles."}

    result = {"category": category, "brand_matches": None, "comparison": None, "best": None}

    # Filter by brand if provided
    if brand_query:
        if category == 'headphones':
            brand_matches = headphones_cleaned[(headphones_cleaned['Company'].str.lower().str.contains(brand_query.lower())) & (headphones_cleaned['Price'] <= budget)]
        elif category == 'laptops':
            brand_matches = laptops_cleaned[(laptops_cleaned['Brand'].str.lower().str.contains(brand_query.lower())) & (laptops_cleaned['Prices'] <= budget)]
        elif category == 'mobiles':
            brand_matches = mobiles_cleaned[(mobiles_cleaned['Brand'].str.lower().str.contains(brand_query.lower())) & (mobiles_cleaned['price'] <= budget)]
        else:
            brand_matches = pd.DataFrame()

        result["brand_matches"] = brand_matches.to_dict(orient="records") if not brand_matches.empty else []

    # Match and Score
    if category == 'headphones':
        model_names = headphones_cleaned['Description'].tolist()
        matched = improved_fuzzy_match(selected_models, model_names)
        filtered = headphones_cleaned[(headphones_cleaned['Description'].isin(matched)) & (headphones_cleaned['Price'] <= budget)]
        if not filtered.empty:
            scored = calculate_scores(filtered, 'Rate', 'Price')
            result["comparison"] = scored.to_dict(orient="records")
            result["best"] = scored.iloc[0].to_dict()
        else:
            result["comparison"] = []

    elif category == 'laptops':
        model_names = laptops_cleaned['Product Name'].tolist()
        matched = improved_fuzzy_match(selected_models, model_names)
        filtered = laptops_cleaned[(laptops_cleaned['Product Name'].isin(matched)) & (laptops_cleaned['Prices'] <= budget)]
        if not filtered.empty:
            scored = calculate_scores(filtered, 'Ratings', 'Prices')
            result["comparison"] = scored.to_dict(orient="records")
            result["best"] = scored.iloc[0].to_dict()
        else:
            result["comparison"] = []

    elif category == 'mobiles':
        model_names = mobiles_cleaned['name'].tolist()
        matched = improved_fuzzy_match(selected_models, model_names)
        filtered = mobiles_cleaned[(mobiles_cleaned['name'].isin(matched)) & (mobiles_cleaned['price'] <= budget)]
        if not filtered.empty:
            scored = calculate_scores(filtered, 'ratings', 'price')
            result["comparison"] = scored.to_dict(orient="records")
            result["best"] = scored.iloc[0].to_dict()
        else:
            result["comparison"] = []

    return result

