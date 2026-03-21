# ============================================
# app.py — KrishiSetu Platform
# ============================================

import os
import requests
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from flask import (Flask, render_template,
                   request, redirect,
                   url_for, session, flash)
from werkzeug.utils import secure_filename
import tensorflow as tf
import cv2
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

app = Flask(__name__)
app.secret_key = 'KrishiSetu_secret_2024'

# --- Config ---
app.config['UPLOAD_FOLDER']      = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMG_SIZE   = 224
MODEL_PATH = '../model/mobilenet_finetuned.keras'
NAMES_PATH = '../model/class_names.json'

# --- Load Model ---
print("📂 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
with open(NAMES_PATH, 'r') as f:
    class_names = json.load(f)
print(f"✅ Model loaded! Classes: {len(class_names)}")

# ============================================
# Shop Products Database
# ============================================
PRODUCTS = {
    'seeds': [
        {
            'id': 's001',
            'name': 'Tomato Hybrid Seeds',
            'brand': 'AgroPlus',
            'price': 299,
            'unit': '100g pack',
            'rating': 4.5,
            'image': '🍅',
            'category': 'seeds',
            'description': 'High-yield, disease-resistant hybrid variety',
            'diseases': ['Tomato___Early_blight',
                        'Tomato___Late_blight',
                        'Tomato___healthy']
        },
        {
            'id': 's002',
            'name': 'Potato Certified Seeds',
            'brand': 'FarmKing',
            'price': 450,
            'unit': '1kg pack',
            'rating': 4.3,
            'image': '🥔',
            'category': 'seeds',
            'description': 'Blight-resistant certified seed potatoes',
            'diseases': ['Potato___Early_blight',
                        'Potato___Late_blight',
                        'Potato___healthy']
        },
        {
            'id': 's003',
            'name': 'Bell Pepper F1 Seeds',
            'brand': 'SeedTech',
            'price': 199,
            'unit': '50 seeds',
            'rating': 4.6,
            'image': '🫑',
            'category': 'seeds',
            'description': 'Bacterial spot resistant F1 hybrid',
            'diseases': ['Pepper,_bell___Bacterial_spot',
                        'Pepper,_bell___healthy']
        },
        {
            'id': 's004',
            'name': 'Corn Hybrid Seeds',
            'brand': 'AgroPlus',
            'price': 380,
            'unit': '500g pack',
            'rating': 4.4,
            'image': '🌽',
            'category': 'seeds',
            'description': 'Northern leaf blight resistant variety',
            'diseases': ['Corn_(maize)___Northern_Leaf_Blight',
                        'Corn_(maize)___Common_rust_',
                        'Corn_(maize)___healthy']
        },
        {
            'id': 's005',
            'name': 'Grape Vine Cuttings',
            'brand': 'VineGrow',
            'price': 850,
            'unit': 'per cutting',
            'rating': 4.2,
            'image': '🍇',
            'category': 'seeds',
            'description': 'Disease-resistant table grape variety',
            'diseases': ['Grape___Black_rot',
                        'Grape___healthy']
        },
        {
            'id': 's006',
            'name': 'Apple Rootstock',
            'brand': 'OrchardPro',
            'price': 1200,
            'unit': 'per plant',
            'rating': 4.7,
            'image': '🍎',
            'category': 'seeds',
            'description': 'Scab and rust resistant rootstock',
            'diseases': ['Apple___Apple_scab',
                        'Apple___Cedar_apple_rust',
                        'Apple___healthy']
        },
    ],
    'medicines': [
        {
            'id': 'm001',
            'name': 'Mancozeb Fungicide',
            'brand': 'CropShield',
            'price': 320,
            'unit': '250g pack',
            'rating': 4.6,
            'image': '🧴',
            'category': 'medicines',
            'description': 'Broad-spectrum fungicide for blight control',
            'diseases': ['Tomato___Early_blight',
                        'Tomato___Late_blight',
                        'Potato___Early_blight',
                        'Potato___Late_blight']
        },
        {
            'id': 'm002',
            'name': 'Copper Hydroxide Spray',
            'brand': 'BioProtect',
            'price': 480,
            'unit': '500ml bottle',
            'rating': 4.4,
            'image': '🧪',
            'category': 'medicines',
            'description': 'Effective against bacterial and fungal diseases',
            'diseases': ['Tomato___Bacterial_spot',
                        'Pepper,_bell___Bacterial_spot',
                        'Peach___Bacterial_spot']
        },
        {
            'id': 'm003',
            'name': 'Neem Oil Pesticide',
            'brand': 'NatureCure',
            'price': 280,
            'unit': '200ml bottle',
            'rating': 4.5,
            'image': '🌿',
            'category': 'medicines',
            'description': 'Organic solution for spider mites and pests',
            'diseases': ['Tomato___Spider_mites Two-spotted_spider_mite',
                        'Squash___Powdery_mildew']
        },
        {
            'id': 'm004',
            'name': 'Chlorothalonil Spray',
            'brand': 'CropShield',
            'price': 390,
            'unit': '250ml bottle',
            'rating': 4.3,
            'image': '💉',
            'category': 'medicines',
            'description': 'Multi-purpose fungicide for leaf spot diseases',
            'diseases': ['Tomato___Septoria_leaf_spot',
                        'Tomato___Target_Spot',
                        'Strawberry___Leaf_scorch']
        },
        {
            'id': 'm005',
            'name': 'Metalaxyl Fungicide',
            'brand': 'PhytoGuard',
            'price': 560,
            'unit': '100g pack',
            'rating': 4.7,
            'image': '⚗️',
            'category': 'medicines',
            'description': 'Systemic fungicide for late blight control',
            'diseases': ['Potato___Late_blight',
                        'Tomato___Late_blight']
        },
        {
            'id': 'm006',
            'name': 'Sulfur Fungicide Powder',
            'brand': 'BioProtect',
            'price': 220,
            'unit': '500g pack',
            'rating': 4.2,
            'image': '🔬',
            'category': 'medicines',
            'description': 'Controls powdery mildew effectively',
            'diseases': ['Cherry_(including_sour)___Powdery_mildew',
                        'Squash___Powdery_mildew']
        },
    ],
    'fertilizers': [
        {
            'id': 'f001',
            'name': 'NPK 19-19-19 Fertilizer',
            'brand': 'GrowMax',
            'price': 650,
            'unit': '1kg pack',
            'rating': 4.8,
            'image': '🌱',
            'category': 'fertilizers',
            'description': 'Balanced fertilizer for all crop stages',
            'diseases': []
        },
        {
            'id': 'f002',
            'name': 'Potassium Nitrate',
            'brand': 'AgriBoost',
            'price': 480,
            'unit': '500g pack',
            'rating': 4.5,
            'image': '⚡',
            'category': 'fertilizers',
            'description': 'Boosts plant immunity and disease resistance',
            'diseases': ['Tomato___Early_blight',
                        'Tomato___Late_blight']
        },
        {
            'id': 'f003',
            'name': 'Calcium Boron Spray',
            'brand': 'NutriGrow',
            'price': 350,
            'unit': '250ml bottle',
            'rating': 4.4,
            'image': '💧',
            'category': 'fertilizers',
            'description': 'Strengthens cell walls, prevents tip burn',
            'diseases': ['Pepper,_bell___Bacterial_spot',
                        'Tomato___Bacterial_spot']
        },
        {
            'id': 'f004',
            'name': 'Organic Compost Mix',
            'brand': 'EarthCare',
            'price': 299,
            'unit': '2kg pack',
            'rating': 4.6,
            'image': '🪱',
            'category': 'fertilizers',
            'description': 'Premium organic matter for soil health',
            'diseases': []
        },
        {
            'id': 'f005',
            'name': 'Micronutrient Mix',
            'brand': 'GrowMax',
            'price': 420,
            'unit': '250g pack',
            'rating': 4.3,
            'image': '🔋',
            'category': 'fertilizers',
            'description': 'Zinc, iron, manganese for healthy growth',
            'diseases': ['Orange___Haunglongbing_(Citrus_greening)']
        },
        {
            'id': 'f006',
            'name': 'Bio Stimulant Spray',
            'brand': 'NatureCure',
            'price': 520,
            'unit': '500ml bottle',
            'rating': 4.7,
            'image': '🌊',
            'category': 'fertilizers',
            'description': 'Seaweed-based growth booster',
            'diseases': []
        },
    ]
}

# ============================================
# Remedies Database
# ============================================
REMEDIES = {
    "Apple___Apple_scab": {
        "symptoms": "Olive-green to brown spots on leaves and fruit.",
        "remedy": ["Apply fungicides containing captan or myclobutanil",
                   "Remove and destroy infected leaves",
                   "Prune trees to improve air circulation",
                   "Avoid overhead irrigation"],
        "severity": "Moderate"
    },
    "Apple___Black_rot": {
        "symptoms": "Brown circular lesions on fruit, purple spots on leaves.",
        "remedy": ["Remove mummified fruits from tree and ground",
                   "Apply copper-based fungicides",
                   "Prune dead or diseased wood",
                   "Maintain proper tree nutrition"],
        "severity": "High"
    },
    "Apple___Cedar_apple_rust": {
        "symptoms": "Bright orange-yellow spots on upper leaf surface.",
        "remedy": ["Apply fungicides at pink bud stage",
                   "Remove nearby juniper/cedar trees if possible",
                   "Use rust-resistant apple varieties"],
        "severity": "Moderate"
    },
    "Apple___healthy": {
        "symptoms": "No symptoms detected.",
        "remedy": ["Maintain regular watering schedule",
                   "Apply balanced fertilizer seasonally",
                   "Monitor regularly for early disease signs"],
        "severity": "None"
    },
    "Blueberry___healthy": {
        "symptoms": "No symptoms detected.",
        "remedy": ["Maintain soil pH between 4.5-5.5",
                   "Mulch to retain moisture",
                   "Prune annually for good air flow"],
        "severity": "None"
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "symptoms": "White powdery coating on leaves and shoots.",
        "remedy": ["Apply sulfur-based fungicides early",
                   "Use potassium bicarbonate sprays",
                   "Improve air circulation by pruning",
                   "Avoid excess nitrogen fertilization"],
        "severity": "Moderate"
    },
    "Cherry_(including_sour)___healthy": {
        "symptoms": "No symptoms detected.",
        "remedy": ["Water at base, avoid wetting foliage",
                   "Fertilize in early spring",
                   "Monitor for pests regularly"],
        "severity": "None"
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "symptoms": "Rectangular gray to tan lesions on leaves.",
        "remedy": ["Plant resistant corn hybrids",
                   "Apply strobilurin fungicides",
                   "Rotate crops — avoid continuous corn",
                   "Improve field drainage"],
        "severity": "High"
    },
    "Corn_(maize)___Common_rust_": {
        "symptoms": "Small oval brick-red pustules on both leaf surfaces.",
        "remedy": ["Plant rust-resistant hybrids",
                   "Apply fungicides if infection is severe",
                   "Early planting to avoid peak rust season"],
        "severity": "Moderate"
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "symptoms": "Long cigar-shaped gray-green lesions on leaves.",
        "remedy": ["Use resistant hybrids",
                   "Apply fungicides at tasseling stage",
                   "Practice crop rotation",
                   "Till crop residue after harvest"],
        "severity": "High"
    },
    "Corn_(maize)___healthy": {
        "symptoms": "No symptoms detected.",
        "remedy": ["Ensure proper spacing for airflow",
                   "Apply balanced NPK fertilizer",
                   "Monitor soil moisture"],
        "severity": "None"
    },
    "Grape___Black_rot": {
        "symptoms": "Brown circular leaf spots with black borders, shriveled fruit.",
        "remedy": ["Apply mancozeb or myclobutanil fungicides",
                   "Remove infected berries and leaves immediately",
                   "Prune for better air circulation"],
        "severity": "High"
    },
    "Grape___Esca_(Black_Measles)": {
        "symptoms": "Tiger-stripe pattern on leaves, dark spots on berries.",
        "remedy": ["Manage by pruning infected wood",
                   "Apply wound sealants after pruning",
                   "Remove severely infected vines"],
        "severity": "High"
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "symptoms": "Dark brown irregular spots with yellow halo on leaves.",
        "remedy": ["Apply copper-based fungicides",
                   "Remove infected plant debris",
                   "Improve canopy management"],
        "severity": "Moderate"
    },
    "Grape___healthy": {
        "symptoms": "No symptoms detected.",
        "remedy": ["Prune annually for airflow",
                   "Apply balanced fertilizer",
                   "Monitor for early disease signs"],
        "severity": "None"
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "symptoms": "Yellow shoots, blotchy mottled leaves, small lopsided fruit.",
        "remedy": ["Remove infected trees to prevent spread",
                   "Control Asian citrus psyllid with insecticides",
                   "Plant certified disease-free nursery stock"],
        "severity": "Critical"
    },
    "Peach___Bacterial_spot": {
        "symptoms": "Water-soaked spots on leaves, cracked lesions on fruit.",
        "remedy": ["Apply copper hydroxide sprays",
                   "Use resistant peach varieties",
                   "Avoid overhead irrigation"],
        "severity": "Moderate"
    },
    "Peach___healthy": {
        "symptoms": "No symptoms detected.",
        "remedy": ["Thin fruit for better quality",
                   "Apply dormant oil sprays",
                   "Monitor for borers and aphids"],
        "severity": "None"
    },
    "Pepper,_bell___Bacterial_spot": {
        "symptoms": "Water-soaked lesions on leaves and fruit, yellowing.",
        "remedy": ["Apply copper-based bactericides",
                   "Use certified disease-free seeds",
                   "Rotate crops with non-host plants"],
        "severity": "Moderate"
    },
    "Pepper,_bell___healthy": {
        "symptoms": "No symptoms detected.",
        "remedy": ["Water consistently at base",
                   "Fertilize with calcium-rich fertilizer"],
        "severity": "None"
    },
    "Potato___Early_blight": {
        "symptoms": "Dark brown concentric ring spots on leaves.",
        "remedy": ["Apply chlorothalonil or mancozeb fungicides",
                   "Remove lower infected leaves early",
                   "Practice 3-year crop rotation"],
        "severity": "Moderate"
    },
    "Potato___Late_blight": {
        "symptoms": "Water-soaked dark lesions on leaves, white mold on underside.",
        "remedy": ["Apply metalaxyl or cymoxanil fungicides immediately",
                   "Destroy infected plants",
                   "Hill soil around plants to protect tubers"],
        "severity": "Critical"
    },
    "Potato___healthy": {
        "symptoms": "No symptoms detected.",
        "remedy": ["Use certified disease-free seed potatoes",
                   "Hill soil regularly"],
        "severity": "None"
    },
    "Raspberry___healthy": {
        "symptoms": "No symptoms detected.",
        "remedy": ["Prune old canes after harvest",
                   "Mulch to retain moisture"],
        "severity": "None"
    },
    "Soybean___healthy": {
        "symptoms": "No symptoms detected.",
        "remedy": ["Rotate with non-legume crops",
                   "Monitor for aphids and whitefly"],
        "severity": "None"
    },
    "Squash___Powdery_mildew": {
        "symptoms": "White powdery patches on upper leaf surfaces.",
        "remedy": ["Apply neem oil or potassium bicarbonate",
                   "Use resistant squash varieties",
                   "Space plants for good air circulation"],
        "severity": "Moderate"
    },
    "Strawberry___Leaf_scorch": {
        "symptoms": "Small purple spots that enlarge with gray centers.",
        "remedy": ["Apply captan or thiram fungicides",
                   "Remove infected leaves promptly",
                   "Avoid overhead watering"],
        "severity": "Moderate"
    },
    "Strawberry___healthy": {
        "symptoms": "No symptoms detected.",
        "remedy": ["Mulch with straw to prevent soil splash",
                   "Renovate beds annually"],
        "severity": "None"
    },
    "Tomato___Bacterial_spot": {
        "symptoms": "Small water-soaked spots on leaves and fruit, yellowing.",
        "remedy": ["Apply copper bactericides at first sign",
                   "Use disease-free transplants",
                   "Rotate tomatoes with non-solanaceous crops"],
        "severity": "Moderate"
    },
    "Tomato___Early_blight": {
        "symptoms": "Dark concentric rings on lower leaves, yellowing around spots.",
        "remedy": ["Apply chlorothalonil or mancozeb fungicides",
                   "Remove infected lower leaves immediately",
                   "Mulch around plants to prevent soil splash"],
        "severity": "Moderate"
    },
    "Tomato___Late_blight": {
        "symptoms": "Greasy gray-green spots on leaves, white mold, brown stems.",
        "remedy": ["Apply copper-based or chlorothalonil fungicides",
                   "Remove and destroy all infected plant parts",
                   "Avoid overhead irrigation"],
        "severity": "Critical"
    },
    "Tomato___Leaf_Mold": {
        "symptoms": "Yellow patches on upper leaf, olive-green mold below.",
        "remedy": ["Improve greenhouse ventilation",
                   "Apply mancozeb or copper fungicides",
                   "Reduce humidity below 85%"],
        "severity": "Moderate"
    },
    "Tomato___Septoria_leaf_spot": {
        "symptoms": "Small circular spots with dark borders and gray centers.",
        "remedy": ["Apply chlorothalonil fungicides at first sign",
                   "Remove infected leaves immediately",
                   "Avoid wetting foliage when watering"],
        "severity": "Moderate"
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "symptoms": "Tiny yellow stippling on leaves, fine webbing underneath.",
        "remedy": ["Apply insecticidal soap or neem oil sprays",
                   "Introduce predatory mites",
                   "Increase humidity around plants"],
        "severity": "Moderate"
    },
    "Tomato___Target_Spot": {
        "symptoms": "Circular brown spots with concentric rings on leaves.",
        "remedy": ["Apply azoxystrobin or chlorothalonil fungicides",
                   "Remove heavily infected leaves",
                   "Improve air circulation"],
        "severity": "Moderate"
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "symptoms": "Yellowing and upward curling of leaves, stunted growth.",
        "remedy": ["Control whitefly vectors with insecticides",
                   "Use reflective mulches to deter whiteflies",
                   "Plant TYLCV-resistant tomato varieties"],
        "severity": "Critical"
    },
    "Tomato___Tomato_mosaic_virus": {
        "symptoms": "Mosaic pattern of light and dark green on leaves, distortion.",
        "remedy": ["Remove and destroy infected plants",
                   "Wash hands and tools thoroughly",
                   "Control aphid vectors"],
        "severity": "High"
    },
    "Tomato___healthy": {
        "symptoms": "No symptoms detected.",
        "remedy": ["Water consistently at base",
                   "Support plants with stakes or cages",
                   "Monitor weekly for early disease signs"],
        "severity": "None"
    }
}

# ============================================
# Mandi Prices Data
# ============================================
from datetime import datetime

MANDI_DATA = [
    # Madhya Pradesh
    {
        'crop': 'Tomato', 'emoji': '🍅',
        'market': 'Indore', 'state': 'Madhya Pradesh',
        'min': 800, 'max': 1400, 'modal': 1100,
        'trend': 'up', 'unit': 'quintal'
    },
    {
        'crop': 'Potato', 'emoji': '🥔',
        'market': 'Indore', 'state': 'Madhya Pradesh',
        'min': 600, 'max': 900, 'modal': 750,
        'trend': 'stable', 'unit': 'quintal'
    },
    {
        'crop': 'Onion', 'emoji': '🧅',
        'market': 'Indore', 'state': 'Madhya Pradesh',
        'min': 1000, 'max': 1800, 'modal': 1400,
        'trend': 'up', 'unit': 'quintal'
    },
    {
        'crop': 'Wheat', 'emoji': '🌾',
        'market': 'Bhopal', 'state': 'Madhya Pradesh',
        'min': 2100, 'max': 2400, 'modal': 2250,
        'trend': 'stable', 'unit': 'quintal'
    },
    {
        'crop': 'Soybean', 'emoji': '🫘',
        'market': 'Ujjain', 'state': 'Madhya Pradesh',
        'min': 4200, 'max': 4800, 'modal': 4500,
        'trend': 'up', 'unit': 'quintal'
    },
    {
        'crop': 'Garlic', 'emoji': '🧄',
        'market': 'Mandsaur', 'state': 'Madhya Pradesh',
        'min': 3000, 'max': 8000, 'modal': 5500,
        'trend': 'up', 'unit': 'quintal'
    },
    {
        'crop': 'Corn', 'emoji': '🌽',
        'market': 'Indore', 'state': 'Madhya Pradesh',
        'min': 1800, 'max': 2200, 'modal': 2000,
        'trend': 'stable', 'unit': 'quintal'
    },
    # Maharashtra
    {
        'crop': 'Tomato', 'emoji': '🍅',
        'market': 'Pune', 'state': 'Maharashtra',
        'min': 900, 'max': 1600, 'modal': 1200,
        'trend': 'up', 'unit': 'quintal'
    },
    {
        'crop': 'Onion', 'emoji': '🧅',
        'market': 'Nashik', 'state': 'Maharashtra',
        'min': 1200, 'max': 2000, 'modal': 1600,
        'trend': 'up', 'unit': 'quintal'
    },
    {
        'crop': 'Grape', 'emoji': '🍇',
        'market': 'Sangli', 'state': 'Maharashtra',
        'min': 3000, 'max': 6000, 'modal': 4500,
        'trend': 'stable', 'unit': 'quintal'
    },
    {
        'crop': 'Sugarcane', 'emoji': '🎋',
        'market': 'Kolhapur', 'state': 'Maharashtra',
        'min': 280, 'max': 320, 'modal': 300,
        'trend': 'stable', 'unit': 'quintal'
    },
    # Punjab
    {
        'crop': 'Wheat', 'emoji': '🌾',
        'market': 'Amritsar', 'state': 'Punjab',
        'min': 2150, 'max': 2350, 'modal': 2250,
        'trend': 'stable', 'unit': 'quintal'
    },
    {
        'crop': 'Rice', 'emoji': '🌾',
        'market': 'Ludhiana', 'state': 'Punjab',
        'min': 2000, 'max': 2300, 'modal': 2150,
        'trend': 'down', 'unit': 'quintal'
    },
    {
        'crop': 'Potato', 'emoji': '🥔',
        'market': 'Jalandhar', 'state': 'Punjab',
        'min': 500, 'max': 800, 'modal': 650,
        'trend': 'down', 'unit': 'quintal'
    },
    # Uttar Pradesh
    {
        'crop': 'Potato', 'emoji': '🥔',
        'market': 'Agra', 'state': 'Uttar Pradesh',
        'min': 550, 'max': 850, 'modal': 700,
        'trend': 'stable', 'unit': 'quintal'
    },
    {
        'crop': 'Sugarcane', 'emoji': '🎋',
        'market': 'Lucknow', 'state': 'Uttar Pradesh',
        'min': 350, 'max': 400, 'modal': 375,
        'trend': 'up', 'unit': 'quintal'
    },
    {
        'crop': 'Wheat', 'emoji': '🌾',
        'market': 'Kanpur', 'state': 'Uttar Pradesh',
        'min': 2100, 'max': 2350, 'modal': 2200,
        'trend': 'stable', 'unit': 'quintal'
    },
    # Karnataka
    {
        'crop': 'Tomato', 'emoji': '🍅',
        'market': 'Bangalore', 'state': 'Karnataka',
        'min': 1000, 'max': 2000, 'modal': 1500,
        'trend': 'up', 'unit': 'quintal'
    },
    {
        'crop': 'Corn', 'emoji': '🌽',
        'market': 'Davangere', 'state': 'Karnataka',
        'min': 1700, 'max': 2100, 'modal': 1900,
        'trend': 'stable', 'unit': 'quintal'
    },
    # Gujarat
    {
        'crop': 'Cotton', 'emoji': '🌸',
        'market': 'Rajkot', 'state': 'Gujarat',
        'min': 6000, 'max': 7500, 'modal': 6800,
        'trend': 'up', 'unit': 'quintal'
    },
    {
        'crop': 'Groundnut', 'emoji': '🥜',
        'market': 'Junagadh', 'state': 'Gujarat',
        'min': 5500, 'max': 6500, 'modal': 6000,
        'trend': 'stable', 'unit': 'quintal'
    },
]

@app.route('/mandi')
def mandi():
    
    search   = request.args.get('search', '').strip().lower()
    state    = request.args.get('state', 'all')

    # Get unique states for filter
    states = sorted(set(p['state'] for p in MANDI_DATA))

    # Filter data
    filtered = MANDI_DATA

    if state != 'all':
        filtered = [p for p in filtered
                    if p['state'] == state]

    if search:
        filtered = [p for p in filtered
                    if search in p['crop'].lower()
                    or search in p['market'].lower()]

    return render_template('mandi.html',
                           prices=filtered,
                           states=states,
                           selected_state=state,
                           search=search,
                           last_updated=datetime.now().strftime(
                               "%d %b %Y, %I:%M %p"), )


# ============================================
# Crop Calendar Data
# ============================================
CROP_CALENDAR = [
    {
        'crop': 'Wheat',
        'emoji': '🌾',
        'category': 'Rabi',
        'sowing_months': ['October', 'November', 'December'],
        'harvesting_months': ['March', 'April'],
        'duration': '120-150 days',
        'season': 'Winter',
        'soil': 'Loamy, Well-drained',
        'water': 'Moderate',
        'states': 'Punjab, Haryana, MP, UP',
        'tips': 'Sow after first rains. Avoid waterlogging.'
    },
    {
        'crop': 'Rice',
        'emoji': '🌾',
        'category': 'Kharif',
        'sowing_months': ['June', 'July'],
        'harvesting_months': ['October', 'November'],
        'duration': '120-150 days',
        'season': 'Monsoon',
        'soil': 'Clay, Water-retentive',
        'water': 'High',
        'states': 'West Bengal, Punjab, AP, Tamil Nadu',
        'tips': 'Requires standing water. Transplant after 25-30 days.'
    },
    {
        'crop': 'Tomato',
        'emoji': '🍅',
        'category': 'Zaid',
        'sowing_months': ['June', 'July', 'November', 'December'],
        'harvesting_months': ['September', 'October', 'February', 'March'],
        'duration': '60-80 days',
        'season': 'All Season',
        'soil': 'Sandy Loam, Well-drained',
        'water': 'Moderate',
        'states': 'Karnataka, AP, Maharashtra, MP',
        'tips': 'Stake plants for support. Watch for early blight.'
    },
    {
        'crop': 'Potato',
        'emoji': '🥔',
        'category': 'Rabi',
        'sowing_months': ['October', 'November'],
        'harvesting_months': ['January', 'February', 'March'],
        'duration': '90-120 days',
        'season': 'Winter',
        'soil': 'Sandy Loam, Loose soil',
        'water': 'Moderate',
        'states': 'UP, Punjab, West Bengal, MP',
        'tips': 'Hill soil around plants. Avoid excessive moisture.'
    },
    {
        'crop': 'Onion',
        'emoji': '🧅',
        'category': 'Rabi',
        'sowing_months': ['October', 'November', 'December'],
        'harvesting_months': ['February', 'March', 'April', 'May'],
        'duration': '120-130 days',
        'season': 'Winter',
        'soil': 'Well-drained, Fertile loam',
        'water': 'Moderate',
        'states': 'Maharashtra, Karnataka, MP, Gujarat',
        'tips': 'Stop irrigation 2 weeks before harvest.'
    },
    {
        'crop': 'Corn',
        'emoji': '🌽',
        'category': 'Kharif',
        'sowing_months': ['June', 'July'],
        'harvesting_months': ['September', 'October'],
        'duration': '90-110 days',
        'season': 'Monsoon',
        'soil': 'Loamy, Well-drained',
        'water': 'Moderate',
        'states': 'Karnataka, MP, Rajasthan, Bihar',
        'tips': 'Plant in rows for cross-pollination. Watch for rust.'
    },
    {
        'crop': 'Soybean',
        'emoji': '🫘',
        'category': 'Kharif',
        'sowing_months': ['June', 'July'],
        'harvesting_months': ['September', 'October'],
        'duration': '90-120 days',
        'season': 'Monsoon',
        'soil': 'Well-drained loam',
        'water': 'Moderate',
        'states': 'Madhya Pradesh, Maharashtra, Rajasthan',
        'tips': 'Use rhizobium culture for better yield.'
    },
    {
        'crop': 'Cotton',
        'emoji': '🌸',
        'category': 'Kharif',
        'sowing_months': ['April', 'May', 'June'],
        'harvesting_months': ['October', 'November', 'December'],
        'duration': '150-180 days',
        'season': 'Summer/Monsoon',
        'soil': 'Black cotton soil, Deep loam',
        'water': 'Moderate',
        'states': 'Gujarat, Maharashtra, Telangana, Punjab',
        'tips': 'Requires long frost-free period.'
    },
    {
        'crop': 'Sugarcane',
        'emoji': '🎋',
        'category': 'Annual',
        'sowing_months': ['February', 'March', 'October', 'November'],
        'harvesting_months': ['November', 'December', 'January', 'February'],
        'duration': '10-12 months',
        'season': 'All Season',
        'soil': 'Loamy, Well-drained',
        'water': 'High',
        'states': 'UP, Maharashtra, Karnataka, Tamil Nadu',
        'tips': 'Ratoon crop possible for 2-3 years.'
    },
    {
        'crop': 'Groundnut',
        'emoji': '🥜',
        'category': 'Kharif',
        'sowing_months': ['June', 'July'],
        'harvesting_months': ['October', 'November'],
        'duration': '100-130 days',
        'season': 'Monsoon',
        'soil': 'Sandy loam, Light soil',
        'water': 'Low to Moderate',
        'states': 'Gujarat, Rajasthan, AP, Tamil Nadu',
        'tips': 'Needs calcium for pod development.'
    },
    {
        'crop': 'Mustard',
        'emoji': '🌿',
        'category': 'Rabi',
        'sowing_months': ['October', 'November'],
        'harvesting_months': ['February', 'March'],
        'duration': '110-140 days',
        'season': 'Winter',
        'soil': 'Loamy, Well-drained',
        'water': 'Low',
        'states': 'Rajasthan, UP, Haryana, MP',
        'tips': 'Cold tolerant. Good for dry regions.'
    },
    {
        'crop': 'Garlic',
        'emoji': '🧄',
        'category': 'Rabi',
        'sowing_months': ['October', 'November'],
        'harvesting_months': ['February', 'March', 'April'],
        'duration': '130-180 days',
        'season': 'Winter',
        'soil': 'Loamy, Well-drained',
        'water': 'Moderate',
        'states': 'MP, Gujarat, Rajasthan, UP',
        'tips': 'Plant cloves 5cm deep. Stop water before harvest.'
    },
    {
        'crop': 'Grape',
        'emoji': '🍇',
        'category': 'Perennial',
        'sowing_months': ['January', 'February'],
        'harvesting_months': ['March', 'April', 'May'],
        'duration': '2-3 years to establish',
        'season': 'Winter/Spring',
        'soil': 'Well-drained sandy loam',
        'water': 'Moderate',
        'states': 'Maharashtra, Karnataka, Tamil Nadu',
        'tips': 'Prune annually for good yield.'
    },
    {
        'crop': 'Apple',
        'emoji': '🍎',
        'category': 'Perennial',
        'sowing_months': ['December', 'January', 'February'],
        'harvesting_months': ['August', 'September', 'October'],
        'duration': '3-5 years to bear fruit',
        'season': 'Winter',
        'soil': 'Loamy, Well-drained',
        'water': 'Moderate',
        'states': 'Himachal Pradesh, Jammu & Kashmir, Uttarakhand',
        'tips': 'Requires chilling hours. Altitude above 1500m.'
    },
    {
        'crop': 'Pepper (Bell)',
        'emoji': '🫑',
        'category': 'Zaid',
        'sowing_months': ['June', 'July', 'November', 'December'],
        'harvesting_months': ['September', 'October', 'February', 'March'],
        'duration': '65-80 days',
        'season': 'All Season',
        'soil': 'Sandy loam, Well-drained',
        'water': 'Moderate',
        'states': 'Karnataka, HP, Maharashtra, MP',
        'tips': 'Avoid waterlogging. Support with stakes.'
    },
    {
        'crop': 'Strawberry',
        'emoji': '🍓',
        'category': 'Rabi',
        'sowing_months': ['September', 'October'],
        'harvesting_months': ['December', 'January', 'February', 'March'],
        'duration': '60-90 days',
        'season': 'Winter',
        'soil': 'Sandy loam, Well-drained',
        'water': 'Moderate',
        'states': 'HP, Maharashtra, UP, Karnataka',
        'tips': 'Mulch with straw. Cool climate preferred.'
    },
]

# Month helper
MONTHS = ['January', 'February', 'March', 'April',
          'May', 'June', 'July', 'August',
          'September', 'October', 'November', 'December']

@app.route('/calendar')
def calendar():
    
    category = request.args.get('category', 'all')
    month    = request.args.get('month', 'all')
    search   = request.args.get('search', '').strip().lower()

    filtered = CROP_CALENDAR

    if category != 'all':
        filtered = [c for c in filtered
                    if c['category'] == category]

    if month != 'all':
        filtered = [c for c in filtered
                    if month in c['sowing_months']
                    or month in c['harvesting_months']]

    if search:
        filtered = [c for c in filtered
                    if search in c['crop'].lower()]

    # Get current month
    current_month = datetime.now().strftime('%B')

    # Crops to sow this month
    sow_now = [c for c in CROP_CALENDAR
               if current_month in c['sowing_months']]

    # Crops to harvest this month
    harvest_now = [c for c in CROP_CALENDAR
                   if current_month in c['harvesting_months']]

    return render_template('calendar.html',
                           crops=filtered,
                           months=MONTHS,
                           categories=['Kharif', 'Rabi',
                                      'Zaid', 'Annual',
                                      'Perennial'],
                           selected_category=category,
                           selected_month=month,
                           search=search,
                           current_month=current_month,
                           sow_now=sow_now,
                           harvest_now=harvest_now, )

# ============================================
# Government Schemes Data
# ============================================
GOVT_SCHEMES = [
    {
        'id': 1,
        'name': 'PM-KISAN',
        'full_name': 'Pradhan Mantri Kisan Samman Nidhi',
        'emoji': '💰',
        'category': 'financial',
        'benefit': '₹6,000/year direct income support',
        'description': 'Direct income support of ₹6,000 per year to small and marginal farmers in three equal installments of ₹2,000 each.',
        'eligibility': [
            'Small and marginal farmers',
            'Land holding up to 2 hectares',
            'Valid Aadhaar card required',
            'Bank account linked to Aadhaar'
        ],
        'documents': ['Aadhaar Card', 'Land Records', 'Bank Passbook'],
        'apply_link': 'https://pmkisan.gov.in',
        'deadline': 'Open throughout the year',
        'ministry': 'Ministry of Agriculture',
        'tag': 'Most Popular'
    },
    {
        'id': 2,
        'name': 'PM Fasal Bima Yojana',
        'full_name': 'Pradhan Mantri Fasal Bima Yojana',
        'emoji': '🛡️',
        'category': 'insurance',
        'benefit': 'Crop insurance at 1.5-2% premium',
        'description': 'Provides financial support to farmers suffering crop loss or damage due to unforeseen events like natural calamities, pests and diseases.',
        'eligibility': [
            'All farmers growing notified crops',
            'Both loanee and non-loanee farmers',
            'Sharecroppers and tenant farmers'
        ],
        'documents': ['Aadhaar Card', 'Land Records', 'Bank Account', 'Sowing Certificate'],
        'apply_link': 'https://pmfby.gov.in',
        'deadline': 'Before sowing season',
        'ministry': 'Ministry of Agriculture',
        'tag': 'Insurance'
    },
    {
        'id': 3,
        'name': 'Kisan Credit Card',
        'full_name': 'Kisan Credit Card Scheme',
        'emoji': '💳',
        'category': 'financial',
        'benefit': 'Credit up to ₹3 lakh at 4% interest',
        'description': 'Provides adequate and timely credit support to farmers for their agricultural operations at very low interest rates.',
        'eligibility': [
            'All farmers including tenant farmers',
            'Self Help Groups of farmers',
            'Joint Liability Groups'
        ],
        'documents': ['Aadhaar Card', 'Land Records', 'Passport Photo', 'Bank Account'],
        'apply_link': 'https://www.nabard.org',
        'deadline': 'Open throughout the year',
        'ministry': 'Ministry of Finance',
        'tag': 'Credit'
    },
    {
        'id': 4,
        'name': 'PMKSY',
        'full_name': 'PM Krishi Sinchayee Yojana',
        'emoji': '💧',
        'category': 'irrigation',
        'benefit': 'Subsidy on drip/sprinkler irrigation',
        'description': 'Aims to enhance water use efficiency at farm level through micro irrigation like drip and sprinkler systems with up to 55% subsidy.',
        'eligibility': [
            'All categories of farmers',
            'Priority to small and marginal farmers',
            'SC/ST farmers get extra 10% subsidy'
        ],
        'documents': ['Aadhaar Card', 'Land Records', 'Bank Account', 'Quotation from vendor'],
        'apply_link': 'https://pmksy.gov.in',
        'deadline': 'Open throughout the year',
        'ministry': 'Ministry of Agriculture',
        'tag': 'Irrigation'
    },
    {
        'id': 5,
        'name': 'Soil Health Card',
        'full_name': 'Soil Health Card Scheme',
        'emoji': '🌱',
        'category': 'technology',
        'benefit': 'Free soil testing and nutrient advice',
        'description': 'Provides soil health cards to farmers with crop-wise recommendations of nutrients and fertilizers for individual farms.',
        'eligibility': [
            'All farmers across India',
            'Free of cost service',
            'No land size restriction'
        ],
        'documents': ['Aadhaar Card', 'Land Records'],
        'apply_link': 'https://soilhealth.dac.gov.in',
        'deadline': 'Open throughout the year',
        'ministry': 'Ministry of Agriculture',
        'tag': 'Free Service'
    },
    {
        'id': 6,
        'name': 'e-NAM',
        'full_name': 'National Agriculture Market',
        'emoji': '🏪',
        'category': 'market',
        'benefit': 'Sell crops online at best prices',
        'description': 'Online trading platform for agricultural commodities to help farmers get better prices by connecting them to buyers across India.',
        'eligibility': [
            'All farmers with valid Aadhaar',
            'Must be registered at local APMC',
            'Bank account required'
        ],
        'documents': ['Aadhaar Card', 'Bank Account', 'APMC Registration'],
        'apply_link': 'https://enam.gov.in',
        'deadline': 'Open throughout the year',
        'ministry': 'Ministry of Agriculture',
        'tag': 'Digital Market'
    },
    {
        'id': 7,
        'name': 'PKVY',
        'full_name': 'Paramparagat Krishi Vikas Yojana',
        'emoji': '🌿',
        'category': 'organic',
        'benefit': '₹50,000/hectare for organic farming',
        'description': 'Promotes organic farming by providing financial assistance to farmers for adopting organic cultivation practices.',
        'eligibility': [
            'Farmers willing to adopt organic farming',
            'Must form clusters of 50 farmers',
            'Minimum 50 acres of contiguous land'
        ],
        'documents': ['Aadhaar Card', 'Land Records', 'Group Formation Certificate'],
        'apply_link': 'https://pgsindia-ncof.gov.in',
        'deadline': 'As per state government notification',
        'ministry': 'Ministry of Agriculture',
        'tag': 'Organic'
    },
    {
        'id': 8,
        'name': 'SMAM',
        'full_name': 'Sub-Mission on Agricultural Mechanization',
        'emoji': '🚜',
        'category': 'equipment',
        'benefit': 'Up to 50% subsidy on farm equipment',
        'description': 'Provides financial assistance to farmers for purchase of agricultural machinery and equipment at subsidized rates.',
        'eligibility': [
            'All categories of farmers',
            'Priority to small and marginal farmers',
            'Women farmers get additional benefits'
        ],
        'documents': ['Aadhaar Card', 'Land Records', 'Bank Account', 'Quotation'],
        'apply_link': 'https://agrimachinery.nic.in',
        'deadline': 'Open throughout the year',
        'ministry': 'Ministry of Agriculture',
        'tag': 'Equipment'
    },
    {
        'id': 9,
        'name': 'RKVY',
        'full_name': 'Rashtriya Krishi Vikas Yojana',
        'emoji': '📈',
        'category': 'development',
        'benefit': 'State-specific agricultural development',
        'description': 'Provides flexibility to states to plan and execute schemes for development of agriculture sector based on local needs.',
        'eligibility': [
            'Varies by state and project',
            'Contact state agriculture department',
            'Priority to small and marginal farmers'
        ],
        'documents': ['Aadhaar Card', 'Land Records', 'Project Proposal'],
        'apply_link': 'https://rkvy.nic.in',
        'deadline': 'As per state government notification',
        'ministry': 'Ministry of Agriculture',
        'tag': 'Development'
    },
    {
        'id': 10,
        'name': 'PM Kisan MaanDhan',
        'full_name': 'PM Kisan Maandhan Yojana',
        'emoji': '👴',
        'category': 'pension',
        'benefit': '₹3,000/month pension after age 60',
        'description': 'Voluntary and contributory pension scheme for small and marginal farmers providing monthly pension of ₹3,000 after age 60.',
        'eligibility': [
            'Small and marginal farmers',
            'Age between 18-40 years',
            'Land holding up to 2 hectares'
        ],
        'documents': ['Aadhaar Card', 'Land Records', 'Bank Account', 'Age Proof'],
        'apply_link': 'https://maandhan.in',
        'deadline': 'Open throughout the year',
        'ministry': 'Ministry of Agriculture',
        'tag': 'Pension'
    },
    {
        'id': 11,
        'name': 'DBT Agriculture',
        'full_name': 'Direct Benefit Transfer for Agriculture',
        'emoji': '🏦',
        'category': 'financial',
        'benefit': 'Direct subsidy transfer to bank account',
        'description': 'Ensures subsidies on fertilizers, seeds and other inputs are transferred directly to farmers bank accounts without middlemen.',
        'eligibility': [
            'All registered farmers',
            'Valid Aadhaar linked bank account',
            'Registered on DBT portal'
        ],
        'documents': ['Aadhaar Card', 'Bank Account', 'Land Records'],
        'apply_link': 'https://dbtbharat.gov.in',
        'deadline': 'Open throughout the year',
        'ministry': 'Ministry of Finance',
        'tag': 'Subsidy'
    },
    {
        'id': 12,
        'name': 'MIDH',
        'full_name': 'Mission for Integrated Development of Horticulture',
        'emoji': '🍎',
        'category': 'horticulture',
        'benefit': 'Up to 50% subsidy for horticulture',
        'description': 'Provides financial assistance for development of horticulture crops including fruits, vegetables, spices and flowers.',
        'eligibility': [
            'Farmers growing horticultural crops',
            'Registered with state horticulture department',
            'Land ownership or lease documents required'
        ],
        'documents': ['Aadhaar Card', 'Land Records', 'Bank Account', 'Horticulture Plan'],
        'apply_link': 'https://midh.gov.in',
        'deadline': 'As per state notification',
        'ministry': 'Ministry of Agriculture',
        'tag': 'Horticulture'
    },
]

@app.route('/schemes')
def schemes():
    
    category = request.args.get('category', 'all')
    search   = request.args.get('search', '').strip().lower()

    # Get unique categories
    categories = sorted(set(s['category'] for s in GOVT_SCHEMES))

    # Filter
    filtered = GOVT_SCHEMES
    if category != 'all':
        filtered = [s for s in filtered if s['category'] == category]
    if search:
        filtered = [s for s in filtered
                    if search in s['name'].lower()
                    or search in s['full_name'].lower()
                    or search in s['description'].lower()]

    return render_template('schemes.html',
                           schemes=filtered,
                           categories=categories,
                           selected_category=category,
                           search=search,
                           total=len(GOVT_SCHEMES), )


# ============================================
# Helper Functions
# ============================================

def allowed_file(filename):
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower()
            in ALLOWED_EXTENSIONS)


def get_recommended_products(raw_class):
    """Get products recommended for a specific disease"""
    recommended = []
    all_products = (PRODUCTS['medicines'] +
                    PRODUCTS['seeds'] +
                    PRODUCTS['fertilizers'])
    for product in all_products:
        if raw_class in product.get('diseases', []):
            recommended.append(product)
    return recommended[:4]  # Max 4 recommendations


def is_leaf_image(image_path):
    """
    Basic check to detect if image contains
    a leaf/plant using green color dominance.
    Real leaves have significant green content.
    """
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize for speed
    img_small = cv2.resize(img_rgb, (100, 100))

    # Extract color channels
    r = img_small[:,:,0].astype(float)
    g = img_small[:,:,1].astype(float)
    b = img_small[:,:,2].astype(float)

    # Count pixels where green is dominant
    # A leaf pixel: green > red AND green > blue
    green_dominant = np.sum((g > r) & (g > b))
    total_pixels   = 100 * 100

    green_ratio = green_dominant / total_pixels

    # Also check average green value
    avg_green = np.mean(g)
    avg_red   = np.mean(r)
    avg_blue  = np.mean(b)

    # Leaf condition:
    # At least 15% pixels are green dominant
    # OR average green is highest channel
    is_leaf = (green_ratio > 0.15) or (avg_green > avg_red and avg_green > avg_blue)

    return is_leaf, round(green_ratio * 100, 1)


def predict_disease(image_path):

    # --- Check if image is a leaf ---
    leaf_detected, green_ratio = is_leaf_image(image_path)

    if not leaf_detected:
        return {
            'error':      True,
            'message':    'No leaf detected in this image.',
            'suggestion': 'Please upload a clear photo of a plant leaf.',
            'green_ratio': green_ratio
        }


    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.expand_dims(img, axis=0).astype('float32')

    predictions   = model.predict(img, verbose=0)
    predicted_idx = int(np.argmax(predictions[0]))
    confidence    = float(predictions[0][predicted_idx] * 100)

    # Top 3 predictions
    top3_idx = np.argsort(predictions[0])[::-1][:3]
    top3 = []
    for idx in top3_idx:
        parts = class_names[idx].split('___')
        top3.append({
            'plant':      parts[0].replace('_', ' '),
            'disease':    parts[1].replace('_', ' ') if len(parts) > 1 else 'Unknown',
            'confidence': round(float(predictions[0][idx] * 100), 2),
            'raw':        class_names[idx]
        })

    # ← Fix is here — define these BEFORE the return
    raw_class    = class_names[predicted_idx]
    main_parts   = raw_class.split('___')
    plant_name   = main_parts[0].replace('_', ' ')
    disease_name = main_parts[1].replace('_', ' ') if len(main_parts) > 1 else 'Unknown'
    is_healthy   = 'healthy' in disease_name.lower()
    is_reliable  = confidence >= 70.0

    remedy_info  = REMEDIES.get(raw_class, {
        "symptoms": "Symptoms information not available.",
        "remedy":   ["Consult a local agricultural expert"],
        "severity": "Unknown"
    })

    recommended_products = get_recommended_products(raw_class)

    return {
        'plant':       plant_name,
        'disease':     disease_name,
        'confidence':  round(confidence, 2),
        'is_healthy':  is_healthy,
        'is_reliable': is_reliable,
        'top3':        top3,
        'raw_class':   raw_class,
        'symptoms':    remedy_info['symptoms'],
        'remedy':      remedy_info['remedy'],
        'severity':    remedy_info['severity'],
        'recommended': recommended_products
    }

# ============================================
# Scan History
# ============================================
HISTORY_FILE = 'scan_history.json'

def load_history():
    """Load scan history from JSON file"""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

def save_to_history(result, image_path):
    """Save a scan result to history"""
    history = load_history()
    entry = {
        'id':         len(history) + 1,
        'timestamp':  datetime.now().strftime("%d %b %Y, %I:%M %p"),
        'date':       datetime.now().strftime("%Y-%m-%d"),
        'plant':      result['plant'],
        'disease':    result['disease'],
        'confidence': result['confidence'],
        'is_healthy': result['is_healthy'],
        'severity':   result.get('severity', 'Unknown'),
        'image_path': image_path,
    }
    history.insert(0, entry)  # newest first
    history = history[:50]    # keep last 50 scans
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

# Weather API
WEATHER_API_KEY = 'b1b4361c8dd702bb742fb8d80515e0e8'
WEATHER_BASE    = 'https://api.openweathermap.org/data/2.5'

def get_farming_advice(weather_data):
    """
    Generate farming advice based on weather conditions
    """
    advice = []
    alerts = []

    temp     = weather_data['main']['temp']
    humidity = weather_data['main']['humidity']
    wind     = weather_data['wind']['speed']
    condition = weather_data['weather'][0]['main'].lower()

    # Temperature advice
    if temp > 35:
        alerts.append("🌡️ Extreme heat — water crops early morning or evening")
        advice.append("Avoid pesticide spraying in high heat")
    elif temp < 10:
        alerts.append("🥶 Cold temperature — protect sensitive crops from frost")
        advice.append("Cover young plants overnight")
    else:
        advice.append("✅ Temperature is ideal for most crop activities")

    # Humidity advice
    if humidity > 80:
        alerts.append("💧 High humidity — high risk of fungal diseases!")
        advice.append("Inspect leaves for early signs of blight or mold")
        advice.append("Avoid overhead irrigation today")
    elif humidity < 30:
        alerts.append("🏜️ Low humidity — crops may need extra watering")
        advice.append("Mulch around plants to retain soil moisture")
    else:
        advice.append("✅ Humidity levels are comfortable for crops")

    # Wind advice
    if wind > 10:
        alerts.append("💨 Strong winds — avoid spraying pesticides today")
        advice.append("Check for physical damage to tall crops like corn")
    else:
        advice.append("✅ Low wind — good conditions for pesticide spraying")

    # Rain/condition advice
    if 'rain' in condition:
        alerts.append("🌧️ Rain detected — avoid harvesting today")
        advice.append("Check drainage around crop fields")
        advice.append("Good natural irrigation — reduce watering schedule")
    elif 'clear' in condition:
        advice.append("☀️ Clear skies — ideal day for field inspection")
        advice.append("Good conditions for harvesting if crops are ready")
    elif 'cloud' in condition:
        advice.append("⛅ Cloudy — good day for transplanting seedlings")

    return advice, alerts


def get_weather(city):
    """Fetch current weather and 5-day forecast"""
    try:
        # Current weather
        current_url = (f"{WEATHER_BASE}/weather"
                      f"?q={city}"
                      f"&appid={WEATHER_API_KEY}"
                      f"&units=metric")
        current_res = requests.get(current_url, timeout=5)
        current     = current_res.json()

        if current.get('cod') != 200:
            return None, "City not found. Please try again."

        # 5-day forecast
        forecast_url = (f"{WEATHER_BASE}/forecast"
                       f"?q={city}"
                       f"&appid={WEATHER_API_KEY}"
                       f"&units=metric")
        forecast_res  = requests.get(forecast_url, timeout=5)
        forecast_data = forecast_res.json()

        # Get one forecast per day (every 8th entry = 24hrs)
        forecast_list = forecast_data.get('list', [])
        daily_forecast = forecast_list[::8][:5]

        # Get farming advice
        advice, alerts = get_farming_advice(current)

        # Weather icon mapping
        icon_map = {
            'clear':       '☀️',
            'clouds':      '⛅',
            'rain':        '🌧️',
            'drizzle':     '🌦️',
            'thunderstorm':'⛈️',
            'snow':        '❄️',
            'mist':        '🌫️',
            'fog':         '🌫️',
            'haze':        '🌫️',
        }

        condition = current['weather'][0]['main'].lower()
        icon      = icon_map.get(condition, '🌤️')

        return {
            'city':        current['name'],
            'country':     current['sys']['country'],
            'temp':        round(current['main']['temp']),
            'feels_like':  round(current['main']['feels_like']),
            'humidity':    current['main']['humidity'],
            'wind':        round(current['wind']['speed'] * 3.6),  # m/s to km/h
            'condition':   current['weather'][0]['description'].title(),
            'icon':        icon,
            'pressure':    current['main']['pressure'],
            'visibility':  current.get('visibility', 0) // 1000,
            'forecast':    daily_forecast,
            'advice':      advice,
            'alerts':      alerts,
            'icon_map':    icon_map,
        }, None

    except requests.exceptions.Timeout:
        return None, "Request timed out. Check your internet connection."
    except Exception as e:
        return None, f"Weather service unavailable. Try again later."
    


# ============================================
# Database Setup
# ============================================
DATABASE = 'krishisetu.db'

def get_db():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create tables if they don't exist"""
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            email       TEXT UNIQUE NOT NULL,
            password    TEXT NOT NULL,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()
print("✅ Database initialized!")

# ============================================
# Auth Decorator
# ============================================
def login_required(f):
    """Redirect to login if not logged in"""
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


# ============================================
# Routes
# ============================================

@app.route('/')
def index():
    
    return render_template('index.html', )


@app.route('/detect')
@login_required
def detect():
    
    return render_template('detect.html', )


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files:
        return redirect(url_for('detect'))

    file = request.files['file']

    if file.filename == '':
        return redirect(url_for('detect'))

    if not allowed_file(file.filename):
        return render_template('detect.html',
                               error="Please upload a JPG or PNG image.")

    filename    = secure_filename(file.filename)
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(upload_path)

    result = predict_disease(upload_path)

    if result.get('error'):
        return render_template('detect.html',
                               error=result['message'] +
                               ' ' + result['suggestion'])

    image_path = f"uploads/{filename}"
    result['image_path'] = image_path

    # ← Save to history
    save_to_history(result, image_path)

    return render_template('result.html', result=result)


@app.route('/shop')
def shop():
    
    category = request.args.get('category', 'all')
    search   = request.args.get('search', '')

    if category == 'all':
        products = (PRODUCTS['seeds'] +
                    PRODUCTS['medicines'] +
                    PRODUCTS['fertilizers'])
    else:
        products = PRODUCTS.get(category, [])

    if search:
        products = [p for p in products
                    if search.lower() in p['name'].lower()
                    or search.lower() in p['description'].lower()]

    return render_template('shop.html',
                           products=products,
                           category=category,
                           search=search,
                           counts={
                               'all':         len(PRODUCTS['seeds']) +
                                              len(PRODUCTS['medicines']) +
                                              len(PRODUCTS['fertilizers']),
                               'seeds':       len(PRODUCTS['seeds']),
                               'medicines':   len(PRODUCTS['medicines']),
                               'fertilizers': len(PRODUCTS['fertilizers'])
                           }, )

@app.route('/weather', methods=['GET', 'POST'])
def weather():
    
    weather_data = None
    error        = None
    city         = 'Bhopal'  # Default city

    if request.method == 'POST':
        city = request.form.get('city', 'Bhopal').strip()

    weather_data, error = get_weather(city)

    return render_template('weather.html',
                           weather=weather_data,
                           error=error,
                           city=city, )


@app.route('/history')
@login_required
def history():
    
    scans    = load_history()
    total    = len(scans)
    healthy  = sum(1 for s in scans if s['is_healthy'])
    diseased = total - healthy

    # Group by plant
    plants = {}
    for scan in scans:
        p = scan['plant']
        plants[p] = plants.get(p, 0) + 1

    top_plant = max(plants, key=plants.get) if plants else 'None'

    return render_template('history.html',
                           scans=scans,
                           total=total,
                           healthy=healthy,
                           diseased=diseased,
                           top_plant=top_plant, )

@app.route('/history/clear')
def clear_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    return redirect(url_for('history'))


# ============================================
# Auth Routes
# ============================================

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if 'user_id' in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        name     = request.form.get('name', '').strip()
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm  = request.form.get('confirm', '')

        # Validation
        if not name or not email or not password:
            flash('All fields are required.', 'error')
            return render_template('signup.html')

        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
            return render_template('signup.html')

        if password != confirm:
            flash('Passwords do not match.', 'error')
            return render_template('signup.html')

        # Check if email exists
        conn = get_db()
        existing = conn.execute(
            'SELECT id FROM users WHERE email = ?', (email,)
        ).fetchone()

        if existing:
            flash('Email already registered. Please login.', 'error')
            conn.close()
            return render_template('signup.html')

        # Create user
        hashed = generate_password_hash(password)
        conn.execute(
            'INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
            (name, email, hashed)
        )
        conn.commit()
        conn.close()

        flash('Account created successfully! Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        if not email or not password:
            flash('Please enter email and password.', 'error')
            return render_template('login.html')

        conn = get_db()
        user = conn.execute(
            'SELECT * FROM users WHERE email = ?', (email,)
        ).fetchone()
        conn.close()

        if not user or not check_password_hash(user['password'], password):
            flash('Invalid email or password.', 'error')
            return render_template('login.html')

        # Set session
        session['user_id']   = user['id']
        session['user_name'] = user['name']
        session['user_email'] = user['email']

        flash(f'Welcome back, {user["name"]}!', 'success')
        return redirect(url_for('index'))

    return render_template('login.html')


@app.route('/logout')
def logout():
    name = session.get('user_name', 'User')
    session.clear()
    flash(f'Goodbye, {name}! You have been logged out.', 'success')
    return redirect(url_for('login'))



# ============================================
# Run
# ============================================
if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    print("\n🌐 Starting KrishiSetu Platform...")
    print("   Open http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)