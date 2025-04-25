from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import os
from werkzeug.security import generate_password_hash, check_password_hash
import mysql.connector
import re
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import traceback

app = Flask(__name__)
app.secret_key = 'Crop_Disease_Prediction'  

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
MODEL_PATH = 'crop_disease_model.h5'
try:
    print("Loading model...")
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
    print(model.summary())
    print(f"Model input shape: {model.input_shape}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    exit(1)

# Define class names
CLASS_NAMES = [
    'Cashew healthy',
    'Cashew red rust',
    'Cassava brown spot',
    'Cassava healthy',
    'Guava scab',
    'Maize healthy',
    'Maize leaf blight',
    'Maize streak virus',
    'Pumpkin healthy',
    'Pumpkin powdery mildew',
    'Tomato healthy',
    'Tomato leaf curl',
    'Tomato septoria'
]

# Define class names and their treatment recommendations
DISEASE_INFO = {
    'Cashew healthy': {
        'recommendations': [
            'Continue regular maintenance and monitoring',
            'Maintain proper watering schedule',
            'Apply balanced fertilizer as needed',
            'Regular pruning to maintain plant health'
        ]
    },
    'Cashew red rust': {
        'recommendations': [
            'Apply copper-based fungicides',
            'Remove and destroy infected leaves',
            'Improve air circulation through pruning',
            'Maintain proper plant spacing',
            'Apply neem oil as preventive measure'
        ]
    },
    'Cassava brown spot': {
        'recommendations': [
            'Remove and destroy infected leaves',
            'Apply fungicides containing mancozeb',
            'Practice crop rotation',
            'Maintain proper field sanitation',
            'Use disease-free planting materials'
        ]
    },
    'Cassava healthy': {
        'recommendations': [
            'Continue regular maintenance',
            'Monitor for early signs of disease',
            'Maintain proper soil moisture',
            'Apply balanced fertilizer',
            'Control weeds regularly'
        ]
    },
    'Guava scab': {
        'recommendations': [
            'Apply copper-based fungicides',
            'Prune affected branches',
            'Remove fallen leaves and fruits',
            'Maintain proper tree spacing',
            'Use resistant varieties if available'
        ]
    },
    'Maize healthy': {
        'recommendations': [
            'Continue regular monitoring',
            'Maintain proper irrigation',
            'Apply balanced NPK fertilizer',
            'Control weeds regularly',
            'Practice crop rotation'
        ]
    },
    'Maize leaf blight': {
        'recommendations': [
            'Apply fungicides containing mancozeb',
            'Remove infected plant debris',
            'Practice crop rotation',
            'Use resistant varieties',
            'Maintain proper plant spacing'
        ]
    },
    'Maize streak virus': {
        'recommendations': [
            'Control vector (leafhoppers) with insecticides',
            'Remove and destroy infected plants',
            'Use resistant varieties',
            'Practice early planting',
            'Maintain proper field hygiene'
        ]
    },
    'Pumpkin healthy': {
        'recommendations': [
            'Continue regular maintenance',
            'Monitor for pests and diseases',
            'Maintain proper watering',
            'Apply balanced fertilizer',
            'Control weeds regularly'
        ]
    },
    'Pumpkin powdery mildew': {
        'recommendations': [
            'Apply sulfur-based fungicides',
            'Improve air circulation',
            'Water plants at the base',
            'Remove infected leaves',
            'Use resistant varieties if available'
        ]
    },
    'Tomato healthy': {
        'recommendations': [
            'Continue regular monitoring',
            'Maintain proper irrigation',
            'Apply balanced fertilizer',
            'Prune for better air circulation',
            'Control weeds regularly'
        ]
    },
    'Tomato leaf curl': {
        'recommendations': [
            'Control whitefly vectors with insecticides',
            'Remove and destroy infected plants',
            'Use resistant varieties',
            'Practice crop rotation',
            'Maintain proper field hygiene'
        ]
    },
    'Tomato septoria': {
        'recommendations': [
            'Apply copper-based fungicides',
            'Remove infected leaves',
            'Improve air circulation',
            'Water at the base of plants',
            'Practice crop rotation'
        ]
    }
}

# MySQL Configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'crop_disease_db'
}

def get_db_connection():
    try:
        conn = mysql.connector.connect(**db_config)
        return conn
    except mysql.connector.Error as err:
        print(f"Error connecting to MySQL: {err}")
        return None

def init_db():
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        cursor.close()
        conn.close()

init_db()

def validate_password(password):
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r"\d", password):
        return False, "Password must contain at least one number"
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "Password must contain at least one special character"
    return True, ""

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def preprocess_image(img):
    try:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        input_shape = model.input_shape
        if len(input_shape) == 4:
            IMG_SIZE = (input_shape[1], input_shape[2])
            print(f"Adjusting image size to: {IMG_SIZE}")
            img = img.resize(IMG_SIZE)
        
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        return img_array
    except Exception as e:
        print(f"Error during image preprocessing: {str(e)}")
        raise e

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction')
def prediction():
    if 'user_id' not in session:
        flash('Please sign in to access the prediction page.', 'warning')
        return redirect(url_for('signin'))
    return render_template('prediction.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username'].strip()
        email = request.form['email'].strip()
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if not username or not email or not password:
            flash('All fields are required!', 'error')
            return redirect(url_for('signup'))

        if not validate_email(email):
            flash('Invalid email format!', 'error')
            return redirect(url_for('signup'))

        if password != confirm_password:
            flash('Passwords do not match!', 'error')
            return redirect(url_for('signup'))

        is_valid, message = validate_password(password)
        if not is_valid:
            flash(message, 'error')
            return redirect(url_for('signup'))

        try:
            conn = get_db_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute('INSERT INTO users (username, email, password) VALUES (%s, %s, %s)',
                             (username, email, generate_password_hash(password)))
                conn.commit()
                cursor.close()
                conn.close()
                flash('Account created successfully! Please sign in.', 'success')
                return redirect(url_for('signin'))
        except mysql.connector.Error as err:
            if err.errno == 1062:
                flash('Username or email already exists!', 'error')
            else:
                flash('An error occurred. Please try again.', 'error')
            return redirect(url_for('signup'))

    return render_template('signup.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        email = request.form['email'].strip()
        password = request.form['password']
        remember = True if request.form.get('remember') else False

        if not email or not password:
            flash('Please enter both email and password!', 'error')
            return redirect(url_for('signin'))

        try:
            conn = get_db_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
                user = cursor.fetchone()
                cursor.close()
                conn.close()

                if user and check_password_hash(user[3], password):
                    session['user_id'] = user[0]
                    session['username'] = user[1]
                    if remember:
                        session.permanent = True
                    flash('Successfully signed in!', 'success')
                    return redirect(url_for('home'))
                else:
                    flash('Invalid email or password!', 'error')
                    return redirect(url_for('signin'))
        except mysql.connector.Error as err:
            flash('An error occurred. Please try again.', 'error')
            return redirect(url_for('signin'))

    return render_template('signin.html')

@app.route('/signout')
def signout():
    session.clear()
    flash('You have been signed out.', 'info')
    return redirect(url_for('home'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return jsonify({'error': 'Please sign in to make predictions'}), 401

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    prompt = request.form.get('prompt', '')

    if image_file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        img_bytes = image_file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        img_array = preprocess_image(img)
        
        predictions = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        recommendations = DISEASE_INFO[predicted_class]['recommendations']

        response = {
            'disease': predicted_class,
            'confidence': confidence,
            'prompt': prompt,
            'recommendations': recommendations
        }

        return jsonify(response)

    except Exception as e:
        print("Prediction error:", traceback.format_exc())
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True) 