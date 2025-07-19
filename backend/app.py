from flask import Flask, request, jsonify, send_from_directory, send_file, make_response
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity, get_jwt, get_jwt_identity
import mysql.connector
from mysql.connector import Error
import bcrypt
from datetime import datetime, timedelta
import os
import uuid
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import json
import joblib
import datetime
import re
from functools import lru_cache
from bs4 import BeautifulSoup

def clean_html_content(html_content):
    """
    Clean HTML content by removing potentially dangerous elements and attributes
    while preserving basic formatting.
    """
    if not html_content:
        return ''
    
    # List of allowed HTML tags
    ALLOWED_TAGS = [
        'p', 'br', 'strong', 'em', 'u', 's', 'blockquote', 'ul', 'ol', 'li',
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'a'
    ]
    
    # List of allowed attributes
    ALLOWED_ATTRS = {
        'a': ['href', 'title', 'target'],
        'p': ['style'],
        'span': ['style']
    }
    
    # Clean using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove all script tags and other potentially dangerous elements
    for tag in soup.find_all(True):
        # Remove disallowed tags
        if tag.name not in ALLOWED_TAGS:
            tag.unwrap()  # Remove the tag but keep its contents
            continue
            
        # Remove disallowed attributes
        attrs = list(tag.attrs.keys())
        for attr in attrs:
            if tag.name in ALLOWED_ATTRS and attr in ALLOWED_ATTRS[tag.name]:
                # Only allow specific attributes for specific tags
                continue
            del tag.attrs[attr]
    
    # Convert back to string and clean up any remaining issues
    cleaned = str(soup)
    
    # Additional cleaning for common issues
    cleaned = re.sub(r'<p[^>]*>\s*<br\s*/?>\s*</p>', '', cleaned)  # Empty paragraphs
    cleaned = re.sub(r'\s+', ' ', cleaned)  # Collapse multiple spaces
    cleaned = cleaned.replace('&nbsp;', ' ')  # Replace non-breaking spaces
    
    return cleaned.strip()

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder='../frontend/public', static_url_path='')

# Configure CORS
CORS(app, resources={
    r"/account/delete": {"origins": "*", "methods": ["DELETE"], "allow_headers": ["Content-Type", "Authorization"]},
    r"/api/*": {"origins": "*"}
}, supports_credentials=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Configure file uploads
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads', 'blog_images')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config.update(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max file size
    MAX_CONTENT_PATH=16 * 1024 * 1024,    # 16MB max file size
    UPLOAD_EXTENSIONS=ALLOWED_EXTENSIONS,
    UPLOAD_PATH=UPLOAD_FOLDER,
    # Increase buffer size for large files
    MAX_BUFFER_SIZE=16 * 1024 * 1024,  # 16MB
    # Configure Flask to handle large file uploads
    JSONIFY_PRETTYPRINT_REGULAR=True,
    JSON_SORT_KEYS=False,
    # Configure request timeout (in seconds)
    REQUEST_TIMEOUT=300  # 5 minutes
)

# Configure werkzeug to handle large file uploads
from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app)

# Database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',  # Add password if you have one
    'database': 'ml_project'
}

# Configure CORS with credentials support
cors = CORS()
cors.init_app(
    app,
    resources={
        r"/*": {
            "origins": ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5000", "http://127.0.0.1:5000"],
            "methods": ["GET", "HEAD", "POST", "OPTIONS", "PUT", "PATCH", "DELETE"],
            "allow_headers": ["Content-Type", "Authorization"],
            "supports_credentials": True,
            "expose_headers": ["Content-Type", "Authorization"]
        }
    },
    supports_credentials=True
)

# Image upload configuration
UPLOAD_FOLDER = os.path.abspath(os.path.join('frontend', 'public', 'assets', 'images', 'blog'))
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max file size

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Set CORS headers for all responses - simplified as CORS middleware handles most headers
@app.after_request
def after_request(response):
    # These headers are now handled by the CORS middleware
    # We only need to add any additional headers not covered by CORS
    return response

# JWT Configuration
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-in-production')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
app.config['JWT_TOKEN_LOCATION'] = ['headers', 'cookies']
app.config['JWT_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['JWT_COOKIE_CSRF_PROTECT'] = False  # Set to True in production
app.config['JWT_CSRF_CHECK_FORM'] = True

# Initialize JWT Manager
jwt = JWTManager(app)

# Add request logging middleware
@app.before_request
def log_request_info():
    print('Headers:', dict(request.headers))
    print('Body:', request.get_data())
    print('URL:', request.url)
    print('Method:', request.method)

# JWT Configuration
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-secret-key-123')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
app.config['JWT_TOKEN_LOCATION'] = ['headers']
app.config['JWT_HEADER_NAME'] = 'Authorization'
app.config['JWT_HEADER_TYPE'] = 'Bearer'
app.config['JWT_IDENTITY_CLAIM'] = 'identity'

jwt = JWTManager(app)

@jwt.user_identity_loader
def user_identity_lookup(user):
    # Return just the user ID as a string
    if isinstance(user, dict):
        return str(user.get('id', ''))
    return str(user) if user else ''

@jwt.user_lookup_loader
def user_lookup_callback(_jwt_header, jwt_data):
    # Return a dictionary with the user ID
    identity = jwt_data.get('sub', '')
    return {'id': str(identity) if identity else ''}

@jwt.additional_claims_loader
def add_claims_to_access_token(identity):
    # Add standard JWT claims
    now = datetime.datetime.utcnow()
    user_id = str(identity)  # identity is now just the user ID string
    return {
        'sub': user_id,  # Subject (must be included)
        'iat': now,  # Issued At
        'exp': now + timedelta(hours=1),  # Expiration Time
        'iss': 'ml-project-api',  # Issuer
        'jti': str(uuid.uuid4()),  # JWT ID
        'user_id': user_id  # For easy access to user ID
    }

# Database configuration
db_config = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'ml_project')
}

# Load the model and target encodings
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ml', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'random_forest.pkl')

# Load target encodings from the training data
def load_target_encodings():
    try:
        # Load the original training data
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ml', 'data', 'raw', 'locust_dataset.csv')
        print(f"Loading dataset from: {data_path}")
        df = pd.read_csv(data_path)
        
        # Preprocess the data similar to training
        df['REGION'] = df['REGION'].str.strip().str.upper()
        df['COUNTRYNAME'] = df['COUNTRYNAME'].str.strip().str.upper()
        df['LOCUSTPRESENT'] = df['LOCUSTPRESENT'].str.strip().str.upper()
        df['LOCUSTPRESENT'] = df['LOCUSTPRESENT'].map({'YES': 1, 'NO': 0})
        
        # Calculate target encodings
        country_target_means = df.groupby('COUNTRYNAME')['LOCUSTPRESENT'].mean()
        region_target_means = df.groupby('REGION')['LOCUSTPRESENT'].mean()
        
        return country_target_means, region_target_means
    except Exception as e:
        print(f"Error loading target encodings: {e}")
        # Return default values if loading fails
        return pd.Series(), pd.Series()

# Load model and target encodings at startup
model = None
country_encodings, region_encodings = pd.Series(), pd.Series()

# Try to load the default model, but don't fail if it's not available
try:
    # First try to load the default model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Default ML model loaded successfully.")
    else:
        print(f"Warning: Default model not found at {MODEL_PATH}")
        
    # Always try to load the target encodings
    country_encodings, region_encodings = load_target_encodings()
    if not country_encodings.empty and not region_encodings.empty:
        print("Target encodings loaded successfully.")
    else:
        print("Warning: Could not load target encodings or they are empty")
        
except Exception as e:
    print(f"Warning: Error during model/encoding initialization: {str(e)}")
    print("The application will continue but some features may not work correctly.")
    print("Please check the model files in the ml/models directory.")

def init_db():
    """Initialize the database with required tables if they don't exist."""
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        # Create users table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(80) UNIQUE NOT NULL,
            email VARCHAR(120) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        ''')
        
        # Create predictions table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            region VARCHAR(100) NOT NULL,
            country_name VARCHAR(100) NOT NULL,
            prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            locust_present TINYINT(1) NOT NULL,
            probability FLOAT,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            INDEX (user_id, prediction_date)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        ''')
        
        # Create blog_posts table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS blog_posts (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            title VARCHAR(255) NOT NULL,
            content LONGTEXT NOT NULL,
            region VARCHAR(100),
            country VARCHAR(100),
            date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            author VARCHAR(100),
            tags TEXT,
            image_url VARCHAR(255),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        ''')
        
        conn.commit()
        print("Database tables verified/created successfully")
        
    except mysql.connector.Error as e:
        print(f"Error initializing database: {e}")
        if conn and conn.is_connected():
            conn.rollback()
        raise
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()

def get_db_connection():
    """Get a database connection and ensure tables exist."""
    try:
        connection = mysql.connector.connect(**db_config)
        print('Successfully connected to database')
        return connection
    except mysql.connector.Error as e:
        print(f'Error connecting to database: {e}')
        raise

# Serve frontend files
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

# Contact Form Submission
@app.route('/api/contact', methods=['POST'])
def save_contact_message():
    try:
        data = request.get_json()
        name = data.get('name')
        email = data.get('email')
        phone = data.get('phone')
        message = data.get('message')

        if not all([name, email, phone, message]):
            return jsonify({'error': 'All fields are required'}), 400

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO contact_messages (name, email, phone, message) VALUES (%s, %s, %s, %s)',
            (name, email, phone, message)
        )
        conn.commit()
        return jsonify({'message': 'Message submitted successfully'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()

# User Registration
@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        full_name = data.get('full_name')
        email = data.get('email')
        password = data.get('password')
        security_question = data.get('security_question')
        security_answer = data.get('security_answer')

        if not all([full_name, email, password, security_question, security_answer]):
            return jsonify({'error': 'All fields are required'}), 400

        # Hash password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        conn = get_db_connection()
        cursor = conn.cursor()

        # Check if email already exists
        cursor.execute('SELECT id FROM users WHERE email = %s', (email,))
        if cursor.fetchone():
            return jsonify({'error': 'Email already registered'}), 400

        # Insert new user with security question and answer
        cursor.execute(
            'INSERT INTO users (full_name, email, password, security_question, security_answer) VALUES (%s, %s, %s, %s, %s)',
            (full_name, email, hashed_password, security_question, security_answer)
        )
        conn.commit()

        return jsonify({'message': 'User registered successfully'}), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        conn.close()

# User Login
@app.route('/api/login', methods=['POST'])
def login():
    conn = None
    cursor = None
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        if not all([email, password]):
            return jsonify({'error': 'Email and password are required'}), 400

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Get user
        cursor.execute('SELECT id, email, password, full_name FROM users WHERE email = %s', (email,))
        user = cursor.fetchone()

        if not user or not bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            return jsonify({'error': 'Invalid email or password'}), 401

        # Use the user ID as the identity (must be a string)
        user_id = str(user['id'])
        user_data = {
            'id': user_id,
            'email': user['email'],
            'full_name': user['full_name']
        }
        
        # Create access token with user ID as identity
        # Additional user data will be included via the additional_claims_loader
        access_token = create_access_token(identity=user_id)
        
        # Return token and user data
        response_data = {
            'access_token': access_token,
            'user': user_data,
            'message': 'Login successful',
            'status': 'success'
        }
        print(f"[DEBUG] Login response: {response_data}")
        return jsonify(response_data), 200

    except Exception as e:
        print(f"Login error: {str(e)}")  # Add debug logging
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# List of allowed models and their display names
ALLOWED_MODELS = {
    'random_forest.pkl': 'Random Forest',
    'random_forest_tuned.pkl': 'Random Forest (Tuned)',
    'gradient_boosting.pkl': 'Gradient Boosting',
    'lightgbm.pkl': 'LightGBM',
    'xgboost.pkl': 'XGBoost'
}

@lru_cache(maxsize=8)
def load_model(model_filename):
    # Block loading Decision Tree model
    if 'decision_tree' in model_filename.lower():
        raise ValueError("Decision Tree model is not available")
        
    model_path = os.path.join(MODEL_DIR, model_filename)
    
    # Double-check the model being loaded
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    # Load the model and verify it's not a Decision Tree
    model = joblib.load(model_path)
    model_class = str(type(model).__name__).lower()
    
    if 'decisiontree' in model_class:
        raise ValueError(f"Attempted to load a Decision Tree model: {model_filename}")
        
    return model

@app.route('/api/models', methods=['GET'])
def get_models():
    """Return the list of available models for prediction."""
    response = jsonify([
        {'value': fname, 'label': label}
        for fname, label in ALLOWED_MODELS.items()
    ])
    # Add headers to prevent caching
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/api/predict', methods=['POST'])
def make_prediction():
    try:
        # Get data from request
        data = request.get_json()
        print("Received data:", data)
        
        # Hardcode the model name to ensure only Random Forest is used
        model_name = 'random_forest.pkl'
        model_path = os.path.join(MODEL_DIR, model_name)
        
        # Verify the model file exists
        if not os.path.exists(model_path):
            error_msg = f"Random Forest model file not found at {model_path}"
            print(error_msg)
            return jsonify({
                'error': 'Prediction model not available. Please contact support.',
                'details': error_msg
            }), 500
            
        try:
            # Load the model directly without using the load_model function
            # to ensure no other model can be loaded
            model_to_use = joblib.load(model_path)
            print("Using Random Forest model for prediction")
            
            # Double-check the loaded model is not a Decision Tree
            model_class = str(type(model_to_use).__name__).lower()
            if 'decisiontree' in model_class:
                raise ValueError(f"Invalid model type loaded: {model_class}")
                
        except Exception as e:
            error_msg = f"Error loading Random Forest model: {str(e)}"
            print(error_msg)
            return jsonify({
                'error': 'Failed to load the prediction model. Please contact support.',
                'details': error_msg
            }), 500
        # Create DataFrame
        input_data = pd.DataFrame({
            'REGION': [data['REGION'].strip().upper()],
            'COUNTRYNAME': [data['COUNTRYNAME'].strip().upper()],
            'STARTYEAR': [int(data['STARTYEAR'])],
            'STARTMONTH': [int(data['STARTMONTH'])],
            'PPT': [float(data['PPT'])],
            'TMAX': [float(data['TMAX'])],
            'SOILMOISTURE': [float(data['SOILMOISTURE'])]
        })
        print("Input data before preprocessing:", input_data)
        # Apply target encoding for REGION and COUNTRYNAME
        input_data['REGION'] = input_data['REGION'].map(region_encodings)
        input_data['COUNTRYNAME'] = input_data['COUNTRYNAME'].map(country_encodings)
        # Handle unknown regions/countries by using mean encoding or default value
        if input_data['REGION'].isna().any():
            input_data['REGION'] = input_data['REGION'].fillna(region_encodings.mean())
        if input_data['COUNTRYNAME'].isna().any():
            input_data['COUNTRYNAME'] = input_data['COUNTRYNAME'].fillna(country_encodings.mean())
        print("Input data after preprocessing:", input_data)
        # Make prediction
        prediction = model_to_use.predict(input_data)
        probabilities = model_to_use.predict_proba(input_data)
        print("Prediction:", prediction)
        print("Probabilities:", probabilities)
        # Return prediction and probability
        return jsonify({
            'prediction': 'yes' if prediction[0] == 1 else 'no',
            'probability': float(probabilities[0][1]),  # Probability of class 1 (yes)
            'matched_region': data['REGION'].strip().upper(),
            'matched_country': data['COUNTRYNAME'].strip().upper(),
            'model_used': model_name
        })
    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({'error': str(e)}), 500

# Global error handler for /api/predict
@app.errorhandler(422)
def handle_422(err):
    messages = ['An unprocessable entity error occurred.']
    if hasattr(err, 'data') and 'messages' in err.data:
        messages = err.data['messages']
    print(f"422 Error: {messages}")

# Save Prediction
@app.route('/api/save_prediction', methods=['POST'])
def save_prediction():
    cursor = None
    connection = None
    try:
        data = request.get_json()
        print('Received data:', data)

        # Get user email from Authorization header
        auth_header = request.headers.get('Authorization')
        user_email = None
        if auth_header:
            try:
                user_data = json.loads(auth_header)
                if user_data.get('isLoggedIn'):
                    user_email = user_data.get('email')
            except Exception as e:
                print('Error parsing auth header:', str(e))
                pass

        if not user_email:
            return jsonify({'error': 'Not authenticated'}), 401

        connection = get_db_connection()
        if not connection:
            return jsonify({'error': 'Database connection failed'}), 500

        # First get user_id from email
        cursor = connection.cursor(dictionary=True)
        cursor.execute('SELECT id FROM users WHERE email = %s', (user_email,))
        user = cursor.fetchone()
        if not user:
            return jsonify({'error': 'User not found'}), 404
        user_id = user['id']

        cursor = connection.cursor()
        sql = """
            INSERT INTO predictions 
            (user_id, region, country_name, start_year, start_month, 
            soil_moisture, tmax, ppt, locust_present, probability)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (
            user_id,
            data['region_name'],
            data['country_name'],
            data['start_year'],
            data['start_month'],
            data['soil_moisture_percent'],
            data['temperature_celsius'],
            data['precipitation_mm'],
            data['prediction_result'],
            data.get('probability', None)
        )

        print('Executing SQL with values:', values)
        cursor.execute(sql, values)
        connection.commit()

        return jsonify({
            'message': 'Prediction saved successfully',
            'id': cursor.lastrowid
        }), 200
    except mysql.connector.Error as e:
        print('Database error:', str(e))
        return jsonify({'error': f'Database error: {str(e)}'}), 500
    except Exception as e:
        print('Error saving prediction:', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

# Get User's Predictions
@app.route('/api/predictions/<int:prediction_id>', methods=['DELETE'])
@jwt_required()
def delete_prediction(prediction_id):
    conn = None
    cursor = None
    try:
        # Get the current user ID from JWT
        user_id = get_jwt_identity()
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Check if the prediction exists and belongs to the user
        cursor.execute('SELECT id FROM predictions WHERE id = %s AND user_id = %s', (prediction_id, user_id))
        prediction = cursor.fetchone()
        
        if not prediction:
            return jsonify({
                'status': 'error',
                'error': 'Prediction not found or access denied'
            }), 404
        
        # Delete the prediction
        cursor.execute('DELETE FROM predictions WHERE id = %s', (prediction_id,))
        conn.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'Prediction deleted successfully'
        })
        
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error deleting prediction: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': 'Failed to delete prediction',
            'details': str(e)
        }), 500
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()

@app.route('/api/predictions/<int:prediction_id>', methods=['GET'])
@jwt_required()
def get_prediction(prediction_id):
    conn = None
    cursor = None
    try:
        print(f"[DEBUG] Fetching prediction with ID: {prediction_id}")
        
        # Get the current user ID from JWT
        user_id = get_jwt_identity()
        print(f"[DEBUG] Current user ID: {user_id}")
        
        if not user_id:
            print("[ERROR] No user ID in token")
            return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
            
        try:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            
            # First, check if the prediction exists and user has access
            query = """
                SELECT p.*, u.full_name, u.email
                FROM predictions p
                JOIN users u ON p.user_id = u.id
                WHERE p.id = %s
            """
            print(f"[DEBUG] Executing query with prediction_id: {prediction_id}")
            cursor.execute(query, (prediction_id,))
            prediction = cursor.fetchone()
            
            if not prediction:
                print(f"[DEBUG] No prediction found with ID: {prediction_id}")
                return jsonify({'status': 'error', 'message': 'Prediction not found'}), 404
                
            print(f"[DEBUG] Found prediction: {prediction}")
            
            # Check if user has permission to view this prediction
            # Since we don't have roles, only allow users to view their own predictions
            if str(prediction['user_id']) != str(user_id):
                print(f"[DEBUG] Access denied for user {user_id} to prediction {prediction_id}")
                return jsonify({
                    'status': 'error', 
                    'message': 'Access denied. You can only view your own predictions.'
                }), 403
            
            # Convert Decimal to float for JSON serialization
            result = {}
            for key, value in prediction.items():
                if hasattr(value, 'to_eng_string'):
                    result[key] = float(value)
                else:
                    result[key] = value
            
            print(f"[DEBUG] Returning prediction data")
            return jsonify({
                'status': 'success',
                'data': result
            })
            
        except Exception as db_error:
            print(f"[ERROR] Database error: {str(db_error)}")
            print(f"[ERROR] Query: {query}")
            print(f"[ERROR] Params: ({prediction_id},)")
            raise
            
    except Exception as e:
        print(f"[ERROR] Failed to fetch prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error', 
            'message': 'Failed to fetch prediction details',
            'debug': str(e)
        }), 500
        
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()

@app.route('/api/predictions', methods=['GET'])
@jwt_required()
def get_predictions():
    conn = None
    cursor = None
    try:
        # Get the JWT identity (now just the user ID string)
        user_id = get_jwt_identity()
        print(f"[DEBUG] Current user ID from token: {user_id}")
        
        if not user_id:
            print("[ERROR] No user ID found in token")
            return jsonify({
                'status': 'error',
                'message': 'No user ID found in token'
            }), 401
            
        if not user_id:
            print("[ERROR] No user ID found in token")
            return jsonify({
                'status': 'error',
                'message': 'No user ID found in token'
            }), 401
            
        print(f"[DEBUG] Fetching predictions for user ID: {user_id}")
        
        # Initialize database to ensure tables exist
        try:
            init_db()
        except Exception as e:
            print(f"[WARNING] Database initialization warning: {e}")
            # Continue even if init fails, as tables might already exist
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # First, check if user exists
        cursor.execute("SELECT id FROM users WHERE id = %s", (user_id,))
        user_exists = cursor.fetchone()
        
        if not user_exists:
            return jsonify({
                'status': 'error',
                'message': 'User not found',
                'user_id': user_id
            }), 404
            
        try:
            # Fetch user's predictions with actual column names
            cursor.execute("""
                SELECT 
                    id, 
                    region as region_name, 
                    country_name, 
                    start_year, 
                    start_month, 
                    soil_moisture as soil_moisture_percent, 
                    tmax as temperature_celsius, 
                    ppt as precipitation_mm,
                    locust_present as prediction_result,
                    probability,
                    prediction_date as created_at,
                    feedback,
                    feedback_timestamp
                FROM predictions 
                WHERE user_id = %s
                ORDER BY prediction_date DESC
            """, (user_id,))
            
            predictions = cursor.fetchall()
            print(f"[DEBUG] Found {len(predictions)} predictions for user {user_id}")
            
            # Convert datetime objects to strings for JSON serialization
            formatted_predictions = []
            for pred in predictions:
                prediction = dict(pred)
                for key, value in prediction.items():
                    if value is not None and hasattr(value, 'isoformat'):
                        prediction[key] = value.isoformat()
                formatted_predictions.append(prediction)

            return jsonify({
                'status': 'success',
                'data': formatted_predictions,
                'count': len(formatted_predictions)
            }), 200
            
        except mysql.connector.Error as db_err:
            print(f"[ERROR] Database error in get_predictions: {db_err}")
            if 'Table' in str(db_err) and 'doesn\'t exist' in str(db_err):
                # If table doesn't exist, return empty array
                return jsonify({'predictions': []}), 200
            raise

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[ERROR] get_predictions failed: {str(e)}\n{error_details}")
        return jsonify({
            'error': 'Failed to fetch predictions',
            'details': str(e)
        }), 500
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()

@app.route('/api/options', methods=['GET'])
def get_options():
    try:
        # Assuming load_target_encodings also makes the full dataframe available or
        # we can re-read it here. For simplicity, let's assume we can access the data
        # or re-load relevant columns. Re-loading is safer if the global vars aren't
        # guaranteed to be populated correctly or if the data is too large.
        # Let's read only the necessary columns to be efficient.

        # Look for dataset in the ml/data/raw directory and other possible locations
        possible_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ml', 'data', 'raw', 'locust_dataset.csv'),  # ml/data/raw/locust_dataset.csv
            os.path.join(os.path.dirname(__file__), 'data', 'locust_dataset.csv'),  # backend/data/locust_dataset.csv
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'locust_dataset.csv'),  # data/locust_dataset.csv
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'locust_dataset.csv')  # root directory
        ]
        
        df_options = None
        for path in possible_paths:
            if os.path.exists(path):
                df_options = pd.read_csv(path, usecols=['REGION', 'COUNTRYNAME'])
                break
                
        if df_options is None:
            raise FileNotFoundError("Could not find locust_dataset.csv in any expected location")

        # Clean and get unique values
        regions = df_options['REGION'].str.strip().str.upper().unique().tolist()
        countries = df_options['COUNTRYNAME'].str.strip().str.upper().unique().tolist()

        # Sort for better user experience in the dropdowns
        regions.sort()
        countries.sort()

        return jsonify({
            'regions': regions,
            'countries': countries
        }), 200

    except FileNotFoundError:
        return jsonify({'error': 'Dataset file not found to load options.'}), 500
    except Exception as e:
        print(f"Error fetching options: {str(e)}")
        return jsonify({'error': f'Error loading options: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify database connectivity and table structure."""
    try:
        # Test database connection
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Check if tables exist
        cursor.execute("SHOW TABLES")
        tables = [table[f'Tables_in_{db_config["database"]}'] for table in cursor.fetchall()]
        
        # Check if predictions table exists and get its structure
        predictions_columns = []
        if 'predictions' in tables:
            cursor.execute("SHOW COLUMNS FROM predictions")
            predictions_columns = [col['Field'] for col in cursor.fetchall()]
        
        return jsonify({
            'status': 'ok',
            'database': db_config['database'],
            'tables': tables,
            'predictions_columns': predictions_columns,
            'tables_exist': {
                'users': 'users' in tables,
                'predictions': 'predictions' in tables
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn and conn.is_connected():
            conn.close()

@app.route('/api/analytics/prediction_summary', methods=['GET'])
@jwt_required()
def get_prediction_summary():
    conn = None
    cursor = None
    try:
        current_user_id = get_jwt_identity()
        if not current_user_id:
            return jsonify({'error': 'User not authenticated'}), 401

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute('''
            SELECT locust_present, COUNT(*) as count
            FROM predictions
            WHERE user_id = %s
            GROUP BY locust_present
        ''', (current_user_id,))

        summary_data = cursor.fetchall()

        # Format the data into a dictionary for easier frontend use
        summary = {'yes': 0, 'no': 0}
        for row in summary_data:
            if row['locust_present'] == 1:
                summary['yes'] = row['count']
            else:
                summary['no'] = row['count']

        return jsonify({'status': 'success', 'data': summary}), 200

    except Exception as e:
        print(f"Error fetching prediction summary: {str(e)}")
        return jsonify({'error': 'Error fetching prediction summary', 'details': str(e)}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

@app.route('/api/analytics/predictions_over_time', methods=['GET'])
@jwt_required()
def get_predictions_over_time():
    conn = None
    cursor = None
    try:
        current_user_id = get_jwt_identity()
        if not current_user_id:
            return jsonify({'error': 'User not authenticated'}), 401

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Extract month and year from prediction_date and count predictions
        # Note: Date functions vary by SQL dialect. This is for MySQL.
        # For other databases (e.g., PostgreSQL, SQLite), functions might differ.
        cursor.execute('''
            SELECT
                YEAR(prediction_date) as year,
                MONTH(prediction_date) as month,
                COUNT(*) as count
            FROM predictions
            WHERE user_id = %s
            GROUP BY YEAR(prediction_date), MONTH(prediction_date)
            ORDER BY year, month
        ''', (current_user_id,))

        time_series_data = cursor.fetchall()

        # Format the data for frontend, e.g., [{month: 'Jan 2023', count: 10}, ...]
        formatted_data = []
        for row in time_series_data:
            month_name = datetime.date(1900, row['month'], 1).strftime('%b') # Get short month name
            formatted_data.append({
                'month_year': f"{month_name} {row['year']}",
                'count': row['count']
            })

        return jsonify({'status': 'success', 'data': formatted_data}), 200

    except Exception as e:
        print(f"Error fetching predictions over time: {str(e)}")
        return jsonify({'error': 'Error fetching predictions over time', 'details': str(e)}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

@app.route('/api/analytics/predictions_by_location', methods=['GET'])
@jwt_required()
def get_predictions_by_location():
    conn = None
    cursor = None
    try:
        current_user_id = get_jwt_identity()
        if not current_user_id:
            return jsonify({'error': 'User not authenticated'}), 401
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute('''
            SELECT
                region,
                country_name,
                COUNT(*) as total_count,
                SUM(CASE WHEN locust_present = 1 THEN 1 ELSE 0 END) as positive_count
            FROM predictions
            WHERE user_id = %s
            GROUP BY region, country_name
            ORDER BY country_name, region
        ''', (current_user_id,))

        location_data = cursor.fetchall()

        # Optional: Format data into a nested structure (e.g., by country then region)
        # For simplicity, let's return a flat list for now, frontend can process
        return jsonify({'status': 'success', 'data': location_data}), 200

    except Exception as e:
        print(f"Error fetching predictions by location: {str(e)}")
        return jsonify({'error': 'Error fetching predictions by location', 'details': str(e)}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

@app.route('/api/analytics/environmental_factors_summary', methods=['GET'])
@jwt_required()
def get_environmental_factors_summary():
    conn = None
    cursor = None
    try:
        current_user_id = get_jwt_identity()
        if not current_user_id:
            return jsonify({'error': 'User not authenticated'}), 401

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Calculate average environmental factors grouped by prediction outcome
        cursor.execute('''
            SELECT
                locust_present,
                AVG(ppt) as avg_ppt,
                AVG(tmax) as avg_tmax,
                AVG(soil_moisture) as avg_soil_moisture
            FROM predictions
            WHERE user_id = %s 
            GROUP BY locust_present
        ''', (current_user_id,))
        
        factors_data = cursor.fetchall()

        # Format the data for frontend, converting Decimal to float
        formatted_data = {}
        for row in factors_data:
            outcome = 'yes' if row['locust_present'] == 1 else 'no'
            formatted_data[outcome] = {
                'avg_ppt': float(row['avg_ppt']) if row['avg_ppt'] is not None else None,
                'avg_tmax': float(row['avg_tmax']) if row['avg_tmax'] is not None else None,
                'avg_soil_moisture': float(row['avg_soil_moisture']) if row['avg_soil_moisture'] is not None else None,
            }

        # Ensure both 'yes' and 'no' keys exist even if no data for one outcome
        if 'yes' not in formatted_data: formatted_data['yes'] = {'avg_ppt': None, 'avg_tmax': None, 'avg_soil_moisture': None}
        if 'no' not in formatted_data: formatted_data['no'] = {'avg_ppt': None, 'avg_tmax': None, 'avg_soil_moisture': None}

        return jsonify({'status': 'success', 'data': formatted_data}), 200

    except Exception as e:
        print(f"Error fetching environmental factors summary: {str(e)}")
        return jsonify({'error': 'Error fetching environmental factors summary', 'details': str(e)}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

@app.route('/api/user', methods=['GET'])
@jwt_required()
def get_user_details():
    conn = None
    cursor = None
    try:
        current_user_id = get_jwt_identity()
        if not current_user_id:
            return jsonify({'error': 'User not authenticated'}), 401

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Fetch user details
        cursor.execute(
            """
            SELECT id, full_name, email, 
                   DATE_FORMAT(created_at, '%Y-%m-%d') as created_at
            FROM users 
            WHERE id = %s
            """, 
            (current_user_id,)
        )
        user = cursor.fetchone()

        if not user:
            return jsonify({'error': 'User not found'}), 404
            
        # Get user stats
        cursor.execute(
            """
            SELECT 
                COUNT(*) as total_predictions,
                SUM(CASE WHEN feedback = 'correct' THEN 1 ELSE 0 END) as correct_predictions
            FROM predictions 
            WHERE user_id = %s
            """,
            (current_user_id,)
        )
        stats = cursor.fetchone()
        
        # Close database connection
        cursor.close()
        conn.close()
        
        # Prepare response
        response = {
            'id': user['id'],
            'full_name': user['full_name'],
            'email': user['email'],
            'created_at': user['created_at'],
            'total_predictions': stats['total_predictions'] or 0,
            'correct_predictions': stats['correct_predictions'] or 0,
            'accuracy': round((stats['correct_predictions'] / stats['total_predictions'] * 100), 2) 
                        if stats['total_predictions'] > 0 else 0
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"Error fetching user details: {str(e)}")
        return jsonify({'error': 'Failed to fetch user details'}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

@app.route('/api/user/profile', methods=['PUT'])
@jwt_required()
def update_user_profile():
    conn = None
    cursor = None
    try:
        current_user_id = get_jwt_identity()
        if not current_user_id:
            return jsonify({'error': 'User not authenticated'}), 401

        data = request.get_json()
        full_name = data.get('full_name')
        email = data.get('email')

        if not all([full_name, email]):
            return jsonify({'error': 'Full name and email are required'}), 400

        conn = get_db_connection()
        cursor = conn.cursor()

        # Check if email already exists for another user
        cursor.execute('SELECT id FROM users WHERE email = %s AND id != %s', (email, current_user_id))
        if cursor.fetchone():
            return jsonify({'error': 'Email already in use by another account'}), 400

        cursor.execute(
            'UPDATE users SET full_name = %s, email = %s WHERE id = %s',
            (full_name, email, current_user_id)
        )
        conn.commit()

        return jsonify({'status': 'success', 'message': 'Profile updated successfully'}), 200

    except Exception as e:
        print(f"Error updating user profile: {str(e)}")
        if conn: conn.rollback()
        return jsonify({'error': 'Error updating profile', 'details': str(e)}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

@app.route('/api/user/password', methods=['PUT'])
@jwt_required()
def change_password():
    conn = None
    cursor = None
    try:
        current_user_id = get_jwt_identity()
        if not current_user_id:
            return jsonify({'error': 'User not authenticated'}), 401

        data = request.get_json()
        current_password = data.get('current_password')
        new_password = data.get('new_password')

        if not all([current_password, new_password]):
            return jsonify({'error': 'Current password and new password are required'}), 400

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Fetch user to verify current password
        cursor.execute('SELECT id, password FROM users WHERE id = %s', (current_user_id,))
        user = cursor.fetchone()

        if not user or not bcrypt.checkpw(current_password.encode('utf-8'), user['password'].encode('utf-8')):
            return jsonify({'error': 'Invalid current password'}), 401

        # Hash the new password
        new_password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())

        # Update the password
        cursor.execute('UPDATE users SET password = %s WHERE id = %s', (new_password_hash, current_user_id))
        conn.commit()

        return jsonify({'status': 'success', 'message': 'Password changed successfully'}), 200

    except Exception as e:
        print(f"Error changing password: {str(e)}")
        if conn:
            conn.rollback()
        return jsonify({'error': 'Failed to change password'}), 500
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()

@app.route('/api/users/me', methods=['DELETE'])
@jwt_required()
def delete_account():
    conn = None
    cursor = None
    try:
        user_id = get_jwt_identity()
        if not user_id:
            return jsonify({'error': 'User not authenticated'}), 401

        conn = get_db_connection()
        cursor = conn.cursor()

        # Delete user's predictions first due to foreign key constraint
        cursor.execute('DELETE FROM predictions WHERE user_id = %s', (user_id,))
        
        # Delete user's account
        cursor.execute('DELETE FROM users WHERE id = %s', (user_id,))
        
        if cursor.rowcount == 0:
            return jsonify({'error': 'User not found'}), 404
            
        conn.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'Account and all associated data have been deleted successfully'
        }), 200

    except mysql.connector.Error as err:
        if conn:
            conn.rollback()
        print(f"Database error: {err}")
        return jsonify({'error': 'Database error while deleting account'}), 500
    except Exception as e:
        print(f"Error deleting account: {str(e)}")
        return jsonify({'error': 'Failed to delete account'}), 500
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()

# User registration endpoint is already defined above

# This route handles the frontend's delete account request
@app.route('/account/delete', methods=['DELETE'])
@jwt_required()
def delete_account_legacy():
    # Simply call the existing delete_account function
    return delete_account()

@app.route('/api/user/factory-reset', methods=['POST'])
@jwt_required()
def factory_reset():
    """
    Deletes all predictions for the authenticated user, but does not delete the user account.
    """
    conn = None
    cursor = None
    try:
        user_id = get_jwt_identity()
        if not user_id:
            return jsonify({'success': False, 'message': 'Unauthorized'}), 401
        conn = get_db_connection()
        cursor = conn.cursor()
        # Delete all predictions for this user
        cursor.execute('DELETE FROM predictions WHERE user_id = %s', (user_id,))
        conn.commit()
        return jsonify({'success': True, 'message': 'All your predictions have been deleted.'}), 200
    except Exception as e:
        if conn:
                conn.rollback()
        print(f"Error during factory reset: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()

# Forgot Password - Step 1: Get Security Question
@app.route('/api/auth/forgot-password', methods=['POST'])
def forgot_password():
    print("\n=== Forgot Password Request ===")
    print(f"Request Headers: {dict(request.headers)}")
    print(f"Request Data: {request.get_data()}")
    
    conn = None
    cursor = None
    try:
        # Get data from JSON request
        data = request.get_json()
        print(f"Parsed JSON Data: {data}")
        
        if not data:
            print("Error: No data provided")
            return jsonify({'error': 'No data provided'}), 400
            
        email = data.get('email')
        print(f"Looking up email: {email}")
        
        if not email:
            print("Error: Email is required")
            return jsonify({'error': 'Email is required'}), 400

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # First, check if any users exist in the database
        cursor.execute('SELECT COUNT(*) as count FROM users')
        user_count = cursor.fetchone()['count']
        print(f"Total users in database: {user_count}")
        
        # Debug: List all users in the database
        if user_count > 0:
            cursor.execute('SELECT id, email FROM users LIMIT 10')
            all_users = cursor.fetchall()
            print("First 10 users in database:")
            for u in all_users:
                print(f"ID: {u['id']}, Email: {u['email']}")
        
        # Get the user's security question (case-insensitive search)
        query = 'SELECT id, email, security_question FROM users WHERE LOWER(email) = LOWER(%s)'
        print(f"Executing query: {query} with email: {email}")
        cursor.execute(query, (email,))
        user = cursor.fetchone()
        
        if not user:
            print(f"No user found with email: {email}")
            return jsonify({'error': 'No account found with this email'}), 404
        
        print(f"Found user: ID={user['id']}, Email={user['email']}")
                    
        # Return the security question in JSON format
        response = {
            'success': True,
            'security_question': user['security_question']
        }
        print(f"Returning response: {response}")
        return jsonify(response)

    except Exception as e:
        print(f"Error in forgot-password: {str(e)}")
        return jsonify({
            'error': 'An error occurred while processing your request',
            'details': str(e)
        }), 500
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Forgot Password - Step 2: Verify Security Answer
@app.route('/api/auth/verify-answer', methods=['POST'])
def verify_answer():
    conn = None
    cursor = None
    try:
        # Get data from JSON request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        email = data.get('email')
        answer = data.get('answer')
        
        if not all([email, answer]):
            return jsonify({'error': 'Email and answer are required'}), 400

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Verify the security answer (case-insensitive)
        cursor.execute(
            'SELECT id FROM users WHERE email = %s AND LOWER(security_answer) = LOWER(%s)',
            (email, answer.strip())
        )
        user = cursor.fetchone()
        
        if not user:
            return jsonify({'error': 'Incorrect answer. Please try again.'}), 401
            
        # Return success response
        return jsonify({
            'success': True,
            'message': 'Answer verified successfully'
        })
        
    except Exception as e:
        print(f"Error in verify-answer: {str(e)}")
        return jsonify({
            'error': 'An error occurred while verifying your answer',
            'details': str(e)
        }), 500
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Forgot Password - Step 3: Reset Password
@app.route('/api/auth/reset-password', methods=['POST'])
def reset_password():
    print("\n=== Reset Password Request ===")
    print(f"Request Headers: {dict(request.headers)}")
    print(f"Request Data: {request.get_data()}")
    
    conn = None
    cursor = None
    try:
        # Get data from JSON request
        data = request.get_json()
        print(f"Parsed JSON Data: {data}")
        
        if not data:
            print("Error: No data provided")
            return jsonify({'error': 'No data provided'}), 400
            
        email = data.get('email')
        new_password = data.get('new_password')
        
        print(f"Email: {email}, Password Length: {len(new_password) if new_password else 0}")
        
        if not all([email, new_password]):
            print("Error: Missing email or password")
            return jsonify({'error': 'Email and new password are required'}), 400
            
        if len(new_password) < 8:
            print("Error: Password too short")
            return jsonify({'error': 'Password must be at least 8 characters long'}), 400

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # First, check if user exists
        cursor.execute('SELECT id FROM users WHERE email = %s', (email,))
        user = cursor.fetchone()
        print(f"User lookup result: {user}")
        
        if not user:
            print(f"Error: No user found with email: {email}")
            return jsonify({'error': 'User not found'}), 404
        
        # Hash the new password
        hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
        print(f"Hashed password: {hashed_password}")
        
        # Update the password
        try:
            cursor.execute(
                'UPDATE users SET password = %s WHERE email = %s',
                (hashed_password, email)
            )
            print(f"Password update query executed. Rows affected: {cursor.rowcount}")
            
            if cursor.rowcount == 0:
                print("Error: No rows were updated")
                return jsonify({'error': 'Failed to update password. No changes made.'}), 500
            
            conn.commit()
            print("Database changes committed successfully")
            
            # Return success response
            response = {
                'success': True,
                'message': 'Password updated successfully. You can now log in with your new password.'
            }
            print(f"Returning success response: {response}")
            return jsonify(response)
            
        except Exception as e:
            print(f"Database error during password update: {str(e)}")
            if conn:
                conn.rollback()
            raise
        
    except Exception as e:
        print(f"Error in reset-password: {str(e)}")
        if conn:
            conn.rollback()
        return jsonify({'error': 'An error occurred while resetting your password'}), 500
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# --- USER API ---
# This function was moved to the USER API section below

# --- BLOG POSTS API ---
@app.route('/api/uploads/blog_images/<filename>')
def serve_blog_image(filename):
    """Serve uploaded blog images"""
    try:
        return send_from_directory(
            app.config['UPLOAD_FOLDER'],
            filename,
            as_attachment=False
        )
    except FileNotFoundError:
        return jsonify({'error': 'Image not found'}), 404

@app.route('/api/blogposts/upload', methods=['POST'])
@jwt_required()
def upload_blog_image():
    """
    Upload an image for a blog post
    ---
    tags:
      - Blog Posts
    security:
      - JWT: []
    consumes:
      - multipart/form-data
    parameters:
      - in: formData
        name: image
        type: file
        required: true
        description: The image file to upload (max 16MB)
    responses:
      200:
        description: Image uploaded successfully
        schema:
          type: object
          properties:
            url:
              type: string
              description: URL to access the uploaded image
      400:
        description: No file part or invalid file
      401:
        description: Unauthorized
      413:
        description: File too large (max 16MB)
    """
    try:
        # Check if the post request has the file part
        if 'image' not in request.files:
            app.logger.warning('No file part in the request')
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['image']
        if not file or file.filename == '':
            app.logger.warning('No selected file')
            return jsonify({'error': 'No selected file'}), 400
            
        # Check file extension
        if not allowed_file(file.filename):
            app.logger.warning(f'File type not allowed: {file.filename}')
            return jsonify({
                'error': 'File type not allowed',
                'allowed': list(ALLOWED_EXTENSIONS)
            }), 400
            
        try:
            # Generate a secure filename with timestamp and UUID
            from datetime import datetime
            import uuid
            
            # Get file extension
            file_ext = os.path.splitext(file.filename)[1].lower()
            # Create a unique filename
            filename = f"{uuid.uuid4().hex}{file_ext}"
            
            # Ensure upload directory exists
            upload_dir = app.config['UPLOAD_FOLDER']
            os.makedirs(upload_dir, exist_ok=True)
            
            # Check available disk space (at least 50MB free space required)
            import shutil
            total, used, free = shutil.disk_usage(upload_dir)
            if free < 50 * 1024 * 1024:  # 50MB
                app.logger.error(f'Insufficient disk space: {free/1024/1024:.2f}MB free')
                return jsonify({'error': 'Insufficient disk space'}), 507
            
            # Save the file in chunks to prevent memory issues
            filepath = os.path.join(upload_dir, filename)
            chunk_size = 1024 * 1024  # 1MB chunks
            
            with open(filepath, 'wb') as f:
                while True:
                    chunk = file.stream.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
            
            # Verify the file was saved correctly
            if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                raise Exception('Failed to save file')
            
            # Create a URL that points to the uploaded file
            file_url = f"/api/uploads/blog_images/{filename}"
            
            app.logger.info(f'File uploaded successfully: {file_url} ({os.path.getsize(filepath)/1024:.2f} KB)')
            return jsonify({
                'message': 'File uploaded successfully',
                'url': file_url,
                'filename': filename,
                'imageUrl': file_url
            }), 200
            
        except Exception as e:
            app.logger.error(f'Error processing file upload: {str(e)}', exc_info=True)
            # Clean up partially uploaded file if it exists
            if 'filepath' in locals() and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception as cleanup_error:
                    app.logger.error(f'Error cleaning up file {filepath}: {str(cleanup_error)}')
            
            return jsonify({
                'error': 'Failed to process file upload',
                'details': str(e)
            }), 500
            
    except Exception as e:
        app.logger.error(f'Unexpected error in upload_blog_image: {str(e)}', exc_info=True)
        return jsonify({
            'error': 'An unexpected error occurred',
            'details': str(e)
        }), 500

@app.route('/api/blogposts', methods=['GET'])
@jwt_required()
def get_blog_posts():
    conn = None
    cursor = None
    try:
        # Get the current user's ID from the JWT token
        current_user = get_jwt_identity()
        if not current_user:
            return jsonify({'error': 'User not authenticated'}), 401
            
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Only fetch posts for the current user
        cursor.execute("""
            SELECT id, user_id, title, content, region, country, date, 
                   author, tags, image_url 
            FROM blog_posts 
            WHERE user_id = %s
            ORDER BY date DESC 
            LIMIT 20
        """, (current_user,))
        
        posts = cursor.fetchall()
        # Format date to ISO string
        for post in posts:
            if post['date'] and hasattr(post['date'], 'isoformat'):
                post['date'] = post['date'].isoformat()
        return jsonify(posts), 200
    except Exception as e:
        print(f"Error fetching user's blog posts: {str(e)}")
        return jsonify({'error': 'Failed to fetch blog posts'}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

@app.route('/api/blogposts/public', methods=['GET'])
def get_public_blog_posts():
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Fetch all public posts (no user filter)
        cursor.execute("""
            SELECT id, user_id, title, content, region, country, date, 
                   author, tags, image_url 
            FROM blog_posts 
            ORDER BY date DESC 
            LIMIT 50
        """)
        
        posts = cursor.fetchall()
        # Format date to ISO string
        for post in posts:
            if post['date'] and hasattr(post['date'], 'isoformat'):
                post['date'] = post['date'].isoformat()
        return jsonify(posts), 200
    except Exception as e:
        print(f"Error fetching public blog posts: {str(e)}")
        return jsonify({'error': 'Failed to fetch public blog posts'}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

@app.route('/api/blogposts/<int:post_id>', methods=['GET'])
def get_blog_post(post_id):
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT * FROM blog_posts WHERE id = %s", 
            (post_id,)
        )
        post = cursor.fetchone()
        if not post:
            return jsonify({'error': 'Blog post not found'}), 404
            
        # Format date
        if post['date'] and hasattr(post['date'], 'isoformat'):
            post['date'] = post['date'].isoformat()
            
        return jsonify(post), 200
    except Exception as e:
        print(f"Error fetching blog post: {str(e)}")
        return jsonify({'error': 'Failed to fetch blog post'}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

@app.route('/api/blogposts', methods=['POST'])
@jwt_required()
def create_blog_post():
    conn = None
    cursor = None
    try:
        # Get user ID from JWT token
        current_user_id = get_jwt_identity()
        
        if not current_user_id:
            return jsonify({'error': 'User not authenticated'}), 401
            
        print(f"[DEBUG] Creating blog post for user ID: {current_user_id}")
        
        # Accept JSON or form-data
        if request.is_json:
            data = request.get_json()
            title = data.get('title')
            raw_content = data.get('content')
            tags = data.get('tags')
            region = data.get('region')
            country = data.get('country')
            author = data.get('author')
            image_url = data.get('image_url')
        else:
            title = request.form.get('title')
            raw_content = request.form.get('content')
            tags = request.form.get('tags')
            region = request.form.get('region')
            country = request.form.get('country')
            author = request.form.get('author')
            image_file = request.files.get('image')
            image_url = None
            
            if image_file:
                # Save image to frontend/public/assets/blog_images
                img_dir = os.path.join(os.path.dirname(__file__), '../frontend/public/assets/blog_images')
                img_dir = os.path.abspath(img_dir)
                os.makedirs(img_dir, exist_ok=True)
                ext = os.path.splitext(image_file.filename)[-1]
                img_name = f"blog_{uuid.uuid4().hex}{ext}"
                img_path = os.path.join(img_dir, img_name)
                image_file.save(img_path)
                image_url = f"/assets/blog_images/{img_name}"
                print(f"[BLOG IMAGE] Saved to: {img_path}")
        
        # Clean the HTML content before saving
        content = clean_html_content(raw_content) if raw_content else ''
        
        # Process tags - ensure it's a string of comma-separated values
        processed_tags = ''
        if tags:
            if isinstance(tags, list):
                processed_tags = ','.join([str(tag).strip() for tag in tags if str(tag).strip()])
            else:
                processed_tags = str(tags).strip()

        if not title or not content:
            return jsonify({'error': 'Title and content are required'}), 400
            
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get user's full name for the author field if not provided
        if not author:
            cursor.execute("SELECT full_name FROM users WHERE id = %s", (current_user_id,))
            user = cursor.fetchone()
            if user and user[0]:
                author = user[0]
        
        cursor.execute("""
            INSERT INTO blog_posts 
            (user_id, title, content, region, country, author, tags, image_url) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (current_user_id, title, content, region, country, author, processed_tags, image_url))
        
        post_id = cursor.lastrowid
        conn.commit()
        
        return jsonify({
            'message': 'Blog post created successfully',
            'post_id': post_id
        }), 201
        
    except Exception as e:
        print(f"Error creating blog post: {str(e)}")
        return jsonify({'error': 'Failed to create blog post'}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

@app.route('/api/blogposts/<int:post_id>', methods=['PUT'])
@jwt_required()
def update_blog_post(post_id):
    conn = None
    cursor = None
    try:
        # Get user ID from JWT token
        current_user_id = get_jwt_identity()
        
        if not current_user_id:
            return jsonify({'error': 'User not authenticated'}), 401
            
        print(f"[DEBUG] Updating blog post {post_id} for user ID: {current_user_id}")
        
        # Check if post exists and belongs to user
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT user_id FROM blog_posts WHERE id = %s", 
            (post_id,)
        )
        post = cursor.fetchone()
        
        if not post:
            return jsonify({'error': 'Blog post not found'}), 404
            
        # Convert both to integers for comparison
        try:
            post_user_id = int(post.get('user_id'))
            current_user_id_int = int(current_user_id)
            
            if post_user_id != current_user_id_int:
                return jsonify({'error': 'Unauthorized to update this post'}), 403
                
        except (ValueError, TypeError) as e:
            print(f"Error comparing user IDs: {e}")
            return jsonify({'error': 'Invalid user ID format'}), 400
        
        # Get update data
        data = request.get_json() if request.is_json else request.form
        
        # Clean the content if it's being updated
        if 'content' in data and data['content']:
            data['content'] = clean_html_content(data['content'])
            
        # Process tags if they're being updated
        if 'tags' in data and data['tags'] is not None:
            if isinstance(data['tags'], list):
                data['tags'] = ','.join([str(tag).strip() for tag in data['tags'] if str(tag).strip()])
            else:
                data['tags'] = str(data['tags']).strip()
        
        # Build dynamic update query
        update_fields = []
        update_values = []
        
        # Handle image_url separately as it might come from file upload
        if 'image_url' in data and data['image_url'] is not None:
            update_fields.append("image_url = %s")
            update_values.append(data['image_url'])
        
        # Handle other fields
        for field in ['title', 'content', 'tags', 'region', 'country']:
            if field in data and data[field] is not None:
                update_fields.append(f"{field} = %s")
                update_values.append(data[field])
        
        if not update_fields:
            return jsonify({'error': 'No fields to update'}), 400
            
        # Add post_id to values
        update_values.append(post_id)
        
        # Execute update
        query = f"UPDATE blog_posts SET {', '.join(update_fields)} WHERE id = %s"
        print(f"Executing query: {query} with values: {update_values}")  # Debug log
        cursor.execute(query, update_values)
        conn.commit()
        
        return jsonify({'message': 'Blog post updated successfully'}), 200
        
    except Exception as e:
        print(f"Error updating blog post: {str(e)}")
        return jsonify({'error': 'Failed to update blog post'}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

@app.route('/api/blogposts/<int:post_id>', methods=['DELETE'])
@jwt_required()
def delete_blog_post(post_id):
    conn = None
    cursor = None
    try:
        # Get the JWT identity (should be the user ID)
        current_user_id = get_jwt_identity()
        
        if not current_user_id:
            return jsonify({'error': 'Invalid authentication'}), 401
            
        print(f"Current user ID from JWT: {current_user_id}")
        
        # Check if post exists and belongs to user
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT id, user_id, image_url FROM blog_posts WHERE id = %s", 
            (post_id,)
        )
        post = cursor.fetchone()
        
        if not post:
            print(f"Post {post_id} not found")
            return jsonify({'error': 'Blog post not found'}), 404
            
        print(f"Post found: {post}")
        print(f"Post user_id: {post.get('user_id')}, Current user ID: {current_user_id}")
        print(f"Type comparison - post_user_id: {type(post.get('user_id'))}, current_user_id: {type(current_user_id)}")
            
        # Convert both to integers for comparison to ensure type consistency
        try:
            post_user_id = int(post.get('user_id'))
            current_user_id_int = int(current_user_id)
            
            if post_user_id != current_user_id_int:
                print(f"Unauthorized: User {current_user_id_int} cannot delete post {post_id} owned by {post_user_id}")
                return jsonify({'error': 'Unauthorized to delete this post'}), 403
                
        except (ValueError, TypeError) as e:
            print(f"Error comparing user IDs: {e}")
            return jsonify({'error': 'Invalid user ID format'}), 400
        
        # Delete image file if exists
        if post['image_url']:
            try:
                img_path = os.path.join(
                    os.path.dirname(__file__), 
                    f"../frontend/public{post['image_url']}"
                )
                if os.path.exists(img_path):
                    os.remove(img_path)
            except Exception as e:
                print(f"Error deleting image file: {str(e)}")
        
        # Delete post
        cursor.execute("DELETE FROM blog_posts WHERE id = %s", (post_id,))
        conn.commit()
        
        return jsonify({'message': 'Blog post deleted successfully'}), 200
        
    except Exception as e:
        print(f"Error deleting blog post: {str(e)}")
        return jsonify({'error': 'Failed to delete blog post'}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

@app.route('/api/users/me/blogposts', methods=['GET'])
@jwt_required()
def get_current_user_blog_posts():
    conn = None
    cursor = None
    try:
        # Get the current user ID from the JWT token
        current_user_id = get_jwt_identity()
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT id, title, content, region, country, date, author, image_url 
            FROM blog_posts 
            WHERE user_id = %s 
            ORDER BY date DESC
        """, (current_user_id,))
        posts = cursor.fetchall()
        
        # Format dates
        for post in posts:
            if post['date'] and hasattr(post['date'], 'isoformat'):
                post['date'] = post['date'].isoformat()
                
        return jsonify(posts), 200
        
    except Exception as e:
        print(f"Error fetching user blog posts: {str(e)}")
        return jsonify({'error': 'Failed to fetch user blog posts'}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

import glob

@app.route('/api/debug/blog_images')
def debug_blog_images():
    static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../frontend/public'))
    img_dir = os.path.join(static_dir, 'assets/blog_images')
    files = glob.glob(os.path.join(img_dir, '*'))
    files = [os.path.basename(f) for f in files]
    return {
        'static_dir': static_dir,
        'img_dir': img_dir,
        'files': files,
        'url_example': '/assets/blog_images/' + files[0] if files else None
    }

@app.route('/api/predictions/<int:prediction_id>/feedback', methods=['POST'])
@jwt_required()
def set_prediction_feedback(prediction_id):
    user_id = get_jwt_identity()
    data = request.get_json()
    feedback = data.get('feedback')
    if feedback not in ['correct', 'incorrect']:
        return jsonify({'success': False, 'message': 'Invalid feedback value'}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Only allow feedback on user's own predictions and if feedback is not already set
        cursor.execute('SELECT feedback FROM predictions WHERE id = %s AND user_id = %s', (prediction_id, user_id))
        row = cursor.fetchone()
        if not row:
            return jsonify({'success': False, 'message': 'Prediction not found or access denied'}), 404
        if row[0] is not None:
            return jsonify({'success': False, 'message': 'Feedback already submitted for this prediction'}), 400

        cursor.execute(
            'UPDATE predictions SET feedback = %s, feedback_timestamp = NOW() WHERE id = %s',
            (feedback, prediction_id)
        )
        conn.commit()
        return jsonify({'success': True, 'message': 'Feedback recorded'})
    except Exception as e:
        conn.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/api/analytics/feedback', methods=['GET'])
@jwt_required()
def get_feedback_analytics():
    user_id = get_jwt_identity()
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        # Total feedback, correct, incorrect
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(feedback = 'correct') as correct_count,
                SUM(feedback = 'incorrect') as incorrect_count
            FROM predictions
            WHERE user_id = %s AND feedback IS NOT NULL
        ''', (user_id,))
        row = cursor.fetchone()
        total = row['total'] or 0
        correct = row['correct_count'] or 0
        incorrect = row['incorrect_count'] or 0
        correct_pct = round((correct / total) * 100, 1) if total else 0.0
        incorrect_pct = round((incorrect / total) * 100, 1) if total else 0.0

        # Feedback over time (weekly)
        cursor.execute('''
            SELECT DATE_FORMAT(feedback_timestamp, '%Y-%u') as week,
                   SUM(feedback = 'correct') as correct,
                   SUM(feedback = 'incorrect') as incorrect
            FROM predictions
            WHERE user_id = %s AND feedback IS NOT NULL
            GROUP BY week
            ORDER BY week DESC
            LIMIT 12
        ''', (user_id,))
        feedback_over_time = [
            {'period': row['week'], 'correct': row['correct'], 'incorrect': row['incorrect']}
            for row in cursor.fetchall()
        ]
        feedback_over_time.reverse()  # chronological order

        # Feedback by region
        cursor.execute('''
            SELECT region,
                   SUM(feedback = 'correct') as correct,
                   SUM(feedback = 'incorrect') as incorrect
            FROM predictions
            WHERE user_id = %s AND feedback IS NOT NULL
            GROUP BY region
            ORDER BY region
        ''', (user_id,))
        feedback_by_region = [
            {'region': row['region'], 'correct': row['correct'], 'incorrect': row['incorrect']}
            for row in cursor.fetchall()
        ]

        # Recent feedback entries
        cursor.execute('''
            SELECT id as prediction_id, feedback_timestamp as date, region, country_name as country, 
                   CASE WHEN locust_present = 1 THEN 'Yes' ELSE 'No' END as result, feedback
            FROM predictions
            WHERE user_id = %s AND feedback IS NOT NULL
            ORDER BY feedback_timestamp DESC
            LIMIT 10
        ''', (user_id,))
        recent_feedback = []
        for row in cursor.fetchall():
            # Format date as ISO string if possible
            if row['date'] and hasattr(row['date'], 'isoformat'):
                row['date'] = row['date'].isoformat()
            recent_feedback.append(row)

        return jsonify({
            'total_feedback': total,
            'correct_count': correct,
            'incorrect_count': incorrect,
            'correct_pct': correct_pct,
            'incorrect_pct': incorrect_pct,
            'feedback_over_time': feedback_over_time,
            'feedback_by_region': feedback_by_region,
            'recent_feedback': recent_feedback
        }), 200
    except Exception as e:
        print(f"Error in feedback analytics: {e}")
        return jsonify({'error': 'Failed to fetch feedback analytics', 'details': str(e)}), 500
    finally:
        cursor.close()
        conn.close()

if __name__ == '__main__':
    # Initialize database on startup
    try:
        init_db()
        print("Database initialization completed successfully")
    except Exception as e:
        print(f"Warning: Database initialization failed: {e}")
    
    app.run(debug=True, port=5000)