from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import mysql.connector
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

# Load environment variables
load_dotenv()

# Database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',  # Add password if you have one
    'database': 'ml_project'
}

app = Flask(__name__, static_folder='../frontend/public', static_url_path='')

# Configure CORS with credentials support
cors = CORS(app, 
     resources={
         r"/api/*": {
             "origins": ["http://localhost:5000", "http://127.0.0.1:5000", "http://localhost:3000", "http://localhost:8000", "http://127.0.0.1:8000"],
             "supports_credentials": True,
             "allow_headers": ["Content-Type", "Authorization", "Accept"],
             "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
             "expose_headers": ["Content-Type", "Authorization"]
         }
     })

# Set CORS headers for all responses - simplified as CORS middleware handles most headers
@app.after_request
def after_request(response):
    # These headers are now handled by the CORS middleware
    # We only need to add any additional headers not covered by CORS
    return response

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
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'thesis-project', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'random_forest.pkl')

# Load target encodings from the training data
def load_target_encodings():
    try:
        # Load the original training data
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'thesis-project', 'data', 'raw', 'locust_dataset.csv')
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
try:
    model = joblib.load(MODEL_PATH)
    print("ML model loaded successfully.")
    country_encodings, region_encodings = load_target_encodings()
    print("Target encodings loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    print("Please make sure the 'models' directory and 'random_forest.pkl' exist in the backend folder.")
    model = None
    country_encodings, region_encodings = pd.Series(), pd.Series()
except Exception as e:
    print(f"Error loading model or encodings: {e}")
    model = None
    country_encodings, region_encodings = pd.Series(), pd.Series()

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

# Make Prediction
@app.route('/api/predict', methods=['POST'])
def make_prediction():
    try:
        # Get data from request
        data = request.get_json()
        print("Received data:", data)
        
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
        # If region/country not in encodings, use mean or default value
        input_data['REGION'] = input_data['REGION'].map(region_encodings)
        input_data['COUNTRYNAME'] = input_data['COUNTRYNAME'].map(country_encodings)
        
        # Handle unknown regions/countries by using mean encoding or default value
        if input_data['REGION'].isna().any():
            input_data['REGION'] = input_data['REGION'].fillna(region_encodings.mean())
        if input_data['COUNTRYNAME'].isna().any():
            input_data['COUNTRYNAME'] = input_data['COUNTRYNAME'].fillna(country_encodings.mean())
        
        print("Input data after preprocessing:", input_data)
        
        # Make prediction
        prediction = model.predict(input_data)
        probabilities = model.predict_proba(input_data)
        
        print("Prediction:", prediction)
        print("Probabilities:", probabilities)
        
        # Return prediction and probability
        return jsonify({
            'prediction': 'yes' if prediction[0] == 1 else 'no',
            'probability': float(probabilities[0][1]),  # Probability of class 1 (yes)
            'matched_region': data['REGION'].strip().upper(),
            'matched_country': data['COUNTRYNAME'].strip().upper()
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
            soil_moisture, tmax, ppt, locust_present)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
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
            data['prediction_result']
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
                    prediction_date as created_at
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

        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'thesis-project', 'data', 'raw', 'locust_dataset.csv')
        df_options = pd.read_csv(data_path, usecols=['REGION', 'COUNTRYNAME'])

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

        cursor.execute('SELECT id, full_name, email, created_at FROM users WHERE id = %s', (current_user_id,))
        user = cursor.fetchone()

        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Format created_at to string
        if user.get('created_at') and hasattr(user['created_at'], 'isoformat'):
            user['created_at'] = user['created_at'].isoformat()

        return jsonify({'status': 'success', 'data': user}), 200

    except Exception as e:
        print(f"Error fetching user details: {str(e)}")
        return jsonify({'error': 'Error fetching user details', 'details': str(e)}), 500
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
        if conn: conn.rollback()
        return jsonify({'error': 'Error changing password', 'details': str(e)}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

@app.route('/api/account/delete', methods=['DELETE'])
@jwt_required()
def delete_account():
    """
    Delete the authenticated user's account and all associated data.
    Requires the user to be authenticated via JWT.
    """
    conn = None
    cursor = None
    try:
        # Get user ID from JWT
        user_id = get_jwt_identity()
        if not user_id:
            return jsonify({'error': 'User not authenticated'}), 401
            
        # Get database connection
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Start transaction
            conn.start_transaction()
            
            # 1. Delete user's predictions (or set to NULL if you want to keep the data)
            cursor.execute('DELETE FROM predictions WHERE user_id = %s', (user_id,))
            
            # 2. Delete user's session data (if you have a sessions table)
            # cursor.execute('DELETE FROM user_sessions WHERE user_id = %s', (user_id,))
            
            # 3. Delete user's profile data (if you have a separate profiles table)
            # cursor.execute('DELETE FROM user_profiles WHERE user_id = %s', (user_id,))
            
            # 4. Finally, delete the user account
            cursor.execute('DELETE FROM users WHERE id = %s', (user_id,))
            
            # Check if any row was affected
            if cursor.rowcount == 0:
                conn.rollback()
                return jsonify({'success': False, 'message': 'User not found'}), 404
            
            # Commit the transaction
            conn.commit()
            
            return jsonify({
                'success': True,
                'message': 'Account and all associated data have been permanently deleted.'
            }), 200
            
        except Exception as e:
            if conn:
                conn.rollback()
            print(f"Error deleting account: {str(e)}")
            return jsonify({
                'success': False,
                'message': 'Failed to delete account',
                'error': str(e)
            }), 500
            
    except Exception as e:
        print(f"Unexpected error in delete_account: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'An error occurred while processing your request',
            'error': str(e)
        }), 500
        
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

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

if __name__ == '__main__':
    # Initialize database on startup
    try:
        init_db()
        print("Database initialization completed successfully")
    except Exception as e:
        print(f"Warning: Database initialization failed: {e}")
    
    app.run(debug=True, port=5000)