import mysql.connector
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def check_user(email):
    try:
        # Database configuration
        db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'user': os.getenv('DB_USER', 'root'),
            'password': os.getenv('DB_PASSWORD', ''),
            'database': os.getenv('DB_NAME', 'ml_project')
        }
        
        # Connect to the database
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        
        # Check if user exists
        cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
        user = cursor.fetchone()
        
        if user:
            print(f"User found: {user}")
        else:
            print(f"No user found with email: {email}")
            
            # List all users in the database
            cursor.execute('SELECT id, email, username FROM users LIMIT 10')
            all_users = cursor.fetchall()
            print("\nFirst 10 users in database:")
            for u in all_users:
                print(f"ID: {u['id']}, Email: {u['email']}, Username: {u['username']}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Check for the specific email
    email = "mo@gmail.com"
    print(f"Checking for email: {email}")
    check_user(email)
