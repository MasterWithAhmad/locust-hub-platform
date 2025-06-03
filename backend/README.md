# LocustHub Backend Service

## Overview
This is the backend service for the Locust Prediction System, built with Python and Flask. It provides RESTful APIs for user authentication, environmental data processing, and locust swarm prediction.

## Features
- User authentication and authorization
- Environmental data processing
- Machine learning model integration for locust prediction
- RESTful API endpoints
- MySQL database integration
- Data visualization endpoints
- Prediction history tracking

## Prerequisites
- Python 3.8+
- pip (Python package manager)
- MySQL 8.0+

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/MasterWithAhmad/locust-hub-platform.git
   cd locust-hub-platform/backend
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory with the following variables:
   ```
   # MySQL Database Configuration
      DB_HOST=localhost
      DB_PORT=3306
      DB_USER=your_mysql_username
      DB_PASSWORD=your_mysql_password
      DB_NAME=locusthub
      
   # Make sure MySQL server is running and the database exists

   # JWT Configuration
      JWT_SECRET_KEY='your secret key here'

   # Flask Configuration
      FLASK_ENV=development
      FLASK_DEBUG=1
   ```


## Running the Application

```bash
python app.py
```

The server will start on `http://localhost:5000` by default.

## Project Structure

```
backend/
├── .env                     # Environment variables
├── app.py                   # Main Flask application
├── check_user.py            # User verification utility
├── requirements.txt         # Python dependencies
├── schema.sql              # Database schema definition
├── insert_test_predictions*.sql  # Test data scripts
├── routes/
│   └── auth.js             # Authentication routes and logic
└── README.md               # This file
```

## API Endpoints

### Authentication
- `POST /api/auth/register` - Register a new user
- `POST /api/auth/login` - User login
- `POST /api/auth/logout` - User logout


## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
Distributed under the MIT License. See `LICENSE` for more information.
