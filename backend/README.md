# Backend Service

## Overview
This is the backend service for the News Recommendation System, built with Python and Flask. It provides RESTful APIs for user authentication, news processing, and recommendation generation.

## Features
- User authentication and authorization
- News article processing
- Recommendation generation
- RESTful API endpoints
- SQLite database integration

## Prerequisites
- Python 3.8+
- pip (Python package manager)
- SQLite3

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd backend
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
   # Database Configuration
      DB_HOST=localhost
      DB_USER=root
      DB_PASSWORD= ''   (if you have a password, put it there)
      DB_NAME='database name goes here'

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
