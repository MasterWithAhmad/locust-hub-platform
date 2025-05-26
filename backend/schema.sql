-- Create the database
CREATE DATABASE IF NOT EXISTS ml_project;
USE ml_project;

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    full_name VARCHAR(100) NOT NULL,
    email VARCHAR(100) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    security_question VARCHAR(255) NOT NULL,
    security_answer VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    region VARCHAR(100) NOT NULL,
    country_name VARCHAR(100) NOT NULL,
    start_year INT NOT NULL,
    start_month INT NOT NULL,
    ppt FLOAT NOT NULL,
    tmax FLOAT NOT NULL,
    soil_moisture FLOAT NOT NULL,
    locust_present BOOLEAN NOT NULL,
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Add indexes for better query performance
CREATE INDEX idx_user_email ON users(email);
CREATE INDEX idx_prediction_user ON predictions(user_id);
CREATE INDEX idx_prediction_date ON predictions(prediction_date); 