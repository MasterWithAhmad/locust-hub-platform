# LocustHub Frontend Application

## Overview
This is the frontend application for the Locust Prediction System, built with Node.js, Express, and Bootstrap 5. It provides an interactive dashboard for monitoring and predicting locust swarms with a clean, responsive user interface.

## Features
- Responsive design using Bootstrap 5
- Real-time data visualization with Chart.js
- Interactive prediction interface
- User authentication and authorization
- Dashboard with key metrics and alerts
- Environmental data visualization
- Historical prediction tracking

## Prerequisites
- Node.js (v14+)
- npm (v6+)
- Backend API service (see backend README)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/MasterWithAhmad/locust-hub-platform.git
   cd locust-hub-platform/frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Configure environment**
   Create a `.env` file in the root directory if needed for any environment-specific configurations.

## Project Structure

```
frontend/
├── public/                 # Static files
│   ├── assets/             # Images, CSS, JS, and other static assets
│   ├── forms/              # Form-related HTML files
│   ├── analytics.html      # Analytics page
│   ├── blogs.html          # Blog posts and articles
│   ├── dashboard.html      # Main dashboard
│   ├── forgot-password.html # Password recovery
│   ├── index.html          # Home/Landing page
│   ├── login.html          # Login page
│   ├── predict.html        # Prediction interface
│   ├── reports.html        # Reports and analytics
│   ├── settings.html       # User settings
│   └── signup.html         # User registration
├── server.js              # Server entry point (Express setup)
├── package.json           # Project dependencies and scripts
└── README.md              # This file
```

## Available Scripts

### `npm start`
Starts the Express server in production mode.

### `npm run dev`
Starts the server using nodemon for development with auto-reload.

## Development

1. **Start the development server**
   ```bash
   npm run dev
   ```
   The application will be available at ` http://127.0.0.1:5000

2. **Making Changes**
   - Add or modify HTML files in the `public` directory
   - Place static assets in the `public/assets` directory
   - Update routes in `app.js` as needed

## Deployment

1. **Install production dependencies**
   ```bash
   npm install --production
   ```

2. **Start the server**
   ```bash
   npm start
   ```

   The application will be served on the specified port (default: 3000).

## Dependencies

- **express**: Web framework for Node.js
- **bootstrap**: Frontend CSS framework
- **nodemon**: Development dependency for auto-reloading
- **concurrently**: For running multiple commands concurrently

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
Distributed under the MIT License. See `LICENSE` for more information.
