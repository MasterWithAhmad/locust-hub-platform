# LocustHub Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Locust Prediction System developed as a final year thesis project. This platform leverages machine learning to predict locust swarms, providing early warnings and analysis tools to agricultural communities.

## 🚀 Features

- **Real-time Prediction**: Machine learning models for accurate locust swarm forecasting
- **Interactive Dashboard**: Visualize predictions and historical data
- **User Management**: Secure authentication and authorization system
- **Data Analysis**: Tools for analyzing environmental factors affecting locust behavior
- **Responsive Design**: Accessible on desktop and mobile devices

## 📁 Project Structure

```
.
├── backend/           # Backend API and services
│   ├── app.py         # Main application entry point
│   ├── routes/        # API route definitions
│   ├── models/        # Database models
│   └── README.md      # Backend documentation
│
├── frontend/         # Frontend web application
│   ├── public/        # Static files
│   ├── app.js         # Express server
│   └── README.md      # Frontend documentation
│
└── thesis-project/   # Thesis documentation and research
    ├── data/          # Research datasets
    ├── models/        # ML model files
    └── README.md      # Thesis documentation
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Node.js 14+
- SQLite (or your preferred database)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/MasterWithAhmad/locust-hub-platform.git
   cd locust-hub-platform
   ```

2. **Set up the backend**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up the frontend**
   ```bash
   cd ../frontend
   npm install
   ```

## 🚦 Running the Application

1. **Start the backend server**
   ```bash
   cd backend
   python app.py
   ```

2. **Start the frontend server**
   ```bash
   cd ../frontend
   npm start
   ```

3. Access the application at `http://localhost:3000`

## 📚 Documentation

- [Backend Documentation](./backend/README.md)
- [Frontend Documentation](./frontend/README.md)
- [Thesis Documentation](./thesis-project/README.md)

## 🤝 Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) to get started.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## 📧 Contact

Ahmad - [ahmad.netdev@gmail.com](mailto:ahmad.netdev@gmail.com)

Project Link: [https://github.com/MasterWithAhmad/locust-hub-platform](https://github.com/MasterWithAhmad/locust-hub-platform)

## 🙏 Acknowledgments

### Technologies & Libraries
- [Flask](https://flask.palletsprojects.com/) - Python web framework
- [Express.js](https://expressjs.com/) - Node.js web application framework
- [Bootstrap 5](https://getbootstrap.com/) - Frontend component library
- [SQLite](https://www.sqlite.org/) - Database engine
- [Chart.js](https://www.chartjs.org/) - Data visualization
- [Font Awesome](https://fontawesome.com/) - Icons and UI components

### Data & Research
- [FAO Locust Watch](http://www.fao.org/ag/locusts/en/info/info/faq/index.html) - For locust behavior and prediction methodologies
- [Meteorological data sources](https://www.ncdc.noaa.gov/) - Weather and climate data