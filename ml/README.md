# Locust Prediction System - Thesis Project

## 📝 Project Overview
This repository contains the implementation and documentation of a Locust Prediction System developed as a final year thesis project at Jamhuriya University. The system leverages machine learning techniques to predict locust swarms, providing early warnings to agricultural communities and helping mitigate potential crop damage.

## 🏛️ Academic Information
- **University**: Jamhuriya University of Science and Technology
- **Faculty**: Faculty of Computing and ICT
- **Department**: Information Technology
- **Supervisor**: Eng. Suldaanka (Abdi Rahman Omar Mohamud)
- **Thesis Committee**:
  - Eng. Abdulahi Hashi Abdi
  - Eng. Ayanle Ahmed Adow
  - Eng. Sulieman Ali Abshir

## 📁 Project Structure
```
thesis-project/
├── data/               # Datasets and processed data
│   ├── raw/           # Raw datasets
│   └── processed/     # Processed and cleaned data
├── models/            # Trained model files and model definitions
│   ├── saved_models/  # Serialized model files
│   └── training/      # Training scripts and logs
├── notebooks/         # Jupyter notebooks for analysis
│   ├── EDA.ipynb      # Exploratory Data Analysis
│   └── model_training.ipynb  # Model development
├── reports/           # Project reports and documentation
│   ├── final/        # Final thesis document
│   └── presentations/ # Presentation materials
├── .gitignore         # Git ignore file
├── LICENSE            # MIT License
└── requirements.txt   # Python dependencies
```

## 🛠️ Setup and Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Jupyter Notebook (for running analysis)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/MasterWithAhmad/locust-hub-platform.git
   cd locust-hub-platform/thesis-project
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

## 🚀 Usage

### Running Jupyter Notebooks
```bash
jupyter notebook notebooks/
```

### Training Models
```bash
python models/train.py --config config/training_config.yaml
```

### Generating Reports
```bash
python reports/generate_reports.py
```

## 📊 Features
- **Data Collection**: Automated data gathering from various environmental sources
- **Preprocessing**: Data cleaning and feature engineering pipelines
- **ML Models**: Implementation of various prediction models
- **Evaluation**: Comprehensive model evaluation metrics
- **Visualization**: Interactive plots and dashboards

## 📂 Data
- **Sources**:
  - Meteorological data
  - Historical locust swarm data
  - Satellite imagery
- **Location**: Primarily focused on the Horn of Africa region
- **Preprocessing**: See `notebooks/data_preprocessing.ipynb` for data cleaning steps

## 🤖 Models
- **Implemented Algorithms**:
  - Random Forest Classifier
  - XGBoost
  - Decision Tree and more
- **Model Persistence**: All trained models are saved in `models/saved_models/`
- **Evaluation**: Performance metrics and comparison reports in `reports/`

## 📚 Related Documents
- [Thesis Document](./reports/final/thesis.pdf)
- [Project Proposal](./reports/proposal.pdf)
- [Final Presentation](./reports/presentations/final_presentation.pdf)

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.

## 📧 Contact
Ahmad - [ahmad.netdev@gmail.com](mailto:ahmad.netdev@gmail.com)

Project Link: [https://github.com/MasterWithAhmad/locust-hub-platform](https://github.com/MasterWithAhmad/locust-hub-platform)

## 🙏 Acknowledgments

### Academic Support
- **Supervisor**: Eng. Suldaanka (Abdi Rahman Omar Mohamud) - Jamhuriya University
- **Thesis Committee**:
  - Eng. Abdulahi Hashi Abdi
  - Eng. Ayanle Ahmed Adow
  - Eng. Sulieman Ali Abshir
- Special thanks to the faculty of Jamhuriya University for their guidance and support

### Data Sources
- [FAO Locust Watch](http://www.fao.org/ag/locusts/)
- [NOAA Climate Data](https://www.ncdc.noaa.gov/)
- [NASA EarthData](https://earthdata.nasa.gov/)

### Technologies Used
- Python Data Stack (Pandas, NumPy, Scikit-learn)
- Jupyter Notebooks
- Matplotlib/Seaborn for visualizations
- Git for version control
