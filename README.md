# 🚀 Interactive Skills Portfolio

A Flask web application showcasing technical skills through interactive demonstrations.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.3.3-green?style=flat-square&logo=flask)
![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-yellow?style=flat-square&logo=javascript)

## ✨ Features

- **⚡ Embedded Systems**: Live C/C++ code generation with sensor simulation
- **🌐 IoT Dashboard**: Real-time sensor monitoring and device control
- **🧠 Machine Learning**: Interactive model training and predictions
- **📊 Data Science**: Dynamic analysis with visualizations

## 🚀 Quick Start

1. **Setup**
   ```bash
   mkdir flask_skills_portfolio && cd flask_skills_portfolio
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install**
   ```bash
   pip install Flask==2.3.3 numpy==1.24.3
   mkdir templates
   ```

3. **Run**
   ```bash
   python app.py
   ```
   Open `http://localhost:5000`

## 🎮 What It Does

### Embedded Systems
- Generates real C++ Arduino code with live sensor readings
- Simulates temperature, light, and moisture sensors
- Shows hardware interfacing examples

### IoT Dashboard  
- Live sensor data (temperature, humidity, pressure)
- Device control simulation (LED, fan, pump)
- Auto-refreshing every 10 seconds

### Machine Learning
- Train Neural Networks, Decision Trees, SVM
- Real-time training progress
- Make predictions with confidence scores

### Data Science
- Analyze sales and random datasets  
- Generate interactive charts
- Statistical insights and correlations

## 🔧 Key APIs

```bash
# Get embedded sensor simulation
GET /api/embedded/simulate

# Get IoT sensor data
GET /api/iot/sensors

# Train ML model
POST /api/ml/train
{"model": "neural_network", "epochs": 25}

# Analyze data
POST /api/data/analyze  
{"type": "sales", "size": 1000}
```

## 📂 Structure

```
flask_skills_portfolio/
├── app.py              # Flask backend
├── templates/
│   └── index.html     # Frontend
└── requirements.txt   # Dependencies
```

## 🛠️ Technologies

**Backend**: Flask, Python, NumPy  
**Frontend**: HTML5, CSS3, JavaScript (ES6+)  
**Skills**: Embedded C/C++, IoT, Machine Learning, Data Science

## 🎯 Why This Portfolio?

Instead of just listing skills, this portfolio **demonstrates** them through working code:
- Real API endpoints you can test
- Interactive ML training you can watch
- Live data simulations that update
- Generated code examples with actual values

Built with Flask to showcase full-stack Python development alongside domain expertise.

---

**Live Demo**: Interactive skills showcase at `localhost:5000` 🚀
