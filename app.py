from flask import Flask, render_template, jsonify, request
import random
import time
import json
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Simulated sensor data storage
sensor_data = {
    'temperature': 24.0,
    'humidity': 65.0,
    'pressure': 1013.25,
    'last_updated': datetime.now()
}

# ML model simulation data
ml_models = {
    'neural_network': {'accuracy': 0.0, 'epochs': 0, 'status': 'untrained'},
    'decision_tree': {'accuracy': 0.0, 'depth': 0, 'status': 'untrained'},
    'svm': {'accuracy': 0.0, 'kernel': 'rbf', 'status': 'untrained'}
}

# Data analysis results storage
analysis_results = {
    'dataset_size': 0,
    'correlations': [],
    'insights': [],
    'last_analysis': None
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/embedded/simulate')
def simulate_embedded():
    """Simulate embedded system sensor readings"""
    # Generate realistic sensor data
    adc_value = random.randint(0, 1023)
    voltage = round(adc_value * 5.0 / 1023.0, 2)
    
    # Simulate different sensor types
    sensors = {
        'temperature': {
            'raw': adc_value,
            'voltage': voltage,
            'celsius': round((voltage - 0.5) * 100, 1),
            'status': 'active'
        },
        'light': {
            'raw': adc_value,
            'voltage': voltage,
            'lux': round(adc_value * 0.5, 0),
            'status': 'active'
        },
        'moisture': {
            'raw': adc_value,
            'voltage': voltage,
            'percentage': round((1023 - adc_value) / 1023 * 100, 1),
            'status': 'active'
        }
    }
    
    # Generate C++ code with actual values
    cpp_code = f"""// Real-time multi-sensor reading
#include <Arduino.h>

void setup() {{
  Serial.begin(9600);
  pinMode(A0, INPUT); // Temperature
  pinMode(A1, INPUT); // Light
  pinMode(A2, INPUT); // Moisture
}}

void loop() {{
  // Temperature sensor (LM35)
  int temp_raw = {sensors['temperature']['raw']};
  float temp_voltage = {sensors['temperature']['voltage']};
  float celsius = {sensors['temperature']['celsius']};
  
  // Light sensor (LDR)
  int light_raw = {sensors['light']['raw']};
  float light_lux = {sensors['light']['lux']};
  
  // Soil moisture
  int moisture_raw = {sensors['moisture']['raw']};
  float moisture_pct = {sensors['moisture']['percentage']};
  
  Serial.print("Temp: "); Serial.print(celsius); Serial.println("°C");
  Serial.print("Light: "); Serial.print(light_lux); Serial.println(" lux");
  Serial.print("Moisture: "); Serial.print(moisture_pct); Serial.println("%");
  
  delay(1000);
}}"""
    
    return jsonify({
        'sensors': sensors,
        'code': cpp_code,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/iot/sensors')
def get_iot_sensors():
    """Get current IoT sensor data"""
    global sensor_data
    
    # Simulate realistic sensor drift
    sensor_data['temperature'] += random.uniform(-0.5, 0.5)
    sensor_data['humidity'] += random.uniform(-2, 2)
    sensor_data['pressure'] += random.uniform(-1, 1)
    
    # Keep values in realistic ranges
    sensor_data['temperature'] = max(15, min(35, sensor_data['temperature']))
    sensor_data['humidity'] = max(30, min(90, sensor_data['humidity']))
    sensor_data['pressure'] = max(990, min(1030, sensor_data['pressure']))
    
    sensor_data['last_updated'] = datetime.now()
    
    # Add connection status simulation
    wifi_strength = random.choice(['weak', 'medium', 'strong'])
    
    return jsonify({
        'temperature': round(sensor_data['temperature'], 1),
        'humidity': round(sensor_data['humidity'], 1),
        'pressure': round(sensor_data['pressure'], 1),
        'wifi_strength': wifi_strength,
        'uptime': int(time.time() % 86400),  # Seconds since midnight
        'last_updated': sensor_data['last_updated'].isoformat()
    })

@app.route('/api/iot/control', methods=['POST'])
def control_iot_device():
    """Control IoT devices"""
    data = request.json
    device = data.get('device')
    action = data.get('action')
    
    responses = {
        'led': f'LED {action} successfully',
        'fan': f'Fan {action} - Speed: {random.randint(1, 10)}',
        'pump': f'Water pump {action} - Flow rate: {random.randint(50, 100)}L/h',
        'heater': f'Heater {action} - Target temp: {data.get("value", 25)}°C'
    }
    
    return jsonify({
        'status': 'success',
        'message': responses.get(device, 'Unknown device'),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/ml/train', methods=['POST'])
def train_ml_model():
    """Simulate ML model training"""
    data = request.json
    model_type = data.get('model', 'neural_network')
    epochs = data.get('epochs', 50)
    
    # Simulate training process
    training_progress = []
    for epoch in range(1, epochs + 1):
        # Simulate accuracy improvement with some noise
        base_accuracy = min(0.95, 0.5 + (epoch / epochs) * 0.4)
        noise = random.uniform(-0.05, 0.05)
        accuracy = max(0.1, min(0.99, base_accuracy + noise))
        
        training_progress.append({
            'epoch': epoch,
            'accuracy': round(accuracy, 4),
            'loss': round(1 - accuracy + random.uniform(0, 0.1), 4)
        })
    
    # Update model status
    final_accuracy = training_progress[-1]['accuracy']
    ml_models[model_type] = {
        'accuracy': final_accuracy,
        'epochs': epochs,
        'status': 'trained',
        'training_time': round(random.uniform(10, 60), 1)
    }
    
    return jsonify({
        'model': model_type,
        'final_accuracy': final_accuracy,
        'training_progress': training_progress[-10:],  # Last 10 epochs
        'status': 'completed',
        'model_info': ml_models[model_type]
    })

@app.route('/api/ml/predict', methods=['POST'])
def ml_predict():
    """Make predictions using trained models"""
    data = request.json
    model_type = data.get('model', 'neural_network')
    input_data = data.get('input', [])
    
    if ml_models[model_type]['status'] != 'trained':
        return jsonify({'error': 'Model not trained yet'}), 400
    
    # Simulate prediction
    if not input_data:
        input_data = [random.random() for _ in range(4)]
    
    # Generate realistic predictions based on model type
    if model_type == 'neural_network':
        prediction = random.choice([0, 1]) if len(input_data) > 2 else random.random()
        confidence = random.uniform(0.7, 0.95)
    else:
        prediction = random.uniform(0, 100)
        confidence = random.uniform(0.6, 0.9)
    
    return jsonify({
        'model': model_type,
        'input': input_data,
        'prediction': round(prediction, 4) if isinstance(prediction, float) else prediction,
        'confidence': round(confidence, 3),
        'model_accuracy': ml_models[model_type]['accuracy']
    })

@app.route('/api/data/analyze', methods=['POST'])
def analyze_data():
    """Perform data science analysis"""
    data = request.json
    dataset_type = data.get('type', 'random')
    size = data.get('size', 1000)
    
    # Generate synthetic dataset
    if dataset_type == 'sales':
        np.random.seed(42)
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        sales_data = {
            'months': months,
            'revenue': [random.randint(10000, 50000) for _ in months],
            'customers': [random.randint(100, 500) for _ in months],
            'products_sold': [random.randint(200, 1000) for _ in months]
        }
        
        # Calculate insights
        avg_revenue = sum(sales_data['revenue']) / len(sales_data['revenue'])
        growth_rate = ((sales_data['revenue'][-1] - sales_data['revenue'][0]) / sales_data['revenue'][0]) * 100
        
        insights = [
            f"Average monthly revenue: ${avg_revenue:,.0f}",
            f"Revenue growth rate: {growth_rate:.1f}%",
            f"Peak sales month: {months[sales_data['revenue'].index(max(sales_data['revenue']))]}",
            f"Customer acquisition trend: {'Positive' if sales_data['customers'][-1] > sales_data['customers'][0] else 'Negative'}"
        ]
        
    else:  # Random data
        # Generate correlation matrix
        features = ['Feature_A', 'Feature_B', 'Feature_C', 'Feature_D']
        correlations = []
        for i in range(len(features)):
            row = []
            for j in range(len(features)):
                if i == j:
                    row.append(1.0)
                else:
                    row.append(round(random.uniform(-0.8, 0.8), 3))
            correlations.append(row)
        
        sales_data = {
            'features': features,
            'correlations': correlations,
            'sample_size': size
        }
        
        insights = [
            f"Dataset contains {size} samples with {len(features)} features",
            f"Strongest correlation: {max([abs(c) for row in correlations for c in row if c != 1.0]):.3f}",
            f"Data distribution: Normal with μ={random.uniform(45, 55):.1f}",
            f"Outlier detection: {random.randint(5, 15)} anomalies found ({random.uniform(0.5, 1.5):.1f}%)"
        ]
    
    # Update global analysis results
    global analysis_results
    analysis_results = {
        'dataset_type': dataset_type,
        'dataset_size': size,
        'data': sales_data,
        'insights': insights,
        'last_analysis': datetime.now().isoformat()
    }
    
    return jsonify(analysis_results)

@app.route('/api/data/visualize')
def get_visualization_data():
    """Generate data for visualization"""
    chart_type = request.args.get('type', 'bar')
    
    if chart_type == 'line':
        # Generate time series data
        dates = [f"2024-0{i+1}-01" for i in range(6)]
        values = [random.randint(20, 100) for _ in dates]
        return jsonify({
            'type': 'line',
            'labels': dates,
            'data': values,
            'title': 'Performance Over Time'
        })
    
    elif chart_type == 'pie':
        # Generate category distribution
        categories = ['Python', 'JavaScript', 'C++', 'SQL', 'R']
        values = [random.randint(10, 40) for _ in categories]
        return jsonify({
            'type': 'pie',
            'labels': categories,
            'data': values,
            'title': 'Technology Usage Distribution'
        })
    
    else:  # bar chart
        # Generate comparative data
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [random.uniform(0.7, 0.95) for _ in metrics]
        return jsonify({
            'type': 'bar',
            'labels': metrics,
            'data': [round(v, 3) for v in values],
            'title': 'Model Performance Metrics'
        })

@app.route('/api/skills/summary')
def skills_summary():
    """Get comprehensive skills summary"""
    return jsonify({
        'embedded_systems': {
            'languages': ['C', 'C++', 'Assembly'],
            'platforms': ['Arduino', 'ESP32', 'STM32', 'Raspberry Pi'],
            'protocols': ['I2C', 'SPI', 'UART', 'CAN'],
            'experience_years': 5
        },
        'iot': {
            'connectivity': ['WiFi', 'Bluetooth', 'LoRaWAN', 'Zigbee'],
            'cloud_platforms': ['AWS IoT', 'Azure IoT', 'Google Cloud IoT'],
            'protocols': ['MQTT', 'CoAP', 'HTTP/HTTPS'],
            'experience_years': 4
        },
        'machine_learning': {
            'frameworks': ['TensorFlow', 'PyTorch', 'Scikit-learn', 'Keras'],
            'algorithms': ['Neural Networks', 'Random Forest', 'SVM', 'XGBoost'],
            'specializations': ['Computer Vision', 'NLP', 'Time Series'],
            'experience_years': 3
        },
        'data_science': {
            'tools': ['Pandas', 'NumPy', 'Matplotlib', 'Seaborn', 'Plotly'],
            'databases': ['PostgreSQL', 'MongoDB', 'InfluxDB'],
            'techniques': ['Statistical Analysis', 'Data Mining', 'Visualization'],
            'experience_years': 4
        },
        'total_projects': random.randint(25, 50),
        'github_commits': random.randint(500, 1500),
        'certifications': random.randint(5, 12)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)