# 🚗 Road Accident Risk Prediction

A machine learning-based web application developed using Python and Flask to predict road accident severity based on accident-related factors.

## 📌 Project Overview

Road accidents can be influenced by multiple factors such as vehicle type, road conditions, weather conditions, traffic conditions, driver information, and other accident-related parameters.

The **Road Accident Risk Prediction** project provides a web-based interface where users can enter relevant accident information and receive a predicted accident severity.

The application integrates a trained machine learning model with a Flask backend and a web-based frontend.

## ✨ Key Features

- 🚘 Road accident severity prediction
- 📊 Accident data analysis and visualization
- 🌦️ Weather and road condition related inputs
- 🚦 Traffic and vehicle-related information
- 👤 Driver-related input parameters
- 🖥️ User-friendly web interface
- ⚙️ Flask backend
- 🤖 Machine learning model integration
- 📈 Accident data graphs and visualizations
- 📱 Responsive web interface

## 🛠️ Technologies Used

- **Python**
- **Flask**
- **Machine Learning**
- **Pandas**
- **HTML5**
- **CSS3**
- **JavaScript**
- **Materialize CSS**

## 📂 Project Structure

```text
Road-Accident-Risk-Prediction/
│
├── app.py
├── Prediction Model.py
├── test1.pkl
├── accidents_india.csv
├── requirements.txt
├── Procfile
├── runtime.txt
├── LICENSE
├── graph.png
│
├── static/
│   ├── css/
│   ├── js/
│   └── img/
│
└── templates/
    ├── base.html
    ├── hm.html
    ├── t.html
    ├── graph.html
    ├── pie.html
    ├── bs.html
    ├── map.html
    └── ur.html
## 🔄 How It Works

1. The user enters the required accident-related information.
2. The Flask application receives and processes the input.
3. The input data is passed to the trained machine learning model.
4. The model analyzes the provided features.
5. The system predicts the accident severity.
6. The prediction is displayed on the web interface.
7. Accident-related visualizations can also be viewed through the application.

## 📊 Dataset

The project uses historical road accident data stored in `accidents_india.csv`.

The dataset is processed using Python and Pandas for analysis and prediction.

## 🚀 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Shashank-coder739/Road-Accident-Risk-Prediction.git
