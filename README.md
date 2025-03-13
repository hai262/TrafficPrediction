# 🚦 Traffic Prediction (GRU)

## Project Overview
Traffic Prediction (GRU) is an advanced deep learning project designed to forecast traffic volumes using Gated Recurrent Units (GRU) on time-series data. The application leverages sophisticated feature engineering techniques—including cyclical encoding of time attributes—to capture temporal patterns in traffic data, enabling accurate predictions to aid urban traffic management and planning.

## Key Features
- **Data Preprocessing:** Automated cleaning and transformation of historical traffic data, including feature scaling and time-based encoding.
- **GRU-based Model:** A deep learning model employing GRU layers to effectively capture temporal dependencies in traffic data.
- **Interactive Dashboard:** A user-friendly Streamlit web interface that visualizes historical traffic trends, training performance, and future predictions.
- **Performance Evaluation:** Model evaluation using metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R² score.
- **Scalability:** Designed to handle large-scale traffic data for real-time forecasting and decision support.

## Technologies & Tools
- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Deep Learning:** TensorFlow, Keras (GRU layers)
- **Visualization:** Matplotlib, Plotly, Streamlit
- **Version Control:** Git, GitHub

## Installation
Follow these steps to set up and run the Traffic Prediction application locally:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/hai262/TrafficPrediction.git
   cd TrafficPrediction
