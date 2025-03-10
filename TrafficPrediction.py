import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

# Set page configuration at the very top
st.set_page_config(page_title="Advanced Traffic Prediction (GRU)", layout="wide")

# Inject custom CSS for background image and styling
st.markdown(
    """
    <style>
    /* Background Image */
    .stApp {
        background: url("https://www.aiplusinfo.com/wp-content/uploads/2022/03/AI-in-Traffic-Management-.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(240, 240, 255, 0.9));
        border-right: 5px solid #0052cc;
    }

    /* Header Styling */
    h1 {
        color: #ff5733;  /* Bright Orange */
        font-weight: bold;
        text-shadow: 2px 2px 5px rgba(0,0,0,0.2);
    }
    h2 {
        color: #008080;  /* Teal */
        font-weight: bold;
    }
    h3 {
        color: #0044cc;  /* Dark Blue */
        font-weight: bold;
    }

    /* Paragraph Styling */
    .stMarkdown p {
        color: #222222;
        font-size: 18px;
        font-weight: 500;
        line-height: 1.5;
    }

    /* Custom Text Colors */
    .highlight-text {
        color: #FFD700; /* Gold */
        font-size: 22px;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .feature-list {
        color: #00FF7F; /* Spring Green */
        font-size: 18px;
        font-weight: bold;
    }

    .upload-message {
        color: #FF4500; /* Orange-Red */
        font-size: 20px;
        font-weight: bold;
    }

    /* Metric Styling */
    .metric-box {
        padding: 10px;
        border-radius: 5px;
        color: white;
        text-align: center;
        margin-bottom: 10px;
    }
    .mse { background-color: #ff4444; }   /* Red */
    .rmse { background-color: #ffbb33; }  /* Orange */
    .mae { background-color: #33b5e5; }   /* Light Blue */
    .r2 { background-color: #00c851; }    /* Green */
    </style>
    """,
    unsafe_allow_html=True
)

# Title and Description
st.title("🚦 Advanced Traffic Prediction with GRU")
st.markdown("""
<div class="highlight-text">This application predicts traffic volume using Deep Learning (GRU) and advanced data analytics.</div>
<div class="highlight-text">It allows users to train models, visualize data, and forecast future trends.</div>

### 🚀 Features:
<div class="feature-list">✅ Advanced feature engineering (time-based attributes)</div>
<div class="feature-list">✅ Customizable deep learning model (GRU)</div>
<div class="feature-list">✅ Performance metrics & evaluation</div>
<div class="feature-list">✅ Future traffic forecasting</div>

<div class="upload-message">📂 Please upload a CSV file to begin.</div>
""", unsafe_allow_html=True)

# Sidebar Configuration for Data Upload
st.sidebar.header("📂 Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file)
        return df

    df = load_data(uploaded_file)
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    # Check for required columns
    if 'DateTime' in df.columns and 'Vehicles' in df.columns:
        # Data Preprocessing
        df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
        df.dropna(subset=['DateTime'], inplace=True)
        df.sort_values('DateTime', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Optional: Filter by Junction if available
        if 'Junction' in df.columns:
            junctions = df['Junction'].unique()
            if len(junctions) > 1:
                selected_junction = st.sidebar.selectbox("Select Junction", junctions)
                df = df[df['Junction'] == selected_junction].reset_index(drop=True)

        # Traffic Trend Plot
        st.subheader("📈 Traffic Volume Over Time")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df['DateTime'], df['Vehicles'], label='Vehicles', color='navy')
        ax.set_xlabel("DateTime")
        ax.set_ylabel("Vehicles")
        ax.set_title("Traffic Volume Trend")
        ax.legend()
        st.pyplot(fig)

        # Feature Engineering: Create time-based features
        def create_time_features(df):
            df['day_of_week'] = df['DateTime'].dt.dayofweek
            df['hour_of_day'] = df['DateTime'].dt.hour
            df['sin_hour'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
            df['cos_hour'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
            return df

        df = create_time_features(df)
        df_model = df[['Vehicles', 'day_of_week', 'hour_of_day', 'sin_hour', 'cos_hour']]

        # Sidebar: Model & Training Parameters
        st.sidebar.header("⚙️ Model Configuration")
        sequence_length = st.sidebar.slider("Sequence Length", 5, 60, 12, step=1)
        epochs = st.sidebar.slider("Epochs", 5, 200, 20, step=5)
        batch_size = st.sidebar.slider("Batch Size", 16, 128, 32, step=16)
        gru_units_1 = st.sidebar.slider("GRU Units (Layer 1)", 16, 128, 64, step=16)
        gru_units_2 = st.sidebar.slider("GRU Units (Layer 2)", 0, 128, 32, step=16)
        use_bidirectional = st.sidebar.checkbox("Use Bidirectional GRU", value=False)
        use_lr_scheduler = st.sidebar.checkbox("Use Learning Rate Scheduler", value=True)

        # Scaling the Data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df_model)  # shape: (num_samples, 5)
        # We'll use all columns: Vehicles (target) + additional features
        combined_scaled = scaled_data

        # Function to create sequences for time series modeling
        def create_sequences(features, target, seq_length):
            xs, ys = [], []
            for i in range(len(features) - seq_length):
                x = features[i: i + seq_length]
                y = target[i + seq_length]
                xs.append(x)
                ys.append(y)
            return np.array(xs), np.array(ys)

        X_seq, y_seq = create_sequences(combined_scaled, combined_scaled[:, -1], sequence_length)
        X_seq = X_seq[:, :, :4]  # Use only feature columns as input

        # Train-test split (80/20)
        split = int(0.8 * len(X_seq))
        X_train, X_test = X_seq[:split], X_seq[split:]
        y_train, y_test = y_seq[:split], y_seq[split:]

        # Build the Advanced GRU Model
        st.subheader("🤖 Building and Training the Model")
        model = Sequential()
        if use_bidirectional:
            model.add(Bidirectional(GRU(gru_units_1, activation='tanh', return_sequences=(gru_units_2 > 0)),
                                    input_shape=(sequence_length, 4)))
        else:
            model.add(GRU(gru_units_1, activation='tanh', return_sequences=(gru_units_2 > 0),
                          input_shape=(sequence_length, 4)))
        model.add(Dropout(0.2))
        if gru_units_2 > 0:
            model.add(GRU(gru_units_2, activation='tanh'))
            model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        # Callbacks for training
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        callbacks_list = [early_stop]
        if use_lr_scheduler:
            lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
            callbacks_list.append(lr_scheduler)

        with st.spinner("Training the model..."):
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                callbacks=callbacks_list,
                verbose=0
            )
        st.success("✅ Model training complete!")

        # Display Model Summary
        st.subheader("📌 Model Summary")
        summary_str = []
        model.summary(print_fn=lambda x: summary_str.append(x))
        st.text("\n".join(summary_str))

        # Predictions on the Test Set
        y_pred_scaled = model.predict(X_test)
        n_test_samples = len(y_pred_scaled)
        reconstruction_pred = np.zeros((n_test_samples, 5))
        reconstruction_pred[:, -1] = y_pred_scaled.flatten()
        pred_inverted = scaler.inverse_transform(reconstruction_pred)[:, -1]
        reconstruction_test = np.zeros((len(y_test), 5))
        reconstruction_test[:, -1] = y_test.flatten()
        y_test_inverted = scaler.inverse_transform(reconstruction_test)[:, -1]

        # Calculate Evaluation Metrics
        mse_val = mean_squared_error(y_test_inverted, pred_inverted)
        rmse_val = math.sqrt(mse_val)
        mae_val = mean_absolute_error(y_test_inverted, pred_inverted)
        r2_val = r2_score(y_test_inverted, pred_inverted)

        st.subheader("📊 Evaluation Metrics")
        st.markdown(f"""
        <div class="metric-box mse"><b>Mean Squared Error (MSE):</b> {mse_val:.4f}</div>
        <div class="metric-box rmse"><b>Root Mean Squared Error (RMSE):</b> {rmse_val:.4f}</div>
        <div class="metric-box mae"><b>Mean Absolute Error (MAE):</b> {mae_val:.4f}</div>
        <div class="metric-box r2"><b>R² Score:</b> {r2_val:.4f}</div>
        """, unsafe_allow_html=True)

        # Plot Predictions vs. Actual
        st.subheader("📈 Predictions vs. Actual (Test Set)")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(range(len(y_test_inverted)), y_test_inverted, label='Actual', color='blue')
        ax2.plot(range(len(pred_inverted)), pred_inverted, label='Predicted', color='red', linestyle="--")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Vehicles")
        ax2.set_title("Traffic Prediction Comparison")
        ax2.legend()
        st.pyplot(fig2)

        # Plot Training History
        st.subheader("📉 Training History")
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.plot(history.history['loss'], label='Training Loss')
        ax3.plot(history.history['val_loss'], label='Validation Loss')
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Loss (MSE)")
        ax3.set_title("Model Loss Curve")
        ax3.legend()
        st.pyplot(fig3)

        # Future Forecasting
        st.subheader("🔮 Future Forecasting")
        forecast_steps = st.sidebar.slider("Forecast Steps", 1, 72, 12, step=1)
        if st.button("Forecast Future"):
            last_seq = combined_scaled[-sequence_length:, :4]  # use only feature columns from the last available sequence
            future_preds = []
            current_seq = np.copy(last_seq)
            for _ in range(forecast_steps):
                current_seq_reshaped = current_seq.reshape(1, sequence_length, 4)
                pred_scaled = model.predict(current_seq_reshaped)
                recon_forecast = np.zeros((1, 5))
                recon_forecast[:, -1] = pred_scaled[0, 0]
                pred_inverted_single = scaler.inverse_transform(recon_forecast)[:, -1][0]
                future_preds.append(pred_inverted_single)
                # Shift the sequence: roll and replace last row (simplified forecasting)
                current_seq = np.roll(current_seq, -1, axis=0)
                current_seq[-1, :] = current_seq[-1, :]
            # Plot future forecast
            fig4, ax4 = plt.subplots(figsize=(10, 4))
            last_points = 30 if len(y_test_inverted) > 30 else len(y_test_inverted)
            x_actual = range(len(y_test_inverted) - last_points, len(y_test_inverted))
            ax4.plot(x_actual, y_test_inverted[-last_points:], label='Recent Actual', color='blue')
            x_forecast = range(len(y_test_inverted), len(y_test_inverted) + forecast_steps)
            ax4.plot(x_forecast, future_preds, label='Forecast', color='green', linestyle="--")
            ax4.set_xlabel("Time Step")
            ax4.set_ylabel("Vehicles")
            ax4.set_title("Future Traffic Forecast")
            ax4.legend()
            st.pyplot(fig4)
            st.write("""
            **Note:** This forecasting approach is simplified and does not dynamically update time-based features.
            For improved accuracy, consider generating future timestamps and recalculating cyclical features.
            """)

    else:
        st.error("❌ The dataset must contain 'DateTime' and 'Vehicles' columns.")
else:
    st.info("📂 Please upload a CSV file to begin.")
