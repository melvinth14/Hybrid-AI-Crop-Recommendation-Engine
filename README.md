# Hybrid AI Crop Recommendation Engine 🌱

A smart, end-to-end Machine Learning web application designed to help farmers in Tamil Nadu select the most profitable and suitable crops. 

Instead of relying solely on static soil data, this project uses a **Hybrid AI Approach**—combining a high-accuracy ML model with real-time weather forecasts to ensure crop viability.

## ✨ Key Features
*   **Dual-Mode Interface:** 
    *   **Beginner Mode:** Predicts crops based on standard district averages.
    *   **Expert Mode:** Allows entry of exact lab-tested soil metrics (N, P, K, pH) for precision recommendations.
*   (Real-Time) Weather Integration:** Connects to the **OpenWeatherMap API** to fetch 5-day forecasts. It automatically warns against planting if upcoming extreme weather (heavy rain/drought) threatens the crop.
*   **Precision Fertilizer Guide:** Calculates exact Nitrogen, Phosphorus, and Potassium deficits based on the user's soil and target crop, providing tailored agronomic advice.
*   **High Accuracy Model:** Powered by an **XGBoost Classifier** achieving **95.6% accuracy**.

## 🛠️ Tech Stack
*   **Language:** Python
*   **Machine Learning:** Scikit-Learn, XGBoost, Pandas, Numpy
*   **Frontend Data Dashboard:** Streamlit
*   **API Integration:** REST API (OpenWeatherMap)

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/melvinth14/Hybrid-AI-Crop-Recommendation-Engine.git
   cd Hybrid-AI-Crop-Recommendation-Engine
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

## 📊 Model Evaluation
*   **XGBoost Accuracy:** 95.6%
*   **Cross-Validation:** Implemented K-Fold cross-validation to prevent overfitting.
*   *Performance charts (Confusion Matrix, Feature Importance) available in the `/outputs` folder.*
