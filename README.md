# 🌿 Vehicle CO₂ Emissions Predictor

An AI-powered web application that predicts the **CO₂ emissions** of a vehicle based on its **fuel consumption** and **fuel type**. Built using **Streamlit** and **Machine Learning**, this tool helps users make informed decisions toward more environmentally friendly transportation.

---

## 🚀 Features

- 🔍 **Instant Predictions** of CO₂ emissions (g/km)
- 📊 **Visual Comparison** with average, hybrid, and electric vehicles
- 💨 Estimates **Annual Emissions** & required **Trees to Offset**
- ✨ Modern UI with dark-friendly, responsive design
- 📚 Built using **Scikit-learn**, **Pandas**, **Plotly**, and **Streamlit**

---

## 🖥 Demo

🔗 Live App: [View on Streamlit](https://your-deployment-url-here)

📸 UI Preview:
![App Screenshot](https://your-screenshot-link-here)

---

## 🧠 Model Details

- Trained on real-world vehicle emissions data
- Inputs:
  - `Fuel Consumption (L/100 km)`
  - `Fuel Type (Gasoline, Diesel, Ethanol, etc.)`
- Output:
  - Estimated CO₂ emissions in grams/km
- Model used: **Linear Regression** (or specify your model here)

---

## 📦 Installation

```bash
git clone https://github.com/Sahil-coder-eng/vehicle-co2-ai-predictor.git
cd vehicle-co2-ai-predictor
pip install -r requirements.txt
streamlit run app.py
