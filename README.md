# 🍃 Air Quality Analytics & Prediction System

An interactive machine learning project that analyzes air pollutant levels (PM2.5, PM10, NO2, SO2, CO, and O3) to evaluate and predict overall air quality categories using a Random Forest Classifier.

---

## 📌 Project Overview

This project provides two distinct interfaces:
1. **CLI & Visualizations (`Project.py`)**: Generates Exploratory Data Analysis (EDA) charts (bar charts, pie charts, correlation heatmaps, confusion matrix) and runs a terminal-based interactive predictor.
2. **Modern GUI (`UI code.py`)**: Built with **CustomTkinter** for an interactive dashboard with dynamic parameter sliders, real-time KPI metrics, and instant air quality predictions.

---

## 📊 Dataset Parameters

The dataset contains environmental pollutant concentrations:
* **PM2.5**: Fine Particulate Matter (µg/m³)
* **PM10**: Coarse Particulate Matter (µg/m³)
* **NO2**: Nitrogen Dioxide (ppb)
* **SO2**: Sulfur Dioxide (ppb)
* **CO**: Carbon Monoxide (ppm)
* **O3**: Ozone (ppb)
* **Quality Category Target Index**:
  * `1` — **Hazardous**
  * `2` — **Poor**
  * `3` — **Moderate**
  * `4` — **Good**
  * `5` — **Excellent**

---

## 🛠️ Features

* **Machine Learning Pipeline**: Uses `scikit-learn`'s Random Forest model with train-test splitting and multi-metric performance reporting.
* **Exploratory Visualizations**: Integrated Matplotlib and Seaborn visualization charts.
* **Graphical Desktop Application**: Built using `CustomTkinter` with live theme toggles (Dark/Light mode).

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed along with the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn customtkinter
