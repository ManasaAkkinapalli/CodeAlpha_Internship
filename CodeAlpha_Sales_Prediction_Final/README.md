# 📊 Sales Prediction using Python
### CodeAlpha Internship — Task 4

> Predict future sales based on advertising spend across **TV, Radio, and Newspaper** platforms using regression and ensemble models.

---

## 📁 Project Structure

```
CodeAlpha_Sales_Prediction_Final/
├── 📊 sales_prediction.py     ← Run first (full ML pipeline + charts)
├── 🖥️  Streamlit_UI.py         ← Interactive dashboard
├── 📓 sales_prediction.ipynb  ← Jupyter notebook (11 sections)
├── 📄 Advertising.csv         ← Raw dataset (200 records)
├── 🗃️  Cleaned_Data.csv        ← Cleaned output
├── 📋 README.md               ← Full documentation (this file)
├── 📦 requirements.txt        ← Dependencies
└── images/
    ├── dashboard.png          ← 8-panel analysis chart
    └── heatmap.png            ← Correlation heatmap
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Main Analysis Script
```bash
python sales_prediction.py
```
Outputs: console report + `images/dashboard.png` + `images/heatmap.png` + `Cleaned_Data.csv`

### 3. Launch the Streamlit Dashboard
```bash
streamlit run Streamlit_UI.py
```
Opens an interactive browser dashboard at `http://localhost:8501`

### 4. Open the Jupyter Notebook
```bash
jupyter notebook sales_prediction.ipynb
```
Run all 11 cells for a step-by-step walkthrough.

---

## 📊 Dataset

| Feature | Description | Range |
|---|---|---|
| **TV** | TV advertising spend ($000s) | 0.7 – 296.4 |
| **Radio** | Radio advertising spend ($000s) | 0.0 – 49.6 |
| **Newspaper** | Newspaper advertising spend ($000s) | 0.3 – 114.0 |
| **Sales** | Product sales (units, 000s) — **target** | 1.6 – 27.0 |

- **200 records**, **0 missing values**
- Source: Classic ISLR Advertising Dataset

---

## 🤖 Models Trained

| Model | R² | RMSE | MAE |
|---|---|---|---|
| Linear Regression | 0.8990 | 1.782 | 1.461 |
| Ridge Regression | 0.8990 | 1.782 | 1.461 |
| Lasso Regression | 0.8996 | 1.781 | 1.460 |
| Random Forest | 0.9810 | 0.769 | 0.620 |
| Gradient Boosting | 0.9830 | 0.730 | 0.619 |
| **Polynomial (deg2)** | **0.9869** | **0.643** | **0.526** |

**Best Model: Polynomial Regression (degree 2)** with R² = 0.9869

---

## 🔬 Feature Importances (Random Forest)

| Feature | Importance |
|---|---|
| TV | 62.5% |
| Radio | 36.2% |
| Newspaper | 1.3% |

---

## 💡 Business Insights

| Strategy | TV | Radio | News | Predicted Sales |
|---|---|---|---|---|
| Low Budget | 50 | 10 | 10 | 7.14 |
| TV-Heavy | 250 | 20 | 20 | 18.00 |
| Radio-Heavy | 100 | 80 | 20 | 22.64 |
| Balanced | 150 | 50 | 50 | 19.29 |
| Digital-Focused | 80 | 90 | 60 | 23.75 |
| **Max Spend** | **300** | **100** | **80** | **35.54** |

### Key Takeaways
- 📺 **TV** is the strongest driver — prioritise for max ROI
- 📻 **Radio + TV together** produce synergy beyond individual spend
- 📰 **Newspaper** has minimal impact — budget better reallocated
- 💡 **Digital-Focused** strategy achieves strong results at lower cost
- 🚀 **Non-linear interactions** between channels amplify combined returns

---

## 🖥️ Streamlit Dashboard Features

- 🔮 **Live Prediction** — adjust sliders for instant sales forecast
- 📊 **Model Comparison** — all 6 models ranked by R² and RMSE
- 🔍 **Prediction Analysis** — Actual vs Predicted & Residual plots
- 🔬 **Feature Insights** — importances + scatter trend lines
- 💡 **Scenario Analysis** — 6 business strategy forecasts

---

## 📈 Visualisations Generated

1. Feature Correlation Matrix
2. Feature Importances (Random Forest)
3. Model R² Comparison
4. Actual vs Predicted — Best Model
5. Residual Plot
6. Sales Distribution
7. TV Spend vs Sales (with trend line)
8. Sales Forecast by Ad Strategy

---

## 📦 Dependencies

```
pandas, numpy, matplotlib, seaborn
scikit-learn, streamlit, jupyter
```

---

*CodeAlpha Data Science Internship — Task 4: Sales Prediction using Python*
