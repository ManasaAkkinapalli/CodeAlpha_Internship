# 📊 Unemployment Analysis with Python 🧠

## Overview
This project analyzes unemployment rate data across different regions of India to understand trends, regional variations, and the impact of COVID-19. Using Python, the data is cleaned, explored, and visualized to uncover patterns that can inform economic and social policy decisions.

---

## 📁 Project Structure

```
CodeAlpha_EDA_Unemployment/
│
├── Unemployment in India.csv       # Raw dataset (CMIE data)
├── Cleaned_Data.csv                # Cleaned & processed dataset
├── unemployment_analysis.py        # Core EDA + visualization script
├── Streamlit_UI.py                 # Interactive Streamlit dashboard
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── images/                         # Saved chart exports
    ├── national_trend.png
    ├── regional_heatmap.png
    ├── covid_impact.png
    └── rural_urban.png
```

---

## 🎯 Objectives

- ✅ Clean and preprocess unemployment rate data
- ✅ Perform Exploratory Data Analysis (EDA)
- ✅ Visualize national & regional unemployment trends
- ✅ Investigate the impact of **COVID-19** on unemployment rates
- ✅ Identify **seasonal patterns** in the data
- ✅ Compare **Rural vs Urban** unemployment
- ✅ Derive **policy-relevant insights**

---

## 📊 Dataset

| Column | Description |
|--------|-------------|
| `Region` | Indian state / union territory |
| `Date` | Month-end date of observation |
| `Frequency` | Monthly |
| `Unemployment_Rate` | % of labour force unemployed |
| `Estimated_Employed` | Estimated number of employed persons |
| `Labour_Participation_Rate` | % of working-age population in labour force |
| `Area` | Rural / Urban |

- **Source**: Centre for Monitoring Indian Economy (CMIE)
- **Period**: May 2019 – June 2020
- **Records**: 740 (after cleaning)
- **States/UTs**: 28

---

## 🔍 Key Findings

| Metric | Value |
|--------|-------|
| Pre-COVID avg unemployment | 9.51% |
| COVID-period avg | 17.77% |
| COVID shock | **+8.26 percentage points (+87%)** |
| National peak (Apr 2020) | ~24.9% |
| Highest state avg | Tripura — 28.4% |
| Lowest state avg | Meghalaya — 4.8% |
| Urban avg | 13.2% |
| Rural avg | 10.3% |

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the EDA script (generates charts)
```bash
python unemployment_analysis.py
```

### 3. Launch the interactive Streamlit dashboard
```bash
streamlit run Streamlit_UI.py
```

Then open your browser at `http://localhost:8501`

---

## 💡 Policy Recommendations

1. **COVID Emergency Response** — Pre-position income-support schemes (PM-KISAN, MGNREGA) before future shocks
2. **High-Unemployment States** — Tripura, Haryana, Jharkhand need Special Economic Zones and skill centres
3. **Urban Informal Workers** — 13.2% urban rate signals urgent need for portable social safety nets
4. **Labour Force Re-engagement** — Active Labour Market Policies to bring discouraged workers back
5. **Seasonal Buffers** — Pre-monsoon (Apr–May) spikes require agri-credit and rural public works
6. **Data Infrastructure** — Expand CMIE/NSO reporting to district level for faster policy response

---

## 🛠 Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Pandas](https://img.shields.io/badge/Pandas-2.x-green)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.9-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![Plotly](https://img.shields.io/badge/Plotly-5.22-purple)

---

*Made as part of CodeAlpha Data Science Internship — Task 2*
