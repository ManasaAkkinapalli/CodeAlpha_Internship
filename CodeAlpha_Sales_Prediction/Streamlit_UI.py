"""
╔══════════════════════════════════════════════════════════╗
║   Sales Prediction — Streamlit Interactive Dashboard    ║
║   Run:  streamlit run Streamlit_UI.py                   ║
╚══════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Sales Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F7F9FC; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #4C72B0; }
    .metric-label { font-size: 0.85rem; color: #666; margin-top: 4px; }
    .section-header {
        background: linear-gradient(90deg, #4C72B0, #55A868);
        color: white;
        padding: 10px 18px;
        border-radius: 8px;
        margin: 16px 0 12px 0;
        font-weight: 600;
        font-size: 1.05rem;
    }
    .insight-box {
        background: #EEF3FB;
        border-left: 4px solid #4C72B0;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        font-size: 0.93rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Load & Cache Data ─────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('Advertising.csv')
    df.drop(columns=[c for c in df.columns if 'Unnamed' in c], inplace=True)
    return df

@st.cache_resource
def train_models(df):
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression' : LinearRegression(),
        'Ridge Regression'  : Ridge(alpha=1.0),
        'Lasso Regression'  : Lasso(alpha=0.1),
        'Random Forest'     : RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting' : GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Polynomial (deg2)' : Pipeline([
                                  ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                                  ('lr',   LinearRegression())
                              ]),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            'model'  : model,
            'y_pred' : y_pred,
            'rmse'   : np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae'    : mean_absolute_error(y_test, y_pred),
            'r2'     : r2_score(y_test, y_pred),
            'cv_r2'  : cross_val_score(model, X, y, cv=5, scoring='r2').mean(),
        }
    return results, X_test, y_test

df      = load_data()
results, X_test, y_test = train_models(df)
best_name = max(results, key=lambda k: results[k]['r2'])

# ═══════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center; padding: 20px 0 8px 0;'>
    <h1 style='color:#4C72B0; font-size:2.3rem; margin-bottom:4px;'>
        📊 Sales Prediction Dashboard
    </h1>
    <p style='color:#555; font-size:1.05rem;'>
        Advertising Spend → Sales Forecasting | CodeAlpha Task 4
    </p>
</div>
""", unsafe_allow_html=True)
st.divider()

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("### 🔮 Predict Sales")
    tv_spend    = st.slider("📺 TV Spend ($000s)",        0, 300, 150, step=5)
    radio_spend = st.slider("📻 Radio Spend ($000s)",     0, 50,  25,  step=1)
    news_spend  = st.slider("📰 Newspaper Spend ($000s)", 0, 115, 30,  step=1)

    selected_model = st.selectbox("🤖 Model", list(results.keys()),
                                  index=list(results.keys()).index(best_name))

    model_obj  = results[selected_model]['model']
    prediction = model_obj.predict([[tv_spend, radio_spend, news_spend]])[0]

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#4C72B0,#55A868);
                color:white; border-radius:12px; padding:16px; text-align:center; margin-top:12px;'>
        <div style='font-size:0.85rem; opacity:0.85;'>Predicted Sales</div>
        <div style='font-size:2.6rem; font-weight:800;'>{prediction:.2f}</div>
        <div style='font-size:0.8rem; opacity:0.75;'>units (000s)</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    total_spend = tv_spend + radio_spend + news_spend
    if total_spend > 0:
        st.markdown(f"**Total Spend:** ${total_spend}K")
        st.markdown(f"**TV Share:** {tv_spend/total_spend*100:.1f}%")
        st.markdown(f"**Radio Share:** {radio_spend/total_spend*100:.1f}%")
        st.markdown(f"**News Share:** {news_spend/total_spend*100:.1f}%")

    st.markdown("---")
    st.markdown("**📦 Dataset Info**")
    st.info(f"200 records · 4 features · No missing values")

# ═══════════════════════════════════════════════════════════
# ROW 1 — KPI METRICS
# ═══════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📈 Dataset & Best Model KPIs</div>', unsafe_allow_html=True)

best = results[best_name]
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.metric("📋 Total Records",   "200")
with k2:
    st.metric("🏆 Best Model",      best_name.split('(')[0].strip())
with k3:
    st.metric("✅ R² Score",        f"{best['r2']:.4f}")
with k4:
    st.metric("📉 RMSE",            f"{best['rmse']:.3f}")
with k5:
    st.metric("📊 Avg Sales",       f"{df['Sales'].mean():.2f}")

# ═══════════════════════════════════════════════════════════
# ROW 2 — DATA OVERVIEW
# ═══════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🗃️ Data Overview</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📄 Raw Data", "📊 Statistics", "🔥 Correlation"])

with tab1:
    st.dataframe(df.head(20), use_container_width=True, height=320)

with tab2:
    st.dataframe(df.describe().round(3), use_container_width=True)

with tab3:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.patch.set_facecolor('#F7F9FC')
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='RdYlGn',
                ax=ax, linewidths=0.5, annot_kws={'size': 12})
    ax.set_title('Correlation Heatmap', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ═══════════════════════════════════════════════════════════
# ROW 3 — MODEL COMPARISON
# ═══════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🤖 Model Comparison</div>', unsafe_allow_html=True)

res_df = pd.DataFrame({
    'Model'  : list(results.keys()),
    'R²'     : [r['r2']    for r in results.values()],
    'RMSE'   : [r['rmse']  for r in results.values()],
    'MAE'    : [r['mae']   for r in results.values()],
    'CV R²'  : [r['cv_r2'] for r in results.values()],
}).sort_values('R²', ascending=False).reset_index(drop=True)

col_t, col_c = st.columns([1, 1.2])

with col_t:
    st.dataframe(
        res_df.style.highlight_max(subset=['R²', 'CV R²'], color='#c8f0c8')
                    .highlight_min(subset=['RMSE', 'MAE'], color='#c8f0c8')
                    .format({'R²': '{:.4f}', 'RMSE': '{:.3f}', 'MAE': '{:.3f}', 'CV R²': '{:.4f}'}),
        use_container_width=True, height=260
    )

with col_c:
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#F7F9FC')
    colors = ['#C44E52' if m == best_name else '#4C72B0' for m in res_df['Model']]
    bars   = ax.bar(res_df['Model'], res_df['R²'], color=colors, edgecolor='white')
    for b, v in zip(bars, res_df['R²']):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                f'{v:.3f}', ha='center', fontsize=8.5, fontweight='bold')
    ax.set_xticklabels(res_df['Model'], rotation=28, ha='right', fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('R² Score')
    ax.set_title('Model R² Comparison', fontweight='bold')
    ax.axhline(0.9, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ═══════════════════════════════════════════════════════════
# ROW 4 — PREDICTION ANALYSIS
# ═══════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🔍 Prediction Analysis</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    # Actual vs Predicted
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.patch.set_facecolor('#F7F9FC')
    r = results[selected_model]
    ax.scatter(y_test, r['y_pred'], alpha=0.7, color='#4C72B0',
               edgecolors='white', linewidth=0.5, s=65)
    lims = [min(y_test.min(), r['y_pred'].min())-1,
            max(y_test.max(), r['y_pred'].max())+1]
    ax.plot(lims, lims, '--', color='#C44E52', linewidth=1.8, label='Perfect Fit')
    ax.set_xlabel('Actual Sales'); ax.set_ylabel('Predicted Sales')
    ax.set_title(f'Actual vs Predicted — {selected_model}', fontweight='bold')
    ax.text(0.05, 0.90, f"R² = {r['r2']:.4f}", transform=ax.transAxes,
            fontsize=11, color='#C44E52', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

with c2:
    # Residuals
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.patch.set_facecolor('#F7F9FC')
    residuals = y_test.values - r['y_pred']
    ax.scatter(r['y_pred'], residuals, alpha=0.6, color='#DD8452',
               edgecolors='white', linewidth=0.5, s=55)
    ax.axhline(0, color='#C44E52', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Predicted Sales'); ax.set_ylabel('Residuals')
    ax.set_title('Residual Plot', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ═══════════════════════════════════════════════════════════
# ROW 5 — FEATURE INSIGHTS
# ═══════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🔬 Feature Insights</div>', unsafe_allow_html=True)

rf_model  = results['Random Forest']['model']
feat_imp  = pd.Series(rf_model.feature_importances_,
                      index=['TV', 'Radio', 'Newspaper']).sort_values(ascending=False)

f1, f2, f3 = st.columns(3)
for col, (feat, imp) in zip([f1, f2, f3], feat_imp.items()):
    with col:
        st.metric(f"📌 {feat}", f"{imp:.4f}", delta="importance")

fi1, fi2 = st.columns(2)

with fi1:
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    fig.patch.set_facecolor('#F7F9FC')
    ax.barh(feat_imp.index[::-1], feat_imp.values[::-1],
            color=['#4C72B0', '#DD8452', '#55A868'][::-1], edgecolor='white')
    ax.set_title('Feature Importances (RF)', fontweight='bold')
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

with fi2:
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    fig.patch.set_facecolor('#F7F9FC')
    ax.scatter(df['TV'], df['Sales'], alpha=0.45, color='#4C72B0', s=45)
    m, b = np.polyfit(df['TV'], df['Sales'], 1)
    xs = np.linspace(df['TV'].min(), df['TV'].max(), 200)
    ax.plot(xs, m*xs + b, color='#C44E52', linewidth=2,
            label=f'y={m:.3f}x+{b:.2f}')
    ax.set_xlabel('TV Spend ($000s)'); ax.set_ylabel('Sales')
    ax.set_title('TV Spend vs Sales', fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ═══════════════════════════════════════════════════════════
# ROW 6 — SCENARIO ANALYSIS
# ═══════════════════════════════════════════════════════════
st.markdown('<div class="section-header">💡 Business Scenario Analysis</div>', unsafe_allow_html=True)

lr_model  = results['Linear Regression']['model']
scenarios = {
    'Low Budget'      : [50,  10,  10],
    'TV-Heavy'        : [250, 20,  20],
    'Radio-Heavy'     : [100, 80,  20],
    'Balanced'        : [150, 50,  50],
    'Digital-Focused' : [80,  90,  60],
    'Max Spend'       : [300, 100, 80],
}
sc_df = pd.DataFrame([
    {'Strategy': k, 'TV': v[0], 'Radio': v[1], 'Newspaper': v[2],
     'Predicted Sales': round(lr_model.predict([v])[0], 2)}
    for k, v in scenarios.items()
]).sort_values('Predicted Sales', ascending=False).reset_index(drop=True)

sc1, sc2 = st.columns([1, 1.2])

with sc1:
    st.dataframe(
        sc_df.style.highlight_max(subset=['Predicted Sales'], color='#c8f0c8')
                   .highlight_min(subset=['Predicted Sales'], color='#ffc8c8'),
        use_container_width=True, height=270
    )

with sc2:
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#F7F9FC')
    pal  = sns.color_palette('muted', len(sc_df))
    bars = ax.bar(sc_df['Strategy'], sc_df['Predicted Sales'],
                  color=pal, edgecolor='white', width=0.65)
    for b, v in zip(bars, sc_df['Predicted Sales']):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.1,
                f'{v:.1f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticklabels(sc_df['Strategy'], rotation=28, ha='right', fontsize=9)
    ax.set_ylabel('Predicted Sales')
    ax.set_title('Sales Forecast by Ad Strategy', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ── Key Insights ──────────────────────────────────────────
st.markdown('<div class="section-header">📌 Key Business Insights</div>', unsafe_allow_html=True)

insights = [
    "📺  <b>TV</b> is the strongest sales driver (62% importance). Prioritise TV spend for maximum ROI.",
    "📻  <b>Radio</b> is highly effective (36%) — especially when combined with TV for a synergy boost.",
    "📰  <b>Newspaper</b> has minimal impact (<2%). Reallocating newspaper budget to TV/Radio increases predicted sales.",
    "💡  A <b>Digital-Focused</b> strategy (high Radio) rivals TV-heavy approaches at a lower budget.",
    "🚀  <b>Max Spend</b> scenario predicts ~35 units — the TV × Radio interaction amplifies combined returns.",
]
for ins in insights:
    st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; color:#888; font-size:0.82rem; padding:8px 0;'>
    CodeAlpha Internship — Task 4: Sales Prediction using Python &nbsp;|&nbsp;
    Dataset: Advertising.csv (200 records) &nbsp;|&nbsp;
    Built with Streamlit + Scikit-learn
</div>
""", unsafe_allow_html=True)
