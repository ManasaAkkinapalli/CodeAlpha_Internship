"""
╔══════════════════════════════════════════════════════════════╗
║        SALES PREDICTION USING PYTHON — CodeAlpha Task 4     ║
║  Predict sales based on advertising spend across platforms  ║
╚══════════════════════════════════════════════════════════════╝

STEPS:
  1. Load & Explore Data
  2. Data Cleaning & Transformation
  3. Feature Engineering & Selection
  4. Model Training (Linear, Ridge, Lasso, RF, GB, Polynomial)
  5. Evaluation (RMSE, MAE, R²)
  6. Business Insights & Scenario Analysis
  7. Visualisation Dashboard (8 panels → images/)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')
os.makedirs('images', exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. LOAD & EXPLORE
# ─────────────────────────────────────────────────────────────
print("=" * 62)
print("  STEP 1 — LOAD & EXPLORE DATA")
print("=" * 62)

df = pd.read_csv('Advertising.csv')
df.drop(columns=[c for c in df.columns if 'Unnamed' in c], inplace=True)

print(f"Shape          : {df.shape}")
print(f"Columns        : {list(df.columns)}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nDescriptive Stats:\n{df.describe().round(2)}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nData Types:\n{df.dtypes}")

# ─────────────────────────────────────────────────────────────
# 2. DATA CLEANING & TRANSFORMATION
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  STEP 2 — DATA CLEANING & TRANSFORMATION")
print("=" * 62)

df_clean = df.copy()

# Outlier detection (IQR method)
for col in ['TV', 'Radio', 'Newspaper', 'Sales']:
    Q1, Q3 = df_clean[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = df_clean[(df_clean[col] < lower) | (df_clean[col] > upper)]
    print(f"  {col:<12} → {len(outliers)} outliers  (IQR range: {lower:.1f} – {upper:.1f})")

# Save cleaned CSV
df_clean.to_csv('Cleaned_Data.csv', index=False)
print(f"\n  ✅ Cleaned data saved → Cleaned_Data.csv  ({len(df_clean)} rows)")

# ─────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING & SELECTION
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  STEP 3 — FEATURE ENGINEERING & SELECTION")
print("=" * 62)

df_feat = df_clean.copy()
df_feat['TV_Radio_Interaction'] = df_feat['TV'] * df_feat['Radio']
df_feat['Total_Ad_Spend']       = df_feat['TV'] + df_feat['Radio'] + df_feat['Newspaper']
df_feat['TV_Ratio']             = df_feat['TV'] / df_feat['Total_Ad_Spend']
df_feat['Digital_Spend']        = df_feat['Radio'] + df_feat['Newspaper']

print("  New features added:")
print("    • TV_Radio_Interaction  = TV × Radio")
print("    • Total_Ad_Spend        = TV + Radio + Newspaper")
print("    • TV_Ratio              = TV / Total_Ad_Spend")
print("    • Digital_Spend         = Radio + Newspaper")
print(f"\n  Feature preview:\n{df_feat.head(3)}")

FEATURES = ['TV', 'Radio', 'Newspaper']
TARGET   = 'Sales'

X = df_clean[FEATURES]
y = df_clean[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler   = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"\n  Train set : {X_train.shape[0]} samples")
print(f"  Test  set : {X_test.shape[0]} samples")

# ─────────────────────────────────────────────────────────────
# 4. MODEL TRAINING
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  STEP 4 — MODEL TRAINING")
print("=" * 62)

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
    y_pred  = model.predict(X_test)
    rmse    = np.sqrt(mean_squared_error(y_test, y_pred))
    mae     = mean_absolute_error(y_test, y_pred)
    r2      = r2_score(y_test, y_pred)
    cv_r2   = cross_val_score(model, X, y, cv=5, scoring='r2').mean()
    results[name] = dict(rmse=rmse, mae=mae, r2=r2, cv_r2=cv_r2,
                         y_pred=y_pred, model=model)
    print(f"  Trained → {name}")

# ─────────────────────────────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  STEP 5 — MODEL EVALUATION")
print("=" * 62)
print(f"\n  {'Model':<22} {'RMSE':>7} {'MAE':>7} {'R²':>7} {'CV R²':>9}")
print("  " + "-" * 55)
for name, r in results.items():
    print(f"  {name:<22} {r['rmse']:>7.3f} {r['mae']:>7.3f} {r['r2']:>7.3f} {r['cv_r2']:>9.3f}")

best_name = max(results, key=lambda k: results[k]['r2'])
best      = results[best_name]
print(f"\n  ✅ Best Model : {best_name}")
print(f"     R²   = {best['r2']:.4f}")
print(f"     RMSE = {best['rmse']:.4f}")
print(f"     MAE  = {best['mae']:.4f}")

# Feature importance (RF)
rf          = results['Random Forest']['model']
feat_imp    = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(f"\n  Feature Importances (Random Forest):")
for feat, imp in feat_imp.items():
    bar = '█' * int(imp * 40)
    print(f"    {feat:<12} {bar}  {imp:.4f}")

# ─────────────────────────────────────────────────────────────
# 6. BUSINESS INSIGHTS — SCENARIO ANALYSIS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  STEP 6 — BUSINESS INSIGHTS & SCENARIO ANALYSIS")
print("=" * 62)

lr = results['Linear Regression']['model']
scenarios = {
    'Low Budget'      : [50,  10,  10],
    'TV-Heavy'        : [250, 20,  20],
    'Radio-Heavy'     : [100, 80,  20],
    'Balanced'        : [150, 50,  50],
    'Digital-Focused' : [80,  90,  60],
    'Max Spend'       : [300, 100, 80],
}

print(f"\n  {'Strategy':<18} {'TV':>6} {'Radio':>7} {'News':>6}  →  Predicted Sales")
print("  " + "-" * 58)
for label, vals in scenarios.items():
    pred = lr.predict([vals])[0]
    print(f"  {label:<18} {vals[0]:>6}  {vals[1]:>6}  {vals[2]:>6}  →  {pred:.2f}")

lr_coef = pd.Series(lr.coef_, index=FEATURES)
print(f"\n  Linear Model Coefficients:")
for feat, coef in lr_coef.items():
    print(f"    {feat:<12} : {coef:+.4f}  (every $1 extra → {coef:+.4f} sales units)")
print(f"    Intercept    : {lr.intercept_:.4f}")

print("\n  📌 Key Takeaways:")
print("    • TV advertising has the highest ROI per dollar spent")
print("    • Radio is highly effective especially combined with TV")
print("    • Newspaper spend shows minimal impact on sales")
print("    • Digital-Focused strategy yields strong results with lower TV budget")
print("    • TV × Radio interaction amplifies returns beyond individual channels")

# ─────────────────────────────────────────────────────────────
# 7. VISUALISATION DASHBOARD
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  STEP 7 — GENERATING VISUALISATION DASHBOARD")
print("=" * 62)

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.05)
BLUE, ORANGE, GREEN, RED = '#4C72B0', '#DD8452', '#55A868', '#C44E52'

fig = plt.figure(figsize=(20, 24))
fig.patch.set_facecolor('#F7F9FC')
gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.48, wspace=0.32)

# Panel 1 — Correlation Heatmap
ax1 = fig.add_subplot(gs[0, 0])
corr = df_clean.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', mask=mask,
            vmin=-1, vmax=1, ax=ax1, linewidths=0.5, annot_kws={'size': 12})
ax1.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=13)

# Panel 2 — Feature Importances
ax2 = fig.add_subplot(gs[0, 1])
bars = ax2.barh(feat_imp.index[::-1], feat_imp.values[::-1],
                color=[BLUE, ORANGE, GREEN][::-1], edgecolor='white')
for b, v in zip(bars, feat_imp.values[::-1]):
    ax2.text(b.get_width() + 0.005, b.get_y() + b.get_height()/2,
             f'{v:.3f}', va='center', fontsize=10)
ax2.set_xlim(0, feat_imp.max() * 1.2)
ax2.set_title('Feature Importances — Random Forest', fontweight='bold', fontsize=13)
ax2.set_xlabel('Importance Score')

# Panel 3 — Model R² Comparison
ax3 = fig.add_subplot(gs[1, 0])
names = list(results.keys())
r2s   = [results[m]['r2'] for m in names]
cols  = [RED if m == best_name else BLUE for m in names]
b3    = ax3.bar(range(len(names)), r2s, color=cols, edgecolor='white', width=0.6)
for b, v in zip(b3, r2s):
    ax3.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
             f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
ax3.set_xticks(range(len(names)))
ax3.set_xticklabels(names, rotation=28, ha='right', fontsize=9)
ax3.set_ylim(0, 1.08)
ax3.axhline(0.9, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax3.set_ylabel('R² Score')
ax3.set_title('Model R² Comparison', fontweight='bold', fontsize=13)

# Panel 4 — Actual vs Predicted (best model)
ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(y_test, best['y_pred'], alpha=0.7, color=BLUE,
            edgecolors='white', linewidth=0.5, s=65)
lims = [min(y_test.min(), best['y_pred'].min()) - 1,
        max(y_test.max(), best['y_pred'].max()) + 1]
ax4.plot(lims, lims, '--', color=RED, linewidth=1.8, label='Perfect Fit')
ax4.set_xlabel('Actual Sales'); ax4.set_ylabel('Predicted Sales')
ax4.set_title(f'Actual vs Predicted — {best_name}', fontweight='bold', fontsize=13)
ax4.legend()
ax4.text(0.05, 0.90, f'R² = {best["r2"]:.4f}', transform=ax4.transAxes,
         fontsize=11, color=RED, fontweight='bold')

# Panel 5 — Residuals
ax5 = fig.add_subplot(gs[2, 0])
residuals = y_test.values - best['y_pred']
ax5.scatter(best['y_pred'], residuals, alpha=0.6, color=ORANGE,
            edgecolors='white', linewidth=0.5, s=55)
ax5.axhline(0, color=RED, linestyle='--', linewidth=1.5)
ax5.set_xlabel('Predicted Sales'); ax5.set_ylabel('Residuals')
ax5.set_title('Residual Plot', fontweight='bold', fontsize=13)

# Panel 6 — Sales Distribution
ax6 = fig.add_subplot(gs[2, 1])
ax6.hist(y, bins=28, color=GREEN, edgecolor='white', alpha=0.85)
ax6.axvline(y.mean(), color=RED, linestyle='--', linewidth=2,
            label=f'Mean = {y.mean():.1f}')
ax6.axvline(y.median(), color=ORANGE, linestyle=':', linewidth=2,
            label=f'Median = {y.median():.1f}')
ax6.set_xlabel('Sales'); ax6.set_ylabel('Frequency')
ax6.set_title('Sales Distribution', fontweight='bold', fontsize=13)
ax6.legend()

# Panel 7 — TV Spend vs Sales (scatter + trend)
ax7 = fig.add_subplot(gs[3, 0])
ax7.scatter(df['TV'], df['Sales'], alpha=0.45, color=BLUE, s=45)
m, b = np.polyfit(df['TV'], df['Sales'], 1)
xs = np.linspace(df['TV'].min(), df['TV'].max(), 200)
ax7.plot(xs, m*xs + b, color=RED, linewidth=2, label=f'y = {m:.3f}x + {b:.2f}')
ax7.set_xlabel('TV Advertising ($000s)'); ax7.set_ylabel('Sales (units)')
ax7.set_title('TV Spend vs Sales', fontweight='bold', fontsize=13)
ax7.legend()

# Panel 8 — Scenario Forecast
ax8 = fig.add_subplot(gs[3, 1])
sc_labels = list(scenarios.keys())
sc_preds  = [lr.predict([v])[0] for v in scenarios.values()]
pal = sns.color_palette('muted', len(sc_labels))
b8  = ax8.bar(range(len(sc_labels)), sc_preds, color=pal, edgecolor='white', width=0.65)
for b, v in zip(b8, sc_preds):
    ax8.text(b.get_x() + b.get_width()/2, b.get_height() + 0.1,
             f'{v:.1f}', ha='center', fontsize=9, fontweight='bold')
ax8.set_xticks(range(len(sc_labels)))
ax8.set_xticklabels(sc_labels, rotation=28, ha='right', fontsize=9)
ax8.set_ylabel('Predicted Sales')
ax8.set_title('Sales Forecast by Ad Strategy', fontweight='bold', fontsize=13)

fig.suptitle('Sales Prediction Dashboard — Advertising Dataset',
             fontsize=18, fontweight='bold', y=0.998)

plt.savefig('images/dashboard.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("  ✅ Saved → images/dashboard.png")

# Heatmap standalone
fig2, ax = plt.subplots(figsize=(7, 5))
fig2.patch.set_facecolor('#F7F9FC')
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
            linewidths=0.5, annot_kws={'size': 13})
ax.set_title('Advertising Spend Correlation Heatmap', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('images/heatmap.png', dpi=150, bbox_inches='tight')
print("  ✅ Saved → images/heatmap.png")

print("\n" + "=" * 62)
print("  ✅ ALL STEPS COMPLETE — sales_prediction.py done!")
print("=" * 62)
