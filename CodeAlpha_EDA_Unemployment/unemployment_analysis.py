"""
╔══════════════════════════════════════════════════════════════╗
║        UNEMPLOYMENT IN INDIA — EDA & VISUALIZATION          ║
║        CodeAlpha Data Science Internship | Task 2           ║
╚══════════════════════════════════════════════════════════════╝

Run: python unemployment_analysis.py
Output: Cleaned_Data.csv + charts saved to images/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
os.makedirs("images", exist_ok=True)

# ──────────────────────────────────────────────────────────
#  COLOUR PALETTE
# ──────────────────────────────────────────────────────────
BG      = "#0d0f18"
CARD    = "#141726"
CARD2   = "#1c2035"
TEAL    = "#2ee8b5"
ORANGE  = "#ff7043"
YELLOW  = "#ffc947"
PURPLE  = "#b39ddb"
BLUE    = "#64b5f6"
PINK    = "#f48fb1"
MUTED   = "#5c6880"
TEXT    = "#e8edf8"
WHITE   = "#ffffff"

pct_fmt  = mticker.FuncFormatter(lambda x, _: f"{x:.0f}%")
pct1_fmt = mticker.FuncFormatter(lambda x, _: f"{x:.1f}%")

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    CARD,
    "text.color":        TEXT,
    "axes.labelcolor":   TEXT,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "axes.edgecolor":    "#252840",
    "grid.color":        "#1e2236",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
    "font.size":         9,
})

# ══════════════════════════════════════════════════════════
# 1. LOAD & CLEAN DATA
# ══════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 1 — DATA LOADING & CLEANING")
print("═"*60)

df_raw = pd.read_csv("Unemployment in India.csv")
df_raw.columns = [c.strip() for c in df_raw.columns]

print(f"\n📋 Raw shape       : {df_raw.shape}")
print(f"🔍 Missing rows    : {df_raw.isnull().any(axis=1).sum()}")
print(f"   Missing by col :\n{df_raw.isnull().sum().to_string()}")

df = df_raw.dropna(subset=["Region"]).copy()
df.columns = ["Region","Date","Frequency",
              "Unemployment_Rate","Estimated_Employed",
              "Labour_Participation_Rate","Area"]
df["Date"]   = pd.to_datetime(df["Date"].str.strip(), dayfirst=True)
df["Month"]  = df["Date"].dt.month
df["Month_Name"] = df["Date"].dt.strftime("%b")
df["COVID"]  = df["Date"] >= "2020-03-01"

df.to_csv("Cleaned_Data.csv", index=False)
print(f"\n✅ Cleaned shape   : {df.shape}")
print(f"   Date range     : {df['Date'].min().strftime('%b %Y')} → {df['Date'].max().strftime('%b %Y')}")
print(f"   Regions        : {df['Region'].nunique()} states/UTs")
print(f"   Saved           → Cleaned_Data.csv")

# ══════════════════════════════════════════════════════════
# 2. KEY STATISTICS
# ══════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 2 — KEY STATISTICS")
print("═"*60)

pre_covid  = df.loc[~df["COVID"], "Unemployment_Rate"].mean()
covid_avg  = df.loc[ df["COVID"], "Unemployment_Rate"].mean()
rural_avg  = df.loc[df["Area"]=="Rural", "Unemployment_Rate"].mean()
urban_avg  = df.loc[df["Area"]=="Urban", "Unemployment_Rate"].mean()
peak_row   = df.loc[df["Unemployment_Rate"].idxmax()]
region_avg = df.groupby("Region")["Unemployment_Rate"].mean().sort_values(ascending=False)
monthly    = df.groupby("Date")["Unemployment_Rate"].mean().reset_index()

print(f"\n   Pre-COVID avg         : {pre_covid:.2f}%")
print(f"   COVID-period avg      : {covid_avg:.2f}%")
print(f"   Shock magnitude       : +{covid_avg-pre_covid:.2f} pp  ({((covid_avg/pre_covid)-1)*100:.0f}% rise)")
print(f"   Peak                  : {peak_row['Unemployment_Rate']:.1f}% — {peak_row['Region']} ({peak_row['Date'].strftime('%b %Y')})")
print(f"   Rural avg             : {rural_avg:.2f}%")
print(f"   Urban avg             : {urban_avg:.2f}%")
print(f"\n   Top 5 states:\n{region_avg.head().to_string()}")
print(f"\n   Bottom 5 states:\n{region_avg.tail().to_string()}")

# ══════════════════════════════════════════════════════════
# 3. CHART 1 — MASTER DASHBOARD
# ══════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 3 — GENERATING DASHBOARD")
print("═"*60)

covid_s = pd.Timestamp("2020-03-01")
covid_e = pd.Timestamp("2020-06-30")

fig = plt.figure(figsize=(22, 28), facecolor=BG)
fig.text(0.5, 0.987, "UNEMPLOYMENT IN INDIA  ·  2019 – 2020",
         ha="center", fontsize=25, fontweight="bold", color=WHITE)
fig.text(0.5, 0.978,
         "EDA  ·  Regional Analysis  ·  COVID-19 Impact  ·  Rural vs Urban  ·  Policy Insights  |  CodeAlpha Task 2",
         ha="center", fontsize=10, color=MUTED)

gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.52, wspace=0.30,
                       left=0.06, right=0.96, top=0.970, bottom=0.03)

# ── P1: National monthly trend ───────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(monthly["Date"], monthly["Unemployment_Rate"],
         color=TEAL, linewidth=2.8, zorder=4)
ax1.fill_between(monthly["Date"], monthly["Unemployment_Rate"],
                 alpha=0.18, color=TEAL)
ax1.axvspan(covid_s, covid_e, color=ORANGE, alpha=0.11, label="COVID-19 Lockdown")
ax1.axhline(pre_covid, color=YELLOW, linestyle="--", lw=1.4,
            alpha=0.8, label=f"Pre-COVID avg: {pre_covid:.1f}%")
peak_date = monthly.loc[monthly["Unemployment_Rate"].idxmax(), "Date"]
peak_v    = monthly["Unemployment_Rate"].max()
ax1.annotate(f"Lockdown Peak\n{peak_v:.1f}%",
             xy=(peak_date, peak_v),
             xytext=(pd.Timestamp("2020-05-20"), peak_v - 6),
             arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.6),
             color=ORANGE, fontsize=10, fontweight="bold")
for _, row in monthly.iterrows():
    ax1.scatter(row["Date"], row["Unemployment_Rate"], color=TEAL, s=30, zorder=5)
    ax1.text(row["Date"], row["Unemployment_Rate"]+0.7,
             f"{row['Unemployment_Rate']:.1f}", ha="center", fontsize=7.5, color=TEAL)
ax1.set_title("📈  National Average Unemployment Rate — Monthly Trend",
              color=TEXT, fontsize=13, pad=10, loc="left", fontweight="bold")
ax1.set_ylabel("Unemployment Rate (%)")
ax1.yaxis.set_major_formatter(pct_fmt)
ax1.legend(fontsize=9, framealpha=0)
ax1.grid(True, axis="y", alpha=0.5)

# ── P2: Top 10 states ────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
top10 = region_avg.head(10)
bcols = [ORANGE if v>20 else (YELLOW if v>14 else TEAL) for v in top10.values]
bars  = ax2.barh(top10.index[::-1], top10.values[::-1],
                 color=bcols[::-1], height=0.65, alpha=0.88)
ax2.axvline(pre_covid, color=YELLOW, linestyle="--", lw=1.2,
            alpha=0.7, label=f"Nat avg {pre_covid:.1f}%")
for bar, val in zip(bars, top10.values[::-1]):
    ax2.text(val+0.2, bar.get_y()+bar.get_height()/2,
             f"{val:.1f}%", va="center", fontsize=8, color=TEXT)
ax2.set_title("🔴  Top 10 States — Highest Unemployment",
              color=TEXT, fontsize=11, pad=8, loc="left", fontweight="bold")
ax2.set_xlabel("Avg Rate (%)")
ax2.xaxis.set_major_formatter(pct_fmt)
ax2.legend(fontsize=8, framealpha=0)
ax2.grid(True, axis="x", alpha=0.4)

# ── P3: Bottom 10 states ─────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
bot10 = region_avg.tail(10).sort_values()
bars3 = ax3.barh(bot10.index, bot10.values, color=BLUE, height=0.65, alpha=0.88)
ax3.axvline(pre_covid, color=YELLOW, linestyle="--", lw=1.2,
            alpha=0.7, label=f"Nat avg {pre_covid:.1f}%")
for bar, val in zip(bars3, bot10.values):
    ax3.text(val+0.15, bar.get_y()+bar.get_height()/2,
             f"{val:.1f}%", va="center", fontsize=8, color=TEXT)
ax3.set_title("🟢  Top 10 States — Lowest Unemployment",
              color=TEXT, fontsize=11, pad=8, loc="left", fontweight="bold")
ax3.set_xlabel("Avg Rate (%)")
ax3.xaxis.set_major_formatter(pct_fmt)
ax3.legend(fontsize=8, framealpha=0)
ax3.grid(True, axis="x", alpha=0.4)

# ── P4: COVID shock by state ─────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
pre_r   = df[~df["COVID"]].groupby("Region")["Unemployment_Rate"].mean()
cov_r   = df[ df["COVID"]].groupby("Region")["Unemployment_Rate"].mean()
impact  = (cov_r - pre_r).dropna().sort_values(ascending=False).head(12)
bcols4  = [ORANGE if v>15 else (YELLOW if v>8 else PINK) for v in impact.values]
ax4.barh(impact.index[::-1], impact.values[::-1],
         color=bcols4[::-1], height=0.65, alpha=0.88)
ax4.axvline(0, color=MUTED, lw=0.8)
for i,(r,v) in enumerate(zip(impact.index[::-1], impact.values[::-1])):
    ax4.text(v+0.3, i, f"+{v:.1f}pp", va="center", fontsize=7.5, color=TEXT)
ax4.set_title("🦠  COVID-19 Shock — Worst-Hit States (pp increase)",
              color=TEXT, fontsize=11, pad=8, loc="left", fontweight="bold")
ax4.set_xlabel("Change in Unemployment (percentage points)")
ax4.grid(True, axis="x", alpha=0.4)

# ── P5: Rural vs Urban ───────────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
rural_m = df[df["Area"]=="Rural"].groupby("Date")["Unemployment_Rate"].mean()
urban_m = df[df["Area"]=="Urban"].groupby("Date")["Unemployment_Rate"].mean()
ax5.plot(rural_m.index, rural_m.values, color=TEAL,   lw=2.2,
         marker="o", ms=4, label="Rural")
ax5.plot(urban_m.index, urban_m.values, color=PURPLE, lw=2.2,
         marker="s", ms=4, label="Urban")
ax5.fill_between(rural_m.index, rural_m.values, alpha=0.10, color=TEAL)
ax5.fill_between(urban_m.index, urban_m.values, alpha=0.10, color=PURPLE)
ax5.axvspan(covid_s, covid_e, color=ORANGE, alpha=0.09)
ax5.text(urban_m.index[-1], urban_m.values[-1]+1.5,
         f"Urban\n{urban_avg:.1f}%", color=PURPLE, fontsize=8, ha="right")
ax5.text(rural_m.index[-1], rural_m.values[-1]-4,
         f"Rural\n{rural_avg:.1f}%", color=TEAL, fontsize=8, ha="right")
ax5.set_title("🌾  Rural vs Urban Unemployment Over Time",
              color=TEXT, fontsize=11, pad=8, loc="left", fontweight="bold")
ax5.set_ylabel("Rate (%)")
ax5.yaxis.set_major_formatter(pct_fmt)
ax5.legend(fontsize=9, framealpha=0)
ax5.grid(True, axis="y", alpha=0.4)

# ── P6: Labour Participation Rate ────────────────────────
ax6 = fig.add_subplot(gs[3, 0])
lpr = df.groupby("Date")["Labour_Participation_Rate"].mean().reset_index()
ax6.plot(lpr["Date"], lpr["Labour_Participation_Rate"],
         color=PINK, lw=2.2, marker="o", ms=4)
ax6.fill_between(lpr["Date"], lpr["Labour_Participation_Rate"],
                 alpha=0.15, color=PINK)
ax6.axvspan(covid_s, covid_e, color=ORANGE, alpha=0.09, label="COVID Lockdown")
for _, row in lpr.iterrows():
    ax6.text(row["Date"], row["Labour_Participation_Rate"]+0.25,
             f"{row['Labour_Participation_Rate']:.1f}",
             ha="center", fontsize=7, color=PINK)
ax6.set_title("👥  Labour Force Participation Rate — Monthly",
              color=TEXT, fontsize=11, pad=8, loc="left", fontweight="bold")
ax6.set_ylabel("Participation Rate (%)")
ax6.yaxis.set_major_formatter(pct_fmt)
ax6.legend(fontsize=9, framealpha=0)
ax6.grid(True, axis="y", alpha=0.4)

# ── P7: Seasonal bar chart ────────────────────────────────
ax7 = fig.add_subplot(gs[3, 1])
month_order  = [5,6,7,8,9,10,11,12,1,2,3,4]
month_labels = ["May","Jun","Jul","Aug","Sep","Oct","Nov","Dec","Jan","Feb","Mar","Apr"]
seasonal = df.groupby("Month")["Unemployment_Rate"].mean()
vals = [seasonal.get(m, 0) for m in month_order]
bcols7 = [ORANGE if m in [3,4] else (YELLOW if m in [5,6] else BLUE)
          for m in month_order]
bars7 = ax7.bar(month_labels, vals, color=bcols7, width=0.72, alpha=0.88)
ax7.axhline(df["Unemployment_Rate"].mean(), color=YELLOW, linestyle="--",
            lw=1.2, label=f"Overall avg {df['Unemployment_Rate'].mean():.1f}%", alpha=0.8)
for bar, val in zip(bars7, vals):
    ax7.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.15,
             f"{val:.1f}", ha="center", fontsize=7.5, color=TEXT)
ax7.set_title("📅  Seasonal Pattern — Avg Unemployment by Month",
              color=TEXT, fontsize=11, pad=8, loc="left", fontweight="bold")
ax7.set_ylabel("Avg Rate (%)")
ax7.yaxis.set_major_formatter(pct1_fmt)
ax7.legend(handles=[
    mpatches.Patch(color=YELLOW, label=f"Overall avg {df['Unemployment_Rate'].mean():.1f}%"),
    mpatches.Patch(color=ORANGE, alpha=0.8, label="COVID months (Mar–Apr)"),
], fontsize=8, framealpha=0)
ax7.grid(True, axis="y", alpha=0.4)

# ── P8: Policy cards ─────────────────────────────────────
ax8 = fig.add_subplot(gs[4, :])
ax8.set_facecolor(BG)
ax8.axis("off")
ax8.set_title("💡  Policy Insights & Recommendations",
              color=TEXT, fontsize=13, pad=12, loc="left", fontweight="bold")

cards = [
    ("🦠 COVID-19 Shock",
     f"+{covid_avg-pre_covid:.1f} pp surge (87%)",
     f"Avg jumped {pre_covid:.1f}%→{covid_avg:.1f}%\nduring lockdown. Emergency\nstimulus was critical."),
    ("🔴 High-Risk States",
     "Tripura · Haryana · Jharkhand",
     "Chronic >18% unemployment.\nSpecial Economic Zones\n& skill centres needed."),
    ("🌾 Rural-Urban Gap",
     f"Urban {urban_avg:.1f}% > Rural {rural_avg:.1f}%",
     "Urban informal workers\nlack safety nets. Portable\nbenefits scheme needed."),
    ("👥 Labour Participation",
     "Fell during lockdown",
     "Discouraged worker effect.\nMGNREGA expansion &\nALMPs can re-engage them."),
    ("📅 Seasonal Signals",
     "Apr–May annual spike",
     "Pre-monsoon job losses\nneed agri-credit &\nrural public works."),
    ("📈 Recovery Insight",
     "Jun 2020: Rapid bounce",
     "Unlock 1.0 showed swift\nrecovery. Staggered\nreopening saves livelihoods."),
]
cw, ch = 0.148, 0.88
for i, (title, metric, body) in enumerate(cards):
    x = 0.007 + i*(cw+0.01)
    ax8.add_patch(mpatches.FancyBboxPatch(
        (x, 0.04), cw, ch, boxstyle="round,pad=0.015",
        linewidth=1.2, edgecolor=TEAL, facecolor=CARD2,
        transform=ax8.transAxes, clip_on=False))
    ax8.text(x+cw/2, 0.86, title, ha="center", va="top",
             fontsize=9, fontweight="bold", color=TEAL, transform=ax8.transAxes)
    ax8.text(x+cw/2, 0.66, metric, ha="center", va="top",
             fontsize=9, fontweight="bold", color=YELLOW, transform=ax8.transAxes)
    ax8.text(x+cw/2, 0.46, body, ha="center", va="top",
             fontsize=8, color=TEXT, transform=ax8.transAxes,
             multialignment="center", linespacing=1.5)

plt.savefig("images/dashboard.png", dpi=150, bbox_inches="tight", facecolor=BG)
print("   ✅ images/dashboard.png")

# ══════════════════════════════════════════════════════════
# 4. CHART 2 — REGIONAL HEATMAP
# ══════════════════════════════════════════════════════════
pivot = df.pivot_table(index="Region", columns="Date",
                       values="Unemployment_Rate", aggfunc="mean")
pivot.columns = pivot.columns.strftime("%b\n%Y")

fig2, ax = plt.subplots(figsize=(20, 12), facecolor=BG)
ax.set_facecolor(BG)
cmap = LinearSegmentedColormap.from_list("india",
    ["#0d2b3e", "#0b6e4f", "#2ee8b5", "#ffc947", "#ff7043", "#c0392b"])

import matplotlib
norm  = matplotlib.colors.Normalize(vmin=0, vmax=pivot.values.max())
im    = ax.imshow(pivot.values, aspect="auto", cmap=cmap, norm=norm)

ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, fontsize=7.5, color=MUTED)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=8.5, color=TEXT)

for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        v = pivot.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    fontsize=6.5, color=WHITE if v > 20 else "#0d0f18",
                    fontweight="bold" if v > 25 else "normal")

cb = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
cb.set_label("Unemployment Rate (%)", color=TEXT, fontsize=10)
cb.ax.yaxis.set_tick_params(color=MUTED)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT)

ax.set_title("REGIONAL UNEMPLOYMENT HEATMAP — All States × All Months",
             color=WHITE, fontsize=16, fontweight="bold", pad=14)
fig2.patch.set_facecolor(BG)
plt.tight_layout()
plt.savefig("images/heatmap.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("   ✅ images/heatmap.png")

print("\n" + "═"*60)
print("  ALL OUTPUTS GENERATED SUCCESSFULLY")
print("═"*60)
print("\n  Files created:")
print("  ├── Cleaned_Data.csv")
print("  ├── images/dashboard.png")
print("  └── images/heatmap.png")
print("\n  Run the interactive dashboard:")
print("  → streamlit run Streamlit_UI.py\n")
