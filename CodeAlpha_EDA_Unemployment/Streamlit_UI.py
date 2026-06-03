"""
╔══════════════════════════════════════════════════════════╗
║   UNEMPLOYMENT IN INDIA — STREAMLIT INTERACTIVE DASHBOARD  ║
║   Run: streamlit run Streamlit_UI.py                     ║
╚══════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="India Unemployment EDA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
  /* ---- global ---- */
  .stApp { background: #0d0f18; color: #e8edf8; }
  section[data-testid="stSidebar"] { background: #141726; }
  
  /* ---- metric cards ---- */
  div[data-testid="metric-container"] {
    background: #1c2035;
    border: 1px solid #2ee8b5;
    border-radius: 12px;
    padding: 16px 20px;
  }
  div[data-testid="metric-container"] label { color: #5c6880 !important; font-size: 0.78rem; }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #2ee8b5 !important; font-size: 1.9rem; font-weight: 800;
  }
  div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {
    color: #ffc947 !important;
  }

  /* ---- headers ---- */
  h1 { color: #ffffff !important; font-weight: 900; letter-spacing: 1px; }
  h2, h3 { color: #2ee8b5 !important; }

  /* ---- sidebar ---- */
  .css-1d391kg { color: #e8edf8 !important; }
  label { color: #b0b8d0 !important; }

  /* ---- tabs ---- */
  button[data-baseweb="tab"] { color: #5c6880 !important; }
  button[data-baseweb="tab"][aria-selected="true"] {
    color: #2ee8b5 !important;
    border-bottom: 2px solid #2ee8b5 !important;
  }

  /* ---- info / warning boxes ---- */
  div[data-testid="stInfo"]    { background: #1c2035; border-left: 3px solid #2ee8b5; }
  div[data-testid="stWarning"] { background: #1c2035; border-left: 3px solid #ffc947; }
  div[data-testid="stSuccess"] { background: #1c2035; border-left: 3px solid #2ee8b5; }

  /* ---- divider ---- */
  hr { border-color: #252840; }
  
  /* ---- dataframe ---- */
  .stDataFrame { background: #141726 !important; }
</style>
""", unsafe_allow_html=True)

# ── Plotly theme defaults ───────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0d0f18",
    plot_bgcolor="#141726",
    font=dict(color="#e8edf8", family="Arial"),
    margin=dict(l=40, r=20, t=50, b=40),
)
AX   = dict(gridcolor="#1e2236", linecolor="#252840")
MUTED = "#5c6880"
# ══════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Cleaned_Data.csv")
        df["Date"] = pd.to_datetime(df["Date"])
    except FileNotFoundError:
        df = pd.read_csv("Unemployment in India.csv")
        df.columns = [c.strip() for c in df.columns]
        df = df.dropna(subset=["Region"])
        df.columns = ["Region","Date","Frequency",
                      "Unemployment_Rate","Estimated_Employed",
                      "Labour_Participation_Rate","Area"]
        df["Date"] = pd.to_datetime(df["Date"].str.strip(), dayfirst=True)
    df["Month"]      = df["Date"].dt.month
    df["Month_Name"] = df["Date"].dt.strftime("%b %Y")
    df["COVID"]      = df["Date"] >= "2020-03-01"
    return df

df = load_data()

pre_covid  = df.loc[~df["COVID"], "Unemployment_Rate"].mean()
covid_avg  = df.loc[ df["COVID"], "Unemployment_Rate"].mean()
rural_avg  = df.loc[df["Area"]=="Rural","Unemployment_Rate"].mean()
urban_avg  = df.loc[df["Area"]=="Urban","Unemployment_Rate"].mean()
region_avg = df.groupby("Region")["Unemployment_Rate"].mean().sort_values(ascending=False)
monthly    = df.groupby("Date")["Unemployment_Rate"].mean().reset_index()

# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Filters")
    st.markdown("---")

    all_regions = sorted(df["Region"].unique())
    selected_regions = st.multiselect(
        "Select States / UTs",
        all_regions,
        default=all_regions[:8],
    )

    area_filter = st.radio("Area Type", ["All", "Rural", "Urban"], horizontal=True)

    date_range = st.date_input(
        "Date Range",
        value=(df["Date"].min().date(), df["Date"].max().date()),
        min_value=df["Date"].min().date(),
        max_value=df["Date"].max().date(),
    )

    st.markdown("---")
    st.markdown("### 📌 About")
    st.markdown("""
    **Dataset**: CMIE — Centre for Monitoring Indian Economy  
    **Period**: May 2019 – Jun 2020  
    **Records**: 740 (cleaned)  
    **States/UTs**: 28  
    """)
    st.markdown("---")
    st.markdown("*CodeAlpha Data Science Internship*  \n**Task 2 — Unemployment EDA**")

# ── Apply filters ──────────────────────────────────────────
dff = df.copy()
if selected_regions:
    dff = dff[dff["Region"].isin(selected_regions)]
if area_filter != "All":
    dff = dff[dff["Area"] == area_filter]
if len(date_range) == 2:
    dff = dff[(dff["Date"] >= pd.Timestamp(date_range[0])) &
              (dff["Date"] <= pd.Timestamp(date_range[1]))]

# ══════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center; padding: 28px 0 10px 0;">
  <h1 style="font-size:2.4rem; margin-bottom:4px;">
    📊 Unemployment in India — EDA Dashboard
  </h1>
  <p style="color:#5c6880; font-size:1rem; margin:0;">
    Exploratory Data Analysis · COVID-19 Impact · Regional & Seasonal Patterns
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── KPI cards ─────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Overall Avg", f"{df['Unemployment_Rate'].mean():.1f}%")
k2.metric("Pre-COVID Avg", f"{pre_covid:.1f}%")
k3.metric("COVID-Period Avg", f"{covid_avg:.1f}%", delta=f"+{covid_avg-pre_covid:.1f}pp")
k4.metric("Peak (Apr 2020)", f"{monthly['Unemployment_Rate'].max():.1f}%")
k5.metric("Rural Avg", f"{rural_avg:.1f}%")
k6.metric("Urban Avg", f"{urban_avg:.1f}%", delta=f"+{urban_avg-rural_avg:.1f}pp vs Rural")

st.markdown("---")

# ══════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Trends", "🗺️ Regional", "🦠 COVID-19",
    "🌾 Rural vs Urban", "📅 Seasonal", "🗃️ Data"
])

# ── TAB 1: National trend ─────────────────────────────────
with tab1:
    st.subheader("National Monthly Unemployment Trend")

    monthly_f = dff.groupby("Date")["Unemployment_Rate"].mean().reset_index()

    fig1 = go.Figure()
    fig1.add_vrect(x0="2020-03-01", x1="2020-06-30",
                   fillcolor="#ff7043", opacity=0.10,
                   layer="below", line_width=0,
                   annotation_text="COVID Lockdown",
                   annotation_position="top left",
                   annotation_font_color="#ff7043")
    fig1.add_hline(y=pre_covid, line_dash="dash",
                   line_color="#ffc947", opacity=0.7,
                   annotation_text=f"Pre-COVID avg: {pre_covid:.1f}%",
                   annotation_font_color="#ffc947")
    fig1.add_trace(go.Scatter(
        x=monthly_f["Date"], y=monthly_f["Unemployment_Rate"],
        mode="lines+markers+text",
        line=dict(color="#2ee8b5", width=3),
        marker=dict(size=8, color="#2ee8b5"),
        text=[f"{v:.1f}%" for v in monthly_f["Unemployment_Rate"]],
        textposition="top center",
        textfont=dict(color="#2ee8b5", size=10),
        name="Unemployment Rate",
        fill="tozeroy", fillcolor="rgba(46,232,181,0.08)",
    ))
    fig1.update_layout(**PLOTLY_LAYOUT, title="Average Unemployment Rate Over Time",
                       height=420, xaxis=AX, yaxis={**AX},
                       yaxis_title="Unemployment Rate (%)",
                       showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Labour Participation Rate")
        lpr_f = dff.groupby("Date")["Labour_Participation_Rate"].mean().reset_index()
        fig_lpr = go.Figure(go.Scatter(
            x=lpr_f["Date"], y=lpr_f["Labour_Participation_Rate"],
            mode="lines+markers",
            line=dict(color="#f48fb1", width=2.5),
            marker=dict(size=6),
            fill="tozeroy", fillcolor="rgba(244,143,177,0.08)",
        ))
        fig_lpr.add_vrect(x0="2020-03-01", x1="2020-06-30",
                          fillcolor="#ff7043", opacity=0.09,
                          layer="below", line_width=0)
        fig_lpr.update_layout(**PLOTLY_LAYOUT, height=320, xaxis=AX, yaxis={**AX},
                              yaxis_title="Participation Rate (%)")
        st.plotly_chart(fig_lpr, use_container_width=True)

    with c2:
        st.subheader("Estimated Employed (millions)")
        emp_f = dff.groupby("Date")["Estimated_Employed"].sum().reset_index()
        emp_f["Employed_M"] = emp_f["Estimated_Employed"] / 1e6
        fig_emp = go.Figure(go.Bar(
            x=emp_f["Date"], y=emp_f["Employed_M"],
            marker_color="#64b5f6", opacity=0.8,
        ))
        fig_emp.add_vrect(x0="2020-03-01", x1="2020-06-30",
                          fillcolor="#ff7043", opacity=0.09,
                          layer="below", line_width=0)
        fig_emp.update_layout(**PLOTLY_LAYOUT, height=320, xaxis=AX, yaxis={**AX},
                              yaxis_title="Employed (millions)")
        st.plotly_chart(fig_emp, use_container_width=True)

# ── TAB 2: Regional ───────────────────────────────────────
with tab2:
    st.subheader("Regional Unemployment Analysis")

    c1, c2 = st.columns(2)
    with c1:
        top_n = st.slider("Show top N states", 5, 28, 10)
    with c2:
        sort_order = st.radio("Sort", ["Highest first", "Lowest first"], horizontal=True)

    reg_f = dff.groupby("Region")["Unemployment_Rate"].mean().sort_values(
        ascending=(sort_order=="Lowest first"))
    reg_plot = reg_f.head(top_n)

    colors_bar = ["#ff7043" if v > 20 else ("#ffc947" if v > 14 else "#2ee8b5")
                  for v in reg_plot.values]
    fig_reg = go.Figure(go.Bar(
        x=reg_plot.values,
        y=reg_plot.index,
        orientation="h",
        marker_color=colors_bar,
        text=[f"{v:.1f}%" for v in reg_plot.values],
        textposition="outside",
        textfont=dict(color="#e8edf8"),
    ))
    fig_reg.add_vline(x=pre_covid, line_dash="dash",
                      line_color="#ffc947",
                      annotation_text=f"Pre-COVID avg {pre_covid:.1f}%",
                      annotation_font_color="#ffc947")
    fig_reg.update_layout(**PLOTLY_LAYOUT, height=500, xaxis=AX, yaxis={**AX},
                          title=f"Average Unemployment — {'Top' if sort_order=='Highest first' else 'Bottom'} {top_n} States",
                          xaxis_title="Avg Unemployment Rate (%)")
    st.plotly_chart(fig_reg, use_container_width=True)

    st.subheader("Regional Heatmap")
    pivot = dff.pivot_table(index="Region", columns="Date",
                            values="Unemployment_Rate", aggfunc="mean")
    pivot.columns = pivot.columns.strftime("%b %Y")
    fig_heat = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[[0,"#0d2b3e"],[0.2,"#0b6e4f"],[0.5,"#2ee8b5"],
                    [0.7,"#ffc947"],[0.85,"#ff7043"],[1.0,"#c0392b"]],
        text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row]
              for row in pivot.values],
        texttemplate="%{text}",
        textfont=dict(size=8),
        colorbar=dict(title="Rate (%)", tickfont=dict(color="#e8edf8"),
                      titlefont=dict(color="#e8edf8")),
    ))
    fig_heat.update_layout(**PLOTLY_LAYOUT, height=650,
                           title="Unemployment Rate — All States × All Months",
                           xaxis=dict(tickangle=-45, gridcolor="#1e2236", linecolor="#252840"),
                           yaxis=dict(gridcolor="#1e2236", linecolor="#252840"))
    st.plotly_chart(fig_heat, use_container_width=True)

# ── TAB 3: COVID Impact ───────────────────────────────────
with tab3:
    st.subheader("COVID-19 Pandemic Impact")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Pre-COVID Avg", f"{pre_covid:.2f}%")
    col_b.metric("COVID-Period Avg", f"{covid_avg:.2f}%", f"+{covid_avg-pre_covid:.2f}pp")
    col_c.metric("Relative Shock", f"{((covid_avg/pre_covid)-1)*100:.0f}% spike")

    st.markdown("---")

    pre_r  = df[~df["COVID"]].groupby("Region")["Unemployment_Rate"].mean()
    cov_r  = df[ df["COVID"]].groupby("Region")["Unemployment_Rate"].mean()
    impact = (cov_r - pre_r).dropna().sort_values(ascending=False)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### Shock by State (pp increase)")
        top12 = impact.head(12)
        cols_impact = ["#ff7043" if v>15 else ("#ffc947" if v>8 else "#f48fb1")
                       for v in top12.values]
        fig_imp = go.Figure(go.Bar(
            x=top12.values, y=top12.index,
            orientation="h",
            marker_color=cols_impact,
            text=[f"+{v:.1f}pp" for v in top12.values],
            textposition="outside",
            textfont=dict(color="#e8edf8"),
        ))
        fig_imp.update_layout(**PLOTLY_LAYOUT, height=420, xaxis=AX, yaxis={**AX},
                              xaxis_title="Unemployment Change (pp)")
        st.plotly_chart(fig_imp, use_container_width=True)

    with c2:
        st.markdown("#### Pre vs During COVID Comparison")
        compare = pd.DataFrame({
            "Pre-COVID": pre_r, "During COVID": cov_r
        }).dropna()
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(
            name="Pre-COVID", x=compare.index,
            y=compare["Pre-COVID"], marker_color="#2ee8b5", opacity=0.85))
        fig_cmp.add_trace(go.Bar(
            name="During COVID", x=compare.index,
            y=compare["During COVID"], marker_color="#ff7043", opacity=0.85))
        fig_cmp.update_layout(**PLOTLY_LAYOUT, barmode="group",
                              height=420, xaxis={**AX, "tickangle": -45}, yaxis={**AX},
                              legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_cmp, use_container_width=True)

    st.markdown("#### Monthly National Rate — Zoomed Around COVID Period")
    fig_zoom = go.Figure()
    fig_zoom.add_vrect(x0="2020-03-01", x1="2020-06-30",
                       fillcolor="#ff7043", opacity=0.10, layer="below", line_width=0)
    fig_zoom.add_trace(go.Scatter(
        x=monthly["Date"], y=monthly["Unemployment_Rate"],
        mode="lines+markers",
        line=dict(color="#2ee8b5", width=2.5),
        fill="tozeroy", fillcolor="rgba(46,232,181,0.07)",
    ))
    fig_zoom.add_hline(y=pre_covid, line_dash="dash",
                       line_color="#ffc947", opacity=0.7)
    fig_zoom.update_layout(**PLOTLY_LAYOUT, height=300,
                           xaxis=dict(range=["2020-01-01","2020-06-30"],
                                      gridcolor="#1e2236", linecolor="#252840"),
                           yaxis=dict(gridcolor="#1e2236", linecolor="#252840"),
                           yaxis_title="Unemployment Rate (%)")
    st.plotly_chart(fig_zoom, use_container_width=True)

# ── TAB 4: Rural vs Urban ─────────────────────────────────
with tab4:
    st.subheader("Rural vs Urban Unemployment")

    rural_m = df[df["Area"]=="Rural"].groupby("Date")["Unemployment_Rate"].mean()
    urban_m = df[df["Area"]=="Urban"].groupby("Date")["Unemployment_Rate"].mean()

    fig_ru = go.Figure()
    fig_ru.add_vrect(x0="2020-03-01", x1="2020-06-30",
                     fillcolor="#ff7043", opacity=0.09, layer="below", line_width=0)
    fig_ru.add_trace(go.Scatter(
        x=rural_m.index, y=rural_m.values,
        name="Rural", mode="lines+markers",
        line=dict(color="#2ee8b5", width=2.5),
        marker=dict(size=7), fill="tozeroy",
        fillcolor="rgba(46,232,181,0.07)"))
    fig_ru.add_trace(go.Scatter(
        x=urban_m.index, y=urban_m.values,
        name="Urban", mode="lines+markers",
        line=dict(color="#b39ddb", width=2.5),
        marker=dict(symbol="square", size=7), fill="tozeroy",
        fillcolor="rgba(179,157,219,0.07)"))
    fig_ru.update_layout(**PLOTLY_LAYOUT, height=380, xaxis=AX, yaxis={**AX},
                         yaxis_title="Unemployment Rate (%)",
                         legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_ru, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Rural vs Urban by State")
        ru_state = df.groupby(["Region","Area"])["Unemployment_Rate"].mean().unstack()
        fig_rub = go.Figure()
        if "Rural" in ru_state.columns:
            fig_rub.add_trace(go.Bar(name="Rural", x=ru_state.index,
                                     y=ru_state["Rural"], marker_color="#2ee8b5", opacity=0.85))
        if "Urban" in ru_state.columns:
            fig_rub.add_trace(go.Bar(name="Urban", x=ru_state.index,
                                     y=ru_state["Urban"], marker_color="#b39ddb", opacity=0.85))
        fig_rub.update_layout(**PLOTLY_LAYOUT, barmode="group", height=380,
                              xaxis={**AX, "tickangle": -45}, yaxis={**AX},
                              legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_rub, use_container_width=True)

    with c2:
        st.markdown("#### Urban–Rural Gap by State")
        gap = (ru_state.get("Urban", pd.Series()) - ru_state.get("Rural", pd.Series())).dropna().sort_values(ascending=False)
        fig_gap = go.Figure(go.Bar(
            x=gap.index, y=gap.values,
            marker_color=["#ff7043" if v > 0 else "#2ee8b5" for v in gap.values],
            opacity=0.85,
        ))
        fig_gap.add_hline(y=0, line_color=MUTED, line_width=0.8)
        fig_gap.update_layout(**PLOTLY_LAYOUT, height=380,
                              xaxis={**AX, "tickangle": -45}, yaxis={**AX},
                              yaxis_title="Urban − Rural (pp)")
        st.plotly_chart(fig_gap, use_container_width=True)

# ── TAB 5: Seasonal ───────────────────────────────────────
with tab5:
    st.subheader("Seasonal & Monthly Patterns")

    month_order  = [5,6,7,8,9,10,11,12,1,2,3,4]
    month_labels = ["May","Jun","Jul","Aug","Sep","Oct","Nov","Dec","Jan","Feb","Mar","Apr"]
    seasonal = df.groupby("Month")["Unemployment_Rate"].mean()
    vals = [seasonal.get(m, 0) for m in month_order]
    cols_s = ["#ff7043" if m in [3,4] else ("#ffc947" if m in [5,6] else "#64b5f6")
              for m in month_order]

    fig_sea = go.Figure(go.Bar(
        x=month_labels, y=vals,
        marker_color=cols_s, opacity=0.88,
        text=[f"{v:.1f}%" for v in vals],
        textposition="outside",
        textfont=dict(color="#e8edf8"),
    ))
    fig_sea.add_hline(y=df["Unemployment_Rate"].mean(), line_dash="dash",
                      line_color="#ffc947",
                      annotation_text=f"Overall avg {df['Unemployment_Rate'].mean():.1f}%",
                      annotation_font_color="#ffc947")
    fig_sea.update_layout(**PLOTLY_LAYOUT, height=380, xaxis=AX, yaxis={**AX},
                          title="Average Unemployment Rate by Calendar Month",
                          yaxis_title="Avg Rate (%)")
    st.plotly_chart(fig_sea, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Violin Plot — Rate Distribution by Area")
        fig_vio = go.Figure()
        for area, color in [("Rural","#2ee8b5"),("Urban","#b39ddb")]:
            subset = df[df["Area"]==area]["Unemployment_Rate"]
            fig_vio.add_trace(go.Violin(
                y=subset, name=area, fillcolor=color,
                line_color=color, opacity=0.6, box_visible=True,
                meanline_visible=True))
        fig_vio.update_layout(**PLOTLY_LAYOUT, height=340, xaxis=AX, yaxis={**AX},
                              yaxis_title="Unemployment Rate (%)",
                              legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_vio, use_container_width=True)

    with c2:
        st.markdown("#### Monthly Box Plot — All States")
        fig_box = go.Figure()
        for i, (m, lbl) in enumerate(zip(month_order, month_labels)):
            subset = df[df["Month"]==m]["Unemployment_Rate"]
            col = "#ff7043" if m in [3,4] else ("#ffc947" if m in [5,6] else "#64b5f6")
            fig_box.add_trace(go.Box(
                y=subset, name=lbl, marker_color=col,
                line_color=col, opacity=0.85, showlegend=False))
        fig_box.update_layout(**PLOTLY_LAYOUT, height=340, xaxis=AX, yaxis={**AX},
                              yaxis_title="Unemployment Rate (%)")
        st.plotly_chart(fig_box, use_container_width=True)

# ── TAB 6: Raw data ───────────────────────────────────────
with tab6:
    st.subheader("🗃️ Cleaned Dataset Explorer")

    c1, c2, c3 = st.columns(3)
    with c1:
        region_filter = st.multiselect("Filter Region",
                                        sorted(df["Region"].unique()),
                                        default=[])
    with c2:
        area_f2 = st.selectbox("Filter Area", ["All","Rural","Urban"])
    with c3:
        covid_f = st.selectbox("COVID Period",
                                ["All","Pre-COVID (< Mar 2020)","During COVID (≥ Mar 2020)"])

    dft = df.copy()
    if region_filter: dft = dft[dft["Region"].isin(region_filter)]
    if area_f2 != "All": dft = dft[dft["Area"]==area_f2]
    if covid_f == "Pre-COVID (< Mar 2020)": dft = dft[~dft["COVID"]]
    if covid_f == "During COVID (≥ Mar 2020)": dft = dft[ dft["COVID"]]

    st.dataframe(
        dft[["Region","Date","Area","Unemployment_Rate",
             "Estimated_Employed","Labour_Participation_Rate"]].reset_index(drop=True),
        use_container_width=True, height=440
    )

    st.markdown(f"**{len(dft):,} rows** displayed")

    col_dl1, col_dl2 = st.columns([1, 5])
    with col_dl1:
        csv = dft.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv,
                           "filtered_unemployment.csv", "text/csv")

# ── Footer ─────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#5c6880; font-size:0.82rem;'>"
    "📊 Unemployment in India EDA · CodeAlpha Data Science Internship · Task 2 · "
    "Data: CMIE (May 2019 – Jun 2020)"
    "</p>", unsafe_allow_html=True
)
