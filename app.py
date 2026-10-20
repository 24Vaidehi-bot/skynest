"""
app.py  --  SkyNest: Airline Flight Delay Analysis and Prediction
=================================================================

# =========================================================
# CASSANDRA FLIGHT EXPLORER — SKYNEST
# =========================================================

st.markdown("---")
st.header("Cassandra Flight Explorer")
st.caption("Browse flight records stored in the SkyNest Cassandra database.")

try:
    from skynest_cassandra import get_sample_flights

    limit = st.selectbox(
        "Number of records to display",
        [10, 25, 50, 100],
        index=1,
        key="cassandra_flight_limit"
    )

    if st.button("Load flights from Cassandra"):
        flights = get_sample_flights(limit)

        if flights.empty:
            st.info("No flight records were returned.")
        else:
            st.dataframe(flights, use_container_width=True)
            st.metric("Records displayed", len(flights))

except Exception as exc:
    st.warning(
        "Could not connect to Cassandra. "
        "Check that the skynest-cassandra container is running."
    )
    st.caption(str(exc))

BDA Lab Prototype | Streamlit 1.65+
Dataset: SkyNest_India_Combined_Flight_Dataset.csv (100,000 rows)
"""

import os
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, date, timedelta

warnings.filterwarnings("ignore")

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="SkyNest - Flight Delay Analysis",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#050d1a 0%,#0a1628 50%,#0d1f3c 100%);color:#e8f0fe;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#060e1e 0%,#0c1a30 100%);border-right:1px solid #1a3a5c;}
[data-testid="stSidebar"] *{color:#b8d0f0 !important;}
.kpi-card{background:linear-gradient(135deg,#0e1f3a 0%,#132b4e 100%);border:1px solid #1e4a7a;
  border-radius:12px;padding:20px 22px;text-align:center;
  box-shadow:0 4px 20px rgba(0,212,255,0.08);transition:transform 0.2s,box-shadow 0.2s;margin-bottom:8px;}
.kpi-card:hover{transform:translateY(-2px);box-shadow:0 8px 28px rgba(0,212,255,0.15);}
.kpi-value{font-size:2.0rem;font-weight:700;color:#00d4ff;line-height:1.1;}
.kpi-label{font-size:0.78rem;color:#7aadcc;text-transform:uppercase;letter-spacing:0.08em;margin-top:6px;}
.kpi-sub{font-size:0.72rem;color:#4a7a9b;margin-top:4px;}
.section-header{font-size:1.25rem;font-weight:600;color:#00d4ff;
  border-bottom:1px solid #1a3a5c;padding-bottom:6px;margin:18px 0 14px 0;}
.wx-card{background:#0e1f3a;border:1px solid #1e4a7a;border-radius:10px;
  padding:16px;text-align:center;margin:4px 0;}
.wx-icon{font-size:1.8rem;}
.wx-val{font-size:1.4rem;font-weight:600;color:#00d4ff;}
.wx-lbl{font-size:0.75rem;color:#7aadcc;text-transform:uppercase;letter-spacing:0.06em;}
.pred-delayed{background:linear-gradient(135deg,#2a0e0e,#3d1515);border:1px solid #8b2020;
  border-radius:12px;padding:20px;text-align:center;}
.pred-ontime{background:linear-gradient(135deg,#0e2a1a,#153d28);border:1px solid #207a40;
  border-radius:12px;padding:20px;text-align:center;}
.pred-label{font-size:1.6rem;font-weight:700;}
.pred-prob{font-size:1.0rem;color:#8ab8c8;margin-top:6px;}
#MainMenu{visibility:hidden;}footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
DATASET_PATH = "SkyNest_India_Combined_Flight_Dataset.csv"
DELAY_THRESHOLD_MIN = 15
SAMPLE_SIZE_FOR_MODEL = 30000

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0a1628", plot_bgcolor="#0d1f3c",
    font=dict(color="#b8d0f0", family="Inter"),
    title_font=dict(color="#00d4ff", size=15),
    legend=dict(bgcolor="rgba(10,22,40,0.8)", bordercolor="#1a3a5c", borderwidth=1),
    margin=dict(l=40, r=20, t=50, b=40),
)
DAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
SEASON_ORDER = ["Winter","Summer","Monsoon","Post-monsoon"]

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading flight dataset...", ttl=3600)
def load_data():
    if not os.path.exists(DATASET_PATH):
        return None, f"Dataset not found at: {os.path.abspath(DATASET_PATH)}"
    try:
        df = pd.read_csv(DATASET_PATH, parse_dates=["flight_date"])
        return df, None
    except Exception as e:
        return None, str(e)

@st.cache_data(show_spinner=False)
def get_airport_reference(df):
    ref = (
        df[["origin_iata","origin_airport","origin_city","origin_state_ut",
            "origin_latitude","origin_longitude"]]
        .drop_duplicates("origin_iata")
        .rename(columns={"origin_iata":"iata","origin_airport":"airport_name",
                         "origin_city":"city","origin_state_ut":"state",
                         "origin_latitude":"latitude","origin_longitude":"longitude"})
        .sort_values("iata").reset_index(drop=True)
    )
    return ref

# ── ML model ──────────────────────────────────────────────────────────────────
_model_df_cache = {}

@st.cache_resource(show_spinner="Training prediction model...")
def train_model(df_hash):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score, f1_score
    from sklearn.preprocessing import LabelEncoder

    df = _model_df_cache[df_hash]
    sampled = len(df) > SAMPLE_SIZE_FOR_MODEL
    df_s = df.sample(n=SAMPLE_SIZE_FOR_MODEL, random_state=42) if sampled else df.copy()

    cat_feats = ["airline","origin_iata","destination_iata","day_of_week",
                 "season","weather_condition","airport_traffic_level"]
    num_feats = ["month","scheduled_duration_min","distance_km_approx",
                 "temperature_c","visibility_km","wind_speed_kmh",
                 "precipitation_mm","public_holiday_flag"]

    df_s = df_s.copy()
    try:
        df_s["departure_hour"] = pd.to_datetime(
            df_s["scheduled_departure"], format="%H:%M", errors="coerce").dt.hour
    except Exception:
        df_s["departure_hour"] = 0
    num_feats = num_feats + ["departure_hour"]
    all_feats = cat_feats + num_feats

    df_m = df_s[all_feats + ["delay_target"]].copy().dropna()
    if len(df_m) < 500:
        return None, None, None, None, None, {"error": "Not enough rows to train."}

    encoders = {}
    for col in cat_feats:
        le = LabelEncoder()
        df_m[col] = le.fit_transform(df_m[col].astype(str))
        encoders[col] = le

    X = df_m[all_feats]; y = df_m["delay_target"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=120, max_depth=12,
                                 class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = round(accuracy_score(y_te, y_pred) * 100, 1)
    f1  = round(f1_score(y_te, y_pred, pos_label=1) * 100, 1)
    report = classification_report(y_te, y_pred,
                                   target_names=["On-Time","Delayed"], output_dict=True)
    info = {
        "sampled": sampled, "sample_n": len(df_s), "total_n": len(df),
        "train_n": len(X_tr), "test_n": len(X_te),
        "feature_importances": dict(zip(all_feats, clf.feature_importances_)),
        "encoders": encoders, "categorical_features": cat_feats,
        "numeric_features": num_feats, "all_features": all_feats,
    }
    return clf, info, report, acc, f1, {}

def get_or_train_model(df):
    df_hash = f"{len(df)}_{list(df.columns)}"
    if df_hash not in _model_df_cache:
        _model_df_cache[df_hash] = df
    return train_model(df_hash)

# ── Shared helpers ────────────────────────────────────────────────────────────
def apply_layout(fig, **kw):
    fig.update_layout(**{**PLOTLY_LAYOUT, **kw})
    fig.update_xaxes(gridcolor="#1a3a5c", zeroline=False)
    fig.update_yaxes(gridcolor="#1a3a5c", zeroline=False)
    return fig

def kpi_card(value, label, sub=""):
    return (f'<div class="kpi-card">'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-label">{label}</div>'
            + (f'<div class="kpi-sub">{sub}</div>' if sub else "")
            + '</div>')

# ── PAGE: Dashboard ───────────────────────────────────────────────────────────
def page_dashboard(df):
    st.markdown('<div class="section-header">✈️ Dashboard Overview</div>', unsafe_allow_html=True)

    total      = len(df)
    delayed    = int((df["delay_target"] == 1).sum())
    on_time    = total - delayed
    delay_rate = round(delayed / total * 100, 1)
    avg_delay  = round(df["departure_delay_min"].mean(), 1)
    avg_del_d  = round(df.loc[df["delay_target"]==1,"departure_delay_min"].mean(), 1)

    cols = st.columns(5)
    for col, (v,l,s) in zip(cols,[
        (f"{total:,}",    "Total Flights",       "2024 - 2026"),
        (f"{delayed:,}",  "Delayed Flights",     f">= {DELAY_THRESHOLD_MIN} min threshold"),
        (f"{on_time:,}",  "On-Time Flights",     "< 15 min delay"),
        (f"{delay_rate}%","Delay Rate",          "All airlines combined"),
        (f"{avg_delay} min","Avg Departure Delay",f"Delayed avg: {avg_del_d} min"),
    ]):
        col.markdown(kpi_card(v,l,s), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Airline delay rate + Severity pie
    c1, c2 = st.columns([3,2])
    with c1:
        st.markdown('<div class="section-header">Airline-wise Delay Rate</div>', unsafe_allow_html=True)
        ag = (df.groupby("airline")
              .agg(total=("flight_id","count"), delayed=("delay_target","sum"))
              .assign(delay_rate=lambda x:(x["delayed"]/x["total"]*100).round(1))
              .sort_values("delay_rate", ascending=True).reset_index())
        fig = px.bar(ag, x="delay_rate", y="airline", orientation="h",
                     color="delay_rate",
                     color_continuous_scale=["#00d4ff","#0077b6","#ff6b6b"],
                     text="delay_rate",
                     labels={"delay_rate":"Delay Rate (%)","airline":""})
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", marker_line_width=0)
        fig.update_coloraxes(showscale=False)
        apply_layout(fig, title="Delay Rate by Airline (>=15 min)", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">Flight Delay Severity</div>', unsafe_allow_html=True)
        sev = df["delay_category"].value_counts().reset_index()
        sev.columns = ["category","count"]
        fig2 = px.pie(sev, values="count", names="category", hole=0.45,
                      color_discrete_sequence=["#00d4ff","#0077b6","#ff9f43","#ff6b6b","#a8edea"])
        fig2.update_traces(textposition="outside", textinfo="percent+label")
        apply_layout(fig2, title="Flight Category Distribution", height=380)
        st.plotly_chart(fig2, use_container_width=True)

    # Monthly trend
    st.markdown('<div class="section-header">Monthly Delay Trend</div>', unsafe_allow_html=True)
    if pd.api.types.is_datetime64_any_dtype(df["flight_date"]):
        mn = (df.assign(ym=df["flight_date"].dt.to_period("M").astype(str))
              .groupby("ym")
              .agg(total=("flight_id","count"), delayed=("delay_target","sum"),
                   avg_delay=("departure_delay_min","mean"))
              .assign(delay_rate=lambda x:(x["delayed"]/x["total"]*100).round(1))
              .sort_values("ym").reset_index())
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=mn["ym"], y=mn["delay_rate"],
            mode="lines+markers", name="Delay Rate (%)",
            line=dict(color="#00d4ff",width=2.5), marker=dict(size=6),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.08)"))
        fig3.add_trace(go.Scatter(x=mn["ym"], y=mn["avg_delay"],
            mode="lines+markers", name="Avg Delay (min)",
            line=dict(color="#ff9f43",width=2,dash="dot"), marker=dict(size=5),
            yaxis="y2"))
        fig3.update_layout(
            yaxis2=dict(
                overlaying="y", side="right",
                title=dict(text="Avg Delay (min)", font=dict(color="#ff9f43")),
                tickfont=dict(color="#ff9f43"),
                gridcolor="rgba(0,0,0,0)",
            )
        )
        apply_layout(fig3, title="Monthly Delay Rate & Average Delay (2024-2026)",
                     height=350, xaxis_title="Month", yaxis_title="Delay Rate (%)")
        st.plotly_chart(fig3, use_container_width=True)

    # Day-of-week + Season
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="section-header">Delay Rate by Day of Week</div>', unsafe_allow_html=True)
        dow = (df.groupby("day_of_week")
               .agg(total=("flight_id","count"), delayed=("delay_target","sum"))
               .assign(delay_rate=lambda x:(x["delayed"]/x["total"]*100).round(1))
               .reindex([d for d in DAY_ORDER if d in df["day_of_week"].unique()])
               .reset_index())
        fig4 = px.bar(dow, x="day_of_week", y="delay_rate",
                      color="delay_rate",
                      color_continuous_scale=["#0077b6","#00d4ff","#ff9f43"],
                      text="delay_rate",
                      labels={"day_of_week":"","delay_rate":"Delay Rate (%)"})
        fig4.update_traces(texttemplate="%{text:.1f}%", textposition="outside", marker_line_width=0)
        fig4.update_coloraxes(showscale=False)
        apply_layout(fig4, title="Delay Rate by Day of Week", height=320)
        st.plotly_chart(fig4, use_container_width=True)

    with c4:
        st.markdown('<div class="section-header">Delay Rate by Season</div>', unsafe_allow_html=True)
        seas = (df.groupby("season")
                .agg(total=("flight_id","count"), delayed=("delay_target","sum"))
                .assign(delay_rate=lambda x:(x["delayed"]/x["total"]*100).round(1))
                .reindex([s for s in SEASON_ORDER if s in df["season"].unique()])
                .reset_index())
        fig5 = px.bar(seas, x="season", y="delay_rate",
                      color="delay_rate",
                      color_continuous_scale=["#00d4ff","#0096c7","#ff6b6b"],
                      text="delay_rate",
                      labels={"season":"","delay_rate":"Delay Rate (%)"})
        fig5.update_traces(texttemplate="%{text:.1f}%", textposition="outside", marker_line_width=0)
        fig5.update_coloraxes(showscale=False)
        apply_layout(fig5, title="Delay Rate by Season", height=320)
        st.plotly_chart(fig5, use_container_width=True)

    # Weather impact
    st.markdown('<div class="section-header">Weather Condition Impact on Delays</div>', unsafe_allow_html=True)
    wx = (df.groupby("weather_condition")
          .agg(total=("flight_id","count"), delayed=("delay_target","sum"),
               avg_delay=("departure_delay_min","mean"))
          .assign(delay_rate=lambda x:(x["delayed"]/x["total"]*100).round(1))
          .sort_values("delay_rate", ascending=False).reset_index())
    fig6 = go.Figure()
    fig6.add_trace(go.Bar(x=wx["weather_condition"], y=wx["delay_rate"],
        name="Delay Rate (%)", marker_color="#00d4ff", yaxis="y", offsetgroup=1,
        text=[f"{v:.1f}%" for v in wx["delay_rate"]], textposition="outside"))
    fig6.add_trace(go.Bar(x=wx["weather_condition"], y=wx["avg_delay"].round(1),
        name="Avg Delay (min)", marker_color="#ff9f43", yaxis="y2", offsetgroup=2,
        text=[f"{v:.1f}" for v in wx["avg_delay"].round(1)], textposition="outside"))
    fig6.update_layout(
        barmode="group",
        yaxis=dict(title=dict(text="Delay Rate (%)")),
        yaxis2=dict(
            overlaying="y", side="right",
            title=dict(text="Avg Delay (min)", font=dict(color="#ff9f43")),
            tickfont=dict(color="#ff9f43"),
            gridcolor="rgba(0,0,0,0)",
        ),
    )
    apply_layout(fig6, title="Weather Condition vs Delay Rate & Average Delay", height=360)
    st.plotly_chart(fig6, use_container_width=True)

    # Departure hour
    st.markdown('<div class="section-header">Scheduled Departure Hour vs Delay Rate</div>', unsafe_allow_html=True)
    try:
        df2 = df.copy()
        df2["departure_hour"] = pd.to_datetime(
            df2["scheduled_departure"], format="%H:%M", errors="coerce").dt.hour
        hg = (df2.groupby("departure_hour")
              .agg(total=("flight_id","count"), delayed=("delay_target","sum"))
              .assign(delay_rate=lambda x:(x["delayed"]/x["total"]*100).round(1))
              .reset_index().dropna(subset=["departure_hour"]))
        hg["departure_hour"] = hg["departure_hour"].astype(int)
        fig7 = px.bar(hg, x="departure_hour", y="delay_rate", color="delay_rate",
                      color_continuous_scale=["#0077b6","#00d4ff","#ff6b6b"],
                      labels={"departure_hour":"Scheduled Departure Hour (IST)",
                              "delay_rate":"Delay Rate (%)"},
                      text="delay_rate")
        fig7.update_traces(texttemplate="%{text:.1f}%", textposition="outside", marker_line_width=0)
        fig7.update_coloraxes(showscale=False)
        apply_layout(fig7, title="Delay Rate by Scheduled Departure Hour", height=330)
        st.plotly_chart(fig7, use_container_width=True)
    except Exception:
        st.info("Departure hour analysis requires a valid scheduled_departure (HH:MM) column.")

# ── PAGE: Detailed Analysis ───────────────────────────────────────────────────
def page_detailed(df):
    st.markdown('<div class="section-header">🔎 Detailed Analysis</div>', unsafe_allow_html=True)

    with st.expander("🔧 Filters", expanded=True):
        fc1,fc2,fc3,fc4 = st.columns(4)
        airlines = sorted(df["airline"].unique().tolist())
        sel_al = fc1.multiselect("Airline", airlines, default=airlines, key="da_al")
        origins = sorted(df["origin_iata"].unique().tolist())
        sel_or = fc2.multiselect("Origin IATA", origins, default=origins[:10], key="da_or")
        dests = sorted(df["destination_iata"].unique().tolist())
        sel_de = fc3.multiselect("Destination IATA", dests, default=dests[:10], key="da_de")
        years = sorted(df["year"].unique().tolist())
        sel_yr = fc4.multiselect("Year", years, default=years, key="da_yr")

    mask = (
        df["airline"].isin(sel_al if sel_al else airlines) &
        df["origin_iata"].isin(sel_or if sel_or else origins) &
        df["destination_iata"].isin(sel_de if sel_de else dests) &
        df["year"].isin(sel_yr if sel_yr else years)
    )
    dff = df[mask].copy()
    st.caption(f"Showing **{len(dff):,}** flights matching current filters.")

    if dff.empty:
        st.warning("No records match the current filters. Try widening your selection.")
        return

    # Airline summary
    st.markdown('<div class="section-header">Airline Summary</div>', unsafe_allow_html=True)
    as_ = (dff.groupby("airline")
           .agg(Total=("flight_id","count"), Delayed=("delay_target","sum"),
                Avg_Delay=("departure_delay_min","mean"))
           .assign(Delay_Rate=lambda x:(x["Delayed"]/x["Total"]*100).round(1),
                   Avg_Delay=lambda x:x["Avg_Delay"].round(1))
           .sort_values("Delay_Rate", ascending=False).reset_index())
    st.dataframe(as_.rename(columns={
        "airline":"Airline","Total":"Total Flights","Delayed":"Delayed (>=15 min)",
        "Avg_Delay":"Avg Delay (min)","Delay_Rate":"Delay Rate (%)"}),
        use_container_width=True, hide_index=True)
    st.download_button("⬇️ Export Airline CSV",
                       as_.to_csv(index=False).encode("utf-8"),
                       "airline_summary.csv","text/csv", key="dl_al")

    # Top 15 origin airports
    st.markdown('<div class="section-header">Top 15 Busiest Origin Airports</div>', unsafe_allow_html=True)
    og = (dff.groupby("origin_iata")
          .agg(total=("flight_id","count"), delayed=("delay_target","sum"))
          .assign(delay_rate=lambda x:(x["delayed"]/x["total"]*100).round(1))
          .sort_values("total", ascending=False).head(15).reset_index())
    fig_og = px.bar(og, x="origin_iata", y=["total","delayed"], barmode="group",
                    color_discrete_map={"total":"#00d4ff","delayed":"#ff6b6b"},
                    labels={"origin_iata":"Origin IATA","value":"Flights","variable":""})
    apply_layout(fig_og, title="Top 15 Origin Airports - Total vs Delayed", height=380)
    st.plotly_chart(fig_og, use_container_width=True)

    # Top 15 destinations by delay rate
    st.markdown('<div class="section-header">Top 15 Destinations by Delay Rate</div>', unsafe_allow_html=True)
    dg = (dff.groupby(["destination_iata","destination_city"])
          .agg(total=("flight_id","count"), delayed=("delay_target","sum"),
               max_delay=("departure_delay_min","max"))
          .assign(delay_rate=lambda x:(x["delayed"]/x["total"]*100).round(1))
          .sort_values("delay_rate", ascending=False).head(15).reset_index())
    fig_dg = px.bar(dg, x="delay_rate", y="destination_iata", orientation="h",
                    color="delay_rate",
                    color_continuous_scale=["#0077b6","#00d4ff","#ff9f43"],
                    text="delay_rate",
                    hover_data={"destination_city":True,"total":True,"max_delay":True},
                    labels={"delay_rate":"Delay Rate (%)","destination_iata":""})
    fig_dg.update_traces(texttemplate="%{text:.1f}%", textposition="outside", marker_line_width=0)
    fig_dg.update_coloraxes(showscale=False)
    apply_layout(fig_dg, title="Top 15 Destinations by Delay Rate (>=15 min)", height=420)
    st.plotly_chart(fig_dg, use_container_width=True)

    # Route bubble chart
    st.markdown('<div class="section-header">Top 20 Routes by Volume</div>', unsafe_allow_html=True)
    rg = (dff.groupby(["origin_iata","destination_iata"])
          .agg(total=("flight_id","count"), delayed=("delay_target","sum"),
               avg_delay=("departure_delay_min","mean"))
          .assign(delay_rate=lambda x:(x["delayed"]/x["total"]*100).round(1))
          .sort_values("total", ascending=False).head(20).reset_index())
    rg["route"] = rg["origin_iata"] + " -> " + rg["destination_iata"]
    fig_rg = px.scatter(rg, x="total", y="delay_rate", text="route",
                        size="avg_delay", color="delay_rate",
                        color_continuous_scale=["#00d4ff","#ff9f43","#ff6b6b"],
                        labels={"total":"Total Flights","delay_rate":"Delay Rate (%)","avg_delay":"Avg Delay (min)"})
    fig_rg.update_traces(textposition="top center")
    apply_layout(fig_rg, title="Top 20 Routes - Volume vs Delay Rate (bubble size = avg delay)", height=450)
    st.plotly_chart(fig_rg, use_container_width=True)

    # Data quality
    st.markdown('<div class="section-header">📋 Data Quality Summary</div>', unsafe_allow_html=True)
    dq1, dq2 = st.columns(2)
    with dq1:
        dq = pd.DataFrame({
            "Metric":["Total Rows","Total Columns","Missing Values (any column)",
                      "Duplicate Rows","Unique Flight IDs","Data Source"],
            "Value":[f"{len(df):,}",f"{len(df.columns)}",
                     f"{df.isnull().sum().sum():,}",f"{df.duplicated().sum():,}",
                     f"{df['flight_id'].nunique():,}","SkyNest India Combined Dataset"]})
        st.dataframe(dq, use_container_width=True, hide_index=True)
    with dq2:
        mv = df.isnull().sum().reset_index()
        mv.columns = ["Column","Missing"]
        mv = mv[mv["Missing"] > 0]
        if mv.empty:
            st.success("✅ No missing values detected in any column.")
        else:
            st.dataframe(mv, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Export Route Summary CSV",
                       rg.to_csv(index=False).encode("utf-8"),
                       "route_summary.csv","text/csv", key="dl_rt")

# ── PAGE: Predict Delay ───────────────────────────────────────────────────────
def page_predict(df):
    st.markdown('<div class="section-header">🤖 Delay Prediction</div>', unsafe_allow_html=True)
    

    with st.spinner("Preparing model..."):
        clf, model_info, report, acc, f1, err = get_or_train_model(df)

    if err:
        st.error(f"Model training failed: {err.get('error','Unknown error')}")
        return

    with st.expander("📊 Model Evaluation (held-out test set)", expanded=False):
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Accuracy", f"{acc}%")
        m2.metric("F1 Score (Delayed)", f"{f1}%")
        m3.metric("Training Rows", f"{model_info['train_n']:,}")
        m4.metric("Test Rows", f"{model_info['test_n']:,}")
        if model_info["sampled"]:
            st.caption(f"Model trained on a random sample of {model_info['sample_n']:,} rows "
                       f"(from {model_info['total_n']:,} total) for performance.")
        st.dataframe(pd.DataFrame(report).T.round(2), use_container_width=True)
        fi = pd.DataFrame(list(model_info["feature_importances"].items()),
                          columns=["Feature","Importance"]).sort_values("Importance", ascending=True)
        fig_fi = px.bar(fi, x="Importance", y="Feature", orientation="h", color="Importance",
                        color_continuous_scale=["#0077b6","#00d4ff"],
                        labels={"Importance":"Feature Importance","Feature":""})
        fig_fi.update_coloraxes(showscale=False)
        apply_layout(fig_fi, title="Feature Importances (Random Forest)", height=400)
        st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown('<div class="section-header">Enter Pre-Departure Flight Details</div>', unsafe_allow_html=True)
    encoders  = model_info["encoders"]
    all_feats = model_info["all_features"]

    ca, cb, cc = st.columns(3)
    with ca:
        sel_airline = st.selectbox("Airline", sorted(df["airline"].unique()), key="p_al")
        sel_origin  = st.selectbox("Origin IATA", sorted(df["origin_iata"].unique()), key="p_or")
        sel_dest    = st.selectbox("Destination IATA", sorted(df["destination_iata"].unique()), key="p_de")
    with cb:
        sel_dow     = st.selectbox("Day of Week", DAY_ORDER, key="p_dow")
        sel_season  = st.selectbox("Season", SEASON_ORDER, key="p_sea")
        sel_month   = st.slider("Month", 1, 12, 6, key="p_mon")
    with cc:
        sel_wx      = st.selectbox("Weather Condition", sorted(df["weather_condition"].unique()), key="p_wx")
        sel_traffic = st.selectbox("Airport Traffic Level", sorted(df["airport_traffic_level"].unique()), key="p_tr")
        sel_holiday = st.checkbox("Public Holiday?", value=False, key="p_hol")

    cd, ce, cf = st.columns(3)
    with cd:
        dep_hour   = st.slider("Departure Hour (IST)", 0, 23, 8, key="p_hr")
        dep_temp   = st.slider("Temperature (deg C)", -5, 48, 28, key="p_tmp")
    with ce:
        dep_vis    = st.slider("Visibility (km)", 0.0, 25.0, 8.0, step=0.5, key="p_vis")
        dep_wind   = st.slider("Wind Speed (km/h)", 0.0, 80.0, 15.0, step=1.0, key="p_wnd")
    with cf:
        dep_precip = st.slider("Precipitation (mm)", 0.0, 50.0, 0.0, step=0.5, key="p_prc")
        dep_dur    = st.slider("Scheduled Duration (min)", 30, 360, 120, key="p_dur")
        dep_dist   = st.slider("Distance (km approx)", 100, 4000, 800, key="p_dst")

    if st.button("🔮 Predict Delay", type="primary", key="pred_btn"):
        row = {
            "month": sel_month, "scheduled_duration_min": dep_dur,
            "distance_km_approx": dep_dist, "temperature_c": dep_temp,
            "visibility_km": dep_vis, "wind_speed_kmh": dep_wind,
            "precipitation_mm": dep_precip, "public_holiday_flag": int(sel_holiday),
            "departure_hour": dep_hour,
        }
        cat_map = {"airline": sel_airline, "origin_iata": sel_origin,
                   "destination_iata": sel_dest, "day_of_week": sel_dow,
                   "season": sel_season, "weather_condition": sel_wx,
                   "airport_traffic_level": sel_traffic}
        for feat, val in cat_map.items():
            le = encoders[feat]
            row[feat] = le.transform([val])[0] if val in le.classes_ else 0

        X_pred     = pd.DataFrame([row])[all_feats]
        pred_class = clf.predict(X_pred)[0]
        pred_proba = clf.predict_proba(X_pred)[0]
        del_prob   = round(float(pred_proba[1]) * 100, 1)
        ok_prob    = round(float(pred_proba[0]) * 100, 1)

        st.markdown("---")
        r1, r2 = st.columns([2,3])
        with r1:
            if pred_class == 1:
                st.markdown(
                    f'<div class="pred-delayed">'
                    f'<div class="pred-label" style="color:#ff6b6b;">⚠️ LIKELY DELAYED</div>'
                    f'<div class="pred-prob">Delay probability: <strong>{del_prob}%</strong></div>'
                    f'<div class="pred-prob">On-time probability: {ok_prob}%</div></div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="pred-ontime">'
                    f'<div class="pred-label" style="color:#7ec850;">✅ LIKELY ON TIME</div>'
                    f'<div class="pred-prob">On-time probability: <strong>{ok_prob}%</strong></div>'
                    f'<div class="pred-prob">Delay probability: {del_prob}%</div></div>',
                    unsafe_allow_html=True)
        with r2:
            gfig = go.Figure(go.Indicator(
                mode="gauge+number", value=del_prob,
                title={"text":"Delay Probability (%)", "font":{"color":"#b8d0f0"}},
                number={"font":{"color":"#00d4ff"}, "suffix":"%"},
                gauge={"axis":{"range":[0,100],"tickcolor":"#7aadcc"},
                       "bar":{"color":"#00d4ff"}, "bgcolor":"#0e1f3a",
                       "bordercolor":"#1a3a5c",
                       "steps":[{"range":[0,30],"color":"#0d3b1f"},
                                 {"range":[30,60],"color":"#2d3a00"},
                                 {"range":[60,100],"color":"#3d1515"}],
                       "threshold":{"line":{"color":"#ff6b6b","width":3},"value":50}}))
            gfig.update_layout(paper_bgcolor="#0a1628", font=dict(color="#b8d0f0"),
                               height=260, margin=dict(l=20,r=20,t=60,b=20))
            st.plotly_chart(gfig, use_container_width=True)

# ── PAGE: Weather ─────────────────────────────────────────────────────────────
def page_weather(df):
    from weather_api import fetch_weather_forecast, get_airport_coordinates_from_df

    st.markdown('<div class="section-header">🌤️ Airport Weather Forecast</div>', unsafe_allow_html=True)
    

    airport_ref = get_airport_reference(df)
    iata_list   = airport_ref["iata"].tolist()
    city_list   = airport_ref["city"].tolist()
    disp_list   = [f"{i} - {c}" for i, c in zip(iata_list, city_list)]
    def_idx     = next((i for i,x in enumerate(disp_list) if x.startswith("DEL")), 0)

    wc1, wc2, wc3 = st.columns(3)
    sel_str  = wc1.selectbox("Select Origin Airport", disp_list, index=def_idx, key="wx_ap")
    sel_iata = sel_str.split(" - ")[0]
    sel_date = wc2.date_input("Forecast Date", value=date.today(),
                              min_value=date.today(),
                              max_value=date.today() + timedelta(days=16),
                              key="wx_dt")
    sel_hour = wc3.slider("Departure Hour (IST)", 0, 23, 8, key="wx_hr")

    coords = get_airport_coordinates_from_df(df, sel_iata)
    if coords is None:
        st.warning(f"Coordinates not available for {sel_iata}. Cannot fetch weather.")
        return

    lat, lon = coords
    arow = airport_ref[airport_ref["iata"] == sel_iata].iloc[0]
    st.caption(f"📍 **{arow['airport_name']}**, {arow['city']}, {arow['state']} "
               f"| Lat: {lat:.4f} N, Lon: {lon:.4f} E")

    if st.button("🌐 Fetch Weather Forecast", type="primary", key="wx_btn"):
        with st.spinner(f"Fetching forecast for {sel_iata} on {sel_date}..."):
            result = fetch_weather_forecast(lat, lon, str(sel_date), sel_hour)

        if not result["success"]:
            st.error(f"**Weather fetch failed:** {result['error']}")
            return

        st.success(f"Forecast retrieved for **{sel_iata}** on **{sel_date}** at **{sel_hour:02d}:00 IST**")

        def wx_card(icon, val, unit, label):
            disp = f"{val}{unit}" if val is not None else "N/A"
            return (f'<div class="wx-card"><div class="wx-icon">{icon}</div>'
                    f'<div class="wx-val">{disp}</div><div class="wx-lbl">{label}</div></div>')

        wx1,wx2,wx3,wx4,wx5 = st.columns(5)
        wx1.markdown(wx_card("🌡️", result["temperature_c"],    "°C",  "Temperature"),  unsafe_allow_html=True)
        wx2.markdown(wx_card("🌧️", result["precipitation_mm"], " mm",      "Precipitation"),unsafe_allow_html=True)
        wx3.markdown(wx_card("💨",       result["wind_speed_kmh"],   " km/h",    "Wind Speed"),   unsafe_allow_html=True)
        wx4.markdown(wx_card("👁️", result["visibility_km"],    " km",      "Visibility"),   unsafe_allow_html=True)
        wx5.markdown(wx_card("🌤️", result["weather_description"], "",      "Condition"),    unsafe_allow_html=True)
        st.caption(f"Source: {result['source']} | Timezone: Asia/Kolkata (IST)")

        st.markdown('<div class="section-header">Historical Weather Impact at This Airport</div>',
                    unsafe_allow_html=True)
        hist = (df[df["origin_iata"] == sel_iata]
                .groupby("weather_condition")
                .agg(total=("flight_id","count"), delayed=("delay_target","sum"),
                     avg_delay=("departure_delay_min","mean"))
                .assign(delay_rate=lambda x:(x["delayed"]/x["total"]*100).round(1))
                .reset_index())
        if hist.empty:
            st.info(f"No historical records found for {sel_iata} in the dataset.")
        else:
            fig_h = px.bar(hist, x="weather_condition", y="delay_rate", color="delay_rate",
                           color_continuous_scale=["#0077b6","#00d4ff","#ff9f43","#ff6b6b"],
                           text="delay_rate",
                           labels={"weather_condition":"Condition","delay_rate":"Delay Rate (%)"})
            fig_h.update_traces(texttemplate="%{text:.1f}%", textposition="outside", marker_line_width=0)
            fig_h.update_coloraxes(showscale=False)
            apply_layout(fig_h, title=f"Historical Delay Rate by Weather Condition - {sel_iata}", height=340)
            st.plotly_chart(fig_h, use_container_width=True)

# ── Sidebar navigation ────────────────────────────────────────────────────────
def sidebar_nav():
    with st.sidebar:
        st.markdown("""
<div style="text-align:center;padding:16px 0 8px 0;">
  <div style="font-size:2.5rem;">✈️</div>
  <div style="font-size:1.5rem;font-weight:700;color:#00d4ff;letter-spacing:0.05em;">SkyNest</div>
  <div style="font-size:0.75rem;color:#4a7a9b;margin-top:2px;">Flight Delay Analysis</div>
</div>
<hr style="border:none;border-top:1px solid #1a3a5c;margin:12px 0;">
""", unsafe_allow_html=True)
        page = st.radio("Navigation",
                        ["📊 Dashboard", "🔎 Detailed Analysis",
                         "🤖 Predict Delay", "🌤️ Weather"],
                        label_visibility="collapsed", key="nav")
        st.markdown("<hr style='border:none;border-top:1px solid #1a3a5c;margin:16px 0;'>",
                    unsafe_allow_html=True)
        st.markdown("""
<div style="font-size:0.72rem;color:#3a6a8a;padding:0 4px;">
  
</div>
""", unsafe_allow_html=True)
    return page

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    df, err = load_data()
    if err:
        st.error(f"**Failed to load dataset:** {err}")
        st.info("Run the app from the project directory:\n"
                "```\ncd sky_nest_flight_delay_analysis\npython -m streamlit run app.py\n```")
        return
    page = sidebar_nav()
    if   page == "📊 Dashboard":         page_dashboard(df)
    elif page == "🔎 Detailed Analysis": page_detailed(df)
    elif page == "🤖 Predict Delay":     page_predict(df)
    elif page == "🌤️ Weather":     page_weather(df)

if __name__ == "__main__":
    main()
