import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
import os

# Page configuration
st.set_page_config(
    page_title="Anomaly Report | Demand Intelligence",
    page_icon="🚨",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1f4068;
        margin-bottom: 0.2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #162447;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e43f5a;
        padding-bottom: 0.3rem;
    }
    .kpi-container {
        background-color: #f8f9fa;
        padding: 1.25rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        text-align: center;
        border-left: 5px solid #e43f5a;
    }
    .kpi-label {
        font-size: 0.9rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🚨 Sales Anomaly Detection Report</div>', unsafe_allow_html=True)
st.markdown("Monitor and detect outlying sales behavior using an unsupervised Isolation Forest machine learning model on weekly aggregated sales.", unsafe_allow_html=True)
st.markdown("---")

# Caching data processing & model fitting
@st.cache_data
def detect_anomalies():
    if not os.path.exists("cleaned_sales.csv"):
        st.error("Error: 'cleaned_sales.csv' not found. Please ensure it is in the project directory.")
        return pd.DataFrame(), pd.DataFrame(), 0.0

    df = pd.read_csv("cleaned_sales.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    
    # Aggregate daily sales into weekly sales (resample with W)
    weekly_sales = df.groupby("Order Date")["Sales"].sum().resample("W").sum()
    weekly_df = weekly_sales.to_frame()
    weekly_df.columns = ["Sales"]
    
    # Train Isolation Forest model
    iso_model = IsolationForest(
        contamination=0.05,
        random_state=42
    )
    weekly_df["Anomaly_Label"] = iso_model.fit_predict(weekly_df[["Sales"]])
    
    # Average weekly sales for comparison
    avg_weekly_sales = weekly_df["Sales"].mean()
    
    # Map anomaly type and business reasons dynamically
    anomalies = weekly_df[weekly_df["Anomaly_Label"] == -1].copy()
    
    reasons = []
    anomaly_types = []
    
    for date, row in anomalies.iterrows():
        sales_val = row["Sales"]
        month = date.month
        
        # High sales anomalies
        if sales_val >= avg_weekly_sales:
            anomaly_types.append("Spike (High Volume)")
            if month in [11, 12]:
                reasons.append("Festive season buying rush (Thanksgiving, Black Friday, Christmas discounts).")
            elif month == 9:
                reasons.append("End-of-quarter volume push / back-to-school marketing promotions.")
            elif month == 6:
                reasons.append("Mid-year inventory liquidation / summer sales clearance.")
            else:
                reasons.append("Flash sales campaign / Corporate bulk purchase or new product launch.")
        # Low sales anomalies
        else:
            anomaly_types.append("Drop (Low Volume)")
            if month == 1:
                reasons.append("Post-holiday market contraction / annual warehouse stocktaking delay.")
            elif month == 2:
                reasons.append("Fewer operational days / seasonal demand cool-off.")
            else:
                reasons.append("Supply chain logistics bottleneck / regional shipping disruption / system outage.")
                
    anomalies["Anomaly Type"] = anomaly_types
    anomalies["Possible Reason"] = reasons
    
    return weekly_df, anomalies, avg_weekly_sales

weekly_df, anomalies, avg_weekly_sales = detect_anomalies()

if weekly_df.empty:
    st.warning("⚠️ Database missing or could not be loaded.")
else:
    # Key Statistics for Page
    num_anomalies = len(anomalies)
    contamination_pct = (num_anomalies / len(weekly_df)) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f'<div class="kpi-container">'
            f'<div class="kpi-label">Total Monitored Weeks</div>'
            f'<div class="kpi-value">{len(weekly_df)}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f'<div class="kpi-container">'
            f'<div class="kpi-label">Flagged Anomalies</div>'
            f'<div class="kpi-value">{num_anomalies} weeks</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f'<div class="kpi-container">'
            f'<div class="kpi-label">Anomaly Ratio</div>'
            f'<div class="kpi-value">{contamination_pct:.1f}%</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        
    st.markdown('<div class="section-header">Weekly Sales Timeline & Flagged Anomalies</div>', unsafe_allow_html=True)
    
    # 1. Visualization using Plotly
    fig = go.Figure()
    
    # Regular sales line
    fig.add_trace(go.Scatter(
        x=weekly_df.index,
        y=weekly_df["Sales"],
        mode='lines',
        name='Weekly Sales',
        line=dict(color='#64748b', width=1.5),
        hovertemplate="Week Ending: %{x|%Y-%m-%d}<br>Sales: $%{{y:,.2f}}<extra></extra>"
    ))
    
    # Anomaly points (Red markers)
    fig.add_trace(go.Scatter(
        x=anomalies.index,
        y=anomalies["Sales"],
        mode='markers',
        name='Isolation Forest Anomaly',
        marker=dict(color='#e43f5a', size=10, symbol='circle', line=dict(color='#000000', width=1.5)),
        hovertemplate="Flagged Date: %{x|%Y-%m-%d}<br>Sales: $%{{y:,.2f}}<extra></extra>"
    ))
    
    # Average baseline
    fig.add_trace(go.Scatter(
        x=weekly_df.index,
        y=[avg_weekly_sales]*len(weekly_df),
        mode='lines',
        name='Historical Average',
        line=dict(color='#0f172a', width=1.5, dash='dash'),
        hovertemplate="Average Sales: $%{{y:,.2f}}<extra></extra>"
    ))
    
    fig.update_layout(
        xaxis_title="Timeline",
        yaxis_title="Weekly Sales ($)",
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=50, b=30, l=10, r=10),
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

    # 2. Table Section
    st.markdown('<div class="section-header">Anomaly Incident Deep Dive</div>', unsafe_allow_html=True)
    
    # Format table data
    table_df = anomalies.reset_index()
    table_df["Order Date"] = table_df["Order Date"].dt.strftime("%Y-%m-%d")
    table_df = table_df.rename(columns={"Order Date": "Date", "Sales": "Sales ($)"})
    table_df["Sales ($)"] = table_df["Sales ($)"].apply(lambda x: f"${x:,.2f}")
    
    # Display table containing required columns: Date, Sales, Anomaly Type, Possible Reason
    st.dataframe(
        table_df[["Date", "Sales ($)", "Anomaly Type", "Possible Reason"]], 
        use_container_width=True, 
        hide_index=True
    )
    
    st.info("💡 **Operational Action Plan:** Positive spikes (high volume spikes) should trigger inventory audits to prevent running out of stock. Negative drops should be investigated by checking local warehouse logs for logistical, weather, or supply delivery constraints on those specific weeks.")
