import streamlit as st
import pandas as pd
import os

# Page configurations - must be the first Streamlit command
st.set_page_config(
    page_title="Demand Intelligence & Forecasting Portal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for custom premium styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1f4068, #162447, #e43f5a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.25rem;
        color: #555555;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    .card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        border-left: 5px solid #1f4068;
    }
    .card-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1f4068;
        margin-bottom: 0.5rem;
    }
    .card-body {
        font-size: 0.95rem;
        color: #444444;
    }
</style>
""", unsafe_allow_html=True)

# Main Title Section
st.markdown('<div class="main-title">End-to-End Sales Forecasting & Demand Intelligence System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Internship Project Portal | Designed for Sales Analytics, Time Series Forecasting & Inventory Strategy Optimization</div>', unsafe_allow_html=True)

# Sidebar branding
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1551836022-d5d88e9218df?w=300", use_container_width=True)
    st.markdown("### **Demand Intelligence System**")
    st.markdown("Developed by: **Aryan Parashar**")
    st.markdown("---")
    st.info("Navigate through the dashboard pages using the menu above to explore Sales Overview, Forecast Explorer, Anomaly Reports, and Product Clustering.")

# Load quick statistics from cleaned sales data
@st.cache_data
def load_quick_stats():
    if os.path.exists("cleaned_sales.csv"):
        df = pd.read_csv("cleaned_sales.csv")
        total_sales = df["Sales"].sum()
        total_orders = df["Order ID"].nunique()
        total_products = df["Product ID"].nunique()
        return total_sales, total_orders, total_products
    return 0, 0, 0

total_sales, total_orders, total_products = load_quick_stats()

# Hero Section Image
st.image(
    "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1200",
    use_container_width=True
)

st.markdown("### 🏢 Executive System Overview")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Accumulated Historical Sales",
        value=f"${total_sales:,.2f}"
    )

with col2:
    st.metric(
        label="Unique Customer Orders",
        value=f"{total_orders:,}"
    )

with col3:
    st.metric(
        label="Monitored Product SKUs",
        value=f"{total_products:,}"
    )

st.markdown("---")

# Feature grid layout
st.markdown("### 🛠️ Key Dashboard Capabilities")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("""
    <div class="card">
        <div class="card-title">📈 1. Sales Overview Dashboard</div>
        <div class="card-body">
            Examine historical sales distribution across categories, sub-categories, and regions.
            Understand seasonal trends and year-over-year performance using advanced filters.
        </div>
    </div>
    <div class="card">
        <div class="card-title">🔮 2. Forecast Explorer (XGBoost)</div>
        <div class="card-body">
            Interact with our top-performing machine learning forecast engine.
            Project future sales up to 3 months ahead by category or region with direct CSV exports.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown("""
    <div class="card">
        <div class="card-title">🚨 3. Anomaly Detection Report</div>
        <div class="card-body">
            Monitor and flags unusual sales volumes using Isolation Forest.
            Explore deep dives into specific dates with automatically suggested operational causes.
        </div>
    </div>
    <div class="card">
        <div class="card-title">📦 4. Product Demand Segments</div>
        <div class="card-body">
            Review sub-category clustering based on sales performance, growth, volatility, and order value.
            Retrieve actionable inventory stocking strategies mapped directly to each cluster.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.success("Demand Intelligence System Initialized Successfully! Use the sidebar navigation to begin.")