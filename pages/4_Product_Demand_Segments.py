import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import os

# Page configuration
st.set_page_config(
    page_title="Product Demand Segments | Demand Intelligence",
    page_icon="📦",
    layout="wide"
)

# Custom premium styling
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
    .strategy-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f4068;
        margin-bottom: 0.75rem;
    }
    .strategy-title {
        font-weight: 700;
        color: #1f4068;
        font-size: 1rem;
        margin-bottom: 0.2rem;
    }
    .strategy-desc {
        font-size: 0.9rem;
        color: #4b5563;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📦 Product Demand Segmentation</div>', unsafe_allow_html=True)
st.markdown("Segment product sub-categories based on total sales, volatility, YoY growth, and average order value using K-Means clustering.", unsafe_allow_html=True)
st.markdown("---")

# Caching the clustering pipeline for performance
@st.cache_data
def perform_demand_clustering():
    if not os.path.exists("cleaned_sales.csv"):
        st.error("Error: 'cleaned_sales.csv' not found. Please ensure it is in the project directory.")
        return pd.DataFrame()
        
    df = pd.read_csv("cleaned_sales.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    
    # 1. Feature Engineering
    # Total Sales
    total_sales = df.groupby("Sub-Category")["Sales"].sum()
    
    # YoY Growth Rate
    yearly_sales = df.groupby(["Sub-Category", "Year"])["Sales"].sum().unstack(fill_value=0)
    # Sort years to ensure correct first and last columns
    yearly_sales = yearly_sales.reindex(columns=sorted(yearly_sales.columns))
    growth_rate = (yearly_sales.iloc[:, -1] - yearly_sales.iloc[:, 0]) / yearly_sales.iloc[:, 0]
    growth_rate = growth_rate.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Volatility (monthly std dev)
    # Try resample freq ME and M for compatibility
    try:
        monthly_subcat = df.groupby(["Sub-Category", pd.Grouper(key="Order Date", freq="ME")])["Sales"].sum()
    except ValueError:
        monthly_subcat = df.groupby(["Sub-Category", pd.Grouper(key="Order Date", freq="M")])["Sales"].sum()
    volatility = monthly_subcat.groupby("Sub-Category").std().fillna(0)
    
    # Average Order Value per subcategory
    average_order = df.groupby("Sub-Category")["Sales"].mean()
    
    # Combine features into dataframe
    cluster_df = pd.DataFrame({
        "Total Sales": total_sales,
        "Growth Rate": growth_rate,
        "Volatility": volatility,
        "Average Order": average_order
    })
    
    # 2. Scaling
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cluster_df)
    
    # 3. KMeans Clustering (n=4, random_state=42)
    kmeans = KMeans(n_clusters=4, random_state=42)
    cluster_df["Cluster"] = kmeans.fit_predict(scaled_features)
    
    # 4. PCA for 2D visualization
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(scaled_features)
    cluster_df["PCA1"] = pca_features[:, 0]
    cluster_df["PCA2"] = pca_features[:, 1]
    
    # 5. Meaningful Cluster Labels
    cluster_names = {
        0: "High Volume, Stable Demand",
        1: "Growing Demand",
        2: "Low Volume, High Volatility",
        3: "Declining Demand"
    }
    cluster_df["Demand Segment"] = cluster_df["Cluster"].map(cluster_names)
    
    # 6. Recommended Stocking Strategy mapping
    strategy_mapping = {
        "High Volume, Stable Demand": "Maintain higher inventory levels to prevent stock-outs.",
        "Growing Demand": "Increase stock gradually and monitor future demand.",
        "Low Volume, High Volatility": "Keep limited stock and reorder based on demand.",
        "Declining Demand": "Reduce inventory and avoid overstocking."
    }
    cluster_df["Recommended Stocking Strategy"] = cluster_df["Demand Segment"].map(strategy_mapping)
    
    return cluster_df

cluster_result = perform_demand_clustering()

if cluster_result.empty:
    st.warning("⚠️ Database missing or could not be loaded.")
else:
    # Page layout columns
    col_plot, col_strategy = st.columns([3, 2])
    
    with col_plot:
        st.markdown('<div class="section-header">PCA Cluster Visualization</div>', unsafe_allow_html=True)
        
        st.markdown("**Sub-categories projected onto 2D PCA space:**")
        
        # Plotly PCA Scatter Plot with Custom Legend
        fig = px.scatter(
            cluster_result.reset_index(),
            x="PCA1",
            y="PCA2",
            color="Demand Segment",
            hover_name="Sub-Category",
            hover_data={
                "Demand Segment": True,
                "Total Sales": ":$,.2f",
                "Growth Rate": ":.2%",
                "Volatility": ":$,.2f",
                "Average Order": ":$,.2f",
                "PCA1": False,
                "PCA2": False
            },
            color_discrete_map={
                "High Volume, Stable Demand": "#0ea5e9", # Sky blue
                "Growing Demand": "#10b981",       # Emerald green
                "Low Volume, High Volatility": "#f59e0b", # Amber
                "Declining Demand": "#ef4444"      # Red
            }
        )
        
        # Style layout
        fig.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='#f1f5f9', zeroline=True, zerolinecolor='#cbd5e1'),
            yaxis=dict(showgrid=True, gridcolor='#f1f5f9', zeroline=True, zerolinecolor='#cbd5e1'),
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
            margin=dict(t=20, b=80, l=10, r=10),
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col_strategy:
        st.markdown('<div class="section-header">Inventory Strategy Guide</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="strategy-card" style="border-left-color: #0ea5e9;">
            <div class="strategy-title" style="color: #0ea5e9;">🌐 High Volume, Stable Demand</div>
            <div class="strategy-desc"><strong>Action:</strong> Maintain higher safety stock levels. Establish continuous replenishment agreements with suppliers.</div>
        </div>
        <div class="strategy-card" style="border-left-color: #10b981;">
            <div class="strategy-title" style="color: #10b981;">📈 Growing Demand</div>
            <div class="strategy-desc"><strong>Action:</strong> Gradually raise order quantities. Conduct monthly demand review sessions to catch upward inflection early.</div>
        </div>
        <div class="strategy-card" style="border-left-color: #f59e0b;">
            <div class="strategy-title" style="color: #f59e0b;">⚡ Low Volume, High Volatility</div>
            <div class="strategy-desc"><strong>Action:</strong> Keep limited physical inventory. Use Just-in-Time (JIT) ordering or drop-shipping methods to limit carrying cost.</div>
        </div>
        <div class="strategy-card" style="border-left-color: #ef4444;">
            <div class="strategy-title" style="color: #ef4444;">📉 Declining Demand</div>
            <div class="strategy-desc"><strong>Action:</strong> Implement promotional markdown pricing to liquidate remainder stock. Avoid placing new purchase orders.</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Interactive Table section
    st.markdown('<div class="section-header">Interactive Demand Segment Details</div>', unsafe_allow_html=True)
    
    # Format and present the interactive table
    table_view = cluster_result.reset_index().copy()
    
    # Pretty formatting for numerical statistics
    table_view["Total Sales"] = table_view["Total Sales"].apply(lambda x: f"${x:,.2f}")
    table_view["Growth Rate"] = table_view["Growth Rate"].apply(lambda x: f"{x:.2%}")
    table_view["Volatility"] = table_view["Volatility"].apply(lambda x: f"${x:,.2f}")
    table_view["Average Order"] = table_view["Average Order"].apply(lambda x: f"${x:,.2f}")
    
    # Re-arrange and rename columns for display
    display_cols = [
        "Sub-Category", 
        "Demand Segment", 
        "Recommended Stocking Strategy",
        "Total Sales", 
        "Growth Rate", 
        "Volatility", 
        "Average Order"
    ]
    
    st.dataframe(
        table_view[display_cols],
        use_container_width=True,
        hide_index=True
    )
    
    st.caption("💡 **Tip:** Click on column headers to sort the table by segment, strategy, sales volume, or growth rate.")
