import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Page configuration
st.set_page_config(
    page_title="Sales Overview | Demand Intelligence",
    page_icon="📈",
    layout="wide"
)

# Custom styling for premium dashboard feel
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
    .metric-container {
        background-color: #f8f9fa;
        padding: 1.25rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📈 Sales Performance Overview</div>', unsafe_allow_html=True)
st.markdown("Monitor historical performance, analyze trend trajectories, and filter across regions and product categories.", unsafe_allow_html=True)
st.markdown("---")

# Data loading function with caching
@st.cache_data
def load_data():
    if not os.path.exists("cleaned_sales.csv"):
        st.error("Error: 'cleaned_sales.csv' not found. Please ensure it is in the project directory.")
        return pd.DataFrame()
    df = pd.read_csv("cleaned_sales.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    return df

df = load_data()

if not df.empty:
    # Sidebar Filters
    with st.sidebar:
        st.header("🔍 Filters")
        
        # Region Filter
        all_regions = sorted(df["Region"].unique())
        selected_regions = st.multiselect(
            "Select Region(s)",
            options=all_regions,
            default=all_regions,
            help="Select one or more geographic regions to filter the metrics and charts."
        )
        
        # Category Filter
        all_categories = sorted(df["Category"].unique())
        selected_categories = st.multiselect(
            "Select Product Category(s)",
            options=all_categories,
            default=all_categories,
            help="Select one or more product categories to filter the metrics and charts."
        )
        
        st.markdown("---")
        st.caption("Filters applied automatically. Clear selections to reset.")

    # Apply Filters
    filtered_df = df[
        (df["Region"].isin(selected_regions)) & 
        (df["Category"].isin(selected_categories))
    ]

    if filtered_df.empty:
        st.warning("⚠️ No data matches the selected filter combinations. Please adjust your filters in the sidebar.")
    else:
        # Calculate Key Metrics
        total_sales = filtered_df["Sales"].sum()
        total_orders = filtered_df["Order ID"].nunique()
        # AOV = Total Sales / Total Orders
        aov = total_sales / total_orders if total_orders > 0 else 0
        total_items = len(filtered_df)

        # Metric Columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f'<div class="metric-container">'
                f'<div class="metric-label">💰 Total Sales</div>'
                f'<div class="metric-value">${total_sales:,.2f}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f'<div class="metric-container">'
                f'<div class="metric-label">📦 Total Orders</div>'
                f'<div class="metric-value">{total_orders:,}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with col3:
            st.markdown(
                f'<div class="metric-container">'
                f'<div class="metric-label">💳 Average Order Value</div>'
                f'<div class="metric-value">${aov:,.2f}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with col4:
            st.markdown(
                f'<div class="metric-container">'
                f'<div class="metric-label">🛒 Items Sold</div>'
                f'<div class="metric-value">{total_items:,}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        # Visualizations Sections
        st.markdown('<div class="section-header">Sales Distribution & Trends</div>', unsafe_allow_html=True)
        
        # Row 1: Charts
        col_chart1, col_chart2 = st.columns([2, 3])

        with col_chart1:
            # Total Sales by Year (bar chart)
            yearly_sales = filtered_df.groupby("Year")["Sales"].sum().reset_index()
            yearly_sales["Year"] = yearly_sales["Year"].astype(str) # category format
            
            fig_year = px.bar(
                yearly_sales,
                x="Year",
                y="Sales",
                text="Sales",
                title="Annual Revenue Distribution",
                labels={"Sales": "Sales ($)", "Year": "Calendar Year"},
                color="Sales",
                color_continuous_scale="Viridis",
            )
            # Styling bars
            fig_year.update_traces(
                texttemplate='$%{text:,.2s}', 
                textposition='outside',
                hovertemplate="Year: %{x}<br>Sales: $%{{y:,.2f}}<extra></extra>"
            )
            fig_year.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                coloraxis_showscale=False,
                margin=dict(t=50, b=30, l=10, r=10),
                height=400
            )
            st.plotly_chart(fig_year, use_container_width=True)

        with col_chart2:
            # Monthly Sales Trend
            monthly_sales = filtered_df.groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"].sum().reset_index()
            # Calculate 3-month moving average for trend line
            monthly_sales["Rolling Mean"] = monthly_sales["Sales"].rolling(window=3, min_periods=1).mean()
            
            fig_trend = go.Figure()
            # Actual sales area
            fig_trend.add_trace(go.Scatter(
                x=monthly_sales["Order Date"],
                y=monthly_sales["Sales"],
                mode='lines+markers',
                name='Monthly Sales',
                line=dict(color='#1f4068', width=2),
                fill='tozeroy',
                fillcolor='rgba(31, 64, 104, 0.1)',
                hovertemplate="Date: %{x|%B %Y}<br>Sales: $%{{y:,.2f}}<extra></extra>"
            ))
            # Rolling mean trend line
            fig_trend.add_trace(go.Scatter(
                x=monthly_sales["Order Date"],
                y=monthly_sales["Rolling Mean"],
                mode='lines',
                name='3-Month Moving Avg',
                line=dict(color='#e43f5a', width=2, dash='dot'),
                hovertemplate="Date: %{x|%B %Y}<br>Rolling Mean: $%{{y:,.2f}}<extra></extra>"
            ))
            
            fig_trend.update_layout(
                title="Monthly Revenue Trajectory & Trend (3-Month MA)",
                xaxis_title="Timeline",
                yaxis_title="Sales ($)",
                plot_bgcolor='rgba(0,0,0,0)',
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=50, b=30, l=10, r=10),
                height=400
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        # Row 2: Categorical Breakdown
        st.markdown('<div class="section-header">Regional & Product Insights</div>', unsafe_allow_html=True)
        col_insight1, col_insight2 = st.columns(2)

        with col_insight1:
            # Sub-category sales
            subcat_sales = filtered_df.groupby("Sub-Category")["Sales"].sum().reset_index()
            subcat_sales = subcat_sales.sort_values(by="Sales", ascending=True)
            
            fig_subcat = px.bar(
                subcat_sales,
                y="Sub-Category",
                x="Sales",
                orientation='h',
                title="Sales Contribution by Sub-Category",
                labels={"Sales": "Sales ($)", "Sub-Category": "Sub-Category"},
                color="Sales",
                color_continuous_scale="Purples",
            )
            fig_subcat.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                coloraxis_showscale=False,
                margin=dict(t=50, b=30, l=10, r=10),
                height=450
            )
            fig_subcat.update_traces(
                hovertemplate="Sub-Category: %{y}<br>Sales: $%{{x:,.2f}}<extra></extra>"
            )
            st.plotly_chart(fig_subcat, use_container_width=True)

        with col_insight2:
            # Region share donut chart
            region_sales = filtered_df.groupby("Region")["Sales"].sum().reset_index()
            
            fig_region = px.pie(
                region_sales,
                values="Sales",
                names="Region",
                hole=0.4,
                title="Revenue Share by Region",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_region.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                hovertemplate="Region: %{label}<br>Sales: $%{{value:,.2f}}<br>Share: %{{percent}}<extra></extra>"
            )
            fig_region.update_layout(
                margin=dict(t=50, b=30, l=10, r=10),
                height=450
            )
            st.plotly_chart(fig_region, use_container_width=True)
else:
    st.info("Please load a valid data source to display the Sales Overview dashboard.")
