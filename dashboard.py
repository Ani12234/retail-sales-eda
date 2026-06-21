import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import json

# Set page layout and config
st.set_page_config(
    page_title="Retail Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Dark Theme and Styling
st.markdown("""
<style>
    /* Google Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', 'Segoe UI', sans-serif;
    }
    
    /* Metric Cards Glassmorphism */
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 700;
        color: #F3F4F6;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card {
        background: rgba(31, 41, 55, 0.4);
        border: 1px solid rgba(75, 85, 99, 0.3);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s, border-color 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(245, 158, 11, 0.6);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #9CA3AF;
        text-transform: uppercase;
        margin-bottom: 8px;
        font-weight: 600;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #FFFFFF;
    }
    .metric-highlight {
        color: #F59E0B; /* Gold Accent */
    }
    .metric-green {
        color: #10B981; /* Emerald Green */
    }
</style>
""", unsafe_allow_html=True)

# Load data helper
@st.cache_data
def load_data():
    clean_csv_path = "retail_sales_clean.csv"
    if os.path.exists(clean_csv_path):
        df = pd.read_csv(clean_csv_path)
        df['order_date'] = pd.to_datetime(df['order_date']).dt.date
        return df
    return None

# Load summary helper
def load_summary():
    summary_path = "analysis_summary.json"
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            return json.load(f)
    return None

# Load dataset and summary
df = load_data()
summary = load_summary()

if df is None:
    st.error("Error: Cleaned dataset not found! Please run the data generation and data cleaning pipelines first.")
    st.info("Run `python data_generation.py` followed by `python data_cleaning.py` to prepare the data.")
else:
    # -------------------------------------------------------------
    # SIDEBAR FILTERING
    # -------------------------------------------------------------
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3222/3222672.png", width=70)
    st.sidebar.title("Dashboard Filters")
    st.sidebar.markdown("Filter sales metrics by categories, segments, or regions.")
    
    # Category filter
    all_categories = sorted(df['product_category'].unique())
    selected_cats = st.sidebar.multiselect(
        "Product Category", 
        options=all_categories, 
        default=all_categories
    )
    
    # Segment filter
    all_segments = sorted(df['customer_segment'].unique())
    selected_segs = st.sidebar.multiselect(
        "Customer Segment", 
        options=all_segments, 
        default=all_segments
    )
    
    # Region filter
    all_regions = sorted(df['region'].unique())
    selected_regions = st.sidebar.multiselect(
        "Sales Region", 
        options=all_regions, 
        default=all_regions
    )
    
    # Filter dataset
    filtered_df = df[
        (df['product_category'].isin(selected_cats)) & 
        (df['customer_segment'].isin(selected_segs)) & 
        (df['region'].isin(selected_regions))
    ]
    
    # -------------------------------------------------------------
    # MAIN HEADER
    # -------------------------------------------------------------
    st.title("📊 Retail Sales Analytics Portfolio Dashboard")
    st.markdown("An interactive, high-fidelity business intelligence dashboard demonstrating synthetic data engineering, quality control, and visual analysis.")
    
    # -------------------------------------------------------------
    # KPI METRICS
    # -------------------------------------------------------------
    # Recalculate based on filters
    filtered_revenue = filtered_df['revenue'].sum()
    filtered_units = filtered_df['units_sold'].sum()
    filtered_aov = filtered_revenue / len(filtered_df) if len(filtered_df) > 0 else 0
    
    # Q4 vs baseline for filtered data
    q_revs = filtered_df.groupby('quarter')['revenue'].sum()
    has_all_quarters = all(q in q_revs for q in ['Q1', 'Q2', 'Q3', 'Q4'])
    if has_all_quarters:
        q_baseline = (q_revs['Q1'] + q_revs['Q2'] + q_revs['Q3']) / 3
        q4_inc = (q_revs['Q4'] - q_baseline) / q_baseline * 100
        q4_inc_str = f"+{q4_inc:.1f}%"
    else:
        q4_inc_str = "N/A"
        
    # Region share for filtered data
    r_revs = filtered_df.groupby('region')['revenue'].sum()
    bottom_3_names = ['Northwest', 'Mountain', 'New England']
    filtered_bottom_3_rev = sum(r_revs.get(r, 0) for r in bottom_3_names)
    filtered_bottom_3_share = (filtered_bottom_3_rev / filtered_revenue * 100) if filtered_revenue > 0 else 0

    cols = st.columns(4)
    with cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Revenue</div>
            <div class="metric-value">${filtered_revenue:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Average Order Value (AOV)</div>
            <div class="metric-value">${filtered_aov:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Q4 Spike vs. Baseline</div>
            <div class="metric-value metric-green">{q4_inc_str}</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[3]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Underperforming Regions Share</div>
            <div class="metric-value metric-highlight">{filtered_bottom_3_share:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # TABS FOR PAGES
    # -------------------------------------------------------------
    tabs = st.tabs(["💡 Sales Performance Insights", "📐 Power BI Rebuild Guide", "📋 Raw Data Explorer"])
    
    with tabs[0]:
        st.subheader("Interactive Business Visualizations")
        
        row1 = st.columns(2)
        
        # Plot 1: Revenue by Category
        with row1[0]:
            st.markdown("#### Revenue by Product Category")
            cat_data = filtered_df.groupby('product_category')['revenue'].sum().reset_index()
            cat_data = cat_data.sort_values(by='revenue', ascending=True)
            
            fig_cat = px.bar(
                cat_data, 
                x='revenue', 
                y='product_category',
                orientation='h',
                color_discrete_sequence=['#3B82F6'], # Slate Blue
                labels={'revenue': 'Revenue ($)', 'product_category': 'Category'}
            )
            fig_cat.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#E5E7EB',
                margin=dict(l=20, r=20, t=20, b=20),
                height=300,
                xaxis=dict(showgrid=True, gridcolor='#374151'),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig_cat, width='stretch')
            
        # Plot 2: Regional Performance Comparison
        with row1[1]:
            st.markdown("#### Regional Sales Performance Comparison")
            reg_data = filtered_df.groupby('region')['revenue'].sum().reset_index()
            reg_data = reg_data.sort_values(by='revenue', ascending=False)
            
            # Map colors: highlight bottom 3 in Accent Gold (#F59E0B), top 7 in Muted Navy (#1F4E78)
            colors = []
            for r in reg_data['region']:
                if r in bottom_3_names:
                    colors.append('#F59E0B') # Gold
                else:
                    colors.append('#1E40AF') # Muted Royal Blue
                    
            fig_reg = go.Figure(data=[go.Bar(
                x=reg_data['region'],
                y=reg_data['revenue'],
                marker_color=colors,
                text=reg_data['revenue'].apply(lambda x: f"${x/1000:,.0f}k"),
                textposition='outside'
            )])
            fig_reg.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#E5E7EB',
                margin=dict(l=20, r=20, t=20, b=20),
                height=300,
                xaxis=dict(tickangle=45, showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#374151', title='Revenue ($)')
            )
            st.plotly_chart(fig_reg, width='stretch')
            
        row2 = st.columns(2)
        
        # Plot 3: Quarterly Sales Trend
        with row2[0]:
            st.markdown("#### Quarterly Performance vs. Baseline")
            q_trend = filtered_df.groupby('quarter')['revenue'].sum().reset_index()
            
            # Line plot
            fig_q = go.Figure()
            
            # Line for quarters
            fig_q.add_trace(go.Scatter(
                x=q_trend['quarter'],
                y=q_trend['revenue'],
                mode='lines+markers',
                line=dict(color='#3B82F6', width=3),
                marker=dict(size=8, color='#3B82F6'),
                name='Quarterly Revenue'
            ))
            
            # Baseline dotted line
            if len(q_trend) >= 3:
                baseline_val = q_trend[q_trend['quarter'].isin(['Q1', 'Q2', 'Q3'])]['revenue'].mean()
                fig_q.add_trace(go.Scatter(
                    x=q_trend['quarter'],
                    y=[baseline_val] * len(q_trend),
                    mode='lines',
                    line=dict(color='#9CA3AF', width=2, dash='dash'),
                    name='Baseline (Avg Q1-Q3)'
                ))
            
            fig_q.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#E5E7EB',
                margin=dict(l=20, r=20, t=20, b=20),
                height=320,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#374151', title='Revenue ($)')
            )
            st.plotly_chart(fig_q, width='stretch')
            
        # Plot 4: Underperforming Regions Shortfall
        with row2[1]:
            st.markdown("#### Bottom 3 Regions Shortfall vs. Avg Region Target")
            
            # Calculate metrics
            total_filtered_rev = filtered_df['revenue'].sum()
            num_regions = filtered_df['region'].nunique()
            
            if num_regions > 0:
                avg_regional_rev = total_filtered_rev / num_regions
                
                # Bottom 3 regions data
                bottom_3_data = reg_data[reg_data['region'].isin(bottom_3_names)].copy()
                bottom_3_data['target'] = avg_regional_rev
                bottom_3_data['shortfall'] = bottom_3_data['target'] - bottom_3_data['revenue']
                
                fig_short = go.Figure()
                
                # Actual
                fig_short.add_trace(go.Bar(
                    name='Actual Revenue',
                    x=bottom_3_data['region'],
                    y=bottom_3_data['revenue'],
                    marker_color='#F59E0B'
                ))
                
                # Target
                fig_short.add_trace(go.Bar(
                    name='Regional Target (Average)',
                    x=bottom_3_data['region'],
                    y=bottom_3_data['target'],
                    marker_color='#4B5563',
                    opacity=0.5
                ))
                
                fig_short.update_layout(
                    barmode='group',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#E5E7EB',
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=320,
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='#374151', title='Revenue ($)')
                )
                st.plotly_chart(fig_short, width='stretch')
            else:
                st.info("Select regions to display shortfall analysis.")
                
    with tabs[1]:
        st.subheader("Power BI Rebuild Specifications")
        st.markdown("""
        Use this guide to replicate the exact dashboard visuals inside **Power BI Desktop**.
        We have exported the full specification file: `power_bi_dashboard_spec.xlsx` which contains the data sheet and visual configuration parameters.
        """)
        
        st.info("📂 **Data Source**: Import sheet `Clean Sales Data` from `power_bi_dashboard_spec.xlsx` or `retail_sales_clean.csv`.")
        
        # Color Palette Specs
        st.markdown("### 🎨 Color Palette Theme (Dark Mode)")
        col_p = st.columns(4)
        col_p[0].markdown("<div style='background:#0F172A;height:40px;border-radius:4px;border:1px solid #FFF;'></div><p style='text-align:center;'>Dark Slate (#0F172A)</p>", unsafe_allow_html=True)
        col_p[1].markdown("<div style='background:#1E3A8A;height:40px;border-radius:4px;border:1px solid #FFF;'></div><p style='text-align:center;'>Muted Royal Blue (#1E3A8A)</p>", unsafe_allow_html=True)
        col_p[2].markdown("<div style='background:#3B82F6;height:40px;border-radius:4px;border:1px solid #FFF;'></div><p style='text-align:center;'>Bright Slate Blue (#3B82F6)</p>", unsafe_allow_html=True)
        col_p[3].markdown("<div style='background:#F59E0B;height:40px;border-radius:4px;border:1px solid #FFF;'></div><p style='text-align:center;'>Accent Gold (#F59E0B)</p>", unsafe_allow_html=True)
        
        # DAX Formulas
        st.markdown("### 📈 Useful DAX Measures")
        st.code("""
-- Total Revenue
Total Revenue = SUM('Clean Sales Data'[revenue])

-- Average Regional Revenue Target
Avg Region Revenue Target = 
CALCULATE(
    [Total Revenue], 
    ALL('Clean Sales Data'[region])
) / DISTINCTCOUNT('Clean Sales Data'[region])

-- Underperforming Region Shortfall
Regional Shortfall = 
VAR AvgTarget = [Avg Region Revenue Target]
VAR ActualRev = [Total Revenue]
RETURN 
IF(ActualRev < AvgTarget, AvgTarget - ActualRev, 0)
        """, language="sql")
        
        # Table of Visual mappings
        st.markdown("### 🖼️ Power BI Visual Field Mapping")
        spec_data = [
            {"Visual Name": "KPI: Total Revenue", "Visual Type": "Card", "Fields": "Value: [Total Revenue]", "Formatting": "Font size 40, Dark Slate background"},
            {"Visual Name": "Revenue by Category", "Visual Type": "Clustered Bar Chart", "Fields": "Y-Axis: product_category, X-Axis: [Total Revenue]", "Formatting": "Color: Slate Blue, Sort descending"},
            {"Visual Name": "Regional Performance", "Visual Type": "Clustered Column Chart", "Fields": "X-Axis: region, Y-Axis: [Total Revenue]", "Formatting": "Conditional Color: Gold for Northwest, Mountain, New England. Royal Blue for others."},
            {"Visual Name": "Quarterly Trend", "Visual Type": "Line Chart", "Fields": "X-Axis: quarter, Y-Axis: [Total Revenue]", "Formatting": "Add Average baseline constant line"},
            {"Visual Name": "Shortfall Analysis", "Visual Type": "Clustered Column Chart", "Fields": "X-Axis: region (filtered bottom 3), Y-Axis: [Total Revenue], [Avg Region Revenue Target]", "Formatting": "Highlight actual bar in Gold"}
        ]
        st.table(spec_data)
        
    with tabs[2]:
        st.subheader("Data Explorer")
        st.markdown("Inspect and query the raw cleaned records. You can download the dataset directly using the download button below.")
        
        # Download buttons
        col_d = st.columns(2)
        with col_d[0]:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Clean Data as CSV",
                data=csv,
                file_name="retail_sales_clean.csv",
                mime="text/csv"
            )
        
        # Display dataset table
        st.dataframe(filtered_df, width='stretch')
