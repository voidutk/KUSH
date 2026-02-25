"""
Streamlit Dashboard for Supply Chain Resilience Engine
Interactive web UI for disruption prediction and what-if analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import sys
import io
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Supply Chain Resilience Engine",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        padding: 1rem 0;
    }
    .risk-critical { color: #DC2626; font-weight: bold; }
    .risk-high { color: #EA580C; font-weight: bold; }
    .risk-medium { color: #D97706; font-weight: bold; }
    .risk-low { color: #16A34A; font-weight: bold; }
    .metric-card {
        background: #F8FAFC;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://127.0.0.1:5000"

# Fallback: Import model directly if API not available
@st.cache_resource
def load_models_direct():
    """Load models directly from model.py"""
    try:
        sys.path.insert(0, '/home/utkarsh/Documents/INDCON ')
        from model import engine, data
        return engine, data
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None, None

# Check API health
def check_api():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

# Dynamic data generation
def generate_fresh_data():
    """Generate new synthetic data with random seed"""
    import random
    import time
    # Use current time as seed for truly random data
    np.random.seed(int(time.time()))
    
    # Re-import and regenerate
    sys.path.insert(0, '/home/utkarsh/Documents/INDCON ')
    from model import generate_synthetic_data
    fresh_data = generate_synthetic_data(1000)
    return fresh_data

# Session state for data
if 'data' not in st.session_state:
    st.session_state.data = None
    st.session_state.engine = None

# Load initial data
engine, data = load_models_direct()
if st.session_state.data is None:
    st.session_state.data = data
    st.session_state.engine = engine

# Header
st.markdown('<p class="main-header">🏭 Supply Chain Resilience Engine</p>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748B;'>AI-Based Disruption Prediction & Decision Intelligence System</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Control Panel")

# Refresh data button
if st.sidebar.button("🔄 Refresh Data", type="secondary"):
    with st.spinner("Generating new data..."):
        fresh_data = generate_fresh_data()
        st.session_state.data = fresh_data
        st.sidebar.success("✅ New data generated!")
        st.rerun()

st.sidebar.divider()

mode = st.sidebar.radio("Select Mode", [
    "📊 Dashboard Overview",
    "🔍 Single SKU Assessment", 
    "📦 Portfolio Risk Analysis",
    "🔮 What-If Scenario",
    "📈 API Status"
])

# Helper functions
def get_risk_color(score):
    if score >= 0.65:
        return "#DC2626"  # Critical - Red
    elif score >= 0.45:
        return "#EA580C"  # High - Orange
    elif score >= 0.25:
        return "#D97706"  # Medium - Amber
    else:
        return "#16A34A"  # Low - Green

def get_risk_class(score):
    if score >= 0.65:
        return "CRITICAL"
    elif score >= 0.45:
        return "HIGH"
    elif score >= 0.25:
        return "MEDIUM"
    else:
        return "LOW"

def predict_risk(sample_dict):
    """Get prediction via API or direct model"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"records": [sample_dict]},
            timeout=5
        )
        if response.status_code == 200:
            return response.json()["predictions"][0]
    except:
        pass
    
    # Fallback to direct model
    if engine:
        sample_df = pd.DataFrame([sample_dict])
        return engine.assess_disruption_risk(sample_df)
    return None

# ==================== DASHBOARD OVERVIEW ====================
if mode == "📊 Dashboard Overview":
    st.header("Supply Chain Risk Dashboard")
    
    if st.session_state.data is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate metrics
        total_skus = len(st.session_state.data)
        avg_demand = st.session_state.data['actual_demand'].mean()
        sole_source_pct = (st.session_state.data['is_sole_source'].sum() / total_skus) * 100
        avg_lead_time = st.session_state.data['lead_time'].mean()
        
        with col1:
            st.metric("Total SKUs", f"{total_skus:,}")
        with col2:
            st.metric("Avg Demand", f"{avg_demand:.0f} units")
        with col3:
            st.metric("Sole Source", f"{sole_source_pct:.1f}%")
        with col4:
            st.metric("Avg Lead Time", f"{avg_lead_time:.1f} days")
        
        # Risk distribution by SKU class
        st.subheader("Risk Distribution by SKU Class")
        
        # Sample predictions for risk distribution
        if st.button("🔄 Analyze Portfolio Risk", key="portfolio_btn"):
            with st.spinner("Analyzing portfolio..."):
                sample_data = st.session_state.data.sample(min(100, len(st.session_state.data)))
                predictions = []
                
                for _, row in sample_data.iterrows():
                    sample_dict = row.drop(['actual_demand', 'delay_happened', 'supplier_id', 'week']).to_dict()
                    pred = predict_risk(sample_dict)
                    if pred:
                        pred['sku_class'] = row['sku_class']
                        pred['region'] = row['region']
                        predictions.append(pred)
                
                if predictions:
                    pred_df = pd.DataFrame(predictions)
                    
                    col_left, col_right = st.columns(2)
                    
                    with col_left:
                        # Risk by SKU class
                        fig1 = px.histogram(
                            pred_df, x="sku_class", color="risk_class",
                            color_discrete_map={
                                "LOW": "#16A34A", "MEDIUM": "#D97706",
                                "HIGH": "#EA580C", "CRITICAL": "#DC2626"
                            },
                            title="Risk Distribution by SKU Class",
                            labels={"sku_class": "SKU Class", "count": "Count"}
                        )
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col_right:
                        # Risk by region
                        fig2 = px.histogram(
                            pred_df, x="region", color="risk_class",
                            color_discrete_map={
                                "LOW": "#16A34A", "MEDIUM": "#D97706",
                                "HIGH": "#EA580C", "CRITICAL": "#DC2626"
                            },
                            title="Risk Distribution by Region",
                            labels={"region": "Region"}
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # Risk score distribution
                    fig3 = px.histogram(
                        pred_df, x="composite_risk_score", nbins=20,
                        color="risk_class",
                        color_discrete_map={
                            "LOW": "#16A34A", "MEDIUM": "#D97706",
                            "HIGH": "#EA580C", "CRITICAL": "#DC2626"
                        },
                        title="Composite Risk Score Distribution",
                        labels={"composite_risk_score": "Risk Score"}
                    )
                    fig3.add_vline(x=0.25, line_dash="dash", line_color="#16A34A")
                    fig3.add_vline(x=0.45, line_dash="dash", line_color="#D97706")
                    fig3.add_vline(x=0.65, line_dash="dash", line_color="#DC2626")
                    st.plotly_chart(fig3, use_container_width=True)
                    
                    # Summary table
                    st.subheader("High Risk Items")
                    high_risk = pred_df[pred_df['composite_risk_score'] >= 0.45].sort_values(
                        'composite_risk_score', ascending=False
                    ).head(10)
                    st.dataframe(
                        high_risk[['sku_class', 'region', 'composite_risk_score', 'risk_class', 'action']],
                        use_container_width=True
                    )
        else:
            st.info("Click 'Analyze Portfolio Risk' to generate risk insights")

# ==================== SINGLE SKU ASSESSMENT ====================
elif mode == "🔍 Single SKU Assessment":
    st.header("Single SKU Risk Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("SKU Configuration")
        sku_class = st.selectbox("SKU Class", ["A-X", "A-Y", "A-Z", "B-X", "B-Y", "B-Z", "C-X", "C-Y", "C-Z"])
        region = st.selectbox("Region", ["NA", "EU", "APAC", "LATAM"])
        
        st.subheader("Demand Parameters")
        demand_last_week = st.number_input("Demand Last Week", min_value=0.0, value=150.0, step=10.0)
        demand_trend = st.slider("Demand Trend", -0.3, 0.5, 0.05, 0.01)
        seasonality = st.slider("Seasonality Index", 0.5, 1.5, 1.1, 0.05)
        demand_volatility = st.slider("Demand Volatility", 0.5, 3.0, 1.0, 0.1)
    
    with col2:
        st.subheader("Supplier Parameters")
        supplier_reliability = st.slider("Supplier Reliability", 0.0, 1.0, 0.75, 0.05)
        avg_transport_delay = st.number_input("Avg Transport Delay (days)", min_value=0, value=3, step=1)
        supplier_tier = st.selectbox("Supplier Tier", [1, 2, 3])
        lead_time = st.number_input("Lead Time (days)", min_value=1, value=7, step=1)
        distance = st.number_input("Distance", min_value=0, value=500, step=50)
        
        is_sole_source = st.checkbox("Is Sole Source Supplier", value=False)
        backup_supplier_count = st.number_input("Backup Supplier Count", min_value=0, value=2, step=1)
        geopolitical_risk = st.slider("Geopolitical Risk", 0.0, 1.0, 0.2, 0.05)
    
    st.subheader("Inventory Parameters")
    col3, col4, col5 = st.columns(3)
    with col3:
        current_inventory = st.number_input("Current Inventory", min_value=0, value=300, step=10)
    with col4:
        incoming_supply = st.number_input("Incoming Supply", min_value=0, value=200, step=10)
    with col5:
        order_book_value = st.number_input("Order Book Value", min_value=0, value=350, step=10)
    
    # Derived fields
    pipeline_inventory = incoming_supply * lead_time * 0.3
    bullwhip_coeff = 1.5 + (1 - supplier_reliability)
    order_volatility = demand_volatility * 0.3
    
    if st.button("🚀 Run Risk Assessment", type="primary"):
        sample = {
            "sku_class": sku_class,
            "region": region,
            "demand_last_week": demand_last_week,
            "demand_trend": demand_trend,
            "seasonality_index": seasonality,
            "demand_volatility": demand_volatility,
            "supplier_reliability": supplier_reliability,
            "avg_transport_delay": avg_transport_delay,
            "supplier_tier": supplier_tier,
            "lead_time": lead_time,
            "distance": distance,
            "is_sole_source": int(is_sole_source),
            "backup_supplier_count": backup_supplier_count,
            "geopolitical_risk": geopolitical_risk,
            "current_inventory": current_inventory,
            "incoming_supply": incoming_supply,
            "pipeline_inventory": pipeline_inventory,
            "bullwhip_coeff": bullwhip_coeff,
            "order_volatility": order_volatility,
            "order_book_value": order_book_value
        }
        
        with st.spinner("Running assessment..."):
            result = predict_risk(sample)
        
        if result:
            st.divider()
            st.subheader("📊 Assessment Results")
            
            # Risk score with gauge
            risk_score = result["composite_risk_score"]
            risk_class = result["risk_class"]
            risk_color = get_risk_color(risk_score)
            
            col_score, col_gauge = st.columns([1, 2])
            
            with col_score:
                st.markdown(f"""
                <div style='text-align: center; padding: 2rem; background: {risk_color}20; border-radius: 12px;'>
                    <h1 style='color: {risk_color}; font-size: 3rem; margin: 0;'>{risk_score:.3f}</h1>
                    <h2 style='color: {risk_color}; margin: 0.5rem 0;'>{risk_class}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col_gauge:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    number={'font': {'size': 40}},
                    gauge={
                        'axis': {'range': [0, 1], 'tickwidth': 1},
                        'bar': {'color': risk_color},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 0.25], 'color': '#DCFCE7'},
                            {'range': [0.25, 0.45], 'color': '#FEF3C7'},
                            {'range': [0.45, 0.65], 'color': '#FFEDD5'},
                            {'range': [0.65, 1], 'color': '#FEE2E2'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': risk_score
                        }
                    }
                ))
                fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)
            
            # Recommendation
            st.markdown(f"""
            <div style='padding: 1.5rem; background: #F1F5F9; border-radius: 8px; border-left: 5px solid {risk_color}; color: #1E293B;'>
                <h3 style='margin-top: 0; color: #1E293B;'>Recommended Action: {result['action']}</h3>
                <p style='color: #334155;'>{result['action_details']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Component breakdown
            col_comp1, col_comp2, col_comp3 = st.columns(3)
            with col_comp1:
                st.metric("Demand Risk", f"{result['demand_risk']:.1%}")
            with col_comp2:
                st.metric("Delay Probability", f"{result['delay_probability']:.1%}")
            with col_comp3:
                st.metric("Stockout Probability", f"{result['inventory_analysis']['stockout_probability']:.1%}")
            
            # Inventory analysis
            st.subheader("📦 Inventory Analysis")
            inv = result['inventory_analysis']
            col_inv1, col_inv2, col_inv3 = st.columns(3)
            with col_inv1:
                st.metric("Predicted Demand", f"{result['predicted_demand']:.0f} units")
            with col_inv2:
                st.metric("Future Inventory", f"{inv['mean_future_inventory']:.0f} units")
            with col_inv3:
                st.metric("Days of Supply", f"{inv['days_of_supply']:.1f} days")

# ==================== PORTFOLIO RISK ANALYSIS ====================
elif mode == "📦 Portfolio Risk Analysis":
    st.header("Portfolio Risk Analysis")
    
    if st.session_state.data is not None:
        num_samples = st.slider("Number of SKUs to Analyze", 10, 500, 100, 10)
        
        if st.button("🔄 Run Portfolio Analysis"):
            with st.spinner(f"Analyzing {num_samples} SKUs..."):
                sample_data = st.session_state.data.sample(min(num_samples, len(st.session_state.data)))
                predictions = []
                
                for _, row in sample_data.iterrows():
                    sample_dict = row.drop(['actual_demand', 'delay_happened', 'supplier_id', 'week']).to_dict()
                    pred = predict_risk(sample_dict)
                    if pred:
                        pred['sku_class'] = row['sku_class']
                        pred['region'] = row['region']
                        pred['is_sole_source'] = row['is_sole_source']
                        predictions.append(pred)
                
                if predictions:
                    pred_df = pd.DataFrame(predictions)
                    
                    # Risk summary
                    risk_counts = pred_df['risk_class'].value_counts()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    for risk_level, col in zip(["LOW", "MEDIUM", "HIGH", "CRITICAL"], [col1, col2, col3, col4]):
                        count = risk_counts.get(risk_level, 0)
                        pct = (count / len(pred_df)) * 100
                        with col:
                            st.metric(f"{risk_level} Risk", f"{count}", f"{pct:.1f}%")
                    
                    # Visualizations
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=("Risk by SKU Class", "Risk by Region", 
                                       "Sole Source vs Risk", "Risk Score Distribution"),
                        specs=[[{"type": "histogram"}, {"type": "histogram"}],
                               [{"type": "box"}, {"type": "histogram"}]]
                    )
                    
                    # Risk by SKU class
                    for risk_class, color in [("LOW", "#16A34A"), ("MEDIUM", "#D97706"), 
                                               ("HIGH", "#EA580C"), ("CRITICAL", "#DC2626")]:
                        df_risk = pred_df[pred_df['risk_class'] == risk_class]
                        if not df_risk.empty:
                            fig.add_trace(
                                go.Histogram(x=df_risk['sku_class'], name=risk_class, 
                                           marker_color=color, showlegend=True),
                                row=1, col=1
                            )
                    
                    # Risk by region
                    for risk_class, color in [("LOW", "#16A34A"), ("MEDIUM", "#D97706"), 
                                               ("HIGH", "#EA580C"), ("CRITICAL", "#DC2626")]:
                        df_risk = pred_df[pred_df['risk_class'] == risk_class]
                        if not df_risk.empty:
                            fig.add_trace(
                                go.Histogram(x=df_risk['region'], name=risk_class, 
                                           marker_color=color, showlegend=False),
                                row=1, col=2
                            )
                    
                    # Sole source box plot
                    fig.add_trace(
                        go.Box(x=pred_df['is_sole_source'].map({0: 'Multi-Source', 1: 'Sole Source'}),
                               y=pred_df['composite_risk_score'], name='Risk Score',
                               marker_color='#3B82F6'),
                        row=2, col=1
                    )
                    
                    # Distribution
                    fig.add_trace(
                        go.Histogram(x=pred_df['composite_risk_score'], nbinsx=20,
                                   marker_color='#3B82F6', name='Distribution', showlegend=False),
                        row=2, col=2
                    )
                    
                    fig.update_layout(height=600, showlegend=True, 
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Data table
                    st.subheader("Detailed Results")
                    display_df = pred_df[['sku_class', 'region', 'composite_risk_score', 'risk_class', 
                                          'action', 'delay_probability']].copy()
                    # Extract stockout_probability from inventory_analysis dict
                    display_df['stockout_probability'] = pred_df['inventory_analysis'].apply(
                        lambda x: x.get('stockout_probability', 0) if isinstance(x, dict) else 0
                    )
                    st.dataframe(
                        display_df.sort_values('composite_risk_score', ascending=False),
                        use_container_width=True
                    )
                    
                    # Export options
                    st.subheader("📥 Export Results")
                    col_csv, col_pdf = st.columns(2)
                    
                    with col_csv:
                        # CSV Export
                        csv_buffer = io.StringIO()
                        display_df.sort_values('composite_risk_score', ascending=False).to_csv(
                            csv_buffer, index=False
                        )
                        st.download_button(
                            label="📄 Download CSV",
                            data=csv_buffer.getvalue(),
                            file_name=f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col_pdf:
                        # PDF Summary Report
                        try:
                            from reportlab.lib.pagesizes import letter
                            from reportlab.pdfgen import canvas
                            from reportlab.lib import colors
                            
                            def create_pdf_report(df, summary_stats):
                                buffer = io.BytesIO()
                                c = canvas.Canvas(buffer, pagesize=letter)
                                width, height = letter
                                
                                # Header
                                c.setFont("Helvetica-Bold", 16)
                                c.drawString(50, height - 50, "Supply Chain Risk Assessment Report")
                                c.setFont("Helvetica", 10)
                                c.drawString(50, height - 70, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                                
                                # Summary
                                c.setFont("Helvetica-Bold", 12)
                                c.drawString(50, height - 100, "Executive Summary")
                                c.setFont("Helvetica", 10)
                                y = height - 120
                                for key, val in summary_stats.items():
                                    c.drawString(50, y, f"{key}: {val}")
                                    y -= 15
                                
                                # High risk items
                                c.setFont("Helvetica-Bold", 12)
                                c.drawString(50, y - 20, "High Risk Items (Top 10)")
                                c.setFont("Helvetica", 8)
                                y -= 40
                                
                                # Table headers
                                headers = ['SKU Class', 'Region', 'Risk Score', 'Risk Class', 'Action']
                                x_pos = [50, 100, 150, 200, 280]
                                for i, header in enumerate(headers):
                                    c.drawString(x_pos[i], y, header)
                                y -= 15
                                c.line(50, y + 10, 550, y + 10)
                                
                                # Data rows
                                high_risk_df = df.sort_values('composite_risk_score', ascending=False).head(10)
                                for _, row in high_risk_df.iterrows():
                                    if y < 50:
                                        c.showPage()
                                        y = height - 50
                                    c.drawString(x_pos[0], y, str(row['sku_class']))
                                    c.drawString(x_pos[1], y, str(row['region']))
                                    c.drawString(x_pos[2], y, f"{row['composite_risk_score']:.3f}")
                                    c.drawString(x_pos[3], y, str(row['risk_class']))
                                    c.drawString(x_pos[4], y, str(row['action'])[:30])
                                    y -= 12
                                
                                c.save()
                                buffer.seek(0)
                                return buffer
                            
                            summary = {
                                'Total SKUs Analyzed': len(display_df),
                                'High Risk Items': len(display_df[display_df['risk_class'].isin(['HIGH', 'CRITICAL'])]),
                                'Sole Source SKUs': len(pred_df[pred_df['is_sole_source'] == 1]),
                                'Avg Risk Score': f"{display_df['composite_risk_score'].mean():.3f}"
                            }
                            
                            pdf_buffer = create_pdf_report(display_df, summary)
                            st.download_button(
                                label="📄 Download PDF Report",
                                data=pdf_buffer.getvalue(),
                                file_name=f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                        except ImportError:
                            st.info("Install reportlab for PDF export: pip install reportlab")

# ==================== WHAT-IF SCENARIO ====================
elif mode == "🔮 What-If Scenario":
    st.header("What-If Scenario Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔹 Baseline Scenario")
        base_reliability = st.slider("Baseline Supplier Reliability", 0.0, 1.0, 0.65, 0.05, key="base_rel")
        base_sole = st.checkbox("Baseline: Is Sole Source", value=True, key="base_sole")
        base_inventory = st.number_input("Baseline Current Inventory", min_value=0, value=400, step=10)
        base_lead = st.number_input("Baseline Lead Time", min_value=1, value=14, step=1)
    
    with col2:
        st.subheader("🔸 Improved Scenario")
        new_reliability = st.slider("New Supplier Reliability", 0.0, 1.0, 0.95, 0.05)
        new_sole = st.checkbox("New: Is Sole Source", value=False)
        new_inventory = st.number_input("New Current Inventory", min_value=0, value=600, step=10)
        new_backup = st.number_input("Backup Supplier Count", min_value=0, value=3, step=1)
    
    if st.button("🚀 Compare Scenarios", type="primary"):
        with st.spinner("Running comparison..."):
            # Baseline
            baseline = {
                "sku_class": "A-Y", "region": "APAC", "demand_last_week": 200.0,
                "demand_trend": 0.08, "seasonality_index": 1.2, "demand_volatility": 1.0,
                "supplier_reliability": base_reliability, "avg_transport_delay": 5, 
                "supplier_tier": 3, "lead_time": base_lead, "distance": 1200, 
                "is_sole_source": int(base_sole), "backup_supplier_count": 0, 
                "geopolitical_risk": 0.5, "current_inventory": base_inventory, 
                "incoming_supply": 300, "pipeline_inventory": 250,
                "bullwhip_coeff": 2.0, "order_volatility": 0.3, "order_book_value": 500
            }
            
            # Scenario
            scenario = {**baseline}
            scenario["supplier_reliability"] = new_reliability
            scenario["is_sole_source"] = int(new_sole)
            scenario["current_inventory"] = new_inventory
            scenario["backup_supplier_count"] = new_backup
            
            # Get predictions
            base_pred = predict_risk(baseline)
            scen_pred = predict_risk(scenario)
            
            if base_pred and scen_pred:
                st.divider()
                
                base_score = base_pred["composite_risk_score"]
                scen_score = scen_pred["composite_risk_score"]
                improvement = base_score - scen_score
                
                col_base, col_arrow, col_new = st.columns([2, 1, 2])
                
                with col_base:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 1.5rem; background: {get_risk_color(base_score)}20; border-radius: 12px;'>
                        <h3 style='margin: 0;'>Baseline</h3>
                        <h1 style='color: {get_risk_color(base_score)}; margin: 0.5rem 0;'>{base_score:.3f}</h1>
                        <p style='margin: 0; color: #64748B;'>{base_pred['risk_class']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_arrow:
                    st.markdown(f"""
                    <div style='text-align: center; padding-top: 3rem;'>
                        <h1 style='color: {"#16A34A" if improvement > 0 else "#DC2626"};'>→</h1>
                        <p style='color: {"#16A34A" if improvement > 0 else "#DC2626"}; font-weight: bold;'>
                            {improvement:+.3f}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_new:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 1.5rem; background: {get_risk_color(scen_score)}20; border-radius: 12px;'>
                        <h3 style='margin: 0;'>Scenario</h3>
                        <h1 style='color: {get_risk_color(scen_score)}; margin: 0.5rem 0;'>{scen_score:.3f}</h1>
                        <p style='margin: 0; color: #64748B;'>{scen_pred['risk_class']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Comparison chart
                categories = ['Risk Score', 'Delay Prob', 'Stockout Prob', 'Demand Risk']
                baseline_vals = [base_score, base_pred['delay_probability'], 
                                base_pred['inventory_analysis']['stockout_probability'],
                                base_pred['demand_risk']]
                scenario_vals = [scen_score, scen_pred['delay_probability'],
                                scen_pred['inventory_analysis']['stockout_probability'],
                                scen_pred['demand_risk']]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(name='Baseline', x=categories, y=baseline_vals,
                                   marker_color='#94A3B8'))
                fig.add_trace(go.Bar(name='Scenario', x=categories, y=scenario_vals,
                                   marker_color='#3B82F6'))
                fig.update_layout(barmode='group', title='Component Comparison',
                                yaxis_title='Probability/Score',
                                yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                col_rec1, col_rec2 = st.columns(2)
                with col_rec1:
                    st.subheader("Baseline Action")
                    st.info(f"**{base_pred['action']}**\n\n{base_pred['action_details']}")
                with col_rec2:
                    st.subheader("Scenario Action")
                    st.success(f"**{scen_pred['action']}**\n\n{scen_pred['action_details']}")

# ==================== API STATUS ====================
elif mode == "📈 API Status":
    st.header("API & System Status")
    
    api_available = check_api()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if api_available:
            st.success("✅ Flask API is running on http://127.0.0.1:5000")
            try:
                response = requests.get(f"{API_BASE_URL}/health", timeout=2)
                status = response.json()
                st.json(status)
            except:
                pass
        else:
            st.error("❌ Flask API is not running")
            st.info("Start the API with: `python api.py`")
    
    with col2:
        if engine:
            st.success("✅ Models loaded directly (fallback available)")
            st.info(f"Data shape: {st.session_state.data.shape if st.session_state.data is not None else 'N/A'}")
        else:
            st.warning("⚠️ Models not loaded")
    
    # Quick API test
    st.subheader("Quick API Test")
    if st.button("🧪 Test /predict Endpoint"):
        try:
            response = requests.get(f"{API_BASE_URL}/predict", timeout=5)
            st.json(response.json())
        except Exception as e:
            st.error(f"Error: {e}")
    
    if st.button("🧪 Test /whatif Endpoint"):
        try:
            response = requests.get(f"{API_BASE_URL}/whatif", timeout=5)
            st.json(response.json())
        except Exception as e:
            st.error(f"Error: {e}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Supply Chain Resilience Engine v1.0**")
st.sidebar.markdown("*Hackathon Demo - INDCON 2025*")
