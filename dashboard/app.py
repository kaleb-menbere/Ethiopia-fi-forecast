# dashboard/app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import os

# Add the project root to the path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Import custom components
from components.overview import render_overview
from components.trends import render_trends
from components.forecasts import render_forecasts
from components.insights import render_insights

# Set page configuration
st.set_page_config(
    page_title="Ethiopia Financial Inclusion Dashboard",
    page_icon="🇪🇹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def load_css():
    css_file = project_root / "dashboard" / "assets" / "style.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        # Basic CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E3A8A;
            text-align: center;
            margin-bottom: 2rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #3B82F6;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }
        .metric-card {
            background-color: #F8FAFC;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #3B82F6;
            margin-bottom: 1rem;
        }
        .warning {
            background-color: #FEF3C7;
            border-left: 5px solid #F59E0B;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        .success {
            background-color: #D1FAE5;
            border-left: 5px solid #10B981;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)

# Load data
@st.cache_data
def load_forecast_data():
    """Load forecast data from Task 4 outputs"""
    try:
        # Load forecast tables
        forecast_path = project_root / "data" / "processed" / "comprehensive_forecasts_2025_2027.csv"
        summary_path = project_root / "reports" / "forecast_summary_2025_2027.csv"
        report_path = project_root / "reports" / "forecasting_final_report.csv"
        
        if forecast_path.exists():
            forecast_df = pd.read_csv(forecast_path)
        else:
            st.error("Forecast data not found. Please run Task 4 forecasting first.")
            return None, None, None
        
        if summary_path.exists():
            summary_df = pd.read_csv(summary_path)
        else:
            summary_df = None
            
        if report_path.exists():
            report_df = pd.read_csv(report_path)
        else:
            report_df = None
            
        return forecast_df, summary_df, report_df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

# Initialize session state
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'overview'

# Main app
def main():
    # Load CSS
    load_css()
    
    # Load data
    forecast_df, summary_df, report_df = load_forecast_data()
    
    if forecast_df is None:
        st.error("Failed to load forecast data. Please run Task 4 forecasting first.")
        return
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/71/Flag_of_Ethiopia.svg/320px-Flag_of_Ethiopia.svg.png", 
                width=150)
        
        st.markdown("## 📊 Navigation")
        
        # Navigation buttons
        view_options = {
            "🏠 Overview": "overview",
            "📈 Trends Analysis": "trends",
            "🔮 Forecasts": "forecasts",
            "💡 Insights & Recommendations": "insights"
        }
        
        for label, view in view_options.items():
            if st.button(label, key=view, use_container_width=True):
                st.session_state.current_view = view
        
        st.markdown("---")
        
        # Data info
        st.markdown("### 📁 Data Info")
        st.info(f"Forecast period: 2025-2027")
        st.info(f"Indicators: {len(forecast_df['indicator'].unique())}")
        
        # Download buttons
        st.markdown("---")
        st.markdown("### 📥 Download Data")
        
        if st.button("Download Forecasts CSV", use_container_width=True):
            csv = forecast_df.to_csv(index=False)
            st.download_button(
                label="Click to download",
                data=csv,
                file_name="ethiopia_fi_forecasts_2025_2027.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # About section
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.caption("""
        Developed by Selam Analytics for the National Bank of Ethiopia consortium.
        
        Data Sources:
        - Global Findex Database
        - National Bank of Ethiopia
        - Mobile Money Operators
        """)
    
    # Main content area
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<h1 class="main-header">🇪🇹 Ethiopia Financial Inclusion Dashboard</h1>', 
                   unsafe_allow_html=True)
        st.markdown("### Tracking Digital Financial Transformation 2025-2027")
    
    # Render selected view
    if st.session_state.current_view == 'overview':
        render_overview(forecast_df, summary_df, report_df)
    elif st.session_state.current_view == 'trends':
        render_trends(forecast_df, summary_df, report_df)
    elif st.session_state.current_view == 'forecasts':
        render_forecasts(forecast_df, summary_df, report_df)
    elif st.session_state.current_view == 'insights':
        render_insights(forecast_df, summary_df, report_df)

if __name__ == "__main__":
    main()