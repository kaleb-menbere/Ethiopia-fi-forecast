# dashboard/components/overview.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

def render_overview(forecast_df, summary_df, report_df):
    """Render overview dashboard"""
    
    st.markdown('<h2 class="sub-header">📊 Executive Summary</h2>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Current account ownership
        acc_data = forecast_df[forecast_df['indicator'] == 'Account Ownership']
        current_acc = acc_data['last_historical_value'].iloc[0] if not acc_data.empty else None
        if current_acc:
            st.metric(
                label="Current Account Ownership",
                value=f"{current_acc:.1f}%",
                delta=f"{current_acc - 46:.1f}% from 2021"
            )
    
    with col2:
        # 2027 forecast
        if not acc_data.empty:
            forecast_2027 = acc_data[acc_data['year'] == 2027]['event_adjusted_forecast'].iloc[0]
            st.metric(
                label="Projected 2027",
                value=f"{forecast_2027:.1f}%",
                delta=f"+{forecast_2027 - current_acc:.1f}pp"
            )
    
    with col3:
        # Target gap
        if not acc_data.empty:
            forecast_2025 = acc_data[acc_data['year'] == 2025]['event_adjusted_forecast'].iloc[0]
            target_gap = 60 - forecast_2025
            st.metric(
                label="Gap to 2025 Target",
                value=f"{target_gap:.1f}pp",
                delta_color="inverse" if target_gap > 0 else "normal"
            )
    
    with col4:
        # Required growth rate
        if current_acc:
            required_growth = 60 - current_acc
            st.metric(
                label="Required 1-year Growth",
                value=f"{required_growth:.1f}pp",
                help="Growth needed from 2024 to meet 2025 target"
            )
    
    # Warning if off-track
    if not acc_data.empty:
        forecast_2025 = acc_data[acc_data['year'] == 2025]['event_adjusted_forecast'].iloc[0]
        if forecast_2025 < 60:
            st.markdown("""
            <div class="warning">
            <strong>⚠️ Attention Needed:</strong> Ethiopia is currently <strong>not on track</strong> 
            to meet the NFIS-II 2025 target of 60% account ownership. 
            Significant acceleration is required.
            </div>
            """, unsafe_allow_html=True)
    
    # Progress bars
    st.markdown("### Progress Toward Targets")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Account ownership progress
        if not acc_data.empty:
            current_acc = acc_data['last_historical_value'].iloc[0]
            target_2025 = 60
            
            progress_pct = min(100, (current_acc / target_2025) * 100)
            
            st.markdown(f"**Account Ownership: {current_acc:.1f}% / {target_2025}%**")
            st.progress(progress_pct / 100)
            
            # Projected progress
            for year in [2025, 2026, 2027]:
                forecast = acc_data[acc_data['year'] == year]['event_adjusted_forecast']
                if not forecast.empty:
                    forecast_value = forecast.iloc[0]
                    forecast_pct = min(100, (forecast_value / target_2025) * 100)
                    st.caption(f"Projected {year}: {forecast_value:.1f}% ({forecast_pct:.0f}% of target)")
    
    with col2:
        # Mobile money progress
        mm_data = forecast_df[forecast_df['indicator'] == 'Mobile Money Accounts']
        if not mm_data.empty:
            current_mm = mm_data['last_historical_value'].iloc[0]
            # Assuming target of 20% by 2027 (hypothetical)
            mm_target = 20
            
            mm_progress = min(100, (current_mm / mm_target) * 100)
            
            st.markdown(f"**Mobile Money: {current_mm:.1f}% / {mm_target}%**")
            st.progress(mm_progress / 100)
            
            # Projected progress
            for year in [2025, 2026, 2027]:
                forecast = mm_data[mm_data['year'] == year]['event_adjusted_forecast']
                if not forecast.empty:
                    forecast_value = forecast.iloc[0]
                    forecast_pct = min(100, (forecast_value / mm_target) * 100)
                    st.caption(f"Projected {year}: {forecast_value:.1f}% ({forecast_pct:.0f}% of target)")
    
    # Quick insights
    st.markdown("### 🚀 Quick Insights")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown("""
        <div class="metric-card">
        <h4>📈 Growth Trajectory</h4>
        <p>Account ownership grew from 46% (2021) to 49% (2024), showing a slowdown. 
        Mobile money accelerated from 4.7% to 9.4% in the same period.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
        <h4>🎯 Target Analysis</h4>
        <p>To reach 60% by 2025, Ethiopia needs 11pp growth in 1 year vs 3pp in last 3 years. 
        This requires 3.7x acceleration.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with insight_col2:
        st.markdown("""
        <div class="metric-card">
        <h4>💡 Event Impact</h4>
        <p>Upcoming events (Fayda ID, CBDC, interoperability) could add +2-3pp to growth rates, 
        but still insufficient for 2025 target.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
        <h4>📱 Mobile Momentum</h4>
        <p>Mobile money is the fastest-growing channel, projected to reach 16.2% by 2027. 
        Key to accelerating inclusion.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Historical context
    st.markdown("### 📜 Historical Context")
    
    # Create a simple historical timeline
    historical_data = {
        'Year': [2011, 2014, 2017, 2021, 2024],
        'Account Ownership': [14, 22, 35, 46, 49],
        'Digital Payments': [5, 12, 22, 32, 35]
    }
    
    hist_df = pd.DataFrame(historical_data)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hist_df['Year'],
        y=hist_df['Account Ownership'],
        mode='lines+markers',
        name='Account Ownership',
        line=dict(color='#3B82F6', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=hist_df['Year'],
        y=hist_df['Digital Payments'],
        mode='lines+markers',
        name='Digital Payments',
        line=dict(color='#10B981', width=3)
    ))
    
    # Add event markers
    events = {
        2021: 'Telebirr Launch',
        2022: 'Safaricom Entry',
        2023: 'M-Pesa Launch'
    }
    
    for year, event in events.items():
        fig.add_annotation(
            x=year,
            y=hist_df[hist_df['Year'] == year]['Account Ownership'].values[0] if year in hist_df['Year'].values else 50,
            text=event,
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40
        )
    
    fig.update_layout(
        title="Historical Financial Inclusion Trends",
        xaxis_title="Year",
        yaxis_title="Percentage (%)",
        template="plotly_white",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Key performance indicators
    st.markdown("### 📊 Key Performance Indicators")
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    with kpi_col1:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #F0F9FF; border-radius: 10px;">
        <h3 style="margin: 0; color: #1E40AF;">3.7x</h3>
        <p style="margin: 0; font-size: 0.9rem;">Required Acceleration</p>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi_col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #F0F9FF; border-radius: 10px;">
        <h3 style="margin: 0; color: #1E40AF;">+2.0pp</h3>
        <p style="margin: 0; font-size: 0.9rem;">Event Impact Potential</p>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi_col3:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #F0F9FF; border-radius: 10px;">
        <h3 style="margin: 0; color: #1E40AF;">7.8pp</h3>
        <p style="margin: 0; font-size: 0.9rem;">2025 Target Gap</p>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi_col4:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #F0F9FF; border-radius: 10px;">
        <h3 style="margin: 0; color: #1E40AF;">2.47pp/yr</h3>
        <p style="margin: 0; font-size: 0.9rem;">Projected Growth Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Data quality note
    st.markdown("---")
    st.info("""
    **Data Note:** Digital payment usage data is limited. The dashboard focuses on account ownership 
    and mobile money as primary indicators. Consider enriching with additional data sources for 
    comprehensive usage analysis.
    """)