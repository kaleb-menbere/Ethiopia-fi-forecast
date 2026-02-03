# dashboard/components/forecasts.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

def render_forecasts(forecast_df, summary_df, report_df):
    """Render forecasts dashboard"""
    
    st.markdown('<h2 class="sub-header">🔮 Forecasts & Projections</h2>', unsafe_allow_html=True)
    
    # Year selector
    st.markdown("### Select Forecast Year")
    
    years = sorted(forecast_df['year'].unique().tolist())
    selected_year = st.select_slider(
        "Forecast Year",
        options=years,
        value=years[-1]  # Default to 2027
    )
    
    # Get data for selected year
    year_data = forecast_df[forecast_df['year'] == selected_year].copy()
    
    # 1. Forecast Summary Cards
    st.markdown("### Forecast Summary")
    
    cols = st.columns(len(year_data))
    
    for idx, (_, row) in enumerate(year_data.iterrows()):
        with cols[idx]:
            # Calculate change from historical
            historical = row['last_historical_value']
            forecast = row['event_adjusted_forecast']
            
            if pd.notna(historical) and pd.notna(forecast):
                change = forecast - historical
                change_pct = (change / historical * 100) if historical > 0 else 0
                
                st.metric(
                    label=row['indicator'],
                    value=f"{forecast:.1f}%",
                    delta=f"{change:+.1f}pp ({change_pct:+.1f}%)",
                    delta_color="normal" if change > 0 else "inverse"
                )
                
                # Confidence interval
                lower = row.get('event_adjusted_lower_80')
                upper = row.get('event_adjusted_upper_80')
                
                if pd.notna(lower) and pd.notna(upper):
                    st.caption(f"80% CI: {lower:.1f}% - {upper:.1f}%")
    
    # 2. Forecast Comparison Chart
    st.markdown("### Forecast Comparison Across Years")
    
    # Prepare comparison data
    comparison_data = []
    
    for indicator in forecast_df['indicator'].unique():
        indicator_data = forecast_df[forecast_df['indicator'] == indicator]
        
        # Historical
        if not indicator_data.empty:
            hist_year = indicator_data['last_historical_year'].iloc[0]
            hist_value = indicator_data['last_historical_value'].iloc[0]
            
            comparison_data.append({
                'Year': hist_year,
                'Value': hist_value,
                'Indicator': indicator,
                'Type': 'Historical'
            })
        
        # Forecasts
        for _, row in indicator_data.iterrows():
            comparison_data.append({
                'Year': row['year'],
                'Value': row['event_adjusted_forecast'],
                'Indicator': indicator,
                'Type': 'Forecast'
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Line chart
    fig = px.line(
        comparison_df,
        x='Year',
        y='Value',
        color='Indicator',
        line_dash='Type',
        markers=True,
        title=f"Forecast Trajectory: 2024-{years[-1]}"
    )
    
    # Add target line for account ownership
    if selected_year >= 2025:
        fig.add_hline(
            y=60,
            line_dash="dash",
            line_color="red",
            annotation_text="NFIS-II 2025 Target (60%)",
            annotation_position="bottom right"
        )
    
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Percentage (%)",
        template="plotly_white",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 3. Uncertainty Analysis
    st.markdown("### Uncertainty Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confidence intervals
        st.markdown("#### Confidence Intervals")
        
        ci_data = []
        for _, row in year_data.iterrows():
            if pd.notna(row.get('event_adjusted_lower_80')) and pd.notna(row.get('event_adjusted_upper_80')):
                ci_width = row['event_adjusted_upper_80'] - row['event_adjusted_lower_80']
                ci_data.append({
                    'Indicator': row['indicator'],
                    'Lower Bound': row['event_adjusted_lower_80'],
                    'Upper Bound': row['event_adjusted_upper_80'],
                    'Width': ci_width,
                    'Midpoint': (row['event_adjusted_lower_80'] + row['event_adjusted_upper_80']) / 2
                })
        
        if ci_data:
            ci_df = pd.DataFrame(ci_data)
            
            fig = go.Figure()
            
            for _, row in ci_df.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row['Indicator'], row['Indicator']],
                    y=[row['Lower Bound'], row['Upper Bound']],
                    mode='lines',
                    line=dict(width=10, color='lightblue'),
                    showlegend=False,
                    name=row['Indicator']
                ))
                
                fig.add_trace(go.Scatter(
                    x=[row['Indicator']],
                    y=[row['Midpoint']],
                    mode='markers',
                    marker=dict(size=12, color='blue'),
                    showlegend=False,
                    name=row['Indicator']
                ))
            
            fig.update_layout(
                title=f"80% Confidence Intervals for {selected_year}",
                yaxis_title="Percentage (%)",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Monte Carlo results if available
        st.markdown("#### Monte Carlo Simulation Results")
        
        mc_data = []
        for _, row in year_data.iterrows():
            if pd.notna(row.get('mc_10th_percentile')) and pd.notna(row.get('mc_90th_percentile')):
                mc_data.append({
                    'Indicator': row['indicator'],
                    '10th Percentile': row['mc_10th_percentile'],
                    '50th Percentile': row.get('mc_50th_percentile', row['event_adjusted_forecast']),
                    '90th Percentile': row['mc_90th_percentile'],
                    'Std Deviation': row.get('mc_std', 0)
                })
        
        if mc_data:
            mc_df = pd.DataFrame(mc_data)
            
            # Create box plot
            fig = go.Figure()
            
            for _, row in mc_df.iterrows():
                fig.add_trace(go.Box(
                    y=[row['10th_percentile'], row['50th_percentile'], row['90th_percentile']],
                    name=row['Indicator'],
                    boxpoints=False,
                    whiskerwidth=0.2,
                    marker_size=2,
                    line_width=2
                ))
            
            fig.update_layout(
                title=f"Monte Carlo Simulation: Percentile Ranges for {selected_year}",
                yaxis_title="Percentage (%)",
                template="plotly_white",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Monte Carlo simulation results not available in the data.")
    
    # 4. Event Impact Analysis
    st.markdown("### Event Impact Analysis")
    
    # Calculate event impacts
    impact_data = []
    
    for _, row in year_data.iterrows():
        baseline = row['baseline_forecast']
        event_adj = row['event_adjusted_forecast']
        
        if pd.notna(baseline) and pd.notna(event_adj):
            impact = event_adj - baseline
            
            impact_data.append({
                'Indicator': row['indicator'],
                'Baseline Forecast': baseline,
                'Event-Adjusted Forecast': event_adj,
                'Event Impact': impact,
                'Impact Percentage': (impact / baseline * 100) if baseline > 0 else 0
            })
    
    if impact_data:
        impact_df = pd.DataFrame(impact_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Impact bar chart
            fig = px.bar(
                impact_df,
                x='Indicator',
                y='Event Impact',
                title=f"Event Impact on {selected_year} Forecasts",
                labels={'Event Impact': 'Impact (percentage points)'},
                color='Event Impact',
                color_continuous_scale=['red', 'yellow', 'green']
            )
            
            fig.update_layout(
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Impact table
            st.markdown("#### Impact Details")
            
            display_df = impact_df.copy()
            display_df['Baseline Forecast'] = display_df['Baseline Forecast'].apply(lambda x: f"{x:.1f}%")
            display_df['Event-Adjusted Forecast'] = display_df['Event-Adjusted Forecast'].apply(lambda x: f"{x:.1f}%")
            display_df['Event Impact'] = display_df['Event Impact'].apply(lambda x: f"{x:+.1f}pp")
            display_df['Impact Percentage'] = display_df['Impact Percentage'].apply(lambda x: f"{x:+.1f}%")
            
            st.table(display_df[['Indicator', 'Baseline Forecast', 
                                'Event-Adjusted Forecast', 'Event Impact', 
                                'Impact Percentage']])
    
    # 5. Target Achievement Analysis
    if selected_year >= 2025:
        st.markdown("### Target Achievement Analysis")
        
        target_data = []
        
        for _, row in year_data.iterrows():
            if row['indicator'] == 'Account Ownership':
                forecast = row['event_adjusted_forecast']
                target_2025 = 60  # NFIS-II target
                
                if pd.notna(forecast):
                    gap = target_2025 - forecast
                    achievement_pct = (forecast / target_2025 * 100) if target_2025 > 0 else 0
                    
                    target_data.append({
                        'Year': selected_year,
                        'Forecast': forecast,
                        'Target': target_2025,
                        'Gap': gap,
                        'Achievement': achievement_pct,
                        'On Track': achievement_pct >= 100
                    })
        
        if target_data:
            target_df = pd.DataFrame(target_data)
            
            # Gauge chart for target achievement
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=target_df['Achievement'].iloc[0],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"NFIS-II Target Achievement ({selected_year})"},
                delta={'reference': 100, 'increasing': {'color': "green"}},
                gauge={
                    'axis': {'range': [None, 120]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 80], 'color': "lightgray"},
                        {'range': [80, 100], 'color': "gray"},
                        {'range': [100, 120], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 100
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Target details
            st.markdown(f"""
            **Target Details for {selected_year}:**
            - **Forecast:** {target_df['Forecast'].iloc[0]:.1f}%
            - **NFIS-II Target:** {target_df['Target'].iloc[0]}%
            - **Gap:** {target_df['Gap'].iloc[0]:.1f} percentage points
            - **Achievement:** {target_df['Achievement'].iloc[0]:.1f}% of target
            - **Status:** {'✅ ON TRACK' if target_df['On Track'].iloc[0] else '⚠️ NEEDS ACCELERATION'}
            """)
    
    # 6. Export Forecasts
    st.markdown("---")
    with st.expander("📤 Export Forecasts"):
        st.markdown("### Download Forecast Data")
        
        # Filter for download
        download_df = forecast_df[[
            'indicator', 'indicator_code', 'year',
            'last_historical_value', 'last_historical_year',
            'baseline_forecast', 'event_adjusted_forecast',
            'optimistic_scenario', 'pessimistic_scenario'
        ]].copy()
        
        # Convert to CSV
        csv = download_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download All Forecasts (CSV)",
            data=csv,
            file_name=f"ethiopia_fi_forecasts_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Preview
        st.markdown("#### Data Preview")
        st.dataframe(download_df.head(10))