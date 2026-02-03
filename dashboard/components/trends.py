# dashboard/components/trends.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def render_trends(forecast_df, summary_df, report_df):
    """Render trends analysis dashboard"""
    
    st.markdown('<h2 class="sub-header">📈 Trends Analysis</h2>', unsafe_allow_html=True)
    
    # Filter controls
    st.markdown("### Data Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        indicators = forecast_df['indicator'].unique().tolist()
        selected_indicators = st.multiselect(
            "Select Indicators",
            options=indicators,
            default=indicators[:2] if len(indicators) >= 2 else indicators
        )
    
    with col2:
        years = sorted(forecast_df['year'].unique().tolist())
        selected_years = st.multiselect(
            "Select Years",
            options=years,
            default=years
        )
    
    with col3:
        forecast_types = ['baseline_forecast', 'event_adjusted_forecast', 
                         'optimistic_scenario', 'pessimistic_scenario']
        selected_forecast = st.selectbox(
            "Forecast Type",
            options=['Event-Adjusted Forecast', 'Baseline Forecast', 
                    'Scenario Comparison', 'All Forecasts'],
            index=0
        )
    
    # Filter data
    filtered_df = forecast_df[
        forecast_df['indicator'].isin(selected_indicators) &
        forecast_df['year'].isin(selected_years)
    ].copy()
    
    if filtered_df.empty:
        st.warning("No data available for selected filters.")
        return
    
    # 1. Time Series Trend Chart
    st.markdown("### Time Series Trends")
    
    # Prepare data for plotting
    plot_data = []
    
    for indicator in selected_indicators:
        indicator_data = filtered_df[filtered_df['indicator'] == indicator]
        
        # Historical data
        if not indicator_data.empty:
            hist_year = indicator_data['last_historical_year'].iloc[0]
            hist_value = indicator_data['last_historical_value'].iloc[0]
            
            plot_data.append({
                'Year': hist_year,
                'Value': hist_value,
                'Indicator': indicator,
                'Type': 'Historical'
            })
        
        # Forecast data
        for _, row in indicator_data.iterrows():
            if selected_forecast == 'Event-Adjusted Forecast':
                plot_data.append({
                    'Year': row['year'],
                    'Value': row['event_adjusted_forecast'],
                    'Indicator': indicator,
                    'Type': 'Forecast'
                })
            elif selected_forecast == 'Baseline Forecast':
                plot_data.append({
                    'Year': row['year'],
                    'Value': row['baseline_forecast'],
                    'Indicator': indicator,
                    'Type': 'Forecast'
                })
            elif selected_forecast == 'All Forecasts':
                # Add all forecast types
                for forecast_type in ['baseline_forecast', 'event_adjusted_forecast']:
                    if pd.notna(row[forecast_type]):
                        plot_data.append({
                            'Year': row['year'],
                            'Value': row[forecast_type],
                            'Indicator': f"{indicator} ({forecast_type.replace('_', ' ')})",
                            'Type': 'Forecast'
                        })
    
    plot_df = pd.DataFrame(plot_data)
    
    if not plot_df.empty:
        # Create interactive plot
        fig = px.line(
            plot_df, 
            x='Year', 
            y='Value', 
            color='Indicator',
            line_dash='Type',
            title="Financial Inclusion Trends & Forecasts",
            markers=True,
            line_shape='linear'
        )
        
        # Add confidence intervals if available
        if selected_forecast in ['Event-Adjusted Forecast', 'Baseline Forecast']:
            for indicator in selected_indicators:
                indicator_data = filtered_df[filtered_df['indicator'] == indicator]
                
                for _, row in indicator_data.iterrows():
                    if selected_forecast == 'Event-Adjusted Forecast':
                        lower = row.get('event_adjusted_lower_80')
                        upper = row.get('event_adjusted_upper_80')
                    else:
                        lower = row.get('baseline_lower_80')
                        upper = row.get('baseline_upper_80')
                    
                    if pd.notna(lower) and pd.notna(upper):
                        fig.add_trace(go.Scatter(
                            x=[row['year'], row['year']],
                            y=[lower, upper],
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=[row['year']],
                            y=[(lower + upper) / 2],
                            mode='markers',
                            marker=dict(
                                size=10,
                                color='rgba(255,255,255,0)',
                                line=dict(width=2, color='rgba(0,0,0,0.1)')
                            ),
                            error_y=dict(
                                type='data',
                                symmetric=True,
                                array=[(upper - lower) / 2],
                                arrayminus=[(upper - lower) / 2],
                                thickness=1.5,
                                width=3
                            ),
                            showlegend=False,
                            name=f"{indicator} CI"
                        ))
        
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Percentage (%)",
            hovermode='x unified',
            template="plotly_white",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # 2. Growth Rate Analysis
    st.markdown("### Growth Rate Analysis")
    
    growth_data = []
    
    for indicator in selected_indicators:
        indicator_data = filtered_df[filtered_df['indicator'] == indicator].sort_values('year')
        
        if len(indicator_data) > 1:
            # Calculate growth rates
            for i in range(1, len(indicator_data)):
                year1 = indicator_data.iloc[i-1]['year']
                year2 = indicator_data.iloc[i]['year']
                
                if selected_forecast == 'Event-Adjusted Forecast':
                    value1 = indicator_data.iloc[i-1]['event_adjusted_forecast']
                    value2 = indicator_data.iloc[i]['event_adjusted_forecast']
                else:
                    value1 = indicator_data.iloc[i-1]['baseline_forecast']
                    value2 = indicator_data.iloc[i]['baseline_forecast']
                
                if pd.notna(value1) and pd.notna(value2):
                    growth_rate = ((value2 - value1) / (year2 - year1))
                    
                    growth_data.append({
                        'Period': f"{year1}-{year2}",
                        'Growth Rate': growth_rate,
                        'Indicator': indicator,
                        'Years': year2 - year1
                    })
    
    if growth_data:
        growth_df = pd.DataFrame(growth_data)
        
        # Bar chart
        fig = px.bar(
            growth_df,
            x='Period',
            y='Growth Rate',
            color='Indicator',
            barmode='group',
            title="Annual Growth Rates by Period",
            labels={'Growth Rate': 'Growth (percentage points/year)'}
        )
        
        fig.update_layout(
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Growth summary table
        st.markdown("#### Growth Summary")
        
        summary_stats = []
        for indicator in selected_indicators:
            indicator_growth = growth_df[growth_df['Indicator'] == indicator]
            if not indicator_growth.empty:
                avg_growth = indicator_growth['Growth Rate'].mean()
                total_growth = indicator_growth['Growth Rate'].sum()
                
                summary_stats.append({
                    'Indicator': indicator,
                    'Avg Annual Growth': f"{avg_growth:.2f} pp",
                    'Total Projected Growth': f"{total_growth:.2f} pp",
                    'Growth Periods': len(indicator_growth)
                })
        
        if summary_stats:
            st.table(pd.DataFrame(summary_stats))
    
    # 3. Scenario Comparison
    if 'Scenario Comparison' in selected_forecast:
        st.markdown("### Scenario Comparison")
        
        scenario_data = []
        
        for indicator in selected_indicators:
            indicator_data = filtered_df[filtered_df['indicator'] == indicator]
            
            for _, row in indicator_data.iterrows():
                if pd.notna(row.get('optimistic_scenario')) and pd.notna(row.get('pessimistic_scenario')):
                    # Add optimistic
                    scenario_data.append({
                        'Year': row['year'],
                        'Value': row['optimistic_scenario'],
                        'Indicator': indicator,
                        'Scenario': 'Optimistic'
                    })
                    
                    # Add baseline/event-adjusted
                    scenario_data.append({
                        'Year': row['year'],
                        'Value': row['event_adjusted_forecast'] if pd.notna(row.get('event_adjusted_forecast')) else row['baseline_forecast'],
                        'Indicator': indicator,
                        'Scenario': 'Base'
                    })
                    
                    # Add pessimistic
                    scenario_data.append({
                        'Year': row['year'],
                        'Value': row['pessimistic_scenario'],
                        'Indicator': indicator,
                        'Scenario': 'Pessimistic'
                    })
        
        if scenario_data:
            scenario_df = pd.DataFrame(scenario_data)
            
            fig = px.line(
                scenario_df,
                x='Year',
                y='Value',
                color='Indicator',
                line_dash='Scenario',
                title="Scenario Analysis: Optimistic vs Pessimistic",
                markers=True
            )
            
            # Add shaded area for scenario range
            for indicator in selected_indicators:
                ind_scenarios = scenario_df[scenario_df['Indicator'] == indicator]
                
                if not ind_scenarios.empty:
                    # Get min and max for each year
                    years = ind_scenarios['Year'].unique()
                    
                    for year in years:
                        year_data = ind_scenarios[ind_scenarios['Year'] == year]
                        if len(year_data) >= 3:  # Has all scenarios
                            min_val = year_data['Value'].min()
                            max_val = year_data['Value'].max()
                            
                            fig.add_trace(go.Scatter(
                                x=[year, year],
                                y=[min_val, max_val],
                                mode='lines',
                                line=dict(width=8, color='rgba(128,128,128,0.2)'),
                                showlegend=False,
                                hoverinfo='skip'
                            ))
            
            fig.update_layout(
                template="plotly_white",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Scenario impact table
            st.markdown("#### Scenario Impact Analysis")
            
            impact_data = []
            for indicator in selected_indicators:
                indicator_data = filtered_df[filtered_df['indicator'] == indicator]
                
                for _, row in indicator_data.iterrows():
                    if pd.notna(row.get('optimistic_scenario')) and pd.notna(row.get('pessimistic_scenario')):
                        base_value = row['event_adjusted_forecast'] if pd.notna(row.get('event_adjusted_forecast')) else row['baseline_forecast']
                        
                        if pd.notna(base_value):
                            optimistic_impact = row['optimistic_scenario'] - base_value
                            pessimistic_impact = row['pessimistic_scenario'] - base_value
                            range_width = row['optimistic_scenario'] - row['pessimistic_scenario']
                            
                            impact_data.append({
                                'Year': row['year'],
                                'Indicator': indicator,
                                'Base Forecast': f"{base_value:.1f}%",
                                'Optimistic': f"{row['optimistic_scenario']:.1f}%",
                                'Pessimistic': f"{row['pessimistic_scenario']:.1f}%",
                                'Optimistic Impact': f"+{optimistic_impact:.1f}pp",
                                'Pessimistic Impact': f"{pessimistic_impact:.1f}pp",
                                'Uncertainty Range': f"{range_width:.1f}pp"
                            })
            
            if impact_data:
                st.table(pd.DataFrame(impact_data))
    
    # 4. Data Quality Information
    st.markdown("---")
    with st.expander("📊 Data Quality Information"):
        st.markdown("""
        **Data Sources & Confidence Levels:**
        
        | Indicator | Data Points | Time Period | Confidence | Notes |
        |-----------|-------------|-------------|------------|-------|
        | Account Ownership | 5 | 2011-2024 | High | Global Findex survey data |
        | Mobile Money | 2 | 2021-2024 | Medium | Operator reports, estimated |
        | Digital Payments | Limited | 2011-2024 | Low | Estimated from account data |
        
        **Limitations:**
        - Digital payment data is sparse and estimated
        - Mobile money data only available from 2021
        - Historical data points are 3 years apart
        - Event impacts are modeled estimates
        
        **Recommendations for Improvement:**
        1. Collect more frequent survey data
        2. Integrate operator transaction data
        3. Establish real-time monitoring systems
        4. Improve gender and regional disaggregation
        """)