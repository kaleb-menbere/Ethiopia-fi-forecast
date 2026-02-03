# dashboard/components/insights.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def render_insights(forecast_df, summary_df, report_df):
    """Render insights and recommendations dashboard"""
    
    st.markdown('<h2 class="sub-header">💡 Insights & Recommendations</h2>', unsafe_allow_html=True)
    
    # 1. Key Findings
    st.markdown("### 🔍 Key Findings")
    
    findings = [
        {
            "title": "Growth Slowdown",
            "description": "Account ownership growth slowed from +11pp (2017-2021) to +3pp (2021-2024) despite mobile money expansion.",
            "impact": "High",
            "trend": "⚠️ Concerning"
        },
        {
            "title": "Mobile Money Acceleration",
            "description": "Mobile money accounts grew from 4.7% to 9.4% (2021-2024), showing strong momentum as an inclusion driver.",
            "impact": "High",
            "trend": "📈 Positive"
        },
        {
            "title": "2025 Target Gap",
            "description": "Current trajectory shows a 7.8pp gap to NFIS-II 2025 target of 60% account ownership.",
            "impact": "Critical",
            "trend": "🚨 Urgent"
        },
        {
            "title": "Event Impact Potential",
            "description": "Upcoming events (Fayda ID, CBDC, interoperability) could add +2-3pp to growth rates.",
            "impact": "Medium",
            "trend": "💡 Opportunity"
        },
        {
            "title": "Digital Payments Data Gap",
            "description": "Limited data on digital payment usage limits understanding of the 'usage' pillar.",
            "impact": "Medium",
            "trend": "📊 Data Quality"
        }
    ]
    
    for finding in findings:
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"#### {finding['title']}")
                st.markdown(f"{finding['description']}")
            
            with col2:
                # Color code based on impact
                if finding['impact'] == 'Critical':
                    color = '#EF4444'
                elif finding['impact'] == 'High':
                    color = '#F59E0B'
                elif finding['impact'] == 'Medium':
                    color = '#3B82F6'
                else:
                    color = '#6B7280'
                
                st.markdown(f"""
                <div style="text-align: center; padding: 0.5rem; background: {color}20; 
                            border: 2px solid {color}; border-radius: 5px; margin-top: 0.5rem;">
                <strong style="color: {color};">{finding['impact']} Impact</strong><br>
                <small>{finding['trend']}</small>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
    
    # 2. Recommendations
    st.markdown("### 🎯 Strategic Recommendations")
    
    recommendations = [
        {
            "category": "Urgent Acceleration",
            "items": [
                "Launch national digital inclusion campaign targeting +8pp growth in 2024-2025",
                "Accelerate Fayda digital ID rollout to enable instant account opening",
                "Implement regulatory sandbox for innovative fintech products"
            ]
        },
        {
            "category": "Mobile Money Focus",
            "items": [
                "Expand agent networks in rural areas (target: +50% coverage by 2025)",
                "Promote merchant QR code adoption for digital payments",
                "Develop use cases for government payments (salaries, social benefits)"
            ]
        },
        {
            "category": "Infrastructure & Policy",
            "items": [
                "Improve 4G coverage from current ~60% to 80% by 2025",
                "Reduce mobile data costs to below 2% of monthly income",
                "Implement open banking regulations to promote interoperability"
            ]
        },
        {
            "category": "Data & Monitoring",
            "items": [
                "Establish real-time financial inclusion dashboard with operator data",
                "Conduct annual financial inclusion survey (vs current 3-year cycle)",
                "Develop gender and regional disaggregated tracking"
            ]
        }
    ]
    
    for rec in recommendations:
        st.markdown(f"#### {rec['category']}")
        
        for item in rec['items']:
            st.markdown(f"- {item}")
        
        st.markdown("")
    
    # 3. Impact Priority Matrix
    st.markdown("### 🎯 Impact Priority Matrix")
    
    # Create impact-effort matrix
    initiatives = [
        {"name": "Fayda ID Rollout", "impact": 9, "effort": 7, "timeframe": "Short"},
        {"name": "Agent Network Expansion", "impact": 8, "effort": 6, "timeframe": "Medium"},
        {"name": "QR Merchant Adoption", "impact": 7, "effort": 5, "timeframe": "Short"},
        {"name": "CBDC Pilot", "impact": 6, "effort": 8, "timeframe": "Long"},
        {"name": "Open Banking Regulation", "impact": 8, "effort": 7, "timeframe": "Medium"},
        {"name": "Digital Literacy Campaign", "impact": 7, "effort": 6, "timeframe": "Medium"},
        {"name": "4G Coverage Expansion", "impact": 8, "effort": 9, "timeframe": "Long"}
    ]
    
    initiatives_df = pd.DataFrame(initiatives)
    
    # Create bubble chart
    fig = px.scatter(
        initiatives_df,
        x="effort",
        y="impact",
        size=[20] * len(initiatives_df),
        color="timeframe",
        hover_name="name",
        size_max=50,
        labels={"effort": "Implementation Effort (1-10)", "impact": "Expected Impact (1-10)"},
        title="Initiative Impact vs Effort Matrix"
    )
    
    # Add quadrant lines
    fig.add_hline(y=5, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=5, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Add quadrant labels
    fig.add_annotation(x=2.5, y=7.5, text="Quick Wins", showarrow=False, font=dict(size=12))
    fig.add_annotation(x=7.5, y=7.5, text="Major Projects", showarrow=False, font=dict(size=12))
    fig.add_annotation(x=2.5, y=2.5, text="Fill-Ins", showarrow=False, font=dict(size=12))
    fig.add_annotation(x=7.5, y=2.5, text="Resource Drains", showarrow=False, font=dict(size=12))
    
    fig.update_layout(
        template="plotly_white",
        height=500,
        xaxis_range=[0, 10],
        yaxis_range=[0, 10]
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 4. Roadmap Timeline
    st.markdown("### 🗓️ Implementation Roadmap")
    
    roadmap = [
        {"phase": "Phase 1 (2024)", "initiatives": ["Fayda ID acceleration", "QR code promotion", "Digital literacy pilot"]},
        {"phase": "Phase 2 (2025)", "initiatives": ["Agent network expansion", "Open banking implementation", "CBDC pilot launch"]},
        {"phase": "Phase 3 (2026)", "initiatives": ["4G coverage expansion", "Advanced fintech regulation", "Cross-border payments"]},
        {"phase": "Phase 4 (2027)", "initiatives": ["AI-based services", "Full interoperability", "Sustainability focus"]}
    ]
    
    # Create timeline visualization
    timeline_html = """
    <div style="position: relative; padding: 2rem 0; margin: 2rem 0;">
    """
    
    for i, phase in enumerate(roadmap):
        left_pos = i * 25  # 25% increments
        
        timeline_html += f"""
        <div style="position: absolute; left: {left_pos}%; transform: translateX(-50%); width: 20%;">
            <div style="background: #3B82F6; color: white; padding: 0.5rem; border-radius: 5px; text-align: center;">
                <strong>{phase['phase']}</strong>
            </div>
            <div style="background: #F8FAFC; padding: 0.5rem; margin-top: 0.5rem; border-radius: 5px; border-left: 3px solid #3B82F6;">
                <ul style="margin: 0; padding-left: 1rem; font-size: 0.9rem;">
        """
        
        for initiative in phase['initiatives']:
            timeline_html += f"<li>{initiative}</li>"
        
        timeline_html += """
                </ul>
            </div>
        </div>
        
        <div style="position: absolute; left: {left_pos}%; top: 120px; transform: translateX(-50%); 
                    width: 15px; height: 15px; background: #3B82F6; border-radius: 50%;">
        </div>
        """
    
    timeline_html += """
    <div style="position: absolute; left: 0; top: 127px; width: 100%; height: 2px; background: #E5E7EB;"></div>
    </div>
    """
    
    st.markdown(timeline_html, unsafe_allow_html=True)
    
    # 5. Success Metrics
    st.markdown("### 📊 Success Metrics & KPIs")
    
    metrics = [
        {"metric": "Account Ownership", "target_2025": "60%", "current": "49%", "gap": "11pp"},
        {"metric": "Mobile Money Penetration", "target_2025": "15%", "current": "9.4%", "gap": "5.6pp"},
        {"metric": "Digital Payment Users", "target_2025": "45%", "current": "~35%", "gap": "10pp"},
        {"metric": "Agent Network Density", "target_2025": "50/100k", "current": "~30/100k", "gap": "20/100k"},
        {"metric": "QR Merchants", "target_2025": "500k", "current": "~100k", "gap": "400k"},
        {"metric": "Gender Gap Reduction", "target_2025": "<5pp", "current": "~8pp", "gap": "3pp"}
    ]
    
    metrics_df = pd.DataFrame(metrics)
    
    # Create gauge for each metric
    cols = st.columns(3)
    
    for idx, (_, row) in enumerate(metrics_df.iterrows()):
        with cols[idx % 3]:
            # Calculate progress
            try:
                current = float(row['current'].replace('%', '').replace('~', '').replace('/100k', ''))
                target = float(row['target_2025'].replace('%', '').replace('/100k', ''))
                progress = min(100, (current / target * 100)) if target > 0 else 0
            except:
                progress = 0
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=progress,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': row['metric']},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 33], 'color': "lightgray"},
                        {'range': [33, 66], 'color': "gray"},
                        {'range': [66, 100], 'color': "lightgreen"}
                    ]
                }
            ))
            
            fig.update_layout(height=200, margin=dict(t=50, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption(f"Current: {row['current']} | Target: {row['target_2025']} | Gap: {row['gap']}")
    
    # 6. Risk Assessment
    st.markdown("### ⚠️ Risk Assessment")
    
    risks = [
        {"risk": "Economic Downturn", "probability": "Medium", "impact": "High", "mitigation": "Social protection digitization"},
        {"risk": "Regulatory Delays", "probability": "High", "impact": "Medium", "mitigation": "Stakeholder engagement"},
        {"risk": "Infrastructure Gaps", "probability": "Medium", "impact": "High", "mitigation": "Public-private partnerships"},
        {"risk": "Low Digital Literacy", "probability": "High", "impact": "Medium", "mitigation": "Training programs"},
        {"risk": "Market Concentration", "probability": "Low", "impact": "Medium", "mitigation": "Competition policy"}
    ]
    
    risks_df = pd.DataFrame(risks)
    
    # Create risk matrix
    fig = px.scatter(
        risks_df,
        x="probability",
        y="impact",
        size=[30] * len(risks_df),
        color="probability",
        hover_name="risk",
        hover_data=["mitigation"],
        labels={"probability": "Probability", "impact": "Impact"},
        title="Risk Assessment Matrix"
    )
    
    fig.update_layout(
        template="plotly_white",
        height=400,
        xaxis=dict(categoryorder="array", categoryarray=["Low", "Medium", "High"]),
        yaxis=dict(categoryorder="array", categoryarray=["Low", "Medium", "High"])
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 7. Stakeholder Actions
    st.markdown("### 👥 Stakeholder Actions")
    
    stakeholders = {
        "National Bank of Ethiopia": ["Set regulatory framework", "Monitor progress", "Coordinate stakeholders"],
        "Mobile Money Operators": ["Expand networks", "Reduce costs", "Develop products"],
        "Commercial Banks": ["Enable interoperability", "Develop digital products", "Support agents"],
        "Government Ministries": ["Digitize payments", "Promote digital ID", "Support infrastructure"],
        "Development Partners": ["Provide funding", "Share best practices", "Support capacity building"],
        "Fintech Companies": ["Innovate solutions", "Reach underserved", "Partner with incumbents"]
    }
    
    for stakeholder, actions in stakeholders.items():
        with st.expander(f"**{stakeholder}**"):
            for action in actions:
                st.markdown(f"• {action}")
    
    # 8. Download Insights Report
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 📄 Generate Insights Report")
        report_type = st.selectbox(
            "Report Type",
            ["Executive Summary", "Technical Analysis", "Implementation Plan", "Full Report"]
        )
    
    with col2:
        st.markdown("###")
        if st.button("📥 Generate Report", use_container_width=True):
            st.success(f"✅ {report_type} generated successfully!")
            st.info("In a production system, this would generate a PDF report with all insights and recommendations.")