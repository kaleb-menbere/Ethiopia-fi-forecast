# Forecasting Financial Inclusion in Ethiopia

This project builds a data-driven forecasting system to track and predict Ethiopia’s financial inclusion trajectory, focusing on **Access** and **Usage** as defined by the World Bank’s Global Findex framework.

The system supports policymakers, regulators, and financial service providers by quantifying historical trends, modeling the impact of key events, and projecting future inclusion outcomes under multiple scenarios.

---

## 📌 Project Objectives

- Understand and enrich Ethiopia’s financial inclusion data using a unified schema
- Analyze historical patterns and key drivers of financial inclusion
- Model the impact of policies, product launches, and infrastructure investments
- Forecast financial inclusion outcomes for **2025–2027**
- Present insights through an **interactive dashboard**

---

## 📊 Core Indicators

### Access — Account Ownership Rate  
Share of adults (15+) with an account at a financial institution or who used mobile money in the past 12 months.

### Usage — Digital Payment Adoption Rate  
Share of adults who made or received a digital payment in the past 12 months.

---

## 🗂 Project Structure

ethiopia-fi-forecast/
├── data/
│ ├── raw/
│ │ ├── ethiopia_fi_unified_data.csv
│ │ └── reference_codes.csv
│ └── processed/
│ ├── ethiopia_fi_enriched.csv
│ ├── impact_association_matrix.csv
│ ├── refined_impact_estimates.csv
│ └── confidence_matrix.csv
├── notebooks/
│ ├── task_1_data_enrichment.py
│ ├── task_2_eda.py
│ ├── task_3_impact_modeling.py
│ └── task_4_forecasting.py
├── dashboard/
│ └── app.py
├── reports/
│ ├── figures/
│ └── task*_final_reports.txt
├── models/
├── requirements.txt
└── README.md

---

## 🔍 Methodology Overview

### 1. Data Enrichment
- Integrated additional indicators from IMF FAS, GSMA, NBE, and operator reports
- Added policy, product launch, and infrastructure events
- Linked events to indicators using `impact_link` records

### 2. Exploratory Data Analysis
- Analyzed trends in account ownership and digital payments
- Investigated the 2021–2024 slowdown despite rapid mobile money growth
- Identified infrastructure and usage gaps
- Assessed data quality and coverage limitations

### 3. Event Impact Modeling
- Built an event–indicator association matrix
- Estimated direction, magnitude, and lag of event impacts
- Validated estimates against observed historical outcomes

### 4. Forecasting (Task 4)
- **Access**: Trend-based regression with event adjustments
- **Usage**: Proxy-based forecasting using mobile money and digital payment indicators
- Scenarios: **Optimistic**, **Base**, **Pessimistic**
- Forecast horizon: **2025–2027**
- Explicit uncertainty ranges and assumptions documented

---

## 📈 Key Forecast Insights (Summary)

- Account ownership is projected to continue growing, but at a slower pace than 2011–2021
- Digital payment usage grows faster than access, driven by P2P dominance and interoperability
- Policies and infrastructure investments have the largest marginal impact
- Wide uncertainty bands reflect limited historical survey data

---

## 🖥 Interactive Dashboard (Task 5)

The Streamlit dashboard allows stakeholders to:

- View current financial inclusion metrics
- Explore historical trends interactively
- Compare optimistic, base, and pessimistic forecasts
- Track progress toward the **60% inclusion target**
- Download forecast data for further analysis

### Dashboard Sections
- **Overview**: Key metrics and highlights
- **Trends**: Interactive time-series plots
- **Forecasts**: Scenario-based projections with confidence intervals
- **Inclusion Targets**: Progress visualization toward policy goals

---

## ▶️ How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
### 2. Run Forecasting Script
```bash
python notebooks/task_4_forecasting.py
```
### 3. Launch Dashboard
```bash
streamlit run dashboard/app.py
```
## ⚠️ Limitations

Sparse Global Findex survey points (every 3 years)

Usage forecasts rely on proxy indicators

Event impact magnitudes involve informed assumptions

Forecasts should be interpreted as directional, not exact predictions

## 📚 References

World Bank Global Findex Database

IMF Financial Access Survey

GSMA State of the Industry Reports

National Bank of Ethiopia publications

CGAP and World Bank financial inclusion research