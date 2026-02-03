# dashboard/utils/data_loader.py

import pandas as pd
import numpy as np
from pathlib import Path

def load_data():
    """Load all required data for the dashboard"""
    
    # Define paths
    project_root = Path(__file__).resolve().parents[2]
    
    data_files = {
        'forecasts': project_root / 'data' / 'processed' / 'comprehensive_forecasts_2025_2027.csv',
        'summary': project_root / 'reports' / 'forecast_summary_2025_2027.csv',
        'report': project_root / 'reports' / 'forecasting_final_report.csv',
        'enriched': project_root / 'data' / 'processed' / 'ethiopia_fi_enriched.csv'
    }
    
    data = {}
    
    for name, path in data_files.items():
        if path.exists():
            try:
                data[name] = pd.read_csv(path)
            except Exception as e:
                print(f"Error loading {name}: {e}")
                data[name] = None
        else:
            print(f"File not found: {path}")
            data[name] = None
    
    return data

def get_historical_data():
    """Get historical data for Ethiopia"""
    
    # Historical Findex data
    historical = {
        'Year': [2011, 2014, 2017, 2021, 2024],
        'Account Ownership': [14, 22, 35, 46, 49],
        'Digital Payments': [5, 12, 22, 32, 35],
        'Mobile Money': [0.5, 1.2, 4.7, 4.7, 9.45]
    }
    
    return pd.DataFrame(historical)

def get_event_data():
    """Get event impact data"""
    
    events = [
        {'year': 2021, 'event': 'Telebirr Launch', 'impact_access': 2.5, 'impact_usage': 3.0},
        {'year': 2022, 'event': 'Safaricom Entry', 'impact_access': 1.5, 'impact_usage': 2.0},
        {'year': 2023, 'event': 'M-Pesa Launch', 'impact_access': 2.0, 'impact_usage': 2.5},
        {'year': 2024, 'event': 'Interoperability', 'impact_access': 1.0, 'impact_usage': 1.5},
        {'year': 2025, 'event': 'Fayda ID Rollout', 'impact_access': 3.0, 'impact_usage': 2.0},
        {'year': 2026, 'event': 'CBDC Pilot', 'impact_access': 2.0, 'impact_usage': 3.0},
        {'year': 2027, 'event': 'Full Digital Economy', 'impact_access': 2.5, 'impact_usage': 3.5}
    ]
    
    return pd.DataFrame(events)