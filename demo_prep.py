"""
Quick Demo Script for Supply Chain Resilience Engine
Run this to prepare for presentation
"""

import sys
import time
import webbrowser
from pathlib import Path

sys.path.insert(0, '/home/utkarsh/Documents/INDCON ')

def prepare_demo():
    """Prepare demo environment for presentation"""
    print("🚀 PREPARING SUPPLY CHAIN RESILIENCE ENGINE DEMO")
    print("=" * 60)
    
    # Test model loading
    print("\n1. Testing model loading...")
    try:
        from model import engine, data
        print("   ✅ Models loaded successfully")
        print(f"   📊 Dataset: {data.shape[0]} SKUs, {data.shape[1]} features")
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        return False
    
    # Test edge cases
    print("\n2. Testing edge cases...")
    test_cases = [
        ("Critical A-X SKU", {
            "sku_class": "A-X", "region": "APAC", "demand_last_week": 250.0,
            "demand_trend": 0.15, "seasonality_index": 1.3, "demand_volatility": 2.0,
            "supplier_reliability": 0.55, "avg_transport_delay": 8, "supplier_tier": 3,
            "lead_time": 21, "distance": 3000, "is_sole_source": 1,
            "backup_supplier_count": 0, "geopolitical_risk": 0.7,
            "current_inventory": 150, "incoming_supply": 100, "pipeline_inventory": 200,
            "bullwhip_coeff": 2.5, "order_volatility": 0.5, "order_book_value": 600
        }),
        ("Stable C-Z SKU", {
            "sku_class": "C-Z", "region": "NA", "demand_last_week": 50.0,
            "demand_trend": -0.05, "seasonality_index": 0.9, "demand_volatility": 0.5,
            "supplier_reliability": 0.95, "avg_transport_delay": 1, "supplier_tier": 1,
            "lead_time": 5, "distance": 200, "is_sole_source": 0,
            "backup_supplier_count": 3, "geopolitical_risk": 0.1,
            "current_inventory": 800, "incoming_supply": 400, "pipeline_inventory": 300,
            "bullwhip_coeff": 1.2, "order_volatility": 0.1, "order_book_value": 200
        })
    ]
    
    import pandas as pd
    for name, sample in test_cases:
        result = engine.assess_disruption_risk(pd.DataFrame([sample]))
        score = result['composite_risk_score']
        risk_class = result['risk_class']
        print(f"   {name:20s}: {score:.3f} ({risk_class})")
    
    print("\n3. Checking demo files...")
    demo_files = [
        "streamlit_app.py",
        "api.py", 
        "model.py",
        "model_test.py",
        "PRESENTATION.md",
        "README.md"
    ]
    
    for file in demo_files:
        if Path(file).exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} missing")
    
    print("\n4. Demo URLs:")
    print("   🌐 Streamlit Dashboard: http://localhost:8502")
    print("   🔌 Flask API: http://127.0.0.1:5000")
    
    print("\n5. Quick Demo Commands:")
    print("   # Start Streamlit:")
    print("   venv/bin/streamlit run streamlit_app.py")
    print()
    print("   # Start API (optional):")
    print("   venv/bin/python api.py")
    print()
    print("   # Test edge cases:")
    print("   venv/bin/python model_test.py")
    
    print("\n" + "=" * 60)
    print("✅ DEMO PREPARATION COMPLETE!")
    print("📋 Follow the script in PRESENTATION.md")
    print("🎯 Focus on: Critical risk detection, early warnings, what-if analysis")
    print("⏱️  Total demo time: 5-7 minutes")
    
    return True

if __name__ == "__main__":
    prepare_demo()
