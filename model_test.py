"""
Model Testing Suite - Test edge cases and identify improvement opportunities
"""

import sys
import pandas as pd
import numpy as np

sys.path.insert(0, '/home/utkarsh/Documents/INDCON ')
from model import engine

def test_scenario(name, sample_dict, expected_behavior=""):
    """Run a test scenario and print results"""
    sample_df = pd.DataFrame([sample_dict])
    result = engine.assess_disruption_risk(sample_df)
    
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    print(f"Expected: {expected_behavior}")
    print(f"\nRisk Score: {result['composite_risk_score']:.3f} ({result['risk_class']})")
    print(f"Demand Risk: {result['demand_risk']:.2%}")
    print(f"Delay Probability: {result['delay_probability']:.2%}")
    print(f"Stockout Probability: {result['inventory_analysis']['stockout_probability']:.2%}")
    print(f"Days of Supply: {result['inventory_analysis']['days_of_supply']:.1f}")
    print(f"\nAction: {result['action']}")
    print(f"Details: {result['action_details'][:80]}...")
    return result

# Test Case 1: Critical A-X SKU with sole source, low inventory
print("\n" + "="*70)
print("EDGE CASE TEST SUITE - SUPPLY CHAIN RESILIENCE ENGINE")
print("="*70)

test_scenario(
    "1. Critical A-X SKU + Sole Source + Low Inventory",
    {
        "sku_class": "A-X", "region": "APAC", "demand_last_week": 250.0,
        "demand_trend": 0.15, "seasonality_index": 1.3, "demand_volatility": 2.0,
        "supplier_reliability": 0.55, "avg_transport_delay": 8, "supplier_tier": 3,
        "lead_time": 21, "distance": 3000, "is_sole_source": 1,
        "backup_supplier_count": 0, "geopolitical_risk": 0.7,
        "current_inventory": 150, "incoming_supply": 100, "pipeline_inventory": 200,
        "bullwhip_coeff": 2.5, "order_volatility": 0.5, "order_book_value": 600
    },
    "Expected: CRITICAL risk (high demand, sole source, low inventory, geo risk)"
)

# Test Case 2: Stable C-Z SKU with multiple suppliers
result2 = test_scenario(
    "2. Stable C-Z SKU + Multiple Suppliers + High Inventory",
    {
        "sku_class": "C-Z", "region": "NA", "demand_last_week": 50.0,
        "demand_trend": -0.05, "seasonality_index": 0.9, "demand_volatility": 0.5,
        "supplier_reliability": 0.95, "avg_transport_delay": 1, "supplier_tier": 1,
        "lead_time": 5, "distance": 200, "is_sole_source": 0,
        "backup_supplier_count": 3, "geopolitical_risk": 0.1,
        "current_inventory": 800, "incoming_supply": 400, "pipeline_inventory": 300,
        "bullwhip_coeff": 1.2, "order_volatility": 0.1, "order_book_value": 200
    },
    "Expected: LOW risk (stable demand, reliable suppliers, high inventory)"
)

# Test Case 3: Medium B-Y with borderline risk (should trigger early warning)
result3 = test_scenario(
    "3. Borderline Case - B-Y SKU (Early Warning Test)",
    {
        "sku_class": "B-Y", "region": "EU", "demand_last_week": 120.0,
        "demand_trend": 0.08, "seasonality_index": 1.15, "demand_volatility": 1.2,
        "supplier_reliability": 0.70, "avg_transport_delay": 4, "supplier_tier": 2,
        "lead_time": 10, "distance": 800, "is_sole_source": 0,
        "backup_supplier_count": 1, "geopolitical_risk": 0.3,
        "current_inventory": 250, "incoming_supply": 150, "pipeline_inventory": 180,
        "bullwhip_coeff": 1.6, "order_volatility": 0.25, "order_book_value": 350
    },
    "Expected: MEDIUM or HIGH (should trigger early warning if 0.35-0.45)"
)

# Test Case 4: Extreme values - zero inventory
result4 = test_scenario(
    "4. Zero Inventory (Stockout Imminent)",
    {
        "sku_class": "A-Y", "region": "LATAM", "demand_last_week": 180.0,
        "demand_trend": 0.10, "seasonality_index": 1.0, "demand_volatility": 1.5,
        "supplier_reliability": 0.80, "avg_transport_delay": 5, "supplier_tier": 2,
        "lead_time": 12, "distance": 1500, "is_sole_source": 0,
        "backup_supplier_count": 2, "geopolitical_risk": 0.4,
        "current_inventory": 0, "incoming_supply": 50, "pipeline_inventory": 100,
        "bullwhip_coeff": 1.8, "order_volatility": 0.3, "order_book_value": 400
    },
    "Expected: CRITICAL (zero inventory with ongoing demand)"
)

# Test Case 5: Perfect supplier scenario
result5 = test_scenario(
    "5. Perfect Supplier + Perfect Forecast",
    {
        "sku_class": "B-X", "region": "NA", "demand_last_week": 100.0,
        "demand_trend": 0.0, "seasonality_index": 1.0, "demand_volatility": 0.3,
        "supplier_reliability": 1.0, "avg_transport_delay": 0, "supplier_tier": 1,
        "lead_time": 3, "distance": 100, "is_sole_source": 0,
        "backup_supplier_count": 5, "geopolitical_risk": 0.0,
        "current_inventory": 500, "incoming_supply": 300, "pipeline_inventory": 200,
        "bullwhip_coeff": 1.0, "order_volatility": 0.05, "order_book_value": 300
    },
    "Expected: LOW risk (perfect conditions)"
)

# Test Case 6: Geopolitical hotspot
result6 = test_scenario(
    "6. Geopolitical Hotspot (High Geo Risk)",
    {
        "sku_class": "A-Z", "region": "APAC", "demand_last_week": 200.0,
        "demand_trend": 0.05, "seasonality_index": 1.0, "demand_volatility": 1.0,
        "supplier_reliability": 0.60, "avg_transport_delay": 6, "supplier_tier": 3,
        "lead_time": 18, "distance": 2500, "is_sole_source": 1,
        "backup_supplier_count": 0, "geopolitical_risk": 0.9,
        "current_inventory": 300, "incoming_supply": 200, "pipeline_inventory": 150,
        "bullwhip_coeff": 2.0, "order_volatility": 0.3, "order_book_value": 450
    },
    "Expected: HIGH or CRITICAL (90% geo risk with sole source)"
)

# Test Case 7: Negative trend (declining demand)
result7 = test_scenario(
    "7. Declining Demand (Negative Trend)",
    {
        "sku_class": "C-X", "region": "EU", "demand_last_week": 80.0,
        "demand_trend": -0.20, "seasonality_index": 0.8, "demand_volatility": 0.8,
        "supplier_reliability": 0.75, "avg_transport_delay": 3, "supplier_tier": 2,
        "lead_time": 8, "distance": 600, "is_sole_source": 0,
        "backup_supplier_count": 2, "geopolitical_risk": 0.2,
        "current_inventory": 400, "incoming_supply": 150, "pipeline_inventory": 100,
        "bullwhip_coeff": 1.4, "order_volatility": 0.2, "order_book_value": 250
    },
    "Expected: LOW (declining demand reduces risk despite moderate inventory)"
)

# Test Case 8: Bullwhip effect extreme
result8 = test_scenario(
    "8. Extreme Bullwhip Effect",
    {
        "sku_class": "A-Y", "region": "NA", "demand_last_week": 300.0,
        "demand_trend": 0.25, "seasonality_index": 1.4, "demand_volatility": 2.5,
        "supplier_reliability": 0.65, "avg_transport_delay": 7, "supplier_tier": 3,
        "lead_time": 20, "distance": 2000, "is_sole_source": 1,
        "backup_supplier_count": 0, "geopolitical_risk": 0.3,
        "current_inventory": 200, "incoming_supply": 150, "pipeline_inventory": 250,
        "bullwhip_coeff": 4.0, "order_volatility": 0.8, "order_book_value": 700
    },
    "Expected: CRITICAL (extreme bullwhip, high volatility, sole source)"
)

# Summary
print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)

results = [result2, result3, result4, result5, result6, result7, result8]
test_names = [
    "Stable C-Z", "Borderline B-Y", "Zero Inventory", "Perfect Supplier",
    "Geo Hotspot", "Declining Demand", "Extreme Bullwhip"
]

print("\nRisk Score Distribution:")
for name, res in zip(test_names, results):
    score = res['composite_risk_score']
    risk_class = res['risk_class']
    print(f"  {name:20s}: {score:.3f} ({risk_class})")

# Analyze score spread
scores = [r['composite_risk_score'] for r in results]
print(f"\nScore Range: {min(scores):.3f} - {max(scores):.3f}")
print(f"Average Score: {np.mean(scores):.3f}")
print(f"Score Spread (Std Dev): {np.std(scores):.3f}")

if np.std(scores) < 0.15:
    print("\n⚠️ WARNING: Low score variance detected!")
    print("   Model may not be discriminating between scenarios effectively.")
else:
    print("\n✅ Good score variance - model discriminates well between scenarios.")

# Check for edge cases that should be CRITICAL
print("\n" + "="*70)
print("EDGE CASE VALIDATION")
print("="*70)

should_be_critical = [
    ("Zero Inventory", result4),
    ("Extreme Bullwhip", result8)
]

for name, res in should_be_critical:
    if res['risk_class'] == 'CRITICAL':
        print(f"✅ {name}: Correctly flagged as CRITICAL")
    else:
        print(f"❌ {name}: Should be CRITICAL but got {res['risk_class']}")

# Check for scenarios that should be LOW
should_be_low = [
    ("Stable C-Z", result2),
    ("Perfect Supplier", result5)
]

for name, res in should_be_low:
    if res['risk_class'] == 'LOW':
        print(f"✅ {name}: Correctly flagged as LOW")
    else:
        print(f"⚠️ {name}: Expected LOW but got {res['risk_class']} (may be acceptable)")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
