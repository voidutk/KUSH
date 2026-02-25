"""
Interactive Demo Script for Hackathon Presentation
Demonstrates key capabilities of the Supply Chain Resilience Engine
"""

import pandas as pd
import numpy as np
from model import SupplyChainResilienceEngine, data, X, demand_model, delay_model

def print_section(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def demo_1_basic_assessment():
    """Demo 1: Single risk assessment with full explanation"""
    print_section("DEMO 1: Basic Risk Assessment")
    
    engine = SupplyChainResilienceEngine(demand_model, delay_model)
    sample = X.iloc[[0]]
    
    assessment = engine.assess_disruption_risk(sample)
    explanation = engine.explain_assessment(assessment)
    
    print(explanation)
    print("\n" + "-"*70)
    print("Key Takeaway: System provides probabilistic risk with actionable recommendations")

def demo_2_sole_source_risk():
    """Demo 2: Highlight sole-source supplier vulnerability"""
    print_section("DEMO 2: Sole-Source Supplier Risk (Critical Scenario)")
    
    engine = SupplyChainResilienceEngine(demand_model, delay_model)
    
    # Find a sole-source record
    sole_source_mask = (data["is_sole_source"] == 1) & (data["supplier_reliability"] < 0.75)
    if sole_source_mask.any():
        sample = X[sole_source_mask].iloc[[0]]
        
        print("Input: Sole-source supplier with reliability < 0.75")
        print(f"  Supplier Reliability: {sample['supplier_reliability'].values[0]:.3f}")
        print(f"  Is Sole Source: {sample['is_sole_source'].values[0]}")
        print(f"  Geopolitical Risk: {sample['geopolitical_risk'].values[0]:.3f}")
        
        assessment = engine.assess_disruption_risk(sample)
        print(f"\nOutput:")
        print(f"  Risk Class: {assessment['risk_class']}")
        print(f"  Delay Probability: {assessment['delay_probability']:.1%}")
        print(f"  Composite Score: {assessment['composite_risk_score']:.3f}")
        print(f"  Action: {assessment['action']}")
        print(f"\nNote: Higher delay weight (0.45) applied due to sole-source dependency")

def demo_3_inventory_crisis():
    """Demo 3: Low inventory stockout scenario"""
    print_section("DEMO 3: Inventory Crisis Detection")
    
    engine = SupplyChainResilienceEngine(demand_model, delay_model)
    
    # Simulate low inventory scenario
    sample = X.iloc[[0]].copy()
    sample["current_inventory"] = 50  # Critically low
    sample["incoming_supply"] = 20     # Low incoming
    
    print("Input: Low inventory position")
    print(f"  Current Inventory: {sample['current_inventory'].values[0]} units")
    print(f"  Incoming Supply: {sample['incoming_supply'].values[0]} units")
    
    assessment = engine.assess_disruption_risk(sample)
    inv = assessment['inventory_analysis']
    
    print(f"\nOutput:")
    print(f"  Predicted Demand: {assessment['predicted_demand']} units")
    print(f"  Future Inventory: {inv['mean_future_inventory']:.0f} units")
    print(f"  Stockout Probability: {inv['stockout_probability']:.1%}")
    print(f"  Days of Supply: {inv['days_of_supply']:.1f} days")
    print(f"  Risk Class: {assessment['risk_class']}")
    print(f"  Action: {assessment['action']}")

def demo_4_batch_processing():
    """Demo 4: Process multiple SKUs for portfolio view"""
    print_section("DEMO 4: Portfolio Risk Overview (10 SKUs)")
    
    engine = SupplyChainResilienceEngine(demand_model, delay_model)
    samples = X.sample(10, random_state=42)
    
    results = []
    for idx, row in samples.iterrows():
        sample = pd.DataFrame([row])
        assessment = engine.assess_disruption_risk(sample)
        results.append({
            "SKU": idx,
            "Class": row['sku_class'],
            "Risk_Score": assessment['composite_risk_score'],
            "Risk_Class": assessment['risk_class'],
            "Action": assessment['action'].split()[0] + " " + assessment['action'].split()[1] if len(assessment['action'].split()) > 1 else assessment['action']
        })
    
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    
    print(f"\nSummary:")
    print(f"  LOW risk: {(df_results['Risk_Class'] == 'LOW').sum()} SKUs")
    print(f"  MEDIUM risk: {(df_results['Risk_Class'] == 'MEDIUM').sum()} SKUs")
    print(f"  HIGH/CRITICAL risk: {(df_results['Risk_Class'].isin(['HIGH', 'CRITICAL'])).sum()} SKUs")

def demo_5_what_if_analysis():
    """Demo 5: What-if scenario for decision support"""
    print_section("DEMO 5: What-If Analysis (Decision Support)")
    
    engine = SupplyChainResilienceEngine(demand_model, delay_model)
    
    # Baseline
    sample = X.iloc[[0]].copy()
    baseline = engine.assess_disruption_risk(sample)
    
    print("Baseline Scenario:")
    print(f"  Risk Score: {baseline['composite_risk_score']:.3f}")
    print(f"  Risk Class: {baseline['risk_class']}")
    
    # Scenario: Improve supplier reliability
    sample_improved = sample.copy()
    sample_improved["supplier_reliability"] = 0.95
    improved = engine.assess_disruption_risk(sample_improved)
    
    print(f"\nWhat-If: Improve Supplier Reliability to 0.95")
    print(f"  Risk Score: {improved['composite_risk_score']:.3f}")
    print(f"  Risk Class: {improved['risk_class']}")
    print(f"  Improvement: {(baseline['composite_risk_score'] - improved['composite_risk_score']):.3f} points")
    
    # Scenario: Add backup supplier
    sample_backup = sample.copy()
    sample_backup["is_sole_source"] = 0
    sample_backup["backup_supplier_count"] = 2
    with_backup = engine.assess_disruption_risk(sample_backup)
    
    print(f"\nWhat-If: Add 2 Backup Suppliers (eliminate sole-source)")
    print(f"  Risk Score: {with_backup['composite_risk_score']:.3f}")
    print(f"  Risk Class: {with_backup['risk_class']}")
    print(f"  Improvement: {(baseline['composite_risk_score'] - with_backup['composite_risk_score']):.3f} points")
    print(f"\nNote: Delay weight reduced from {baseline['risk_weights']['delay']:.2f} to {with_backup['risk_weights']['delay']:.2f}")

def run_all_demos():
    """Run complete demonstration suite"""
    print("\n" + "="*70)
    print("  AI-BASED SUPPLY CHAIN DISRUPTION PREDICTOR")
    print("  Industrial Hackathon Demo")
    print("="*70)
    
    demo_1_basic_assessment()
    demo_2_sole_source_risk()
    demo_3_inventory_crisis()
    demo_4_batch_processing()
    demo_5_what_if_analysis()
    
    print_section("Demo Complete")
    print("System Capabilities Demonstrated:")
    print("  ✓ Probabilistic risk quantification")
    print("  ✓ Context-aware recommendations")
    print("  ✓ Sole-source vulnerability detection")
    print("  ✓ Inventory stockout prediction")
    print("  ✓ Portfolio risk overview")
    print("  ✓ What-if decision support")

if __name__ == "__main__":
    run_all_demos()
