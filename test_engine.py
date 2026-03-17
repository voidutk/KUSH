"""
Test script for KUSH Supply Chain Risk Engine
==============================================
Validates that all ML improvements are working correctly.
Run: python test_engine.py  (inside venv)
"""

import sys
import os
import shutil
import numpy as np
import pandas as pd

# Ensure we can import from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from corrected_engine import FastDemoEngine, MODEL_DIR

PASS = "✅"
FAIL = "❌"
results = []


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, condition))
    print(f"  {status} {name}" + (f"  ({detail})" if detail else ""))
    return condition


def test_training():
    print("\n🧪 TEST 1: Training & Metrics")
    engine = FastDemoEngine()
    engine.train_fast_models(force_retrain=True)

    check("Engine is trained", engine.is_trained)
    check("model_metrics is populated", bool(engine.model_metrics))

    m = engine.model_metrics
    check("demand_r2 > 0", m.get("demand_r2", -1) > 0, f"R²={m.get('demand_r2')}")
    check("demand_mae > 0", m.get("demand_mae", -1) > 0, f"MAE={m.get('demand_mae')}")
    check("delay_accuracy > 0.5", m.get("delay_accuracy", 0) > 0.5, f"Acc={m.get('delay_accuracy')}")
    check("delay_f1 > 0", m.get("delay_f1", 0) > 0, f"F1={m.get('delay_f1')}")
    check("delay_roc_auc > 0.5", m.get("delay_roc_auc", 0) > 0.5, f"AUC={m.get('delay_roc_auc')}")
    check("train_samples == 1600", m.get("train_samples") == 1600, f"got {m.get('train_samples')}")
    check("test_samples == 400", m.get("test_samples") == 400, f"got {m.get('test_samples')}")

    return engine


def test_determinism(engine):
    print("\n🧪 TEST 2: Deterministic Outputs (model_confidence not random)")
    sample = pd.DataFrame([{
        "sku_class": "A-X", "region": "APAC",
        "demand_last_week": 200, "supplier_reliability": 0.85,
        "is_sole_source": 0, "current_inventory": 300,
        "geopolitical_risk": 0.3, "lead_time": 10,
        "demand_volatility": 1.5,
    }])

    r1 = engine.assess_risk_fast(sample)
    r2 = engine.assess_risk_fast(sample)

    check("model_confidence identical on repeat", r1["model_confidence"] == r2["model_confidence"],
          f"{r1['model_confidence']:.4f} == {r2['model_confidence']:.4f}")
    check("anomaly_score identical on repeat", r1["anomaly_score"] == r2["anomaly_score"],
          f"{r1['anomaly_score']:.4f} == {r2['anomaly_score']:.4f}")
    check("composite_risk_score identical on repeat", r1["composite_risk_score"] == r2["composite_risk_score"])

    return r1


def test_assessment_values(result):
    print("\n🧪 TEST 3: Assessment Value Ranges")
    for key in ["composite_risk_score", "model_confidence", "anomaly_score",
                 "network_risk", "inventory_risk", "supply_risk",
                 "shortage_ratio", "structural_risk", "operational_risk"]:
        val = result.get(key, -1)
        check(f"{key} in [0, 1]", 0 <= val <= 1, f"{val:.4f}")

    check("risk_class is valid", result["risk_class"] in ("LOW", "MEDIUM", "HIGH"))
    check("feature_importances present", len(result.get("feature_importances", [])) > 0,
          f"{len(result.get('feature_importances', []))} features")
    check("feature_importances have real feature names",
          any("Demand" in fi["feature"] or "Inventory" in fi["feature"]
              for fi in result.get("feature_importances", [])))
    check("days_of_supply > 0", result.get("days_of_supply", 0) > 0)


def test_persistence():
    print("\n🧪 TEST 4: Model Persistence (save/load)")
    # Engine 1 trains and saves
    e1 = FastDemoEngine()
    e1.train_fast_models(force_retrain=True)
    saved_metrics = dict(e1.model_metrics)

    # Engine 2 should load from disk
    e2 = FastDemoEngine()
    e2.train_fast_models(force_retrain=False)  # should load, not retrain

    check("Loaded engine is trained", e2.is_trained)
    check("Metrics match after load", e2.model_metrics == saved_metrics)

    # Both should give same prediction
    sample = pd.DataFrame([{
        "sku_class": "B-Y", "region": "EU",
        "demand_last_week": 150, "supplier_reliability": 0.7,
        "is_sole_source": 1, "current_inventory": 100,
        "geopolitical_risk": 0.5, "lead_time": 15,
        "demand_volatility": 2.0,
    }])

    r1 = e1.assess_risk_fast(sample)
    r2 = e2.assess_risk_fast(sample)
    check("Predictions match after load",
          abs(r1["composite_risk_score"] - r2["composite_risk_score"]) < 1e-6)


def test_edge_cases():
    print("\n🧪 TEST 5: Edge Cases")
    engine = FastDemoEngine()
    engine.train_fast_models(force_retrain=False)

    # Zero inventory → should be HIGH risk
    sample_zero_inv = pd.DataFrame([{
        "sku_class": "C-Z", "region": "NA",
        "demand_last_week": 400, "supplier_reliability": 0.5,
        "is_sole_source": 1, "current_inventory": 0,
        "geopolitical_risk": 0.7, "lead_time": 20,
        "demand_volatility": 2.5,
    }])
    r = engine.assess_risk_fast(sample_zero_inv)
    check("Zero inventory → HIGH risk", r["risk_class"] == "HIGH", f"got {r['risk_class']}")
    check("Zero inventory → shortage_ratio == 1", r["shortage_ratio"] == 1.0)

    # Abundant inventory → should be LOW risk
    sample_high_inv = pd.DataFrame([{
        "sku_class": "A-X", "region": "NA",
        "demand_last_week": 50, "supplier_reliability": 0.95,
        "is_sole_source": 0, "current_inventory": 1000,
        "geopolitical_risk": 0.1, "lead_time": 3,
        "demand_volatility": 0.5,
    }])
    r = engine.assess_risk_fast(sample_high_inv)
    check("High inventory → LOW risk", r["risk_class"] == "LOW", f"got {r['risk_class']}")
    check("High inventory → shortage_ratio == 0", r["shortage_ratio"] == 0.0)


# ---- Run all tests ----
if __name__ == "__main__":
    print("=" * 60)
    print("  KUSH Supply Chain Risk Engine — Test Suite")
    print("=" * 60)

    engine = test_training()
    result = test_determinism(engine)
    test_assessment_values(result)
    test_persistence()
    test_edge_cases()

    # Summary
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"\n{'=' * 60}")
    print(f"  Results: {passed}/{total} passed")
    if passed == total:
        print("  🎉 ALL TESTS PASSED!")
    else:
        print(f"  ⚠️  {total - passed} test(s) failed")
    print(f"{'=' * 60}")

    sys.exit(0 if passed == total else 1)
