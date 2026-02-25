"""
Supply Chain Resilience Engine - Core Module
Extracted from notebook for importability
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV

# Set seed for reproducibility
np.random.seed(42)

# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_synthetic_data(rows=1000):
    """Generate realistic synthetic supply chain data"""
    
    # 50 suppliers with base characteristics
    n_suppliers = 50
    supplier_base_reliability = np.random.beta(7, 2, n_suppliers)
    supplier_base_delay = np.random.gamma(2, 1.5, n_suppliers)
    supplier_tier = np.random.choice([1, 2, 3], n_suppliers, p=[0.2, 0.5, 0.3])
    
    supplier_ids = np.random.randint(0, n_suppliers, rows)
    
    # ABC-XYZ SKU classification
    sku_classes = np.random.choice(
        ["A-X", "A-Y", "A-Z", "B-X", "B-Y", "B-Z", "C-X", "C-Y", "C-Z"],
        rows,
        p=[0.05, 0.08, 0.07, 0.10, 0.15, 0.10, 0.15, 0.15, 0.15]
    )
    
    volatility_map = {"X": 0.5, "Y": 1.0, "Z": 2.0}
    demand_volatility = np.array([volatility_map[cls.split("-")[1]] for cls in sku_classes])
    
    weeks = np.random.randint(1, 53, rows)
    regions = np.random.choice(["NA", "EU", "APAC", "LATAM"], rows, p=[0.3, 0.25, 0.35, 0.1])
    
    supplier_reliability = supplier_base_reliability[supplier_ids]
    avg_transport_delay = np.random.poisson(supplier_base_delay[supplier_ids])
    supplier_tier_col = supplier_tier[supplier_ids]
    
    is_sole_source = np.random.choice([0, 1], rows, p=[0.75, 0.25])
    backup_supplier_count = np.where(is_sole_source == 1, 0, np.random.poisson(2, rows) + 1)
    
    geopolitical_risk = np.where(regions == "APAC", np.random.uniform(0.3, 0.7, rows),
                                  np.where(regions == "LATAM", np.random.uniform(0.4, 0.6, rows),
                                           np.random.uniform(0.1, 0.3, rows)))
    
    base_demand = np.random.normal(150, 40, rows)
    demand_last_week = base_demand + np.random.normal(0, 15 * demand_volatility, rows)
    demand_trend = np.random.uniform(-0.15, 0.25, rows) * demand_volatility
    
    seasonality_phase = {"NA": 0, "EU": 13, "APAC": 26, "LATAM": 39}
    phase_shift = np.array([seasonality_phase[r] for r in regions])
    seasonality_index = 1 + 0.3 * np.sin(2 * np.pi * (weeks - phase_shift) / 52) + np.random.normal(0, 0.1, rows)
    
    lead_time = np.where(supplier_tier_col == 1, np.random.randint(2, 7, rows),
                          np.where(supplier_tier_col == 2, np.random.randint(5, 12, rows),
                                   np.random.randint(8, 20, rows)))
    
    current_inventory = np.random.gamma(4, 75, rows)
    incoming_supply = np.random.gamma(3, 80, rows)
    distance = np.random.gamma(3, 150, rows)
    
    upstream_variance = np.random.uniform(1.0, 3.0, rows)
    bullwhip_coeff = upstream_variance / demand_volatility
    
    order_volatility = np.random.uniform(0.1, 0.5, rows)
    order_book_value = demand_last_week * np.random.uniform(1.2, 2.5, rows)
    pipeline_inventory = incoming_supply * lead_time * 0.3
    
    # Targets
    demand_noise = np.random.normal(0, 10 * demand_volatility, rows)
    actual_demand = (demand_last_week * (1 + demand_trend) * seasonality_index + demand_noise)
    actual_demand = np.maximum(actual_demand, 20)
    
    delay_logit = (-3 + 4 * (1 - supplier_reliability) + 0.3 * avg_transport_delay + 
                   0.5 * geopolitical_risk + 0.8 * is_sole_source - 0.2 * backup_supplier_count)
    delay_prob_true = 1 / (1 + np.exp(-delay_logit))
    delay_happened = (np.random.uniform(0, 1, rows) < delay_prob_true).astype(int)
    
    data = pd.DataFrame({
        "supplier_id": supplier_ids,
        "sku_class": sku_classes,
        "week": weeks,
        "region": regions,
        "demand_last_week": demand_last_week.round(1),
        "demand_trend": demand_trend.round(3),
        "seasonality_index": seasonality_index.round(3),
        "demand_volatility": demand_volatility,
        "supplier_reliability": supplier_reliability.round(3),
        "avg_transport_delay": avg_transport_delay,
        "supplier_tier": supplier_tier_col,
        "lead_time": lead_time,
        "distance": distance.round(0),
        "is_sole_source": is_sole_source,
        "backup_supplier_count": backup_supplier_count,
        "geopolitical_risk": geopolitical_risk.round(3),
        "current_inventory": current_inventory.round(0),
        "incoming_supply": incoming_supply.round(0),
        "pipeline_inventory": pipeline_inventory.round(0),
        "bullwhip_coeff": bullwhip_coeff.round(2),
        "order_volatility": order_volatility.round(3),
        "order_book_value": order_book_value.round(0),
        "actual_demand": actual_demand.round(1),
        "delay_happened": delay_happened
    })
    
    return data

# Generate data on module load
data = generate_synthetic_data(1000)

# Define features
categorical_features = ["sku_class", "region"]
numerical_features = [
    "demand_last_week", "demand_trend", "seasonality_index", "demand_volatility",
    "supplier_reliability", "avg_transport_delay", "supplier_tier", "lead_time", "distance",
    "is_sole_source", "backup_supplier_count", "geopolitical_risk",
    "current_inventory", "incoming_supply", "pipeline_inventory",
    "bullwhip_coeff", "order_volatility", "order_book_value"
]

# Prepare data
X = data.drop(columns=["actual_demand", "delay_happened", "supplier_id", "week"])
y_demand = data["actual_demand"]
y_delay = data["delay_happened"]

# Train-test split
Xd_train, Xd_test, yd_train, yd_test = train_test_split(X, y_demand, test_size=0.2, random_state=42)
Xl_train, Xl_test, yl_train, yl_test = train_test_split(X, y_delay, test_size=0.2, random_state=42, stratify=y_delay)

# Build and train models
demand_preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_features),
    ("num", StandardScaler(), numerical_features)
])

delay_preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_features),
    ("num", StandardScaler(), numerical_features)
])

demand_model = Pipeline([
    ("preprocess", demand_preprocessor),
    ("feature_eng", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ("regressor", GradientBoostingRegressor(
        n_estimators=200,           # Reduced from 500 for speed
        max_depth=6,                 # Reduced depth
        learning_rate=0.08,          # Slightly higher rate
        subsample=0.8,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42
    ))
])

delay_model = Pipeline([
    ("preprocess", delay_preprocessor),
    ("classifier", GradientBoostingClassifier(
        n_estimators=200,           # Reduced from 400
        max_depth=5,                # Reduced depth
        learning_rate=0.1,
        subsample=0.8,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    ))
])

print("Training enhanced demand forecasting model...")
print("  - Using 200 estimators with polynomial feature interactions")
demand_model.fit(Xd_train, yd_train)
print("  - Training complete")

print("\nTraining enhanced delay prediction model...")
print("  - Using Gradient Boosting for better probability calibration")
delay_model.fit(Xl_train, yl_train)
print("  - Training complete")

print("\n✅ Enhanced models trained successfully!")
print("   Model sensitivity improved for industrial-grade decision support")

# ============================================================================
# RESILIENCE ENGINE
# ============================================================================

class SupplyChainResilienceEngine:
    """Industrial-grade supply chain disruption predictor"""
    
    def __init__(self, demand_model, delay_model):
        self.demand_model = demand_model
        self.delay_model = delay_model
        self.demand_uncertainty_factor = 0.12  # Reduced for tighter bounds
        self.sensitivity_threshold = 0.35     # Lower threshold for earlier warnings
        
    def calculate_probabilistic_inventory_risk(self, sample, pred_demand):
        """Enhanced P(stockout) with dynamic uncertainty based on SKU volatility"""
        current_inv = sample["current_inventory"].values[0]
        incoming = sample["incoming_supply"].values[0]
        demand_vol = sample["demand_volatility"].values[0]
        lead_time = sample["lead_time"].values[0]
        
        # Dynamic uncertainty based on SKU class volatility
        dynamic_uncertainty = self.demand_uncertainty_factor * (1 + demand_vol / 2)
        
        mean_future = current_inv + incoming - pred_demand
        demand_std = pred_demand * dynamic_uncertainty
        supply_std = incoming * 0.08  # Tighter supply uncertainty
        lead_time_var = lead_time * 1.5  # Reduced lead time variance
        
        # Bullwhip amplification factor
        bullwhip = sample["bullwhip_coeff"].values[0]
        amplification = 1 + (bullwhip - 1) * 0.3
        
        total_std = np.sqrt(demand_std**2 * amplification + supply_std**2 + lead_time_var)
        
        if total_std > 0:
            z_score = mean_future / total_std
            stockout_prob = 1 - stats.norm.cdf(z_score)
        else:
            stockout_prob = 0 if mean_future > 0 else 1
        
        # Safety stock calculation
        safety_stock = total_std * 1.65  # 95% service level
        
        return {
            "mean_future_inventory": mean_future,
            "inventory_std": total_std,
            "stockout_probability": min(stockout_prob, 0.999),  # Cap at 99.9%
            "days_of_supply": (current_inv + incoming) / pred_demand * 7,
            "safety_stock_recommended": safety_stock,
            "z_score": z_score if total_std > 0 else 0
        }
    
    def calculate_demand_risk(self, pred_demand, sample):
        """Enhanced demand risk with trend sensitivity and capacity constraints"""
        sku_class = sample["sku_class"].values[0]
        volatility = sample["demand_volatility"].values[0]
        trend = sample["demand_trend"].values[0]
        
        criticality = {"A-X": 1.0, "A-Y": 0.9, "A-Z": 0.85,
                      "B-X": 0.7, "B-Y": 0.6, "B-Z": 0.55,
                      "C-X": 0.4, "C-Y": 0.3, "C-Z": 0.25}
        weight = criticality.get(sku_class, 0.5)
        
        # Base demand risk with capacity constraint
        base_demand_risk = min(pred_demand / 300, 1.0)  # More sensitive threshold
        
        # Volatility factor with trend amplification
        trend_factor = 1 + abs(trend) * 2  # Amplify risk if trend is strong
        volatility_factor = min(volatility * trend_factor / 1.5, 1.0)
        
        # Seasonal risk adjustment
        seasonality = sample["seasonality_index"].values[0]
        seasonal_risk = abs(seasonality - 1) * 0.5  # Deviation from baseline
        
        demand_risk = (base_demand_risk * 0.5 + volatility_factor * 0.35 + seasonal_risk * 0.15) * weight
        
        return min(demand_risk, 1.0)
    
    def calculate_composite_risk(self, demand_risk, delay_prob, stockout_prob, sample):
        """Enhanced weighted composite risk with early warning sensitivity"""
        is_sole = sample["is_sole_source"].values[0]
        backup_count = sample["backup_supplier_count"].values[0]
        geo_risk = sample["geopolitical_risk"].values[0]
        reliability = sample["supplier_reliability"].values[0]
        
        # Reliability penalty - lower reliability = higher delay weight
        reliability_penalty = (1 - reliability) * 0.3
        
        # Geopolitical amplification
        geo_amplification = 1 + geo_risk * 0.5
        
        if is_sole:
            delay_weight, demand_weight, inv_weight = 0.50, 0.25, 0.25
        elif backup_count >= 2:
            delay_weight, demand_weight, inv_weight = 0.20, 0.45, 0.35
        else:
            delay_weight, demand_weight, inv_weight = 0.35, 0.35, 0.30
        
        # Apply penalties and amplifications
        adjusted_delay = min(delay_prob * geo_amplification + reliability_penalty, 1.0)
        
        # Direct geo risk factor (independent of ML model)
        direct_geo_risk = geo_risk * 0.15
        
        # Direct supplier risk factor for low reliability
        direct_supplier_risk = (1 - reliability) * 0.10 if reliability < 0.8 else 0
        
        composite = (demand_weight * demand_risk + 
                    delay_weight * adjusted_delay + 
                    inv_weight * stockout_prob +
                    direct_geo_risk + direct_supplier_risk)
        
        # Early warning boost for borderline cases
        if composite > 0.15 and composite < 0.45:
            composite = min(composite * 1.25, 0.44)  # Push toward HIGH threshold
        
        return {
            "score": min(composite, 1.0),
            "weights": {"demand": demand_weight, "delay": delay_weight, "inventory": inv_weight},
            "adjusted_delay_probability": adjusted_delay,
            "direct_factors": {"geo": direct_geo_risk, "supplier": direct_supplier_risk}
        }
    
    def get_recommendation(self, risk_score, risk_components, sample):
        """5-tier recommendation system"""
        delay_prob = risk_components["delay_probability"]
        stockout_prob = risk_components["stockout_probability"]
        days_supply = risk_components["days_of_supply"]
        
        is_sole = sample["is_sole_source"].values[0]
        backup_count = sample["backup_supplier_count"].values[0]
        supplier_rel = sample["supplier_reliability"].values[0]
        
        if risk_score < 0.25:
            risk_class = "LOW"
            action = "Normal Operations"
            details = "Continue standard monitoring. No immediate action required."
        elif risk_score < 0.45:
            risk_class = "MEDIUM"
            if delay_prob > 0.3:
                action = "Monitor Supply"
                details = f"Supplier reliability at {supplier_rel:.2f}. Increase monitoring frequency to daily."
            else:
                action = "Monitor Supply"
                details = f"Inventory covers {days_supply:.1f} days. Watch demand trends closely."
        elif risk_score < 0.65:
            risk_class = "HIGH"
            if stockout_prob > 0.4:
                action = "Increase Reorder Quantity"
                details = f"Stockout probability {stockout_prob:.1%}. Recommend +30% safety stock buffer."
            elif is_sole and delay_prob > 0.4:
                action = "Switch Supplier"
                details = f"Sole-source risk critical. Activate alternative supplier immediately."
            else:
                action = "Increase Reorder Quantity"
                details = f"Pre-position inventory. Expedite {sample['lead_time'].values[0]:.0f}d lead time orders."
        else:
            risk_class = "CRITICAL"
            if is_sole:
                action = "Switch Supplier + Emergency Procurement"
                details = f"CRITICAL: Sole source failure risk. Activate backup suppliers + air freight."
            elif stockout_prob > 0.6:
                action = "Adjust Production + Expedite"
                details = f"Production line at risk. Slow production + expedite incoming supply."
            else:
                action = "Switch Supplier"
                details = f"Primary supplier unreliable. Switch to backup with {backup_count} alternatives."
        
        return {"risk_class": risk_class, "action": action, "action_details": details}
    
    def assess_disruption_risk(self, sample):
        """Main entry point: comprehensive risk assessment"""
        pred_demand = self.demand_model.predict(sample)[0]
        delay_proba = self.delay_model.predict_proba(sample)[0]
        delay_prob = delay_proba[1] if len(delay_proba) > 1 else delay_proba[0]
        
        inventory_analysis = self.calculate_probabilistic_inventory_risk(sample, pred_demand)
        demand_risk = self.calculate_demand_risk(pred_demand, sample)
        
        risk_result = self.calculate_composite_risk(
            demand_risk, delay_prob, inventory_analysis["stockout_probability"], sample
        )
        
        risk_components = {
            "demand_risk": demand_risk,
            "delay_probability": delay_prob,
            "stockout_probability": inventory_analysis["stockout_probability"],
            "days_of_supply": inventory_analysis["days_of_supply"]
        }
        
        recommendation = self.get_recommendation(risk_result["score"], risk_components, sample)
        
        return {
            "predicted_demand": round(pred_demand, 1),
            "delay_probability": round(delay_prob, 3),
            "inventory_analysis": {k: round(v, 3) if isinstance(v, float) else v 
                                   for k, v in inventory_analysis.items()},
            "demand_risk": round(demand_risk, 3),
            "risk_weights": risk_result["weights"],
            "composite_risk_score": round(risk_result["score"], 3),
            **recommendation
        }
    
    def explain_assessment(self, assessment):
        """Generate natural language summary"""
        inv = assessment["inventory_analysis"]
        
        summary = f"""
=== SUPPLY CHAIN RISK ASSESSMENT ===
Risk Level: {assessment['risk_class']} (Score: {assessment['composite_risk_score']})

DEMAND FORECAST:
  Predicted: {assessment['predicted_demand']} units
  Demand Risk: {assessment['demand_risk']:.1%}

SUPPLY RISK:
  Delay Probability: {assessment['delay_probability']:.1%}

INVENTORY POSITION:
  Expected Future Inventory: {inv['mean_future_inventory']:.0f} units
  Stockout Probability: {inv['stockout_probability']:.1%}
  Days of Supply: {inv['days_of_supply']:.1f} days

RECOMMENDED ACTION: {assessment['action']}
{assessment['action_details']}
        """
        return summary.strip()

# Initialize engine
engine = SupplyChainResilienceEngine(demand_model, delay_model)
