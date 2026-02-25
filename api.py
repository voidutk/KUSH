"""
Flask API for Supply Chain Resilience Engine
REST endpoint for real-time disruption risk prediction
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Global model storage
engine = None
demand_model = None
delay_model = None

# Expected feature order (must match training)
FEATURE_ORDER = [
    "sku_class", "region", "demand_last_week", "demand_trend", 
    "seasonality_index", "demand_volatility", "supplier_reliability",
    "avg_transport_delay", "supplier_tier", "lead_time", "distance",
    "is_sole_source", "backup_supplier_count", "geopolitical_risk",
    "current_inventory", "incoming_supply", "pipeline_inventory",
    "bullwhip_coeff", "order_volatility", "order_book_value"
]

CATEGORICAL_FEATURES = ["sku_class", "region"]

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy" if engine else "not_loaded",
        "models_loaded": engine is not None
    }
    return jsonify(status), 200

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Predict disruption risk for single or batch records
    
    GET: Returns sample prediction and usage documentation
    POST: Accepts JSON payload with records to predict
    
    POST Request body (JSON):
    {
        "records": [{"sku_class": "A-X", "region": "NA", ...}]
    }
    """
    if not engine:
        return jsonify({"error": "Models not loaded"}), 503
    
    # GET returns sample/test response
    if request.method == 'GET':
        sample_record = {
            "sku_class": "A-X", "region": "NA", "demand_last_week": 150.0,
            "demand_trend": 0.05, "seasonality_index": 1.1, "demand_volatility": 0.5,
            "supplier_reliability": 0.85, "avg_transport_delay": 3, "supplier_tier": 2,
            "lead_time": 7, "distance": 500, "is_sole_source": 0,
            "backup_supplier_count": 2, "geopolitical_risk": 0.2,
            "current_inventory": 300, "incoming_supply": 200, "pipeline_inventory": 150,
            "bullwhip_coeff": 1.5, "order_volatility": 0.2, "order_book_value": 350
        }
        sample_df = pd.DataFrame([sample_record])
        result = engine.assess_disruption_risk(sample_df)
        return jsonify({
            "message": "This is a GET sample response. Use POST with 'records' array for predictions",
            "sample_record": sample_record,
            "sample_prediction": result,
            "usage": "POST JSON with {'records': [{...}]}"
        }), 200
    
    # POST - actual prediction
    try:
        data = request.get_json()
        records = data.get('records', [])
        
        if not records:
            return jsonify({"error": "No records provided"}), 400
        
        predictions = []
        for record in records:
            # Validate required fields
            missing = set(FEATURE_ORDER) - set(record.keys())
            if missing:
                return jsonify({"error": f"Missing fields: {missing}"}), 400
            
            # Create DataFrame with correct feature order
            sample_df = pd.DataFrame([{k: record[k] for k in FEATURE_ORDER}])
            
            # Predict
            assessment = engine.assess_disruption_risk(sample_df)
            predictions.append(assessment)
        
        return jsonify({"predictions": predictions}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/batch_predict', methods=['GET', 'POST'])
def batch_predict():
    """Batch prediction for multiple records"""
    return predict()

@app.route('/whatif', methods=['GET', 'POST'])
def whatif():
    """
    What-if scenario analysis
    
    GET: Returns sample scenario and usage documentation
    POST: Accepts JSON with baseline and scenarios to test
    
    POST Request:
    {
        "baseline": { ... record ... },
        "scenarios": [
            {"supplier_reliability": 0.95},
            {"is_sole_source": 0, "backup_supplier_count": 2}
        ]
    }
    """
    if not engine:
        return jsonify({"error": "Models not loaded"}), 503
    
    # GET returns sample/test response
    if request.method == 'GET':
        sample_baseline = {
            "sku_class": "A-Y", "region": "APAC", "demand_last_week": 200.0,
            "demand_trend": 0.08, "seasonality_index": 1.2, "demand_volatility": 1.0,
            "supplier_reliability": 0.65, "avg_transport_delay": 5, "supplier_tier": 3,
            "lead_time": 14, "distance": 1200, "is_sole_source": 1,
            "backup_supplier_count": 0, "geopolitical_risk": 0.5,
            "current_inventory": 400, "incoming_supply": 300, "pipeline_inventory": 250,
            "bullwhip_coeff": 2.0, "order_volatility": 0.3, "order_book_value": 500
        }
        baseline_df = pd.DataFrame([sample_baseline])
        baseline_pred = engine.assess_disruption_risk(baseline_df)
        
        # Test a scenario
        scenario = {"supplier_reliability": 0.95, "is_sole_source": 0, "backup_supplier_count": 2}
        scenario_record = {**sample_baseline, **scenario}
        scenario_df = pd.DataFrame([scenario_record])
        scenario_pred = engine.assess_disruption_risk(scenario_df)
        
        return jsonify({
            "message": "This is a GET sample response. Use POST with 'baseline' and 'scenarios' for analysis",
            "sample_baseline": sample_baseline,
            "baseline_prediction": baseline_pred,
            "sample_scenario": scenario,
            "scenario_prediction": scenario_pred,
            "improvement": baseline_pred["composite_risk_score"] - scenario_pred["composite_risk_score"],
            "usage": "POST JSON with {'baseline': {...}, 'scenarios': [{...}]}"
        }), 200
    
    # POST - actual what-if analysis
    try:
        data = request.get_json()
        baseline_record = data.get('baseline')
        scenarios = data.get('scenarios', [])
        
        if not baseline_record:
            return jsonify({"error": "No baseline record provided"}), 400
        
        # Baseline prediction
        baseline_df = pd.DataFrame([{k: baseline_record[k] for k in FEATURE_ORDER}])
        baseline_pred = engine.assess_disruption_risk(baseline_df)
        
        results = {
            "baseline": {
                "risk_score": baseline_pred["composite_risk_score"],
                "risk_class": baseline_pred["risk_class"],
                "action": baseline_pred["action"]
            },
            "scenarios": []
        }
        
        # Test each scenario
        for i, scenario in enumerate(scenarios):
            scenario_record = baseline_record.copy()
            scenario_record.update(scenario)
            
            scenario_df = pd.DataFrame([{k: scenario_record[k] for k in FEATURE_ORDER}])
            scenario_pred = engine.assess_disruption_risk(scenario_df)
            
            results["scenarios"].append({
                "scenario_id": i + 1,
                "changes": scenario,
                "risk_score": scenario_pred["composite_risk_score"],
                "risk_class": scenario_pred["risk_class"],
                "action": scenario_pred["action"],
                "improvement": baseline_pred["composite_risk_score"] - scenario_pred["composite_risk_score"]
            })
        
        return jsonify(results), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def load_models_from_notebook():
    """Load models from the notebook globals"""
    global engine, demand_model, delay_model
    
    try:
        # Import from the notebook/model file
        import sys
        sys.path.insert(0, '/home/utkarsh/Documents/INDCON ')
        
        from model import engine as eng, demand_model as dm, delay_model as dlm
        engine = eng
        demand_model = dm
        delay_model = dlm
        
        print("Models loaded successfully from notebook")
        return True
    except Exception as e:
        print(f"Could not load from notebook: {e}")
        return False

@app.route('/')
def index():
    """API documentation"""
    return jsonify({
        "name": "Supply Chain Resilience Engine API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "GET - Health check",
            "/predict": "GET/POST - Single/batch prediction (GET=sample, POST=actual)",
            "/batch_predict": "GET/POST - Batch prediction",
            "/whatif": "GET/POST - What-if scenario analysis (GET=sample, POST=actual)"
        }
    })

if __name__ == '__main__':
    # Try to load models
    if not load_models_from_notebook():
        print("Warning: Running without loaded models")
    
    print("\nStarting Supply Chain Resilience Engine API...")
    print("Endpoints:")
    print("  http://127.0.0.1:5000/health")
    print("  http://127.0.0.1:5000/predict")
    print("  http://127.0.0.1:5000/whatif")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
