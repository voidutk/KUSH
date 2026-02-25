"""
Supply Chain Resilience Engine - Model Persistence
Save and load trained models for deployment
"""

import pickle
import joblib
import json
from datetime import datetime

class ModelPersistence:
    """Handle saving/loading of trained models and metadata"""
    
    @staticmethod
    def save_models(demand_model, delay_model, engine, filepath_prefix="models/"):
        """Save all models and configuration"""
        import os
        os.makedirs(filepath_prefix, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save models
        demand_path = f"{filepath_prefix}demand_model_{timestamp}.pkl"
        delay_path = f"{filepath_prefix}delay_model_{timestamp}.pkl"
        engine_path = f"{filepath_prefix}engine_config_{timestamp}.json"
        
        joblib.dump(demand_model, demand_path)
        joblib.dump(delay_model, delay_path)
        
        # Save engine configuration
        config = {
            "timestamp": timestamp,
            "demand_uncertainty_factor": engine.demand_uncertainty_factor,
            "version": "1.0.0"
        }
        with open(engine_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Models saved:")
        print(f"  Demand model: {demand_path}")
        print(f"  Delay model: {delay_path}")
        print(f"  Config: {engine_path}")
        
        return {
            "demand_model": demand_path,
            "delay_model": delay_path,
            "config": engine_path
        }
    
    @staticmethod
    def load_models(demand_path, delay_path, config_path):
        """Load saved models and recreate engine"""
        demand_model = joblib.load(demand_path)
        delay_model = joblib.load(delay_path)
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Recreate engine
        from model import SupplyChainResilienceEngine
        engine = SupplyChainResilienceEngine(demand_model, delay_model)
        engine.demand_uncertainty_factor = config.get("demand_uncertainty_factor", 0.15)
        
        print(f"Models loaded from {config['timestamp']}")
        return engine, demand_model, delay_model

# Save current models
if __name__ == "__main__":
    # This will be called after model training
    print("Model persistence utilities ready")
