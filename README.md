# AI-Based Supply Chain Disruption Predictor
## Industrial Decision Intelligence System

**Hackathon:** INDCON 2026  
**Theme:** Supply Chain Optimization & Industrial Data Analytics  
**Team:** [Your Team Name]

---

## 🎯 Executive Summary

This system transforms supply chain management from reactive firefighting to proactive resilience building. Using machine learning and operations research principles, it predicts disruptions before they occur and prescribes specific mitigation actions.

**Key Innovation:** Probabilistic risk quantification with context-aware recommendations that adapt to supplier network structure (sole-source vs. redundant suppliers).

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Data Layer: Synthetic Industrial Supply Chain Data     │
│  (1,000 records, 26 features, ABC-XYZ SKU classification) │
├─────────────────────────────────────────────────────────┤
│  ML Models:                                             │
│    • Gradient Boosting Demand Forecaster (MAPE 7.8%)   │
│    • Random Forest Delay Classifier (AUC-ROC 0.69)      │
├─────────────────────────────────────────────────────────┤
│  Resilience Engine:                                     │
│    • P(stockout) calculation using normal distribution │
│    • Dynamic risk weights by supplier redundancy         │
│    • 5-tier recommendation system                        │
├─────────────────────────────────────────────────────────┤
│  Decision Support:                                      │
│    • Natural language explanations                       │
│    • What-if scenario analysis                           │
│    • Stress testing framework                            │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Performance Metrics

### Demand Forecasting
- **MAE:** 10.98 units
- **MAPE:** 7.78%
- **R²:** 0.93
- **Forecast Value Added:** 63% improvement over naive baseline

### Delay Risk Classification
- **AUC-ROC:** 0.69
- **AUC-PR:** 0.46
- **Brier Score:** 0.18 (well-calibrated probabilities)
- **Early Detection Rate:** 28.8%

### Business Impact
- **False Positive Rate:** 8.1% (minimizes unnecessary alerts)
- **Mean Detection Lead Time:** Days to weeks before disruption

---

## 🔧 Key Features

### 1. Probabilistic Inventory Risk
```python
# Not: inventory_risk = 1 if future_inventory < 0 else 0
# But: P(stockout) = Φ(-μ/σ)
```
Accounts for demand uncertainty, supply variance, and lead time variability.

### 2. Context-Aware Risk Scoring
| Supplier Type | Delay Weight | Rationale |
|--------------|--------------|-----------|
| Sole Source | 0.45 | Single point of failure |
| Single Backup | 0.30 | Limited redundancy |
| 2+ Backups | 0.20 | High resilience |

### 3. Five-Tier Recommendation System
1. **Normal Operations** - Continue standard monitoring
2. **Monitor Supply** - Increase monitoring frequency
3. **Increase Reorder Quantity** - Buffer inventory
4. **Switch Supplier** - Activate alternatives
5. **Adjust Production + Expedite** - Emergency measures

### 4. Stress Testing Framework
Simulates real-world scenarios:
- Tier-2 supplier failures (40% failure rate)
- Demand surge (250% spike on Class A SKUs)
- APAC port closure (3 weeks)
- Black Swan (multiple simultaneous disruptions)

---

## 🚀 Quick Start

### Run the Demo
```bash
cd /home/utkarsh/Documents/INDCON 
source venv/bin/activate
python demo.py
```

### Start the API
```bash
python api.py
# Visit http://127.0.0.1:5000/health
```

### Run Full Notebook
```bash
jupyter notebook model.ipynb
```

---

## 📁 File Structure

| File | Purpose |
|------|---------|
| `model.ipynb` | Main notebook with full system |
| `demo.py` | Interactive hackathon demo |
| `api.py` | Flask REST API for real-time predictions |
| `visualization.py` | Risk dashboards and heatmaps |
| `model_persistence.py` | Model save/load utilities |

---

## 💡 Innovation Highlights

### Industrial-Grade Design Decisions

**1. ABC-XYZ SKU Classification**
- A = High value, X = Stable demand (critical but predictable)
- C = Low value, Z = Volatile demand (manage differently)
- Risk weights adjust by SKU criticality

**2. Multi-Tier Supplier Visibility**
- Tier 1: Strategic partners (2-7 day lead time)
- Tier 2: Regular suppliers (5-12 days)
- Tier 3: Commodity suppliers (8-20 days)

**3. Bullwhip Coefficient**
Measures demand signal distortion upstream in supply chain:
```
Bullwhip = Variance(orders) / Variance(demand)
```

**4. Geopolitical Risk Integration**
Regional risk scoring affects delay predictions:
- NA/EU: Low baseline risk
- APAC: Medium (trade policy, congestion)
- LATAM: Elevated (infrastructure, stability)

---

## 🎓 Technical Approach

### Feature Engineering (23 Features)

**Demand Signals:**
- `demand_last_week` - Baseline demand
- `demand_trend` - Growth/decline rate
- `seasonality_index` - Weekly seasonality
- `demand_volatility` - XYZ classification

**Supply Risk:**
- `supplier_reliability` - Historical performance
- `avg_transport_delay` - Historical delays
- `supplier_tier` - Strategic importance
- `lead_time` - Procurement cycle time

**Network Structure:**
- `is_sole_source` - Single point of failure flag
- `backup_supplier_count` - Redundancy level
- `geopolitical_risk` - Regional stability score
- `distance` - Transport distance (km)

**Inventory Position:**
- `current_inventory` - On-hand stock
- `incoming_supply` - Pipeline orders
- `pipeline_inventory` - In-transit valuation

**Advanced Metrics:**
- `bullwhip_coeff` - Demand amplification
- `order_volatility` - Order pattern stability
- `order_book_value` - Committed orders

### Model Selection

**Demand Forecasting: Gradient Boosting Regressor**
- Handles non-linear demand patterns
- Feature interactions (trend × seasonality)
- Built-in feature importance

**Delay Prediction: Random Forest Classifier**
- Robust to outliers (supplier anomalies)
- Class balancing for rare delay events
- Probability calibration for risk scoring

### Evaluation Philosophy

**Beyond Accuracy:**
- **Calibration:** Probabilities must reflect true likelihood (Brier score)
- **Business Value:** Forecast Value Added vs naive methods
- **Early Detection:** Flag disruptions before they occur
- **Cost Sensitivity:** Minimize false positives (alert fatigue)

---

## 🔮 Future Enhancements

### Phase 2: External Data Integration
- Port congestion APIs (MarineTraffic, AIS data)
- Commodity price volatility (Bloomberg, Yahoo Finance)
- Weather APIs for transport corridor risk
- Geopolitical risk feeds (CountryRisk.io)

### Phase 3: Network Graph Analysis
- Supply chain graph representation
- Centrality measures for supplier criticality
- Cascade failure simulation
- Multi-tier disruption propagation

### Phase 4: Continuous Learning
- Online model updates as new data arrives
- Drift detection for concept shifts
- A/B testing for recommendation effectiveness
- Automated hyperparameter tuning

---

## 🏆 Competitive Advantages

| Aspect | Typical ML Approach | Our Approach |
|--------|---------------------|--------------|
| Risk Scoring | Binary classification | Probabilistic P(stockout) |
| Recommendations | Static thresholds | Context-aware, 5-tier system |
| Explainability | Feature importance | Natural language + counterfactuals |
| Resilience | Point predictions | Stress testing + scenario planning |
| Business Value | Model accuracy | Detection lead time + cost avoidance |

---

## 📧 Contact & Acknowledgments

**Team Members:** [Names]

**Technologies Used:**
- Python 3.13
- scikit-learn (Gradient Boosting, Random Forest)
- pandas, numpy (Data Engineering)
- Flask (API)
- SHAP (Explainability - optional)

**Data:** Synthetic industrial supply chain data with realistic correlations

---

## 📄 License

This project was developed for educational purposes at INDCON 2026 Hackathon.

---

**System Status:** ✅ All tests passing  
**Last Updated:** February 2026
# KUSH
