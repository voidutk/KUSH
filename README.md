# KUSH — Supply Chain Risk Predictor (India)

AI-powered supply chain risk assessment dashboard for Indian market, featuring real-time Gemini AI insights, live currency conversion, and production-grade ML models.

## Project Structure

```
KUSH/
├── .env                           # API key configuration
├── .env.example                   # Template for API keys
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── corrected_engine.py            # Core ML risk assessment engine
├── india_working_dashboard.py     # Main Streamlit dashboard
├── test_engine.py                 # Automated test suite (32 tests)
├── models/                        # Persisted trained models (auto-generated)
└── venv/                          # Python virtual environment
```

## Features

### Machine Learning Engine
- **Two-layer risk model** — structural risk (shortage, sole-source) + operational risk (delays, geo-risk, volatility) with monotonicity constraints
- **Real evaluation metrics** — train/test split (80/20) on 2 000 samples, reporting R², MAE, Accuracy, F1, ROC-AUC
- **Real feature importances** — extracted from GradientBoosting, not hardcoded
- **Deterministic confidence** — model confidence from calibrated class probabilities
- **Anomaly detection** — z-score of demand residuals vs training distribution
- **Model persistence** — trained models saved/loaded via joblib (no redundant retraining)

### AI Integration
- **Google Gemini** — context-aware supply chain insights via `google.genai` SDK
- **Smart fallback** — rule-based insights when Gemini is unavailable or quota exceeded
- **Graceful degradation** — yellow warnings instead of errors on network issues

### Dashboard
- **Risk Assessment** — interactive form → risk gauge, feature importance chart, inventory policy, action plan
- **Scenario Analysis** — create and compare custom scenarios side-by-side
- **Analytics** — cost-benefit, seasonal risk trends, supplier performance, demand forecasting
- **Live INR conversion** — real-time USD→INR via ExchangeRate-API
- **JSON export** — download full risk brief as JSON

## Quick Start

```bash
# Clone
git clone https://github.com/voidutk/KUSH.git
cd KUSH

# Virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Dependencies
pip install -r requirements.txt

# API key (optional — dashboard works without it)
cp .env.example .env
# Edit .env and add your Gemini API key

# Run
streamlit run india_working_dashboard.py
```

Opens at http://localhost:8501

## Model Performance

| Metric | Value |
|---|---|
| Demand R² | 0.75 |
| Demand MAE | 20.23 units |
| Delay Accuracy | 67% |
| Delay F1 | 0.68 |
| Delay ROC-AUC | 0.75 |
| Training samples | 1 600 |
| Test samples | 400 |

## Running Tests

```bash
python test_engine.py
```

Validates training, determinism, value ranges, model persistence, and edge cases (32 tests).

## Deployment

### Streamlit Community Cloud (Free)
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select repo → set main file to `india_working_dashboard.py`
4. Add `GEMINI_API_KEY` in Advanced Settings → Secrets
5. Deploy

## Tech Stack

- **ML**: scikit-learn (GradientBoosting, RandomForest), joblib
- **Frontend**: Streamlit, Plotly
- **AI**: Google Gemini (`google.genai`)
- **Data**: pandas, numpy
- **APIs**: ExchangeRate-API (currency), Gemini (insights)

## License

MIT
