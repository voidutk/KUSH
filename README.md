# India Supply Chain AI Dashboard

A comprehensive supply chain risk prediction dashboard specifically designed for Indian market, featuring real-time AI-powered insights through Google Gemini API integration.

## Project Overview

This dashboard provides advanced supply chain risk assessment with AI-powered recommendations, specifically optimized for Indian market with real-time currency conversion and localized business context.

## Project Structure

```
KUSH/
├── .env                           # API key configuration
├── README.md                      # Project documentation
├── corrected_engine.py            # Core ML risk assessment engine
├── india_working_dashboard.py     # Main Streamlit dashboard
└── venv/                          # Python virtual environment
```

## Features

### Risk Assessment Engine
- Advanced ML Models: Using corrected_engine with monotonicity and structural dominance
- Real-time Analysis: Instant risk scoring and recommendations
- Indian Market Context: Regional factors, monsoon patterns, logistics challenges
- Currency Localization: All amounts in Indian Rupees (₹) with proper formatting

### AI Integration
- Modern Gemini SDK: Using google.genai package (version 1.65.0)
- Working Model: gemini-flash-latest (tested and confirmed)
- Real API Calls: Actual Gemini AI responses, not just fallback
- Smart Fallback: Rule-based insights when API quota exceeded
- Executive Summaries: Management-ready AI-generated insights

### Live Exchange Rates
- Real-time Conversion: Live USD to INR exchange rates
- API Integration: ExchangeRate-API.com for current rates
- Automatic Updates: Rate fetched on each dashboard load
- Fallback System: Uses estimated rate if API fails

### Professional UI
- Perfect Visibility: Black text (#000000) with proper contrast
- Clean Design: Business-ready interface without hackathon elements
- Indian Context: Localized content and examples
- Responsive Layout: Works across devices and screen sizes

## Requirements

- Python 3.8+
- Streamlit
- Google Gemini API key
- Virtual environment

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/voidutk/KUSH.git
cd KUSH
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install streamlit pandas numpy plotly requests python-dotenv google-genai
```

### 4. Set Up API Key
```bash
# Edit .env file and add your Gemini API key:
GEMINI_API_KEY=your_gemini_api_key_here
```

## Run Dashboard

```bash
streamlit run india_working_dashboard.py
```

The dashboard will open in your browser at http://localhost:8501

## Usage Guide

### 1. Initialize System
- Click "Initialize System" to train the risk assessment models
- Wait for training to complete (usually takes a few seconds)

### 2. Risk Assessment
- Navigate to "Risk Assessment" tab
- Enter parameters:
  - Region: Select Indian region (NA, EU, APAC)
  - SKU Class: Choose product category
  - Demand: Enter demand forecast
  - Inventory: Current inventory levels
  - Lead Time: Supply chain lead time
  - Single Source: Whether supplier is sole source
- Click "Assess Risk" for analysis
- View AI-powered insights and recommendations

### 3. Scenario Analysis
- Navigate to "Scenario Analysis" tab
- Create custom scenarios with different parameters
- Compare multiple scenarios side-by-side
- View cost-benefit analysis in INR

### 4. Analytics
- Navigate to "Analytics" tab
- View risk trends and patterns
- Check demand forecasting with seasonal factors
- Review supplier performance metrics
- Access executive summaries

## AI Integration Details

### Gemini API Configuration
- Model: gemini-flash-latest
- API: Google Gemini with modern google.genai SDK
- Authentication: API key from .env file
- Rate Limiting: Smart fallbacks when quota exceeded

### AI Insights
- Context-Aware: Indian market-specific analysis
- Structured Output: JSON format with risk assessment, factors, recommendations
- Executive Summaries: Management-ready insights
- Cost Analysis: Automatic USD to INR conversion

## Currency Conversion

### Live Exchange Rate
- API: ExchangeRate-API.com
- Endpoint: https://v6.exchangerate-api.com/v6/latest/USD
- Update Frequency: Real-time on dashboard load
- Fallback: Uses estimated rate (₹83.0) if API fails

### INR Formatting
- Crores: ₹1.0 Crore for amounts ≥ 1,00,00,000
- Lakhs: ₹10.0 Lakh for amounts ≥ 1,00,000
- Rupees: ₹50,000 for smaller amounts

## Business Value

### For Supply Chain Managers
- Risk Visibility: Clear understanding of supply chain vulnerabilities
- AI Insights: Strategic recommendations from real Gemini AI
- Cost Savings: Potential savings of ₹50.0 Lakhs monthly
- Proactive Management: Address issues before they become disruptions

### For Executive Decision-Making
- Executive Summaries: AI-generated strategic insights
- ROI Analysis: Clear business impact calculations
- Strategic Planning: Scenario analysis for future planning
- Market Intelligence: Indian-specific supply chain factors

## Technical Architecture

### Core Components
- corrected_engine.py: ML risk assessment engine
  - Monotonicity constraints
  - Structural dominance
  - Effective demand calculation
  - Fast model training

- india_working_dashboard.py: Main Streamlit application
  - Modern Gemini AI integration
  - Live exchange rate API
  - Professional UI with perfect visibility
  - Indian market localization

### API Integrations
- Google Gemini API: AI-powered insights
- ExchangeRate-API.com: Live currency conversion
- Smart Fallbacks: Rule-based insights when APIs fail

## Troubleshooting

### Common Issues

1. Gemini API Quota Exceeded
- Solution: Wait for quota reset or upgrade to paid tier
- Fallback: System automatically uses intelligent rule-based insights

2. Exchange Rate API Fails
- Solution: Dashboard uses fallback rate (₹83.0)
- Check: Internet connection and API availability

3. Models Not Trained
- Solution: Click "Initialize System" to train models
- Time: Usually takes 2-5 seconds

4. Text Visibility Issues
- Solution: All text uses black (#000000) with bold font-weight
- Browser: Ensure modern browser with CSS support

### Dependencies Issues
```bash
# If packages not found, install manually:
pip install streamlit pandas numpy plotly requests python-dotenv google-genai

# For Gemini API specifically:
pip install google-genai --upgrade
```

## Support

### Getting Help
1. Check README: Review this documentation
2. Verify Setup: Ensure all dependencies installed
3. API Keys: Confirm Gemini API key is valid
4. Network: Check internet connection for APIs

### Feature Requests
- Open an issue on GitHub for new features
- Report bugs with detailed error messages
- Suggest improvements for Indian market context

---

## Production Ready

This dashboard is fully functional and production-ready with:
- Real AI Integration: Working Gemini API with smart fallbacks
- Live Data: Real-time currency conversion
- Professional UI: Clean, business-ready interface
- Complete Documentation: Clear setup and usage instructions
- Robust Architecture: Error handling and graceful degradation

Ready for immediate deployment and use!
