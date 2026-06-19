# AI Forex Pro

24/7 AI-powered trading signals with 0.5% threshold.

## Data Sources
- **EUR/USD**: deriv API (Professional forex data)
- **BTC/USD**: Binance API (Real-time crypto data)

## Notifications
- SendGrid Email
- Twilio SMS

## Setup

1. Get API keys:
   - OANDA: https://www.deriv.com/demo-account/
   - Binance: https://www.binance.com/en/my/settings/api-management
   - SendGrid: https://app.sendgrid.com/
   - Twilio: https://console.twilio.com/

2. Add keys to `.env` file

3. Run:
   ```bash
   pip install -r requirements.txt
   redis-server
   uvicorn main:app --reload