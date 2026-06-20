from core.technical_analysis import TechnicalAnalyzer
import pandas as pd

# Initialize the analyzer
analyzer = TechnicalAnalyzer()

# Manually create a small set of fake price data
# We need 200 lines to satisfy the code's requirements
data = {
    'open': [100.0] * 200,
    'high': [105.0] * 200,
    'low': [95.0] * 200,
    'close': [102.0] * 200,
    'volume': [1000] * 200
}
df = pd.DataFrame(data)

# Run the test
print("--- Testing the logic ---")
results = analyzer.calculate_indicators(df)

# Print the result
print(f"RSI: {results['rsi']}")
print(f"Signal: {results['smc_suggested_signal']}")