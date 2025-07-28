# Requirements for the LSTM Stock Prediction Project
requirements = [
    "yfinance>=0.2.18",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "scikit-learn>=1.3.0",
    "tensorflow>=2.13.0",
    "plotly>=5.15.0"
]

print("Required packages for LSTM Stock Prediction Project:")
for req in requirements:
    print(f"- {req}")

print("\nInstall with: pip install " + " ".join([req.split(">=")[0] for req in requirements]))
