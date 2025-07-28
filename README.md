# LSTM-Time-Series-Analysis-Project
 Stock Price Prediction using LSTM with Live yfinance Data Technical Stack yfinance: Live stock data retrieval pandas: Data manipulation numpy: Numerical computations matplotlib: Visualization scikit-learn: Preprocessing and metrics tensorflow/keras: LSTM implementation
Complete Implementation

Phase 1: Data Collection (`data_collection.py`)

- ✅ Live stock data retrieval using `yfinance`
- ✅ Basic data exploration and statistics
- ✅ Data visualization (price, volume, returns)
- ✅ Handles missing data and errors


Phase 2: Data Preprocessing (`data_preprocessing.py`)

- ✅ Missing value handling
- ✅ Feature engineering (MA, RSI, volatility, etc.)
- ✅ Data scaling with MinMaxScaler
- ✅ Time series sequence creation
- ✅ Train/validation/test split


Phase 3: LSTM Model (`lstm_model.py`)

- ✅ Multi-layer LSTM architecture
- ✅ Dropout and BatchNormalization for regularization
- ✅ Adam optimizer with learning rate scheduling
- ✅ Early stopping and model checkpointing
- ✅ Model saving/loading functionality


Phase 4: Training & Evaluation

- ✅ Complete training pipeline (`training_pipeline.py`)
- ✅ Comprehensive evaluation metrics (`evaluation_metrics.py`)
- ✅ Full evaluation with visualizations (`complete_evaluation.py`)


Key Metrics Implemented

- **RMSE** (Root Mean Square Error)
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)
- **Directional Accuracy**
- **R² Score**
- **Correlation Analysis**


Key Features

1. **Live Data Integration**: Real-time stock data from Yahoo Finance
2. **Advanced Feature Engineering**: Technical indicators (RSI, Moving Averages, Volatility)
3. **Robust LSTM Architecture**: Multi-layer with regularization
4. **Comprehensive Evaluation**: Multiple metrics and visualizations
5. **Trading Strategy Simulation**: Basic buy/sell signal testing
6. **Modular Design**: Easy to extend and modify


<img width="925" height="295" alt="Screenshot 2025-07-28 205508" src="https://github.com/user-attachments/assets/09ad4e62-0d8b-4136-a066-bfed6e4bbf54" />

How to Run

1. **Quick Demo**: Run `complete_evaluation.py` for a fast demonstration
2. **Full Pipeline**: Run `training_pipeline.py` for complete analysis
3. **Individual Components**: Run each script separately for specific tasks
