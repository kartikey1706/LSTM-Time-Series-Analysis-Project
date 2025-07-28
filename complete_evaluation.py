import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_collection import StockDataCollector
from data_preprocessing import StockDataPreprocessor
from lstm_model import LSTMStockPredictor
from evaluation_metrics import ModelEvaluator

def run_complete_evaluation(symbol="AAPL", model_path=None):
    """Run complete evaluation pipeline"""
    
    print("=== COMPLETE LSTM STOCK PREDICTION EVALUATION ===")
    
    # Step 1: Load or train model
    if model_path and os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        predictor = LSTMStockPredictor()
        predictor.load_model(model_path)
    else:
        print("Training new model...")
        # This would run the full training pipeline
        # For demo, we'll create a simple example
        return run_demo_evaluation(symbol)
    
def run_demo_evaluation(symbol="AAPL"):
    """Run a complete demo evaluation"""
    
    print(f"=== DEMO EVALUATION FOR {symbol} ===")
    
    # Step 1: Collect data
    collector = StockDataCollector(symbol, "2y")
    data = collector.fetch_data()
    
    if data is None:
        print("Failed to collect data")
        return
    
    # Step 2: Preprocess data
    preprocessor = StockDataPreprocessor(data=data, sequence_length=30)
    preprocessor.handle_missing_values()
    preprocessor.feature_engineering()
    
    scaled_data = preprocessor.scale_data()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data()
    
    # Step 3: Train model
    n_features = X_train.shape[2]
    predictor = LSTMStockPredictor(sequence_length=30, n_features=n_features)
    
    model = predictor.build_model(lstm_units=[50, 25], dropout_rate=0.2)
    history = predictor.train_model(X_train, y_train, X_val, y_val, epochs=20, batch_size=16)
    
    # Step 4: Make predictions
    train_pred = predictor.predict(X_train)
    val_pred = predictor.predict(X_val)
    test_pred = predictor.predict(X_test)
    
    # Step 5: Evaluate model
    evaluator = ModelEvaluator(scaler=preprocessor.scaler)
    
    # Calculate metrics for all datasets
    train_metrics = evaluator.calculate_metrics(y_train, train_pred, "Training")
    val_metrics = evaluator.calculate_metrics(y_val, val_pred, "Validation")
    test_metrics = evaluator.calculate_metrics(y_test, test_pred, "Test")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Plot predictions for each dataset
    evaluator.plot_predictions(y_train, train_pred, "Training")
    evaluator.plot_predictions(y_val, val_pred, "Validation")
    evaluator.plot_predictions(y_test, test_pred, "Test")
    
    # Plot directional accuracy
    evaluator.plot_directional_accuracy(y_test, test_pred, "Test")
    
    # Plot training history
    predictor.plot_training_history()
    
    # Generate comprehensive report
    evaluator.generate_report()
    
    # Additional analysis
    print("\n=== ADDITIONAL ANALYSIS ===")
    
    # Feature importance visualization (simplified)
    feature_names = ['Close', 'Volume', 'MA_5', 'MA_10', 'MA_20', 
                    'Price_Change', 'Volatility', 'HL_Spread', 'RSI']
    
    plt.figure(figsize=(12, 6))
    
    # Plot actual vs predicted for test set
    y_test_actual, y_test_pred = evaluator.plot_predictions(y_test, test_pred, "Final Test Results")
    
    # Calculate and display final performance summary
    print("\n" + "="*60)
    print("FINAL PERFORMANCE SUMMARY")
    print("="*60)
    
    test_rmse = np.sqrt(np.mean((y_test_actual - y_test_pred)**2))
    test_mae = np.mean(np.abs(y_test_actual - y_test_pred))
    test_mape = np.mean(np.abs((y_test_actual - y_test_pred) / y_test_actual)) * 100
    
    print(f"Stock Symbol: {symbol}")
    print(f"Model Architecture: LSTM with {n_features} features")
    print(f"Sequence Length: 30 days")
    print(f"Test RMSE: ${test_rmse:.2f}")
    print(f"Test MAE: ${test_mae:.2f}")
    print(f"Test MAPE: {test_mape:.2f}%")
    
    # Trading strategy simulation (simplified)
    print("\n=== SIMPLE TRADING STRATEGY SIMULATION ===")
    
    # Buy when predicted price > current price, sell otherwise
    price_changes = np.diff(y_test_pred)
    buy_signals = price_changes > 0
    actual_returns = np.diff(y_test_actual)
    
    strategy_returns = []
    for i, should_buy in enumerate(buy_signals):
        if should_buy:
            strategy_returns.append(actual_returns[i])
        else:
            strategy_returns.append(-actual_returns[i])  # Short position
    
    total_return = np.sum(strategy_returns)
    win_rate = np.mean(np.array(strategy_returns) > 0) * 100
    
    print(f"Total Strategy Return: ${total_return:.2f}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Average Return per Trade: ${np.mean(strategy_returns):.2f}")
    
    print("\n🎉 Complete evaluation finished!")
    print("Your LSTM model analysis is ready!")
    
    return {
        'model': predictor,
        'evaluator': evaluator,
        'metrics': {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics
        },
        'predictions': {
            'train': (y_train, train_pred),
            'val': (y_val, val_pred),
            'test': (y_test, test_pred)
        }
    }

# Run the complete evaluation
if __name__ == "__main__":
    import os
    
    # Choose stock symbol
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
    print("Available stocks:", symbols)
    
    symbol = input("Enter stock symbol (or press Enter for AAPL): ").strip().upper()
    if not symbol:
        symbol = "AAPL"
    
    print(f"\nRunning complete evaluation for {symbol}...")
    
    # Run evaluation
    results = run_demo_evaluation(symbol)
    
    if results:
        print(f"\n✅ Evaluation completed successfully for {symbol}!")
        print("Check the generated plots and metrics above.")
        print("\nKey files generated:")
        print("- Model weights saved")
        print("- Evaluation metrics calculated")
        print("- Visualizations created")
    else:
        print(f"\n❌ Evaluation failed for {symbol}")
