import numpy as np
import pandas as pd
from data_collection import StockDataCollector
from data_preprocessing import StockDataPreprocessor
from lstm_model import LSTMStockPredictor
import matplotlib.pyplot as plt

class StockPredictionPipeline:
    def __init__(self, symbol="AAPL", period="5y", sequence_length=60):
        self.symbol = symbol
        self.period = period
        self.sequence_length = sequence_length
        self.collector = None
        self.preprocessor = None
        self.predictor = None
        self.data = None
        
    def run_full_pipeline(self):
        """Run the complete pipeline from data collection to model training"""
        print("=== STARTING FULL PIPELINE ===")
        
        # Step 1: Data Collection
        print("\n1. DATA COLLECTION")
        self.collector = StockDataCollector(self.symbol, self.period)
        self.data = self.collector.fetch_data()
        
        if self.data is None:
            print("Failed to collect data. Stopping pipeline.")
            return False
            
        self.collector.basic_exploration()
        
        # Step 2: Data Preprocessing
        print("\n2. DATA PREPROCESSING")
        self.preprocessor = StockDataPreprocessor(
            data=self.data, 
            sequence_length=self.sequence_length
        )
        
        self.preprocessor.handle_missing_values()
        self.preprocessor.feature_engineering()
        
        # Scale data and create sequences
        scaled_data = self.preprocessor.scale_data()
        if scaled_data is None:
            print("Failed to scale data. Stopping pipeline.")
            return False
            
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.split_data()
        
        # Step 3: Model Building and Training
        print("\n3. MODEL TRAINING")
        n_features = X_train.shape[2]
        self.predictor = LSTMStockPredictor(
            sequence_length=self.sequence_length,
            n_features=n_features
        )
        
        # Build model
        model = self.predictor.build_model(
            lstm_units=[100, 50, 25],
            dropout_rate=0.3,
            learning_rate=0.001
        )
        
        # Train model
        history = self.predictor.train_model(
            X_train, y_train, X_val, y_val,
            epochs=100,
            batch_size=32
        )
        
        # Plot training history
        self.predictor.plot_training_history()
        
        # Step 4: Save results
        print("\n4. SAVING RESULTS")
        self.predictor.save_model(f'lstm_{self.symbol}_model.h5')
        
        # Store data for evaluation
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
        return True
    
    def quick_demo(self):
        """Run a quick demo with sample data"""
        print("=== RUNNING QUICK DEMO ===")
        
        # Use a smaller dataset for demo
        self.collector = StockDataCollector(self.symbol, "2y")  # 2 years of data
        self.data = self.collector.fetch_data()
        
        if self.data is None:
            print("Failed to collect data for demo.")
            return False
            
        print(f"Demo data shape: {self.data.shape}")
        
        # Quick preprocessing
        self.preprocessor = StockDataPreprocessor(
            data=self.data, 
            sequence_length=30  # Shorter sequence for demo
        )
        
        self.preprocessor.handle_missing_values()
        self.preprocessor.feature_engineering()
        scaled_data = self.preprocessor.scale_data()
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.split_data()
        
        # Quick model training
        n_features = X_train.shape[2]
        self.predictor = LSTMStockPredictor(
            sequence_length=30,
            n_features=n_features
        )
        
        model = self.predictor.build_model(
            lstm_units=[50, 25],
            dropout_rate=0.2,
            learning_rate=0.001
        )
        
        # Train for fewer epochs
        history = self.predictor.train_model(
            X_train, y_train, X_val, y_val,
            epochs=20,
            batch_size=16
        )
        
        self.predictor.plot_training_history()
        
        print("Demo completed successfully!")
        return True

# Run the pipeline
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = StockPredictionPipeline("AAPL", "3y", 60)
    
    # Ask user for demo or full pipeline
    print("Choose an option:")
    print("1. Run quick demo (faster, less data)")
    print("2. Run full pipeline (complete analysis)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        success = pipeline.quick_demo()
    else:
        success = pipeline.run_full_pipeline()
    
    if success:
        print("\n🎉 Pipeline executed successfully!")
        print("Your LSTM stock prediction model is ready!")
    else:
        print("\n❌ Pipeline failed. Please check the error messages above.")
