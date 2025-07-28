import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class StockDataPreprocessor:
    def __init__(self, data=None, target_column='Close', sequence_length=60):
        self.data = data
        self.target_column = target_column
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.scaled_data = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
    def load_data(self, file_path='stock_data.csv'):
        """Load data from CSV file"""
        try:
            self.data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            print(f"Data loaded successfully: {self.data.shape}")
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def handle_missing_values(self):
        """Handle missing values in the dataset"""
        if self.data is None:
            print("No data available")
            return
            
        print("=== HANDLING MISSING VALUES ===")
        missing_before = self.data.isnull().sum().sum()
        print(f"Missing values before: {missing_before}")
        
        # Forward fill then backward fill
        self.data = self.data.fillna(method='ffill').fillna(method='bfill')
        
        missing_after = self.data.isnull().sum().sum()
        print(f"Missing values after: {missing_after}")
        
    def feature_engineering(self):
        """Create additional features"""
        if self.data is None:
            print("No data available")
            return
            
        print("=== FEATURE ENGINEERING ===")
        
        # Technical indicators
        self.data['MA_5'] = self.data[self.target_column].rolling(window=5).mean()
        self.data['MA_10'] = self.data[self.target_column].rolling(window=10).mean()
        self.data['MA_20'] = self.data[self.target_column].rolling(window=20).mean()
        
        # Price changes
        self.data['Price_Change'] = self.data[self.target_column].diff()
        self.data['Price_Change_Pct'] = self.data[self.target_column].pct_change()
        
        # Volatility (rolling standard deviation)
        self.data['Volatility'] = self.data[self.target_column].rolling(window=10).std()
        
        # High-Low spread
        self.data['HL_Spread'] = self.data['High'] - self.data['Low']
        
        # Volume indicators
        self.data['Volume_MA'] = self.data['Volume'].rolling(window=10).mean()
        
        # RSI (Relative Strength Index)
        delta = self.data[self.target_column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # Remove NaN values created by rolling operations
        self.data = self.data.dropna()
        
        print(f"Features created. New shape: {self.data.shape}")
        print("New features:", [col for col in self.data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']])
        
    def scale_data(self, features_to_use=None):
        """Scale the data using MinMaxScaler"""
        if self.data is None:
            print("No data available")
            return
            
        if features_to_use is None:
            features_to_use = [self.target_column, 'Volume', 'MA_5', 'MA_10', 'MA_20', 
                             'Price_Change', 'Volatility', 'HL_Spread', 'RSI']
        
        # Select features that exist in the data
        available_features = [f for f in features_to_use if f in self.data.columns]
        print(f"Using features: {available_features}")
        
        # Scale the selected features
        self.scaled_data = self.scaler.fit_transform(self.data[available_features])
        print(f"Data scaled. Shape: {self.scaled_data.shape}")
        
        return self.scaled_data
    
    def create_sequences(self):
        """Create sequences for LSTM training"""
        if self.scaled_data is None:
            print("No scaled data available. Please scale data first.")
            return
            
        print("=== CREATING SEQUENCES ===")
        
        X, y = [], []
        
        for i in range(self.sequence_length, len(self.scaled_data)):
            X.append(self.scaled_data[i-self.sequence_length:i])
            y.append(self.scaled_data[i, 0])  # Assuming target is first column (Close price)
        
        X, y = np.array(X), np.array(y)
        print(f"Sequences created - X shape: {X.shape}, y shape: {y.shape}")
        
        return X, y
    
    def split_data(self, test_size=0.2, val_size=0.2):
        """Split data into train, validation, and test sets"""
        X, y = self.create_sequences()
        
        if X is None or y is None:
            print("Failed to create sequences")
            return
            
        print("=== SPLITTING DATA ===")
        
        # First split: separate test set
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, shuffle=False
        )
        
        print(f"Train set: X={self.X_train.shape}, y={self.y_train.shape}")
        print(f"Validation set: X={self.X_val.shape}, y={self.y_val.shape}")
        print(f"Test set: X={self.X_test.shape}, y={self.y_test.shape}")
        
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def visualize_preprocessing(self):
        """Visualize the preprocessing results"""
        if self.data is None:
            print("No data available")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Data Preprocessing Visualization', fontsize=16)
        
        # Original vs Moving Averages
        axes[0, 0].plot(self.data.index, self.data[self.target_column], label='Close Price', alpha=0.7)
        axes[0, 0].plot(self.data.index, self.data['MA_5'], label='MA 5', alpha=0.8)
        axes[0, 0].plot(self.data.index, self.data['MA_20'], label='MA 20', alpha=0.8)
        axes[0, 0].set_title('Price with Moving Averages')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # RSI
        axes[0, 1].plot(self.data.index, self.data['RSI'])
        axes[0, 1].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
        axes[0, 1].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
        axes[0, 1].set_title('RSI (Relative Strength Index)')
        axes[0, 1].set_ylabel('RSI')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Volatility
        axes[1, 0].plot(self.data.index, self.data['Volatility'])
        axes[1, 0].set_title('Price Volatility (10-day rolling std)')
        axes[1, 0].set_ylabel('Volatility')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Volume
        axes[1, 1].plot(self.data.index, self.data['Volume'], alpha=0.7)
        axes[1, 1].plot(self.data.index, self.data['Volume_MA'], color='red', label='Volume MA')
        axes[1, 1].set_title('Trading Volume')
        axes[1, 1].set_ylabel('Volume')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = StockDataPreprocessor(sequence_length=60)
    
    # Load data
    data = preprocessor.load_data('stock_data.csv')
    
    if data is not None:
        # Preprocessing pipeline
        preprocessor.handle_missing_values()
        preprocessor.feature_engineering()
        preprocessor.visualize_preprocessing()
        
        # Scale data and create sequences
        scaled_data = preprocessor.scale_data()
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data()
        
        print("\nPreprocessing completed successfully!")
        print("Data is ready for LSTM training.")
