import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

class StockDataCollector:
    def __init__(self, symbol="AAPL", period="5y"):
        self.symbol = symbol
        self.period = period
        self.data = None
        
    def fetch_data(self):
        """Fetch historical stock data from Yahoo Finance"""
        try:
            print(f"Fetching data for {self.symbol}...")
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=self.period)
            
            if self.data.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")
                
            print(f"Successfully fetched {len(self.data)} records")
            return self.data
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def basic_exploration(self):
        """Perform basic data exploration"""
        if self.data is None:
            print("No data available. Please fetch data first.")
            return
            
        print("\n=== BASIC DATA EXPLORATION ===")
        print(f"Dataset shape: {self.data.shape}")
        print(f"Date range: {self.data.index[0]} to {self.data.index[-1]}")
        print("\nColumn names:", list(self.data.columns))
        print("\nFirst 5 rows:")
        print(self.data.head())
        print("\nLast 5 rows:")
        print(self.data.tail())
        print("\nBasic statistics:")
        print(self.data.describe())
        print("\nMissing values:")
        print(self.data.isnull().sum())
        
    def visualize_data(self):
        """Create basic visualizations"""
        if self.data is None:
            print("No data available. Please fetch data first.")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.symbol} Stock Analysis', fontsize=16)
        
        # Price over time
        axes[0, 0].plot(self.data.index, self.data['Close'], linewidth=1)
        axes[0, 0].set_title('Closing Price Over Time')
        axes[0, 0].set_ylabel('Price ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Volume over time
        axes[0, 1].plot(self.data.index, self.data['Volume'], color='orange', linewidth=1)
        axes[0, 1].set_title('Trading Volume Over Time')
        axes[0, 1].set_ylabel('Volume')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Price distribution
        axes[1, 0].hist(self.data['Close'], bins=50, alpha=0.7, color='green')
        axes[1, 0].set_title('Closing Price Distribution')
        axes[1, 0].set_xlabel('Price ($)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Daily returns
        daily_returns = self.data['Close'].pct_change().dropna()
        axes[1, 1].hist(daily_returns, bins=50, alpha=0.7, color='red')
        axes[1, 1].set_title('Daily Returns Distribution')
        axes[1, 1].set_xlabel('Daily Return')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Additional statistics
        print(f"\n=== PRICE STATISTICS ===")
        print(f"Current Price: ${self.data['Close'][-1]:.2f}")
        print(f"52-week High: ${self.data['Close'].max():.2f}")
        print(f"52-week Low: ${self.data['Close'].min():.2f}")
        print(f"Average Daily Return: {daily_returns.mean():.4f} ({daily_returns.mean()*100:.2f}%)")
        print(f"Volatility (std): {daily_returns.std():.4f} ({daily_returns.std()*100:.2f}%)")

# Example usage
if __name__ == "__main__":
    # Initialize collector
    collector = StockDataCollector("AAPL", "5y")
    
    # Fetch and explore data
    data = collector.fetch_data()
    if data is not None:
        collector.basic_exploration()
        collector.visualize_data()
        
        # Save data for later use
        print("\nSaving data to CSV...")
        data.to_csv('stock_data.csv')
        print("Data saved successfully!")
