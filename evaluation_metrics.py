import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats

class ModelEvaluator:
    def __init__(self, scaler=None):
        self.scaler = scaler
        self.metrics = {}
        
    def calculate_metrics(self, y_true, y_pred, dataset_name=""):
        """Calculate comprehensive evaluation metrics"""
        
        # Inverse transform if scaler is provided
        if self.scaler is not None:
            # Create dummy array for inverse transform
            dummy_true = np.zeros((len(y_true), self.scaler.n_features_in_))
            dummy_pred = np.zeros((len(y_pred), self.scaler.n_features_in_))
            dummy_true[:, 0] = y_true.flatten()
            dummy_pred[:, 0] = y_pred.flatten()
            
            y_true_scaled = self.scaler.inverse_transform(dummy_true)[:, 0]
            y_pred_scaled = self.scaler.inverse_transform(dummy_pred)[:, 0]
        else:
            y_true_scaled = y_true.flatten()
            y_pred_scaled = y_pred.flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_true_scaled, y_pred_scaled)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_scaled, y_pred_scaled)
        mape = np.mean(np.abs((y_true_scaled - y_pred_scaled) / y_true_scaled)) * 100
        
        # Directional accuracy
        true_direction = np.diff(y_true_scaled) > 0
        pred_direction = np.diff(y_pred_scaled) > 0
        directional_accuracy = np.mean(true_direction == pred_direction) * 100
        
        # R-squared
        ss_res = np.sum((y_true_scaled - y_pred_scaled) ** 2)
        ss_tot = np.sum((y_true_scaled - np.mean(y_true_scaled)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Correlation
        correlation, _ = stats.pearsonr(y_true_scaled, y_pred_scaled)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'Directional_Accuracy': directional_accuracy,
            'R2': r2,
            'Correlation': correlation
        }
        
        self.metrics[dataset_name] = metrics
        
        print(f"\n=== {dataset_name.upper()} METRICS ===")
        print(f"MSE: {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"Directional Accuracy: {directional_accuracy:.2f}%")
        print(f"R²: {r2:.4f}")
        print(f"Correlation: {correlation:.4f}")
        
        return metrics
    
    def plot_predictions(self, y_true, y_pred, dataset_name="", dates=None):
        """Plot predictions vs actual values"""
        
        # Inverse transform if scaler is provided
        if self.scaler is not None:
            dummy_true = np.zeros((len(y_true), self.scaler.n_features_in_))
            dummy_pred = np.zeros((len(y_pred), self.scaler.n_features_in_))
            dummy_true[:, 0] = y_true.flatten()
            dummy_pred[:, 0] = y_pred.flatten()
            
            y_true_scaled = self.scaler.inverse_transform(dummy_true)[:, 0]
            y_pred_scaled = self.scaler.inverse_transform(dummy_pred)[:, 0]
        else:
            y_true_scaled = y_true.flatten()
            y_pred_scaled = y_pred.flatten()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{dataset_name} Predictions Analysis', fontsize=16)
        
        # Time series plot
        if dates is not None:
            axes[0, 0].plot(dates, y_true_scaled, label='Actual', alpha=0.7)
            axes[0, 0].plot(dates, y_pred_scaled, label='Predicted', alpha=0.7)
        else:
            axes[0, 0].plot(y_true_scaled, label='Actual', alpha=0.7)
            axes[0, 0].plot(y_pred_scaled, label='Predicted', alpha=0.7)
        
        axes[0, 0].set_title('Actual vs Predicted Prices')
        axes[0, 0].set_ylabel('Price ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Scatter plot
        axes[0, 1].scatter(y_true_scaled, y_pred_scaled, alpha=0.6)
        min_val = min(y_true_scaled.min(), y_pred_scaled.min())
        max_val = max(y_true_scaled.max(), y_pred_scaled.max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        axes[0, 1].set_xlabel('Actual Price ($)')
        axes[0, 1].set_ylabel('Predicted Price ($)')
        axes[0, 1].set_title('Actual vs Predicted Scatter')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = y_true_scaled - y_pred_scaled
        axes[1, 0].scatter(y_pred_scaled, residuals, alpha=0.6)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.8)
        axes[1, 0].set_xlabel('Predicted Price ($)')
        axes[1, 0].set_ylabel('Residuals ($)')
        axes[1, 0].set_title('Residuals Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals histogram
        axes[1, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Residuals ($)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Residuals Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return y_true_scaled, y_pred_scaled
    
    def plot_directional_accuracy(self, y_true, y_pred, dataset_name=""):
        """Plot directional accuracy analysis"""
        
        # Inverse transform if scaler is provided
        if self.scaler is not None:
            dummy_true = np.zeros((len(y_true), self.scaler.n_features_in_))
            dummy_pred = np.zeros((len(y_pred), self.scaler.n_features_in_))
            dummy_true[:, 0] = y_true.flatten()
            dummy_pred[:, 0] = y_pred.flatten()
            
            y_true_scaled = self.scaler.inverse_transform(dummy_true)[:, 0]
            y_pred_scaled = self.scaler.inverse_transform(dummy_pred)[:, 0]
        else:
            y_true_scaled = y_true.flatten()
            y_pred_scaled = y_pred.flatten()
        
        # Calculate directions
        true_direction = np.diff(y_true_scaled) > 0
        pred_direction = np.diff(y_pred_scaled) > 0
        
        # Create comparison
        correct_predictions = true_direction == pred_direction
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Direction comparison over time
        ax1.plot(correct_predictions.astype(int), 'o-', alpha=0.7, markersize=3)
        ax1.set_title(f'{dataset_name} Directional Accuracy Over Time')
        ax1.set_ylabel('Correct (1) / Incorrect (0)')
        ax1.set_xlabel('Time Step')
        ax1.grid(True, alpha=0.3)
        
        # Accuracy summary
        accuracy = np.mean(correct_predictions) * 100
        ax2.bar(['Correct', 'Incorrect'], 
                [accuracy, 100-accuracy], 
                color=['green', 'red'], alpha=0.7)
        ax2.set_title(f'Overall Directional Accuracy: {accuracy:.1f}%')
        ax2.set_ylabel('Percentage')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self):
        """Generate a comprehensive evaluation report"""
        if not self.metrics:
            print("No metrics calculated yet.")
            return
            
        print("\n" + "="*50)
        print("COMPREHENSIVE EVALUATION REPORT")
        print("="*50)
        
        for dataset_name, metrics in self.metrics.items():
            print(f"\n{dataset_name.upper()} DATASET:")
            print("-" * 30)
            for metric_name, value in metrics.items():
                if metric_name in ['MAPE', 'Directional_Accuracy']:
                    print(f"{metric_name}: {value:.2f}%")
                else:
                    print(f"{metric_name}: {value:.6f}")
        
        print("\n" + "="*50)
        print("INTERPRETATION GUIDE:")
        print("="*50)
        print("• RMSE/MAE: Lower is better (prediction error)")
        print("• MAPE: Lower is better (percentage error)")
        print("• R²: Higher is better (0-1, explained variance)")
        print("• Correlation: Higher is better (-1 to 1)")
        print("• Directional Accuracy: Higher is better (trend prediction)")

# Example usage
if __name__ == "__main__":
    print("Model Evaluator class defined successfully!")
    print("This will be used in the complete evaluation pipeline.")
