import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt

class LSTMStockPredictor:
    def __init__(self, sequence_length=60, n_features=9):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.history = None
        
    def build_model(self, lstm_units=[50, 50], dropout_rate=0.2, learning_rate=0.001):
        """Build LSTM model architecture"""
        print("=== BUILDING LSTM MODEL ===")
        
        self.model = Sequential()
        
        # First LSTM layer
        self.model.add(LSTM(
            units=lstm_units[0],
            return_sequences=True,
            input_shape=(self.sequence_length, self.n_features)
        ))
        self.model.add(Dropout(dropout_rate))
        self.model.add(BatchNormalization())
        
        # Second LSTM layer
        if len(lstm_units) > 1:
            self.model.add(LSTM(
                units=lstm_units[1],
                return_sequences=False
            ))
            self.model.add(Dropout(dropout_rate))
            self.model.add(BatchNormalization())
        
        # Dense layers
        self.model.add(Dense(25, activation='relu'))
        self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(1))
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        print("Model architecture:")
        self.model.summary()
        
        return self.model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the LSTM model"""
        if self.model is None:
            print("Model not built. Please build model first.")
            return
            
        print("=== TRAINING LSTM MODEL ===")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'best_lstm_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        return self.history
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MAE plot
        ax2.plot(self.history.history['mae'], label='Training MAE')
        ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            print("Model not trained. Please train model first.")
            return None
            
        predictions = self.model.predict(X)
        return predictions
    
    def save_model(self, filepath='lstm_stock_model.h5'):
        """Save the trained model"""
        if self.model is None:
            print("No model to save")
            return
            
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='lstm_stock_model.h5'):
        """Load a trained model"""
        try:
            self.model = tf.keras.models.load_model(filepath)
            print(f"Model loaded from {filepath}")
            return self.model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

# Example usage
if __name__ == "__main__":
    # This would typically be run after data preprocessing
    print("LSTM Model class defined successfully!")
    print("To use this model:")
    print("1. First run data_collection.py")
    print("2. Then run data_preprocessing.py")
    print("3. Finally use this class to train the model")
    
    # Example of model architecture
    predictor = LSTMStockPredictor(sequence_length=60, n_features=9)
    model = predictor.build_model(lstm_units=[100, 50], dropout_rate=0.3)
