import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.data.database import DatabaseConnection
from src.features.feature_engineering import FeatureEngineering
from models.model import CustomerPurchaseModel
import matplotlib.pyplot as plt
import seaborn as sns

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ['data/raw', 'data/processed', 'models']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    plt.close()

def main():
    # Create necessary directories
    create_directories()
    
    # Initialize database connection
    db = DatabaseConnection()
    try:
        db.connect()
        
        # Get data from database
        print("Fetching data from database...")
        purchase_history = db.get_customer_purchase_history()
        customer_features = db.get_customer_features()
        
        # Save raw data
        purchase_history.to_csv('data/raw/purchase_history.csv', index=False)
        customer_features.to_csv('data/raw/customer_features.csv', index=False)
        
        # Feature engineering
        print("Performing feature engineering...")
        feature_eng = FeatureEngineering()
        features = feature_eng.create_customer_features(purchase_history, customer_features)
        
        # Save processed features
        features.to_csv('data/processed/processed_features.csv', index=False)
        
        # Prepare training data
        X = feature_eng.prepare_training_data(features)
        
        # Create target variable (example: whether customer purchased in target category)
        target_category = feature_eng.target_category
        y = (features[f'{target_category}_spent'] > 0).astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Train model
        print("Training model...")
        model = CustomerPurchaseModel()
        history = model.train(X_train, y_train, X_val, y_val)
        
        # Plot training history
        plot_training_history(history)
        
        # Evaluate model
        print("\nEvaluating model...")
        test_loss, test_accuracy, test_auc = model.evaluate(X_test, y_test)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        
        # Save model
        model.save_model('models/final_model.keras')
        print("\nModel saved successfully!")
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        db.disconnect()

if __name__ == "__main__":
    main() 