import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'northwind'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', '12345')
}

# Model configuration
MODEL_CONFIG = {
    'input_dim': 18,  # Updated to match actual number of features
    'hidden_layers': [64, 32, 16],
    'output_dim': 1,  # Binary classification
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'validation_split': 0.2,
    'early_stopping_patience': 10
}

# Feature engineering configuration
FEATURE_CONFIG = {
    'target_category': 'Beverages',  # Example target category
    'min_purchase_amount': 100,  # Minimum purchase amount to consider
    'time_window_days': 365  # Time window for feature calculation
}

# RED GREEN REFACTOR
