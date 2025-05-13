import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config.config import FEATURE_CONFIG

class FeatureEngineering:
    def __init__(self):
        self.target_category = FEATURE_CONFIG['target_category']
        self.min_purchase_amount = FEATURE_CONFIG['min_purchase_amount']
        self.time_window_days = FEATURE_CONFIG['time_window_days']

    def create_customer_features(self, purchase_history_df, customer_features_df):
        """Create features for customer purchase prediction"""
        print("\nCreating customer features...")
        print(f"Purchase history shape: {purchase_history_df.shape}")
        print(f"Customer features shape: {customer_features_df.shape}")
        
        # Create category-based features
        print("\nCreating category-based features...")
        category_features = self._create_category_features(purchase_history_df)
        print(f"Category features shape: {category_features.shape}")
        
        # Create time-based features
        print("\nCreating time-based features...")
        time_features = self._create_time_features(customer_features_df)
        print(f"Time features shape: {time_features.shape}")
        
        # Create spending pattern features
        print("\nCreating spending pattern features...")
        spending_features = self._create_spending_features(purchase_history_df)
        print(f"Spending features shape: {spending_features.shape}")
        
        # Merge all features
        print("\nMerging all features...")
        final_features = pd.merge(category_features, time_features, on='customer_id', how='left')
        final_features = pd.merge(final_features, spending_features, on='customer_id', how='left')
        print(f"Final features shape: {final_features.shape}")
        print("\nFeature columns:", final_features.columns.tolist())
        
        return final_features

    def _create_category_features(self, df):
        """Create features based on category purchases"""
        # Pivot the data to get category-wise spending
        category_pivot = df.pivot(
            index='customer_id',
            columns='category_name',
            values='total_spent'
        ).fillna(0)
        
        # Calculate category spending ratios
        total_spent = category_pivot.sum(axis=1)
        category_ratios = category_pivot.div(total_spent, axis=0)
        
        # Add target category specific features
        if self.target_category in category_pivot.columns:
            category_pivot[f'{self.target_category}_ratio'] = category_ratios[self.target_category]
            category_pivot[f'{self.target_category}_spent'] = category_pivot[self.target_category]
        
        return category_pivot.reset_index()

    def _create_time_features(self, df):
        """Create time-based features"""
        df = df.copy()
        
        # Convert dates to datetime
        df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])
        df['first_purchase_date'] = pd.to_datetime(df['first_purchase_date'])
        
        # Calculate time-based features
        current_date = datetime.now()
        df['days_since_last_purchase'] = (current_date - df['last_purchase_date']).dt.days
        df['customer_tenure_days'] = (df['last_purchase_date'] - df['first_purchase_date']).dt.days
        df['purchase_frequency'] = df['total_orders'] / df['customer_tenure_days']
        
        return df[['customer_id', 'days_since_last_purchase', 'customer_tenure_days', 'purchase_frequency']]

    def _create_spending_features(self, df):
        """Create spending pattern features"""
        spending_features = df.groupby('customer_id').agg({
            'total_spent': ['sum', 'mean', 'std'],
            'order_count': ['sum', 'mean']
        }).reset_index()
        
        spending_features.columns = ['customer_id', 'total_spent', 'avg_spent_per_category',
                                   'std_spent_per_category', 'total_orders', 'avg_orders_per_category']
        
        return spending_features

    def prepare_training_data(self, features_df):
        """Prepare data for model training"""
        print("\nPreparing training data...")
        # Select features for training
        feature_columns = [col for col in features_df.columns if col != 'customer_id']
        print(f"Number of features: {len(feature_columns)}")
        print("Feature columns:", feature_columns)
        
        # Handle missing values
        features_df[feature_columns] = features_df[feature_columns].fillna(0)
        
        # Normalize numerical features
        for col in feature_columns:
            if features_df[col].dtype in ['float64', 'int64']:
                mean = features_df[col].mean()
                std = features_df[col].std()
                if std != 0:
                    features_df[col] = (features_df[col] - mean) / std
        
        return features_df[feature_columns] 