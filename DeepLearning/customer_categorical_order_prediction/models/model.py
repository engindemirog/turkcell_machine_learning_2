import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from config.config import MODEL_CONFIG

class CustomerPurchaseModel:
    def __init__(self):
        self.model = None
        self.config = MODEL_CONFIG

    def build_model(self):
        """Build the neural network model"""
        model = Sequential()
        
        # Input layer
        model.add(Dense(self.config['hidden_layers'][0], 
                       input_dim=self.config['input_dim'],
                       activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        # Hidden layers
        for units in self.config['hidden_layers'][1:]:
            model.add(Dense(units, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
        
        # Output layer
        model.add(Dense(self.config['output_dim'], activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        self.model = model
        return model

    def train(self, X_train, y_train, X_val, y_val):
        """Train the model"""
        if self.model is None:
            self.build_model()
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True
            ),
            ModelCheckpoint(
                'models/best_model.keras',
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        return history

    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model has not been built or trained yet")
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model has not been built or trained yet")
        return self.model.evaluate(X_test, y_test)

    def save_model(self, filepath):
        """Save model to file"""
        if self.model is None:
            raise ValueError("Model has not been built or trained yet")
        if not filepath.endswith('.keras'):
            filepath = filepath.replace('.h5', '.keras')
        self.model.save(filepath)

    def load_model(self, filepath):
        """Load model from file"""
        if not filepath.endswith('.keras'):
            filepath = filepath.replace('.h5', '.keras')
        self.model = tf.keras.models.load_model(filepath) 