import tensorflow as tf
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

from sklearn.model_selection import train_test_split
from src.config import MODEL_CONFIG

class ReturnRiskModel:
    def __init__(self):
        self.model = None
        self.history = None

    def build_model(self,input_dim):
        model = Sequential(
            [
                Dense(64,activation = "relu",input_shape = (input_dim,)),
                Dropout(0.3),
                Dense(32,activation = "relu"),
                Dropout(0.2),
                Dense(16,activation = "relu"),
                Dense(1,activation ="sigmoid")
            ]
        )

        model.compile(
            optimizer = Adam(),
            loss = "binary_crossentropy",
            metrics = ["accuracy"]
        )

        self.model = model

    def split_data(self,X,y):
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=MODEL_CONFIG["test_size"],random_state=MODEL_CONFIG["random_state"])
        return  X_train,X_test,y_train,y_test

    def train(self, X_train,X_test,y_train,y_test):
        callbacks = [
            EarlyStopping(monitor="val_loss",patience=5,restore_best_weights=True),
            ModelCheckpoint("best_model.keras",monitor="vall_loss",save_best_only=True)
        ]

        self.history =  self.model.fit(
            X_train,y_train,
            epochs= MODEL_CONFIG["epochs"],
            validation_data = (X_test,y_test),
            callbacks = callbacks,
            verbose = 1
            )
    
    def evaluate(self,X_test,y_test):
        return self.model.evaluate(X_test,y_test)
    
    def predict(self,X):
        return self.model.predict(X)