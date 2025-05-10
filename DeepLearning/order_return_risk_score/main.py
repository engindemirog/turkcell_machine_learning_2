from src.database import DatabaseManager
from src.feature_engineering import FeatureEngineer
from src.model import ReturnRiskModel



def main():
    db_manager = DatabaseManager()
    feature_engineer = FeatureEngineer()
    model = ReturnRiskModel()

    print("Fetching order data")
    df = db_manager.get_order_data()

    print("Creating features")
    df_processed =  feature_engineer.create_features(df)

    X,y = feature_engineer.prepare_model_data(df_processed)

    X_train,X_test,y_train,y_test = model.split_data(X,y)

    model.build_model(input_dim=X_train.shape[1])

    model.train(X_train,X_test,y_train,y_test)

    accuracy = model.evaluate(X_test,y_test)

    print(f"Accuracy Score : {accuracy}")

if __name__ == "__main__":
    main()