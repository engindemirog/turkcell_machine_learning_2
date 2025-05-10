import pandas as pd
import numpy as np
from config import FEATURE_CONFIG
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.customer_features = None

    def create_feature(self,df):
        df["total_amount"] = df["unit_price"]*df["quantity"]*(1-df["discount"])
        df["discount_amount"] = df["unit_price"]*df["quantity"]*df["discount"]

        self.customer_features = df.groupby("customer_id").agg(
            {
                "total_amount":["mean","std","sum"],
                "discount":["mean","max"],
                "quantity":["mean","sum"]
            }
        ).reset_index()

        self.customer_features.columns = ["customer_id","avg_order_amount",
                                          "std_order_amount","total_spent","avg_discount",
                                          "max_discount","avg_quantity","total_quantity"]

        df = df.merge(self.customer_features,on="customer_id",how="left")

        high_discount = df["discount"]>df["discount"].quantile(FEATURE_CONFIG["high_discount_threshold"])
        low_amount = df["total_amount"]<df["total_amount"].quantile(FEATURE_CONFIG["low_amount_threshold"])

        df["return_risk"] = (high_discount & low_amount).astype(int)

        return df

    

    