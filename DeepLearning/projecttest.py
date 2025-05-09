import pandas as pd
import numpy as np
import tensorflow as tf
import psycopg2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

connection = psycopg2.connect(host="localhost",dbname="northwind",user="postgres",password="12345",port=5432)

query = """
WITH last_order_date AS (
SELECT MAX(order_date) AS max_date FROM orders
),
customer_order_stats AS (
SELECT
    c.customer_id, 
    COUNT(o.order_id) AS total_orders, -- burada düzeltildi
    SUM(od.unit_price * od.quantity) AS total_spent,
    AVG(od.unit_price * od.quantity) AS avg_order_value,
    MAX(o.order_date) AS last_order_date
FROM customers c
  LEFT JOIN orders o ON c.customer_id = o.customer_id
  LEFT JOIN order_details od ON o.order_id = od.order_id
  GROUP BY c.customer_id
),
label_data AS (
SELECT
    c.customer_id,
CASE
WHEN EXISTS (
        SELECT 1 
        FROM orders o2, last_order_date lod
        WHERE o2.customer_id = c.customer_id
        AND o2.order_date > (lod.max_date - INTERVAL '6 months')
)
      THEN 1 ELSE 0
END AS will_order_again
FROM customers c
)
SELECT
  s.customer_id, 
  s.total_orders, 
  s.total_spent, 
  s.avg_order_value, 
  s.last_order_date, 
  l.will_order_again
FROM customer_order_stats s
JOIN label_data l ON s.customer_id = l.customer_id;
"""

df = pd.read_sql(query,connection)
connection.close()

X = df[["total_orders","total_spent","avg_order_value"]]
y = df[["will_order_again"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(8,activation="relu",input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(4,activation="relu"),
    tf.keras.layers.Dense(1,activation="sigmoid")
])

model.compile(optimizer = "adam", loss="mean_squared_error",metrics=["accuracy"])

model.fit(X_train,y_train,epochs=50,validation_data=(X_test,y_test),verbose=1)

loss,acc = model.evaluate(X_test,y_test)

print(acc)