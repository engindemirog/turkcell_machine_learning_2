import pandas as pd
import psycopg2
from psycopg2 import sql
from config.config import DB_CONFIG

class DatabaseConnection:
    def __init__(self):
        self.conn = None
        self.cursor = None
        self._queries = {
            'customer_purchase_history': """
                WITH customer_category_purchases AS (
                    SELECT 
                        c.customer_id,
                        cat.category_name,
                        SUM(od.quantity * od.unit_price) as total_spent,
                        COUNT(DISTINCT o.order_id) as order_count,
                        MAX(o.order_date) as last_purchase_date
                    FROM customers c
                    JOIN orders o ON c.customer_id = o.customer_id
                    JOIN order_details od ON o.order_id = od.order_id
                    JOIN products p ON od.product_id = p.product_id
                    JOIN categories cat ON p.category_id = cat.category_id
                    GROUP BY c.customer_id, cat.category_name
                )
                SELECT 
                    customer_id,
                    category_name,
                    total_spent,
                    order_count,
                    last_purchase_date
                FROM customer_category_purchases
                ORDER BY customer_id, category_name;
            """,
            'customer_features': """
                WITH customer_features AS (
                    SELECT 
                        c.customer_id,
                        COUNT(DISTINCT o.order_id) as total_orders,
                        SUM(od.quantity * od.unit_price) as total_spent,
                        AVG(od.quantity * od.unit_price) as avg_order_value,
                        MAX(o.order_date) as last_purchase_date,
                        MIN(o.order_date) as first_purchase_date
                    FROM customers c
                    JOIN orders o ON c.customer_id = o.customer_id
                    JOIN order_details od ON o.order_id = od.order_id
                    GROUP BY c.customer_id
                )
                SELECT * FROM customer_features;
            """
        }

    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**DB_CONFIG)
            self.cursor = self.conn.cursor()
            print("Database connection established successfully")
        except Exception as e:
            print(f"Error connecting to database: {e}")
            raise

    def disconnect(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            print("Database connection closed")

    def execute_query(self, query_name, params=None):
        """
        Execute a predefined query and return results as DataFrame
        
        Args:
            query_name (str): Name of the query to execute
            params (dict, optional): Parameters for the query
            
        Returns:
            pd.DataFrame: Query results
        """
        if query_name not in self._queries:
            raise ValueError(f"Query '{query_name}' not found in predefined queries")
        
        try:
            query = self._queries[query_name]
            if params:
                query = sql.SQL(query).format(**params)
            
            self.cursor.execute(query)
            columns = [desc[0] for desc in self.cursor.description]
            data = self.cursor.fetchall()
            return pd.DataFrame(data, columns=columns)
        except Exception as e:
            print(f"Error executing query '{query_name}': {e}")
            raise

    def get_customer_purchase_history(self):
        """Get customer purchase history with category information"""
        return self.execute_query('customer_purchase_history')

    def get_customer_features(self):
        """Get aggregated customer features"""
        return self.execute_query('customer_features')

    def add_query(self, name, query):
        """
        Add a new query to the predefined queries
        
        Args:
            name (str): Name of the query
            query (str): SQL query string
        """
        self._queries[name] = query

    def get_available_queries(self):
        """Get list of available query names"""
        return list(self._queries.keys()) 