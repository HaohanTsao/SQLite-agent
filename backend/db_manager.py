import sqlite3
import pandas as pd

class DBManager:
    def __init__(self, db_name='example.db'):
        # Connect to the database
        self.db_name = db_name
        self.conn = sqlite3.connect(self.db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()

    def create_tables(self):
        # Create 'member', 'product', and 'record' tables
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS member (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            age INTEGER NOT NULL
        )
        ''')
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS product (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            price REAL NOT NULL
        )
        ''')
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS record (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            member_id INTEGER,
            product_id INTEGER,
            number INTEGER,
            FOREIGN KEY (member_id) REFERENCES member(id),
            FOREIGN KEY (product_id) REFERENCES product(id)
        )
        ''')
        self.conn.commit()
        # Check if member table is empty
        self.cursor.execute('SELECT COUNT(*) FROM member')
        if self.cursor.fetchone()[0] == 0:
            self.insert_example_data()

    def insert_example_data(self):
        # Insert some example members
        members = [
            ('Alice Johnson', 'alice@example.com', 25),
            ('Bob Smith', 'bob@example.com', 30),
            ('Charlie Brown', 'charlie@example.com', 22)
        ]
        self.cursor.executemany("INSERT INTO member (name, email, age) VALUES (?, ?, ?)", members)
        
        # Insert some example products
        products = [
            ('Laptop', 999.99),
            ('Smartphone', 499.99),
            ('Headphones', 199.99)
        ]
        self.cursor.executemany("INSERT INTO product (name, price) VALUES (?, ?)", products)

        # Insert some example records
        records = [
            (1, 1, 1),  # Alice buys 1 Laptop
            (2, 2, 2),  # Bob buys 2 Smartphones
            (3, 3, 3)   # Charlie buys 3 Headphones
        ]
        self.cursor.executemany("INSERT INTO record (member_id, product_id, number) VALUES (?, ?, ?)", records)
        
        self.conn.commit()

    def insert_member(self, name, email, age):
        # Insert a new member
        self.cursor.execute("INSERT INTO member (name, email, age) VALUES (?, ?, ?)", (name, email, age))
        self.conn.commit()

    def insert_product(self, name, price):
        # Insert a new product
        self.cursor.execute("INSERT INTO product (name, price) VALUES (?, ?)", (name, price))
        self.conn.commit()

    def insert_record(self, member_id, product_id, number):
        # Insert a new purchase record
        self.cursor.execute("INSERT INTO record (member_id, product_id, number) VALUES (?, ?, ?)", (member_id, product_id, number))
        self.conn.commit()

    def get_member_by_name(self, name):
        # Find a member by name
        self.cursor.execute("SELECT * FROM member WHERE name = ?", (name,))
        return self.cursor.fetchone()

    def get_product_by_name(self, product_name):
        # Find a product by name
        self.cursor.execute("SELECT * FROM product WHERE name = ?", (product_name,))
        return self.cursor.fetchone()

    def get_member_records(self, member_id):
        # Retrieve all records for a specific member
        self.cursor.execute('''
        SELECT record.id, product.name, product.price, record.number, product.price*record.number
        FROM record 
        JOIN product ON record.product_id = product.id
        WHERE record.member_id = ?
        ''', (member_id,))
        return self.cursor.fetchall()
    
    def list_all_members(self):
        # Retrieve all members
        return pd.read_sql_query("SELECT * FROM member", self.conn)
    
    def list_all_products(self):
        # Retrieve all products
        return pd.read_sql_query("SELECT * FROM product", self.conn)
    
    def list_all_records(self):
        # Retrieve all records
        return pd.read_sql_query('''
        SELECT record.id, member.name AS member_name, product.name AS product_name, record.number 
        FROM record 
        JOIN member ON record.member_id = member.id 
        JOIN product ON record.product_id = product.id
        ''', self.conn)

    def close(self):
        # Close the database connection
        self.conn.close()