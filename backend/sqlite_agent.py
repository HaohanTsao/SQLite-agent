# %%
import os
import re
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from backend.db_manager import DBManager

load_dotenv()

# %%
# Initialize OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",
    max_tokens=None,
    timeout=None,
)
ollama = ChatOllama(
    model="llama3.2",
    temperature=0,
)


# %%
# Define the UserInfo schema for extraction
class UserInfo(BaseModel):
    """Information about a user."""

    name: Optional[str] = Field(default=None, description="The name of the user")
    email: Optional[str] = Field(
        default=None, description="The email address of the user"
    )
    age: Optional[int] = Field(default=None, description="The age of the user")


class ProductInfo(BaseModel):
    """Information about a product purchase."""

    name: Optional[str] = Field(default=None, description="The name of the product")
    price: Optional[str] = Field(default=None, description="The price of the product")
    number: Optional[int] = Field(
        default=1, description="The number of products to purchase"
    )


# Create the extraction chain
extraction_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        ("human", "{text}"),
    ]
)

member_extraction_chain = extraction_prompt | ollama.with_structured_output(
    schema=UserInfo
)
product_extraction_chain = extraction_prompt | ollama.with_structured_output(
    schema=ProductInfo
)

# %%
# Initialize DBManager
db_manager = DBManager("customer_database.db")
db_manager.create_tables()


# %%
# Define the tool for extracting and writing user info
class ExtractAndWriteInput(BaseModel):
    text: str = Field(description="The text containing user information")


def extract_and_write_user_info(text: str) -> str:
    """Extract user information and write it to SQLite database."""
    user_info = member_extraction_chain.invoke({"text": text})
    member = db_manager.get_member_by_name(user_info.name)
    if member:
        return f"Member {user_info.name} already exists with ID: {member[0]}"
    else:
        db_manager.insert_member(user_info.name, user_info.email, user_info.age)
        new_member = db_manager.get_member_by_name(user_info.name)
        return f"Extracted and wrote user info: {new_member}"


# %%
# Define the tool for extracting user info and fetching purchase records
class PurchaseRecordInput(BaseModel):
    text: str = Field(description="The text containing user information")


def extract_and_get_purchase_record(text: str) -> str:
    """Extract user information and return their purchase records from SQLite database."""
    user_info = member_extraction_chain.invoke({"text": text})
    member = db_manager.get_member_by_name(user_info.name)

    if not member:
        return f"No member found for name '{user_info.name}'"

    member_id = member[0]
    purchase_records = db_manager.get_member_records(member_id)

    if not purchase_records:
        return (
            f"No purchase records found for member {user_info.name} (ID: {member_id})"
        )

    response = f"Purchase records for {user_info.name} (ID: {member_id}):\n"
    for record in purchase_records:
        response += f"- Record ID: {record[0]}, Product: {record[1]}, Price: {record[2]}, Number: {record[3]}, Payment: {record[2]*record[3]}\n"

    return response


# %%
# Define the tool for purchasing
class PurchaseInput(BaseModel):
    text: str = Field(description="The text containing user and purchase information")


def extract_and_purchase(text: str) -> str:
    """Extract user and purchase information, write it to SQLite database if necessary, and execute the purchase."""
    user_info = member_extraction_chain.invoke({"text": text})
    product_info = product_extraction_chain.invoke({"text": text})

    if user_info.name is None:
        return "User information is incomplete."
    if product_info.name is None:
        return "Product information is incomplete."

    member = db_manager.get_member_by_name(user_info.name)

    if not member:
        # If member doesn't exist, add new member
        db_manager.insert_member(user_info.name, user_info.email, user_info.age)
        member = db_manager.get_member_by_name(user_info.name)

    member_id = member[0]

    # Execute purchase
    product = db_manager.get_product_by_name(product_info.name)
    if not product:
        return f"Sorry, the product '{product_info.name}' does not exist."

    product_id = product[0]
    db_manager.insert_record(member_id, product_id, product_info.number)

    return f"Purchase successful! Member {user_info.name} bought {product_info.number} {product_info.name}(s)."


# %%
class ViewAllProductsInput(BaseModel):
    pass  # No input required for viewing all products


def view_all_products() -> str:
    """Return all products from the SQLite database."""
    products = db_manager.list_all_products()

    return products


# %%
class ViewAllMembersInput(BaseModel):
    pass  # No input required for viewing all members


def view_all_members() -> str:
    """Return all members from the SQLite database."""
    members = db_manager.list_all_members()

    return members


# %%
# Create tools with current descriptions
def create_default_tools():
    extract_and_write_tool = StructuredTool.from_function(
        func=extract_and_write_user_info,
        name="ExtractAndWriteUserInfo",
        description="Extract user information from text and write it to SQLite database",
        args_schema=ExtractAndWriteInput,
        return_direct=True,
    )

    view_all_members_tool = StructuredTool.from_function(
        func=view_all_members,
        name="ViewAllMembers",
        description="View all products in database to answer the user if user asks about members' information",
        args_schema=ViewAllMembersInput,
        return_direct=True,
    )

    view_all_products_tool = StructuredTool.from_function(
        func=view_all_products,
        name="ViewAllProducts",
        description="View all products in database if user asks about products' information",
        args_schema=ViewAllProductsInput,
        return_direct=True,
    )

    purchase_tool = StructuredTool.from_function(
        func=extract_and_purchase,
        name="Purchase",
        description="Call this tool when the user wants to purchase an item. The tool will handle extracting product information from the input and completing the purchase process.",
        args_schema=PurchaseInput,
        return_direct=True,
    )

    purchase_record_tool = StructuredTool.from_function(
        func=extract_and_get_purchase_record,
        name="PurchaseRecordFetcher",
        description="Extract user information from text and fetch purchase records from SQLite database",
        args_schema=PurchaseRecordInput,
        return_direct=True,
    )

    return [
        extract_and_write_tool,
        purchase_record_tool,
        purchase_tool,
        view_all_members_tool,
        view_all_products_tool,
    ]


# %%
system_prompt = """You are a helpful and friendly AI agent designed to assist users with tasks related to managing customer and product information in an SQLite database.

When a user asks you a question, always respond in a polite and friendly manner, guiding them through the process if necessary. If their request requires using one of your tools, call the tool and explain the results clearly and accurately. If the user’s input is unclear, kindly ask them to clarify or provide more information.

When returning information from the database, present it in an easy-to-understand format. If no relevant data is found, respond in a reassuring and supportive way, encouraging the user to try again or offer additional assistance.

Remember to:
- Always maintain a positive and friendly tone.
- Be patient with users and ensure they feel supported throughout their interaction.
- Provide helpful explanations after using the tools, summarizing the outcome or offering next steps.
- Avoid technical jargon unless the user seems to expect or request it.
"""


# %%
# Recreate agent
def recreate_agent(new_tool: StructuredTool = None, llm: str = "openai"):
    if llm == "openai":
        llm = openai
    elif llm == "ollama":
        llm = ollama

    tools = st.session_state.tools
    for tool in tools:
        if tool.name in st.session_state.tool_descriptions:
            tool.description = st.session_state.tool_descriptions[tool.name]

    if new_tool:
        st.session_state.tool_descriptions[new_tool.name] = new_tool.description
        st.session_state.tools.append(new_tool)

    return create_react_agent(llm, st.session_state.tools, state_modifier=system_prompt)


def create_tool_from_code(code: str) -> StructuredTool:
    db_manager_code = """import sqlite3
import pandas as pd

class DBManager:
    def __init__(self, db_name='customer_database.db'):
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
        self.conn.close()\n\n"""
    new_tool_code = re.search(
        r"(new_tool\s*=\s*StructuredTool\.from_function\(.+?\))", code, re.DOTALL
    ).group(1)

    func_name = re.search(r"func=(\w+)", new_tool_code).group(1)
    schema_name = re.search(r"args_schema=(\w+)", new_tool_code).group(1)
    tool_name = re.search(r"name='(.+?)'", new_tool_code).group(1)
    description = re.search(r"description='(.+?)'", new_tool_code).group(1)

    code = db_manager_code + code

    exec(code, globals())

    if func_name in globals() and schema_name in globals():
        new_tool = StructuredTool.from_function(
            func=globals()[func_name],
            name=tool_name,
            description=description,
            args_schema=globals()[schema_name],
            return_direct=True,
        )
        return new_tool
    else:
        raise ValueError(
            f"Provided function {func_name} or schema {schema_name} not found in the code."
        )


# %%
# Function to handle user messages
tools = create_default_tools()


# %%
def process_user_message(message, llm):
    agent = create_react_agent(model=llm, tools=tools, state_modifier=system_prompt)
    events = agent.stream(
        {"messages": [HumanMessage(content=message)]},
        stream_mode="values",
    )
    responses = []
    for event in events:
        event["messages"][-1].pretty_print()
        responses.append(event["messages"][-1])

    return responses


# %%
# testing
# import time

# # test
# messages = [
#     "A new member, Ted Mosbi, 22 years old, email:tuued@gmail.com, write into the database",
#     "Ted Mosbi purchased 3 Laptop.",
#     "Show me purchase records of Ted Mosbi.",
#     "How much is a Smartphone?",
#     "Ted Mosbi is a member. Tell me how old is he?",
# ]

# # for model in models:
# #     print(f"Testing model: {model}")
# for message in messages:
#     start_time = time.time()
#     answer = process_user_message(message=message, llm=openai)
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(f"Time taken: {elapsed_time:.2f} seconds\n")
