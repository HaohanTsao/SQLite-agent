# %%
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from backend.db_manager import DBManager

load_dotenv()

# %%
# Initialize OpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model='gpt-4o-mini',
    max_tokens=None,
    timeout=None,
)
# %%
# Define the UserInfo schema for extraction
class UserInfo(BaseModel):
    '''Information about a user.'''
    name: Optional[str] = Field(default=None, description='The name of the user')
    email: Optional[str] = Field(default=None, description='The email address of the user')
    age: Optional[int] = Field(default=None, description='The age of the user')

class ProductInfo(BaseModel):
    '''Information about a product purchase.'''
    product_name: Optional[str] = Field(default=None, description='The name of the product')
    number: Optional[int] = Field(default=1, description='The number of products to purchase')

# Create the extraction chain
extraction_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert extraction algorithm. "
        "Only extract relevant information from the text. "
        "If you do not know the value of an attribute asked to extract, "
        "return null for the attribute's value.",
    ),
    ("human", "{text}"),
])

member_extraction_chain = extraction_prompt | openai.with_structured_output(schema=UserInfo)
product_extraction_chain = extraction_prompt | openai.with_structured_output(schema=ProductInfo)

# %%
# Initialize DBManager
db_manager = DBManager('customer_database.db')
db_manager.create_tables()

# %%
# Define the tool for extracting and writing user info
class ExtractAndWriteInput(BaseModel):
    text: str = Field(description='The text containing user information')

def extract_and_write_user_info(text: str) -> str:
    '''Extract user information and write it to SQLite database.'''
    user_info = member_extraction_chain.invoke({'text': text})
    member = db_manager.get_member_by_name(user_info.name)
    if member:
        return f'Member {user_info.name} already exists with ID: {member[0]}'
    else:
        db_manager.insert_member(user_info.name, user_info.email, user_info.age)
        new_member = db_manager.get_member_by_name(user_info.name)
        return f'Extracted and wrote user info: {new_member}'

extract_and_write_tool = StructuredTool.from_function(
    func=extract_and_write_user_info,
    name='ExtractAndWriteUserInfo',
    description='Extract user information from text and write it to SQLite database',
    args_schema=ExtractAndWriteInput,
    return_direct=True,
)

# %%
# Define the tool for extracting user info and fetching purchase records
class PurchaseRecordInput(BaseModel):
    text: str = Field(description='The text containing user information')

def extract_and_get_purchase_record(text: str) -> str:
    '''Extract user information and return their purchase records from SQLite database.'''
    user_info = member_extraction_chain.invoke({'text': text})
    member = db_manager.get_member_by_name(user_info.name)
    
    if not member:
        return f"No member found for name '{user_info.name}'"
    
    member_id = member[0]
    purchase_records = db_manager.get_member_records(member_id)
    
    if not purchase_records:
        return f"No purchase records found for member {user_info.name} (ID: {member_id})"
    
    response = f"Purchase records for {user_info.name} (ID: {member_id}):\n"
    for record in purchase_records:
        response += f"- Record ID: {record[0]}, Product: {record[1]}, Price: {record[2]}, Number: {record[3]}, Payment: {record[2]*record[3]}\n"
    
    return response

purchase_record_tool = StructuredTool.from_function(
    func=extract_and_get_purchase_record,
    name='PurchaseRecordFetcher',
    description='Extract user information from text and fetch purchase records from SQLite database',
    args_schema=PurchaseRecordInput,
    return_direct=True,
)
# %%
# Define the tool for purchasing
class PurchaseInput(BaseModel):
    text: str = Field(description='The text containing user and purchase information')

def extract_and_purchase(text: str) -> str:
    '''Extract user and purchase information, write it to SQLite database if necessary, and execute the purchase.'''
    user_info = member_extraction_chain.invoke({'text': text})
    product_info = product_extraction_chain.invoke({'text': text})

    if user_info.name is None:
        return "User information is incomplete."
    if product_info.product_name is None:
        return "Product information is incomplete."

    member = db_manager.get_member_by_name(user_info.name)
    
    if not member:
        # If member doesn't exist, add new member
        db_manager.insert_member(user_info.name, user_info.email, user_info.age)
        member = db_manager.get_member_by_name(user_info.name)
    
    member_id = member[0]
    
    # Execute purchase
    product = db_manager.get_product_by_name(product_info.product_name)
    if not product:
        return f"Sorry, the product '{product_info.product_name}' does not exist."

    product_id = product[0]
    db_manager.insert_record(member_id, product_id, product_info.number)

    return f"Purchase successful! Member {user_info.name} bought {product_info.number} {product_info.product_name}(s)."

purchase_tool = StructuredTool.from_function(
    func=extract_and_purchase,
    name='Purchase',
    description='Extract user and purchase information from text and execute the purchase',
    args_schema=PurchaseInput,
    return_direct=True,
)

# %%
class ViewAllProductsInput(BaseModel):
    pass  # No input required for viewing all products

def view_all_products() -> str:
    '''Return all products from the SQLite database.'''
    products = db_manager.list_all_products()
    
    return products

view_all_products_tool = StructuredTool.from_function(
    func=view_all_products,
    name='ViewAllProducts',
    description="View all products in database if user asks about products' information",
    args_schema=ViewAllProductsInput,
    return_direct=True,
)

# %%
class ViewAllMembersInput(BaseModel):
    pass  # No input required for viewing all members

def view_all_members() -> str:
    '''Return all members from the SQLite database.'''
    members = db_manager.list_all_members()
    
    return members

view_all_members_tool = StructuredTool.from_function(
    func=view_all_members,
    name='ViewAllMembers',
    description="View all products in database to answer the user if user asks about members' information",
    args_schema=ViewAllMembersInput,
    return_direct=True,
)

# %%
system_prompt = '''You are a helpful and friendly AI agent designed to assist users with tasks related to managing customer and product information in an SQLite database.

When a user asks you a question, always respond in a polite and friendly manner, guiding them through the process if necessary. If their request requires using one of your tools, call the tool and explain the results clearly and accurately. If the user’s input is unclear, kindly ask them to clarify or provide more information.

When returning information from the database, present it in an easy-to-understand format. If no relevant data is found, respond in a reassuring and supportive way, encouraging the user to try again or offer additional assistance.

Remember to:
- Always maintain a positive and friendly tone.
- Be patient with users and ensure they feel supported throughout their interaction.
- Provide helpful explanations after using the tools, summarizing the outcome or offering next steps.
- Avoid technical jargon unless the user seems to expect or request it.
'''
# %%
# Create the agent
# tools = [extract_and_write_tool, purchase_record_tool, purchase_tool, view_all_members_tool, view_all_products_tool]
# agent = create_react_agent(openai, tools, state_modifier=system_prompt)
# %%
# Function to handle user messages
# def process_user_message(message):
#     events = agent.stream(
#         {'messages': [HumanMessage(content=message)]},
#         stream_mode="values",
#     )
#     responses = []
#     for event in events:
#         event["messages"][-1].pretty_print()
#         responses.append(event["messages"][-1])

#     return responses

# %%
# Insert member usage
# user_message = "A new member, Tedy Tsao, 22, dkk94729@gmail.com"
# process_user_message(message=user_message)

# %%
# # Get records usage
# user_message = "My name is Andy Tsao, show me my purchase records"
# process_user_message(message=user_message)

# # %%
# # Purchase usage
# user_message = "My name is Andy Tsao, I want to buy 3 iPhone16"
# process_user_message(message=user_message)
# %%
