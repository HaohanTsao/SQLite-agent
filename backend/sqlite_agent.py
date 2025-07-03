from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool

from backend.db_manager import DBManager
import logging
logger = logging.getLogger(__name__)

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


def create_extraction_chain(llm):
    member_extraction_chain = extraction_prompt | llm.with_structured_output(
        schema=UserInfo
    )
    product_extraction_chain = extraction_prompt | llm.with_structured_output(
        schema=ProductInfo
    )

    return {
        "member_extraction_chain": member_extraction_chain,
        "product_extraction_chain": product_extraction_chain,
    }


# %%
# Initialize DBManager
db_manager = DBManager("customer_database.db")
db_manager.create_tables()


# %%
# Define the tool for extracting and writing user info
class ExtractAndWriteInput(BaseModel):
    text: str = Field(description="The text containing user information")


def extract_and_write_user_info(text: str, extraction_chain) -> str:
    """Extract user information and write it to SQLite database."""
    user_info = extraction_chain["member_extraction_chain"].invoke({"text": text})
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


def extract_and_get_purchase_record(text: str, extraction_chain) -> str:
    """Extract user information and return their purchase records from SQLite database."""
    user_info = extraction_chain["member_extraction_chain"].invoke({"text": text})
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


def extract_and_purchase(text: str, extraction_chain) -> str:
    """Extract user and purchase information, write it to SQLite database if necessary, and execute the purchase."""

    user_info = extraction_chain["member_extraction_chain"].invoke({"text": text})
    product_info = extraction_chain["product_extraction_chain"].invoke({"text": text})

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
    return products.to_string(index=False)


# %%
class ViewAllMembersInput(BaseModel):
    pass  # No input required for viewing all members


def view_all_members() -> str:
    """Return all members from the SQLite database."""
    logger.info("Executing view_all_members tool")
    try:
        members = db_manager.list_all_members()
        logger.debug(f"Retrieved {len(members)} members")
        result = members.to_string(index=False)
        logger.debug(f"Converted to string, length: {len(result)}")
        return result
    except Exception as e:
        logger.error(f"Error in view_all_members: {e}", exc_info=True)
        return f"Error retrieving members: {e}"


# %%
# Create tools with current descriptions
def create_default_tools(extraction_chain):
    extract_and_write_tool = StructuredTool.from_function(
        func=lambda text: extract_and_write_user_info(text, extraction_chain),
        name="ExtractAndWriteUserInfo",
        description="Extract user information from text and write it to SQLite database",
        args_schema=ExtractAndWriteInput,
        return_direct=False,
    )

    view_all_members_tool = StructuredTool.from_function(
        func=view_all_members,
        name="ViewAllMembers",
        description="View all products in database to answer the user if user asks about members' information",
        args_schema=ViewAllMembersInput,
        return_direct=False,
    )

    view_all_products_tool = StructuredTool.from_function(
        func=view_all_products,
        name="ViewAllProducts",
        description="View all products in database if user asks about products' information",
        args_schema=ViewAllProductsInput,
        return_direct=False,
    )

    purchase_tool = StructuredTool.from_function(
        func=lambda text: extract_and_purchase(text, extraction_chain),
        name="Purchase",
        description="Call this tool when the user wants to purchase an item. The tool will handle extracting product information from the input and completing the purchase process.",
        args_schema=PurchaseInput,
        return_direct=False,
    )

    purchase_record_tool = StructuredTool.from_function(
        func=lambda text: extract_and_get_purchase_record(text, extraction_chain),
        name="PurchaseRecordFetcher",
        description="Extract user information from text and fetch purchase records from SQLite database",
        args_schema=PurchaseRecordInput,
        return_direct=False,
    )

    return [
        extract_and_write_tool,
        purchase_record_tool,
        purchase_tool,
        view_all_members_tool,
        view_all_products_tool,
    ]