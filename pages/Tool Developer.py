import streamlit as st
from backend.sqlite_agent import create_tool_from_code
from code_editor import code_editor

from backend.sqlite_agent import recreate_agent


# Define the function to remove a tool by its index
def remove_tool(tool_index):
    del st.session_state.tools[tool_index]
    del st.session_state["confirming_deletion"]
    st.rerun()


# Create a confirmation dialog when deleting a tool
def confirm_delete(tool_index):
    st.session_state.confirming_deletion = tool_index


# New section for tool management
st.write("## Tool Management")

# Create a table with tool name, description, and delete button
if "tools" in st.session_state and st.session_state.tools:
    col1, col2, col3 = st.columns([3, 6, 1])

    # Display the tool's name and description
    with col1:
        st.write(f"**Tool Name**")

    with col2:
        st.write("**Description**")

    with col3:
        st.write(f"**Action**")

    for i, tool in enumerate(st.session_state.tools):
        # Create columns for the table (name, description, and delete button)
        col1, col2, col3 = st.columns([3, 6, 1])

        # Display the tool's name and description
        with col1:
            st.write(f"**`{tool.name}`**")
        with col2:
            st.write(tool.description)

        # Add the delete button, which triggers the confirmation
        with col3:
            if st.button("Delete", key=f"delete_{i}"):
                confirm_delete(i)

    # Check if a deletion is being confirmed
    if "confirming_deletion" in st.session_state:
        with st.expander("⚠️ Confirm Deletion", expanded=True):
            tool_index = st.session_state.confirming_deletion
            st.warning(
                f"Are you sure you want to delete `{st.session_state.tools[tool_index].name}`?"
            )

            # Confirm deletion action
            col_confirm, col_cancel = st.columns([1, 1])
            with col_confirm:
                if st.button("Yes, delete", key=f"confirm_delete_{tool_index}"):
                    remove_tool(tool_index)
                    del st.session_state["confirming_deletion"]

            with col_cancel:
                if st.button("Cancel", key=f"cancel_delete_{tool_index}"):
                    del st.session_state["confirming_deletion"]
                    st.rerun()
else:
    st.write("No tools available.")


# Define a custom button in the editor
custom_buttons = [
    {
        "name": "Save Code",  # The name displayed on the button
        "feather": "Save",  # Optional Feather icon (format required)
        "hasText": True,  # Ensures the button shows the text
        "alwaysOn": True,  # Always visible
        "commands": [
            "save-state",
            ["response", "saved"],
        ],  # List of commands (in this case, a custom command)
    }
]

# Add new tool or edit existing tool
st.write("### Add New Tool")
st.write(
    "You can build a custimized tool right here. check out next session to see example."
)
response_dict = code_editor(
    """
from pydantic import BaseModel, Field        
# NOTE: We will capture the new tool based on the structure below, so be careful when you changed the structure.

db_manager = DBManager()

# Define the tool for extracting and inserting product info
class YourInputSchema(BaseModel):
    text: str = Field(description="The text containing users' input")

def your_customized_tool_func(text: str) -> str:
    '''Write the logic of your tool here'''
    
    return

# Create the tool using the StructuredTool wrapper
new_tool = StructuredTool.from_function(
    func=your_customized_tool_func,
    name='NameOfYourTool',
    description='What does your tool do and when should the agent call it',
    args_schema=YourInputSchema,
    return_direct=True,
)
""",
    theme="contrast",
    buttons=custom_buttons,  # Adding custom buttons to the editor
    shortcuts="vscode",
)

# Save button outside the editor
st.write("Before you save the tool, remember to save the code first.")
if st.button("Save Tool"):
    if response_dict:  # Check if the editor returned valid data
        code_content = response_dict.get("text", "")  # Get the code content
        if code_content:
            # Process the code content (for example, save it or validate it)
            new_tool = create_tool_from_code(code_content)
            st.session_state.agent = recreate_agent(new_tool)
            st.rerun()

# Display available functions
st.write("### Available Functions")
st.write("Here are some functions can be used in your tools:")
st.code(
    """
# Database operations
db_manager = DBManager()
member_name = 'andy tsao'
age = 22
email = 'example@example.com'
product_name = 'iPhone'
price = '2000'  
db_manager.insert_member(member_name, age, email)
db_manager.insert_product(product_name, price)
db_manager.get_member_by_name(name)
db_manager.get_product_by_name(product_name)
db_manager.list_all_members()
db_manager.list_all_products()
db_manager.list_all_records()
        
# Extraction chain
'''
Extraction chain can extract information from text and return structured output.
There are two chain used currently, extracting member and product respectively. Below is example usage.
'''
from backend.sqlite_agent import member_extraction_chain, product_extraction_chain

member_in_text = 'Alice Johson is 23 years old and her email is alice@gmail.com'
product_in_text = 'iPhine is $2000.'
member_info = member_extraction_chain.invoke({'text': text})
product_info = product_extraction_chain.invoke({'text': text})
"""
)

# Build InsertProduct as an example

st.write("### Example of Tool Building")
st.write("Here is an example of how to build a `InsertProduct` tool.")

st.code(
    """
from pydantic import BaseModel, Field        
from langchain_core.tools import StructuredTool
        
db_manager = DBManager()

# Define the tool for extracting and inserting product info
class InsertProductInput(BaseModel):
    text: str = Field(description='The text containing product information')

def insert_product(text: str) -> str:
    '''Extract product information from text and insert it into SQLite database.'''
    # Extract product information using the product extraction chain
    product_info = product_extraction_chain.invoke({'text': text})
    
    # Check if the product already exists in the database
    product = db_manager.get_product_by_name(product_info.name)
    if product:
        return f'Product {product_info.name} already exists with ID: {product[0]}'
    else:
        # Insert the extracted product information into the database
        db_manager.insert_product(product_info.name, product_info.price)
        new_product = db_manager.get_product_by_name(product_info.name)
        return f'Extracted and inserted product info: {new_product}'

# Create the tool using the StructuredTool wrapper
new_tool = StructuredTool.from_function(
    func=insert_product,
    name='InsertProduct',
    description='Extract product information from text and insert it into SQLite database',
    args_schema=InsertProductInput,
    return_direct=True,
)
"""
)
