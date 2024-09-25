import streamlit as st
from langgraph.prebuilt import create_react_agent
from langchain.tools import StructuredTool
from backend.sqlite_agent import (
    extract_and_write_user_info, 
    extract_and_get_purchase_record, 
    extract_and_purchase,
    view_all_members,
    view_all_products,
    ExtractAndWriteInput, 
    PurchaseRecordInput, 
    PurchaseInput,
    ViewAllMembersInput,
    ViewAllProductsInput,
    openai
) 

# 設定頁面配置
st.set_page_config(page_title="Tool Description Editor", layout="wide")

# 設置自定義樣式
st.markdown("""
    <style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        font-size: 16px;
        margin-top: 10px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .title-header {
        font-family: 'Arial', sans-serif;
        color: #333;
        font-weight: 700;
        margin-bottom: 30px;
    }
    .tool-description {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title-header'>Tool Description Editor</h1>", unsafe_allow_html=True)

st.write("Edit the descriptions below and click 'Update Agent' to apply changes.")

# Function to create tools with current descriptions
def create_tools():
    return [
        StructuredTool.from_function(
            func=extract_and_write_user_info,
            name='ExtractAndWriteUserInfo',
            description=st.session_state.tool_descriptions['ExtractAndWriteUserInfo'],
            args_schema=ExtractAndWriteInput,
            return_direct=True,
        ),
        StructuredTool.from_function(
            func=extract_and_get_purchase_record,
            name='PurchaseRecordFetcher',
            description=st.session_state.tool_descriptions['PurchaseRecordFetcher'],
            args_schema=PurchaseRecordInput,
            return_direct=True,
        ),
        StructuredTool.from_function(
            func=extract_and_purchase,
            name='Purchase',
            description=st.session_state.tool_descriptions['Purchase'],
            args_schema=PurchaseInput,
            return_direct=True,
        ),
        StructuredTool.from_function(
            func=view_all_members,
            name='ViewAllMembers',
            description="View all products in database if user asks about members' information",
            args_schema=ViewAllMembersInput,
            return_direct=True,
        ),
        StructuredTool.from_function(
            func=view_all_products,
            name='ViewAllProducts',
            description="View all products in database if user asks about products' information",
            args_schema=ViewAllProductsInput,
            return_direct=True,
        )
    ]

# Function to recreate agent with updated tools
def recreate_agent():
    tools = create_tools()
    agent = create_react_agent(openai, tools)
    return agent

# 用 Streamlit 的列佈局來控制排版
col1, col2 = st.columns([3, 2])

default_tool_desc = {
    'ExtractAndWriteUserInfo': 'This tool should be able to identify and extract user details from the provided text and store them in the SQLite database',
    'PurchaseRecordFetcher': 'This tool should be able to extract user information from the text and retrieve their corresponding purchase records from the SQLite database',
    'Purchase': 'This tool should be able to extract both user and purchase details from the text and process the purchase in the SQLite database'
}

# 第一列顯示工具描述的編輯
with col1:
    st.write("## Edit Tool Descriptions")
    st.markdown("<hr>", unsafe_allow_html=True)

    for tool_name, description in st.session_state.tool_descriptions.items():
        with st.expander(f"`{tool_name}` Description", expanded=True):
            st.session_state.tool_descriptions[tool_name] = st.text_area(
                label=f'{default_tool_desc[tool_name]}',
                value=description, 
                key=f"desc_{tool_name}", 
                height=150
            )

# 第二列顯示工具當前的描述以及按鈕操作
with col2:
    st.write("## Current Tool Descriptions")
    st.markdown("<hr>", unsafe_allow_html=True)
    
    for tool_name, description in st.session_state.tool_descriptions.items():
        st.markdown(f"""
            <div class="tool-description">
                <strong>{tool_name}:</strong><br>{description}
            </div>
        """, unsafe_allow_html=True)

    if st.button("Update Agent"):
        st.session_state.agent = recreate_agent()
        st.success("Agent updated with new tool descriptions!")
