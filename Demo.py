import time
import streamlit as st
import sqlite3
import altair as alt
from langchain_core.messages import HumanMessage
from langchain.tools import StructuredTool
from langgraph.prebuilt import create_react_agent
from backend.db_manager import DBManager
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
    openai,
    system_prompt
) 

st.set_page_config(layout="wide")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'tool_descriptions' not in st.session_state:
    st.session_state.tool_descriptions = {
        'ExtractAndWriteUserInfo': 'Extract user information from text and write it to SQLite database',
        'PurchaseRecordFetcher': 'Extract user information from text and fetch purchase records from SQLite database',
        'Purchase': 'Extract user and purchase information from text and execute the purchase'
    }

# Load data from database
def load_data():
    db_manager = DBManager('customer_database.db')
    members = db_manager.list_all_members()
    products = db_manager.list_all_products()
    records = db_manager.list_all_records()
    return members, products, records

# Refresh data
def refresh_data():
    st.session_state.data = load_data()

# Load initial data if not loaded
if st.session_state.data is None:
    refresh_data()

# Create tools with current descriptions
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
            description="View all members in database if user asks about members' information",
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

# Recreate agent
def recreate_agent():
    tools = create_tools()
    return create_react_agent(openai, tools, state_modifier=system_prompt)

# Initialize agent
if 'agent' not in st.session_state:
    st.session_state.agent = recreate_agent()

# App layout
st.markdown("<h1 style='text-align: center;'>SQLite Agent Demo</h1>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    try:
        # Use data from session state
        members, products, records = st.session_state.data

        # Product table
        st.markdown("<h2>📦 Product Table</h2>", unsafe_allow_html=True)
        selected_products = st.multiselect(
            "Choose Products", list(products['name'].unique()), list(products['name'].unique())[:2],
            help="Select products to view from the table"
        )
        
        if not selected_products:
            st.error("Please select at least one product")
        else:
            product_frame = products[products['name'].isin(selected_products)]
            st.dataframe(product_frame, use_container_width=True)

        # Member table
        st.markdown("<h2>👥 Member Table</h2>", unsafe_allow_html=True)
        selected_members = st.multiselect(
            "Choose Members", list(members['name'].unique()), list(members['name'].unique())[:2],
            help="Select members to view from the table"
        )
        
        if not selected_members:
            st.error("Please select at least one member")
        else:
            member_frame = members[members['name'].isin(selected_members)]
            st.dataframe(member_frame, use_container_width=True)

        # Purchase records
        st.markdown("<h2>🛒 Purchase Records</h2>", unsafe_allow_html=True)
        countries = st.multiselect(
            "Choose Members for Purchase Records", list(records['member_name'].unique()), list(records['member_name'].unique())[:2]
        )

        if not countries:
            st.error("Please select at least one member.")
        else:
            data = records[records['member_name'].isin(countries)]

            pivot_data = data.pivot_table(
                index='member_name', columns='product_name', values='number', aggfunc='sum', fill_value=0
            )
            st.markdown("### Records of Selected Members")
            st.dataframe(pivot_data, use_container_width=True)

            data_melted = pivot_data.reset_index().melt(id_vars='member_name', var_name='Product', value_name='Quantity')
            chart = (
                alt.Chart(data_melted)
                .mark_bar()
                .encode(
                    x='Product:N',
                    y='Quantity:Q',
                    color='member_name:N',
                )
            )
            st.altair_chart(chart, use_container_width=True)

    except sqlite3.Error as e:
        st.error(f"Database connection error: {e}")

    # Refresh button
    st.button("🔄 Refresh Data", on_click=refresh_data, use_container_width=True)

with col2:
    st.markdown("<h2>💬 Chat with SQLite Agent!</h2>", unsafe_allow_html=True)

    # Chat container for messages
    chat_container = st.container()

    with chat_container:
        with st.chat_message("assistant"):
            st.markdown("How can I help you today? Try asking me to insert new members into the database or summarize and update one's purchase records.")
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    def response_generator(response):
        for word in response:
            yield word
            time.sleep(0.01)

# Move chat input to the bottom
prompt = st.chat_input("Type your message here...")

if prompt:
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    # Handle streaming messages
    with chat_container:
        with st.chat_message("assistant"):
            response = ''
            for step in st.session_state.agent.stream({'messages': [HumanMessage(content=prompt)]}, stream_mode='updates'):
                if 'agent' in step:
                    messages = step['agent']['messages']
                    for message in messages:
                        if message.tool_calls:
                            tool_call = message.tool_calls[0]
                            step_response = f'**Calling `{tool_call["name"]}` tool...**'
                            st.markdown(step_response)
                        else:
                            step_response = message.content
                            step_response = st.write_stream(response_generator(step_response))
                
                elif 'tools' in step:
                    messages = step['tools']['messages']
                    for message in messages:
                        if 'tool_call' in locals() and tool_call['name'] in ['ViewAllProducts', 'ViewAllMembers']:
                            step_response = '**Tool Message:**' + '\n\n' + 'Retrieving data from database...'
                            step_response = st.write_stream(response_generator(step_response))
                        else:
                            step_response = '**Tool Message:**' + '\n\n' + message.content
                            step_response = st.write_stream(response_generator(step_response))

                        # refresh data
                        if 'tool_call' in locals() and tool_call['name'] in ['ExtractAndWriteUserInfo', 'Purchase']:
                            refresh_data()
                            st.success("Database updated! Data refreshed.")

                if response == '':
                    response += step_response
                else:
                    response += '\n\n' + step_response
            
            st.session_state.messages.append({"role": "assistant", "content": response})

# Style the UI with more spacing and visual separation
st.markdown("<style> .stMarkdown { margin-bottom: 2rem !important; } </style>", unsafe_allow_html=True)
