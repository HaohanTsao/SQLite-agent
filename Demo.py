import logging
import os
import time
import streamlit as st
import sqlite3
import altair as alt
from dotenv import load_dotenv
from backend.db_manager import DBManager
from langchain_openai import AzureChatOpenAI

from backend.sqlite_agent import (
    create_default_tools,
    create_extraction_chain,
)
# Add: Import framework classes
from backend.frameworks.langgraph_framework import LangGraphFramework

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_debug.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

st.set_page_config(layout="wide")

# Initialize session state
if "data" not in st.session_state:
    st.session_state.data = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_created" not in st.session_state:
    st.session_state.agent_created = False


# Load data from database
def load_data():
    db_manager = DBManager("customer_database.db")
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

# st.sidebar.header("Model Configuration")
# model_provider = st.sidebar.selectbox(
#     "Select Model Provider", ["OpenAI", "Ollama", "Bedrock"]
# )

# Add: Framework selector
st.sidebar.header("Agent Framework")
framework_choice = st.sidebar.selectbox(
    "Select Agent Framework", 
    ["LangGraph"],  # Currently only one option, will expand later
    help="Choose the agent framework to use"
)

# Add: Initialize framework instance
if "framework" not in st.session_state:
    st.session_state.framework = LangGraphFramework()

# Add: Display framework information
with st.sidebar.expander("Framework Info", expanded=False):
    framework_info = st.session_state.framework.get_framework_info()
    st.write(f"**Name:** {framework_info['name']}")
    st.write(f"**Description:** {framework_info['description']}")
    st.write(f"**Features:** {', '.join(framework_info['features'])}")


# def openai_inputs():
#     api_key = st.sidebar.text_input("OpenAI API Key", type="password")
#     model_name = st.sidebar.text_input("Model Name", value="gpt-4o-mini")
#     return {"api_key": api_key, "model_name": model_name}


# def ollama_inputs():
#     model_name = st.sidebar.text_input("Model Name", value="llama3.2")
#     st.sidebar.warning(
#         "Please ensure you have pulled the specified model using Ollama locally."
#     )
#     return {"model_name": model_name}


# def bedrock_inputs():
#     aws_region = st.sidebar.text_input("AWS Region")
#     aws_access_key = st.sidebar.text_input("AWS Access Key", type="password")
#     aws_secret_key = st.sidebar.text_input("AWS Secret Access Key", type="password")
#     model_name = st.sidebar.text_input(
#         "Model Name", value="anthropic.claude-3-5-sonnet-20240620-v1:0"
#     )
#     return {
#         "aws_region": aws_region,
#         "aws_access_key": aws_access_key,
#         "aws_secret_key": aws_secret_key,
#         "model_name": model_name,
#     }


# Display appropriate inputs based on selected provider
# if model_provider == "OpenAI":
#     model_args = openai_inputs()
# elif model_provider == "Ollama":
#     model_args = ollama_inputs()
# else:  # Bedrock
#     model_args = bedrock_inputs()


def create_agent():
    with st.spinner("Creating agent..."):
        # Create extraction chain and tools (still need these for tool creation)
        # Use a temporary LLM just for tool setup
        temp_llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            api_key=os.getenv("AZURE_API_KEY"),
            azure_deployment=os.getenv("AZURE_DEPLOYMENT", "gpt-4o"),
            api_version=os.getenv("AZURE_API_VERSION", "2025-04-01-preview"),
            temperature=0,
        )
        
        st.session_state.extraction_chain = create_extraction_chain(llm=temp_llm)
        st.session_state.tools = create_default_tools(st.session_state.extraction_chain)
        # st.session_state.tool_descriptions = {
        #     tool.name: tool.description for tool in st.session_state.tools
        # }
        
        # Define system prompt
        system_prompt = """You are a helpful and friendly AI agent designed to assist users with tasks related to managing customer and product information in an SQLite database.

When a user asks you a question, always respond in a polite and friendly manner, guiding them through the process if necessary. If their request requires using one of your tools, call the tool and explain the results clearly and accurately. If the user's input is unclear, kindly ask them to clarify or provide more information.

When returning information from the database, present it in an easy-to-understand format. If no relevant data is found, respond in a reassuring and supportive way, encouraging the user to try again or offer additional assistance.

Remember to:
- Always maintain a positive and friendly tone.
- Be patient with users and ensure they feel supported throughout their interaction.
- Provide helpful explanations after using the tools, summarizing the outcome or offering next steps.
- Avoid technical jargon unless the user seems to expect or request it."""
        
        # Use framework to create agent (no LLM parameter needed)
        st.session_state.agent = st.session_state.framework.create_agent(
            tools=st.session_state.tools,
            system_prompt=system_prompt
        )
        st.session_state.agent_created = True
    st.success("Agent created successfully!")
    st.rerun()


# Create Agent button
if st.sidebar.button("Create Agent"):
    create_agent()

# App layout
st.markdown(
    "<h1 style='text-align: center;'>SQLite Agent Demo</h1>", unsafe_allow_html=True
)
col1, col2 = st.columns(2)

with col1:
    try:
        # Use data from session state
        members, products, records = st.session_state.data

        # Purchase records
        st.markdown("<h2>🛒 Purchase Records</h2>", unsafe_allow_html=True)
        countries = st.multiselect(
            "Choose Members for Purchase Records",
            list(records["member_name"].unique()),
            list(records["member_name"].unique())[:2],
        )

        if not countries:
            st.error("Please select at least one member.")
        else:
            data = records[records["member_name"].isin(countries)]

            pivot_data = data.pivot_table(
                index="member_name",
                columns="product_name",
                values="number",
                aggfunc="sum",
                fill_value=0,
            )
            st.markdown("### Records of Selected Members")
            st.dataframe(pivot_data, use_container_width=True)

            data_melted = pivot_data.reset_index().melt(
                id_vars="member_name", var_name="Product", value_name="Quantity"
            )
            chart = (
                alt.Chart(data_melted)
                .mark_bar()
                .encode(
                    x="Product:N",
                    y="Quantity:Q",
                    color="member_name:N",
                )
            )
            st.altair_chart(chart, use_container_width=True)

        # Product table
        st.markdown("<h2>📦 Product Table</h2>", unsafe_allow_html=True)
        selected_products = st.multiselect(
            "Choose Products",
            list(products["name"].unique()),
            list(products["name"].unique())[:2],
            help="Select products to view from the table",
        )

        if not selected_products:
            st.error("Please select at least one product")
        else:
            product_frame = products[products["name"].isin(selected_products)]
            st.dataframe(product_frame, use_container_width=True)

        # Member table
        st.markdown("<h2>👥 Member Table</h2>", unsafe_allow_html=True)
        selected_members = st.multiselect(
            "Choose Members",
            list(members["name"].unique()),
            list(members["name"].unique())[:2],
            help="Select members to view from the table",
        )

        if not selected_members:
            st.error("Please select at least one member")
        else:
            member_frame = members[members["name"].isin(selected_members)]
            st.dataframe(member_frame, use_container_width=True)

    except sqlite3.Error as e:
        st.error(f"Database connection error: {e}")

    # Refresh button
    st.button("🔄 Refresh Data", on_click=refresh_data, use_container_width=True)

with col2:
    if st.session_state.agent_created:
        st.markdown("<h2>💬 Chat with SQLite Agent!</h2>", unsafe_allow_html=True)

        # Chat container for messages
        chat_container = st.container()

        with chat_container:
            with st.chat_message("assistant"):
                st.markdown(
                    "How can I help you today? Try asking me to insert new members into the database or summarize and update one's purchase records."
                )

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        def response_generator(response):
            for word in response:
                yield word
                time.sleep(0.01)

    else:
        st.info(
            "Please create an agent using the 'Create Agent' button in the sidebar to start chatting."
        )

# Move chat input to the bottom
prompt = st.chat_input("Type your message here...")

if prompt:
    logger.info(f"User prompt: {prompt}")
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    # Handle streaming messages
    with chat_container:
        with st.chat_message("assistant"):
            response = ""
            logger.info("Starting agent stream...")
            # Change: Use framework's stream_execute method instead of direct agent.stream
            for i, step in enumerate(st.session_state.framework.stream_execute(
                agent=st.session_state.agent, 
                message=prompt
            )):
                logger.debug(f"Stream step {i}: {type(step)} - {list(step.keys()) if isinstance(step, dict) else 'Not dict'}")
                if "agent" in step:
                    logger.info("Processing agent step")
                    messages = step["agent"]["messages"]
                    for j, message in enumerate(messages):
                        logger.debug(f"Agent message {j}: {type(message)}")
                        if message.tool_calls:
                            tool_call = message.tool_calls[0]
                            step_response = f'**Calling `{tool_call["name"]}` tool...**'
                            st.markdown(step_response)
                        else:
                            step_response = message.content
                            step_response = st.write_stream(
                                response_generator(step_response)
                            )

                elif "tools" in step:
                    logger.info("Processing tools step")
                    messages = step["tools"]["messages"]
                    
                    for j, message in enumerate(messages):
                        logger.debug(f"Tool message {j}: {type(message)} - Tool: {message.name}")
                        
                        try:
                            # Handle different tool types with appropriate display
                            if message.name in ["ViewAllProducts", "ViewAllMembers"]:
                                # For query tools, show "retrieving" message first
                                step_response = "**Retrieving data from database...**"
                                st.write_stream(response_generator(step_response))
                                
                                # Then display the actual results
                                step_response = f"**Tool Result ({message.name}):**\n\n```\n{message.content}\n```"
                                step_response = st.write_stream(response_generator(step_response))
                                
                            else:
                                # For other tools, directly show results
                                step_response = f"**Tool Message ({message.name}):**\n\n{message.content}"
                                step_response = st.write_stream(response_generator(step_response))
                                
                                # Refresh data if it's a data-modifying tool
                                if message.name in ["ExtractAndWriteUserInfo", "Purchase"]:
                                    refresh_data()
                                    st.success("✅ Database updated! Data refreshed.")
                            
                            logger.debug(f"Successfully processed tool message: {message.name}")
                            
                        except Exception as e:
                            logger.error(f"Error processing tool message {message.name}: {e}", exc_info=True)
                            step_response = f"**Tool Execution Error:** {str(e)}"
                            step_response = st.write_stream(response_generator(step_response))

                if response == "":
                    response += step_response
                else:
                    response += "\n\n" + step_response

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

# Style the UI with more spacing and visual separation
st.markdown(
    "<style> .stMarkdown { margin-bottom: 2rem !important; } </style>",
    unsafe_allow_html=True,
)
