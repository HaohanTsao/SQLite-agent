# SQLite-agent

<img width="1117" alt="截圖 2024-09-25 下午2 01 16" src="https://github.com/user-attachments/assets/c00d28b7-f6ef-4866-a45d-2c1fd095a975">

## Project Description

`SQLite-agent` is a Langchain-based AI agent designed to manage an SQLite database with CRUD functionalities. This agent is capable of extracting user and product information from natural language text, writing the data into an SQLite database, and performing various database operations like fetching member and product details or processing purchases. Further, you can ask questions about members and products' information. The interface integrates Langchain with Streamlit, making it a powerful and flexible tool for automating user interactions and data management.

### Features:
- Extract user and product information from text using language models.
- Automate database operations such as adding new members, retrieving purchase records, and processing transactions.
- Answer and analysis database to answer user's question.
- Integration with Streamlit for user-friendly interaction.
  
## Setup Guide

Follow these steps to set up and run the `SQLite-agent`:

### Prerequisites

- Python 3.9 or above installed on your system.
- SQLite installed (comes pre-installed on most systems).

### 1. Clone the Repository

```bash
git clone https://github.com/HaohanTsao/SQLite-agent.git
```

### 2. Install Dependencies

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

### 3. Create the .env File
In the root directory of the project, create a .env file and add your OPENAI_API_KEY. This API key will be used by the agent to interact with OpenAI models.

Here’s an example of how the .env file should look:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```
Make sure to replace your_openai_api_key_here with your actual OpenAI API key.

### 4. Set Up the SQLite Database

You don't need to initialize the SQLite database. The database will be built after running the Streamlit app. And there are some initial example data in the database initiallt. If you want to reset the database. Just remove it directly.

### 5. Running the Streamlit App

After setting up the database, you can launch the Streamlit app to interact with the agent. Run the following command:

```bash
streamlit run Demo.py
```

This will start the app, and you can interact with the SQLite-agent via the provided interface.

### 6. Interacting with the Agent

The `SQLite-agent` provides several tools you can use to interact with the SQLite database:

- **Extract and Write User Info**: Extracts user info from text and writes it to the database.
- **Fetch Purchase Records**: Retrieves a member's purchase history from the database.
- **Process Purchases**: Adds new purchases to the database after extracting both user and product information.
- **Retrieve Member Info**: Returns all member information stored in the database.
- **Retrieve Product Info**: Returns all product information stored in the database.

### Example Queries

Here are a few examples of how to interact with the agent:

- **Adding a member**:  
  "Add a new member named John Doe, with email john.doe@example.com and age 30."
  
- **Checking purchase records**:  
  "What are the purchase records for Jane Smith?"
  
- **Processing a purchase**:  
  "John Doe wants to buy 2 Smartphone."

- **Question about member**:
  "How old is John Doe ?"

- **Question about product**:
  "How much is a Smartphone ?"

### 6. Modifying and Extending the Project

If you'd like to add more functionalities or modify the current ones, you can directly edit the corresponding tools and chains in the `sqlite_agent.py` file in backend folder. The `DBManager` class in `db_manager.py` provides functions for interacting with the SQLite database, which can be extended based on your project requirements.

Enjoy building with `SQLite-agent`! If you run into any issues, feel free to submit an issue on the project's repository.
