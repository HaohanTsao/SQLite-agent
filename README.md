# SQLite-agent

<img width="1117" alt="SQLite-agent" src="https://github.com/user-attachments/assets/c00d28b7-f6ef-4866-a45d-2c1fd095a975">

### Features:
- Extract user and product information from text using language models.
- Automate database operations such as adding new members, retrieving purchase records, and processing transactions.
- Analyze the database to answer user questions.
- Integration with Streamlit for user-friendly interaction.
- Develop your own customized tools for the agent.

## Setup Guide

Follow these steps to set up and run the `SQLite-agent`:

### 1. Clone the Repository

```bash
git clone https://github.com/HaohanTsao/SQLite-agent.git
cd SQLite-agent
```

### 2. Install Dependencies

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

### 3. Set Up the SQLite Database

You don't need to initialize the SQLite database manually. The database will be built after running the Streamlit app. Initial example data will be included in the database. If you want to reset the database, simply remove the existing database file.

### 4. Running the Streamlit App

After setting up the database, you can launch the Streamlit app to interact with the agent. Run the following command:

```bash
streamlit run Demo.py
```

This will start the app, and you can interact with the SQLite-agent via the provided interface.

### 5. Interacting with the Agent

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
  "What are the purchase records for Bob Smith?"
  
- **Processing a purchase**:  
  "John Doe wants to buy 2 Smartphones."

- **Question about a member**:  
  "How old is John Doe?"

- **Question about a product**:  
  "How much is a Smartphone?"

### 7. Modifying and Extending the Project

If you'd like to add more functionalities or modify the current ones, you can directly edit the corresponding tools and chains in the `sqlite_agent.py` file in the `backend` folder. The `DBManager` class in `db_manager.py` provides functions for interacting with the SQLite database, which can be extended based on your project requirements.

**Note**: New feature that allows users to create their own tools. Navigate to the 'Tool Developer' tab and follow the instructions and examples provided there.
<img width="1420" alt="截圖 2024-09-26 上午10 48 10" src="https://github.com/user-attachments/assets/fbdd5ad3-0db6-4d9d-bd90-fa8ecdb7dbae">

