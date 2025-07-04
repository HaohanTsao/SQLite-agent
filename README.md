# SQLite-agent: Multi-Framework Comparison Platform

<img width="1117" alt="SQLite-agent" src="https://github.com/user-attachments/assets/c00d28b7-f6ef-4866-a45d-2c1fd095a975">

A comprehensive platform for comparing different AI agent frameworks using the same SQLite database operations. Test and evaluate **LangGraph**, **AutoGen**, **Semantic Kernel**, and **CrewAI** frameworks side-by-side to understand their unique strengths and characteristics.

## 🚀 New Features

### 🔄 Multi-Framework Support
- **LangGraph**: React Agent Pattern with streaming support
- **AutoGen**: Structured output with natural language understanding  
- **Semantic Kernel**: Enterprise-grade AI orchestration with plugin architecture
- **CrewAI**: Multi-agent collaboration with event-driven execution


## 🛠️ Setup Guide

### 1. Clone the Repository

```bash
git clone https://github.com/HaohanTsao/SQLite-agent.git
cd SQLite-agent
```

### 2. Install Dependencies

Install all required dependencies for multiple frameworks:

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the root directory with your Azure OpenAI credentials:

```env
AZURE_ENDPOINT=your_azure_openai_endpoint
AZURE_API_KEY=your_azure_api_key
AZURE_DEPLOYMENT=gpt-4o
AZURE_API_VERSION=2025-04-01-preview
```

### 4. Database Setup

The SQLite database will be automatically created with sample data when you first run the application. No manual setup required.

### 5. Running the Application

Launch the Streamlit app:

```bash
streamlit run Demo.py
```

The app will start at `http://localhost:8501` with the multi-framework interface.

## 💡 Usage Examples

### Framework Selection
1. Choose your framework from the sidebar dropdown
2. Click "Create Agent" to initialize the selected framework
3. Start chatting with framework-specific capabilities

### Common Operations

**Adding a Member:**
```
Add a new member named John Doe, 30 years old, email john@example.com
```

**Processing Purchases:**
```
John Doe wants to buy 2 Smartphones
```

**Retrieving Records:**
```
What are Bob Smith's purchase records?
```

**Database Queries:**
```
How old is Alice Johnson?
What products are available?
```