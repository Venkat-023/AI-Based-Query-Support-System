📘 AI-Based Intelligent Query System and Analytics
An Interactive AI-Powered Data Exploration, Analysis, and Visualization Platform

🚀 Live Demo:
👉 https://ai-based-query-support-system-1.onrender.com/

🌟 Overview

AI-Based Intelligent Query System and Analytics is a full-stack Streamlit application built to help users interact with datasets intelligently using natural language.

It integrates:

Gemini AI for chat-based data querying

Pandas & SQL processing

Automated dashboards and analytics

Interactive charting with multiple graph types

Error-handled execution for safe user experience

Upload a CSV, ask questions in natural language, and get insights instantly — through chat, tables, or visualizations.

✨ Key Features
🔹 1. AI-Driven Chat Analyst

Ask questions about your data in natural language

AI decides between:

<chat> → conversational response

<pandas> → executes Pandas code safely

<sql> → runs SQL queries on your dataset

Strict sandboxing prevents malicious code execution

🔹 2. Intelligent Charts & Visualizations

Choose from multiple chart types:

Scatter Plot

Line Chart

Bar Chart

Histogram

Box Plot

Heatmap

Categoric Count Plot

Pie Chart

The system checks:

Data type compatibility

Missing values

Numeric/categorical requirements

🔹 3. Automated Data Analytics

Summary statistics (describe())

Missing value analysis

Correlation heatmaps

Automated type detection

🔹 4. User-Friendly, Sci-Fi Themed UI

Clean modern design:

Custom styling

Chat bubbles

Sidebar filtering

Responsive layout

🧠 Architecture
User
 │
 │  Natural language queries
 ▼
Gemini Model Selector (auto-switching across multiple models)
 │
 │  Model response in <chat> / <pandas> / <sql>
 ▼
Safe Expression Execution Layer
 │
 ├── Pandas sandbox (blocked builtins)
 ├── SQL in-memory engine (SQLite)
 └── Chat response handler
 │
 ▼
Streamlit UI Renderer (Chat + Dashboard + Analytics)

🛠 Tech Stack
Frontend / UI

Streamlit

Custom CSS theme

Matplotlib

Seaborn

Backend

Python

Pandas

NumPy

SQLite

Requests API Layer

AI Engine

Google Gemini 2.x Model Family

Automatic model fallback & response handler

📂 Project Structure
├── app.py
├── README.md
├── requirements.txt
└── .streamlit/
      └── config.toml

📦 Installation & Running Locally
1. Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

2. Install dependencies
pip install -r requirements.txt

3. Add your Gemini API key

Create .env (optional) or set OS env variable:

export GEMINI_API_KEY="your_key_here"

4. Run the app
streamlit run app.py

🚀 Deployment (Render)
Build command
pip install -r requirements.txt

Start command
streamlit run app.py --server.port $PORT --server.address 0.0.0.0

Environment Variable
GEMINI_API_KEY = <your key>
🔮 Future Enhancements

Multi-dataset support

Automated report generation

Model comparison engine

Exporting charts as images

CSV cleaning assistant
