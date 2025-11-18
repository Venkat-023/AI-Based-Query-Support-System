##############################################
# NEBULA DATASENSE — CHAT + DASHBOARD + ANALYTICS
##############################################

import os
import ast
import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import html
import requests
import seaborn as sns
import matplotlib.pyplot as plt


# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(page_title="Nebula DataSense", layout="wide")



# -----------------------------------------------------------
# AVAILABLE GEMINI MODELS (YOUR ACCOUNT)
# -----------------------------------------------------------
MODEL_PRIORITY = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001",
    "gemini-2.5-flash-lite"
]



# -----------------------------------------------------------
# SCI-FI UI
# -----------------------------------------------------------
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 50% 50%, rgba(0,200,255,0.15), rgba(0,10,20,1));
    color: #e7faff;
    font-family: 'Inter', sans-serif;
}
.title {
    font-size: 40px;
    font-weight: 800;
    color: #00eaff;
    text-align: center;
    margin-top: 10px;
}
.subtitle {
    font-size: 20px;
    color: #e0f7ff;
    text-align: center;
    margin-bottom: 30px;
}
.hidden-upload > div { display: none !important; }
.user-bubble {
    background: #ffffff;
    color: black;
    padding: 12px 16px;
    border-radius: 12px;
    margin: 12px 0;
}
.bot-bubble {
    background: rgba(255,255,255,0.10);
    color: white;
    padding: 14px 18px;
    border-radius: 12px;
    border-left: 4px solid #00eaff;
    margin: 12px 0;
}
.stTextInput > div > div > input {
    background: white !important;
    color: black !important;
    font-size: 16px;
    padding: 12px;
    border-radius: 12px;
    border: 1px solid #00eaff;
}
.plus-btn {
    background-color: #00eaff;
    color: black;
    font-size: 28px;
    font-weight: bold;
    width: 55px;
    height: 50px;
    border-radius: 12px;
    border: none;
    cursor: pointer;
}
.plus-btn:hover {
    background-color: #00bcd4;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------
# TITLE + WELCOME
# -----------------------------------------------------------
st.markdown('<div class="title">✨ Nebula DataSense</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Welcome! How can I assist you with your dataset today?</div>', unsafe_allow_html=True)



# -----------------------------------------------------------
# GEMINI API HANDLER (AUTO MODEL SWITCHING)
# -----------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

def call_gemini(model_name, system_prompt, user_prompt):
    url = f"https://generativelanguage.googleapis.com/v1/models/{model_name}:generateContent?key={GEMINI_API_KEY}"

    body = {
        "contents": [
            {
                "parts": [
                    {"text": system_prompt + "\nUSER:\n" + user_prompt}
                ]
            }
        ],
        "generationConfig": {"temperature": 0.15}
    }

    try:
        r = requests.post(url, json=body)
        return r.status_code, r.json()
    except Exception as e:
        return -1, {"error": str(e)}



def call_gemini_auto(system_prompt, user_prompt):
    for model in MODEL_PRIORITY:
        status, response = call_gemini(model, system_prompt, user_prompt)

        if status == 200 and "candidates" in response:
            return model, response["candidates"][0]["content"]["parts"][0]["text"]

        if status in [500, 503]:
            continue

    return None, "<chat>All models busy. Try again.</chat>"



# -----------------------------------------------------------
# SAFE PANDAS EXECUTION
# -----------------------------------------------------------
ALLOWED = {"df", "pd", "np"}

def safe_eval(expr, df):
    tree = ast.parse(expr, mode="eval")
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and n.id not in ALLOWED:
            raise ValueError("Illegal variable")
    return eval(compile(tree, "<safe>", "eval"), {"__builtins__": {}}, {"df": df, "pd": pd, "np": np})



# -----------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "schema" not in st.session_state:
    st.session_state.schema = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_model" not in st.session_state:
    st.session_state.current_model = MODEL_PRIORITY[0]
if "pending_send" not in st.session_state:
    st.session_state.pending_send = False
if "text_buffer" not in st.session_state:
    st.session_state.text_buffer = ""
if "input_key" not in st.session_state:
    st.session_state.input_key = 0
if "show_upload" not in st.session_state:
    st.session_state.show_upload = False



# ===========================================================
# UI TABS
# ===========================================================
tab_chat, tab_dashboard, tab_analytics = st.tabs([
    "💬 Chat",
    "📊 Dashboard",
    "📈 Analytics"
])



# ===========================================================
# TAB 1 — CHAT MODE
# ===========================================================
with tab_chat:

    # Chat history
    for m in st.session_state.messages:
        if m["role"] == "user":
            st.markdown(f'<div class="user-bubble">👤 {html.escape(m["content"])}</div>', unsafe_allow_html=True)
        else:
            content = m["content"]
            if isinstance(content, dict) and content.get("type") == "table":
                st.markdown('<div class="bot-bubble">🤖 Result:</div>', unsafe_allow_html=True)
                st.dataframe(content["data"])
            else:
                st.markdown(f'<div class="bot-bubble">🤖 {html.escape(str(content))}</div>', unsafe_allow_html=True)

    # Chat input row
    col1, col2 = st.columns([1, 9])

    with col1:
        attach = st.button("+", key="att", help="Upload CSV")

    with col2:
        def on_enter():
            st.session_state.text_buffer = st.session_state[f"input_{st.session_state.input_key}"]
            st.session_state.pending_send = True

        st.text_input(
            "",
            key=f"input_{st.session_state.input_key}",
            placeholder="Ask anything…",
            on_change=on_enter
        )

    if attach:
        st.session_state.show_upload = True

    # Hidden uploader
    if st.session_state.show_upload:
        st.markdown("<div class='hidden-upload'>", unsafe_allow_html=True)
        csv_file = st.file_uploader("Upload CSV", type=["csv"], key="csv_u")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        csv_file = None

    # Process CSV upload
    if csv_file:
        df = pd.read_csv(csv_file)
        st.session_state.df = df
        rows, cols = df.shape
        schema = f"{rows} rows × {cols} columns\n" + "\n".join([f"- {c} ({df[c].dtype})" for c in df.columns])
        st.session_state.schema = schema

        st.markdown("### 📄 Dataset Uploaded")
        st.code(schema)
        st.dataframe(df.head())

    if st.session_state.df is None:
        st.stop()

    # Chat processing
    if st.session_state.pending_send:
        st.session_state.pending_send = False

        msg = st.session_state.text_buffer.strip()
        st.session_state.text_buffer = ""
        st.session_state.input_key += 1

        if msg:
            st.session_state.messages.append({"role": "user", "content": msg})

            system_prompt = f"""
You are a dataset assistant. Respond using EXACTLY one of:

1) <chat>...</chat>
2) <pandas>...</pandas>
3) <sql>...</sql>

Dataset Schema:
{st.session_state.schema}
"""

            model_used, raw = call_gemini_auto(system_prompt, msg)
            st.session_state.current_model = model_used or "N/A"

            df = st.session_state.df

            try:
                if "<chat>" in raw:
                    ans = raw.split("<chat>")[1].split("</chat>")[0]
                    st.session_state.messages.append({"role": "assistant", "content": ans})

                elif "<pandas>" in raw:
                    expr = raw.split("<pandas>")[1].split("</pandas>")[0]
                    res = safe_eval(expr, df)

                    if isinstance(res, pd.DataFrame):
                        st.session_state.messages.append({"role": "assistant", "content": {"type": "table", "data": res}})
                    elif isinstance(res, pd.Series):
                        st.session_state.messages.append({"role": "assistant", "content": {"type": "table", "data": res.to_frame()}})
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": str(res)})

                elif "<sql>" in raw:
                    query = raw.split("<sql>")[1].split("</sql>")[0]
                    conn = sqlite3.connect(":memory:")
                    df.to_sql("data", conn, index=False)
                    res = pd.read_sql_query(query, conn)
                    st.session_state.messages.append({"role": "assistant", "content": {"type": "table", "data": res}})

                else:
                    st.session_state.messages.append({"role": "assistant", "content": raw})

            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})

        st.rerun()



# ===========================================================
# TAB 2 — DASHBOARD MODE
# ===========================================================
with tab_dashboard:

    if st.session_state.df is None:
        st.warning("Upload a dataset first using Chat tab.")
        st.stop()

    df = st.session_state.df

    st.markdown("## 📊 Interactive Dashboard")

    # Column Selector
    st.sidebar.header("Filters")

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'int64']).columns.tolist()

    # Filters
    selected_col = st.sidebar.selectbox("Select column to filter", df.columns)

    if selected_col in numeric_cols:
        min_val = float(df[selected_col].min())
        max_val = float(df[selected_col].max())
        user_range = st.sidebar.slider("Range", min_val, max_val, (min_val, max_val))
        filtered_df = df[(df[selected_col] >= user_range[0]) & (df[selected_col] <= user_range[1])]
    else:
        choices = st.sidebar.multiselect("Select values", df[selected_col].unique())
        filtered_df = df[df[selected_col].isin(choices)] if choices else df

    st.markdown("### 📄 Filtered Table")
    st.dataframe(filtered_df)

    # Chart builder
    st.markdown("### 📈 Chart Builder")
    colA = st.selectbox("X-axis", df.columns)
    colB = st.selectbox("Y-axis", df.columns)

    fig, ax = plt.subplots()
    ax.scatter(df[colA], df[colB], alpha=0.7)
    ax.set_xlabel(colA)
    ax.set_ylabel(colB)
    ax.set_title(f"{colA} vs {colB}")

    st.pyplot(fig)



# ===========================================================
# TAB 3 — ANALYTICS MODE
# ===========================================================
with tab_analytics:

    if st.session_state.df is None:
        st.warning("Upload a dataset first.")
        st.stop()

    df = st.session_state.df

    st.markdown("## 📈 Automated Analytics")

    st.markdown("### Summary Statistics")
    st.dataframe(df.describe())

    st.markdown("### Missing Values")
    st.dataframe(df.isnull().sum())

    st.markdown("### Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
    st.pyplot(fig)

