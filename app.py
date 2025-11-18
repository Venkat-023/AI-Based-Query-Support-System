# nebula_datasense_app.py
# Improved Nebula DataSense — error handling + graph selector

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
from typing import Tuple

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Ai Based Intelligent Query Agent", layout="wide")

# ---------------------------
# Models
# ---------------------------
MODEL_PRIORITY = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001",
    "gemini-2.5-flash-lite"
]

# ---------------------------
# Simple Sci-fi UI (kept, minimal risk)
# ---------------------------
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 50% 50%, rgba(0,200,255,0.12), rgba(0,10,20,1));
    color: #e7faff;
    font-family: 'Inter', sans-serif;
}
.title { font-size: 36px; font-weight: 700; color: #00eaff; text-align:center; }
.subtitle { font-size:16px; color: #dff8ff; text-align:center; margin-bottom:18px; }
.user-bubble{ background:#fff;color:#000;padding:10px;border-radius:10px;margin:8px 0 }
.bot-bubble{ background:rgba(255,255,255,0.08);color:#fff;padding:12px;border-left:4px solid #00eaff;border-radius:10px;margin:8px 0 }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">✨ AI Based Intelligent Query System </div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a CSV, ask questions, or build charts — now with safer error handling.</div>', unsafe_allow_html=True)

# ---------------------------
# Gemini API helper
# ---------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

def call_gemini_endpoint(model_name: str, system_prompt: str, user_prompt: str, use_beta: bool=True) -> Tuple[int, dict]:
    """Try v1beta first if use_beta True, fallback to v1. Returns status, json"""
    base = "v1beta" if use_beta else "v1"
    url = f"https://generativelanguage.googleapis.com/{base}/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
    body = {
        "contents": [{"parts": [{"text": system_prompt + "\nUSER:\n" + user_prompt}]}],
        "generationConfig": {"temperature": 0.15}
    }
    try:
        r = requests.post(url, json=body, timeout=15)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {"error": "invalid-json-response", "text": r.text}
    except requests.exceptions.RequestException as e:
        return -1, {"error": str(e)}


def call_gemini_auto(system_prompt: str, user_prompt: str) -> Tuple[str, str]:
    """Try model list and endpoints; returns (model_used or None, response_text)
    If API key missing, return helpful message."""
    if not GEMINI_API_KEY:
        return None, "<chat>Error: GEMINI_API_KEY not set on server. Add it to environment variables.</chat>"

    last_error = None
    for model in MODEL_PRIORITY:
        status, response = call_gemini_endpoint(model, system_prompt, user_prompt, use_beta=True)
        # common helpful debug
        if status == 200 and isinstance(response, dict) and "candidates" in response:
            try:
                txt = response["candidates"][0]["content"]["parts"][0]["text"]
                return model, txt
            except Exception:
                return model, "<chat>Error: Unexpected response shape from model.</chat>"

        # try fallback to v1 for this model if v1beta returned 404 or similar
        if status in (404, 400) or (isinstance(response, dict) and response.get("error")):
            # try non-beta
            status2, response2 = call_gemini_endpoint(model, system_prompt, user_prompt, use_beta=False)
            if status2 == 200 and isinstance(response2, dict) and "candidates" in response2:
                try:
                    txt = response2["candidates"][0]["content"]["parts"][0]["text"]
                    return model, txt
                except Exception:
                    return model, "<chat>Error: Unexpected response shape from model.</chat>"

        last_error = (status, response)
        # if server error, try next model
        if status in (500, 503):
            continue

    # If loop finishes, show a helpful diagnostic message
    if last_error is None:
        return None, "<chat>All models busy. Try again later.</chat>"
    status, response = last_error
    return None, f"<chat>Model requests failed. Last status={status}, response={response}</chat>"

# ---------------------------
# Safe pandas evaluator
# ---------------------------
ALLOWED = {"df", "pd", "np"}

def safe_eval(expr: str, df: pd.DataFrame):
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {e}")
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and n.id not in ALLOWED:
            raise ValueError(f"Illegal variable or function used: {n.id}")
    try:
        return eval(compile(tree, "<safe>", "eval"), {"__builtins__": {}}, {"df": df, "pd": pd, "np": np})
    except Exception as e:
        raise

# ---------------------------
# Session state defaults
# ---------------------------
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

# ---------------------------
# Tabs
# ---------------------------
tab_chat, tab_dashboard, tab_analytics = st.tabs(["💬 Chat", "📊 Dashboard", "📈 Analytics"])

# ---------------------------
# CHAT TAB
# ---------------------------
with tab_chat:
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

    col1, col2 = st.columns([1, 9])
    with col1:
        attach = st.button("+", key="att", help="Upload CSV")
    with col2:
        def on_enter():
            st.session_state.text_buffer = st.session_state.get(f"input_{st.session_state.input_key}", "")
            st.session_state.pending_send = True
        st.text_input("", key=f"input_{st.session_state.input_key}", placeholder="Ask anything…", on_change=on_enter)

    if attach:
        st.session_state.show_upload = True

    if st.session_state.show_upload:
        st.markdown("<div class='hidden-upload'>", unsafe_allow_html=True)
        try:
            csv_file = st.file_uploader("Upload CSV", type=["csv"], key="csv_u")
        except Exception as e:
            st.error(f"File uploader error: {e}")
            csv_file = None
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        csv_file = None

    if csv_file:
        try:
            df = pd.read_csv(csv_file)
            st.session_state.df = df
            rows, cols = df.shape
            schema = f"{rows} rows × {cols} columns\n" + "\n".join([f"- {c} ({df[c].dtype})" for c in df.columns])
            st.session_state.schema = schema
            st.markdown("### 📄 Dataset Uploaded")
            st.code(schema)
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.session_state.df = None

    if st.session_state.df is None:
        st.info("Upload a dataset in the Chat tab to enable chat and dashboards.")
        st.stop()

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
                    try:
                        res = safe_eval(expr, df)
                        if isinstance(res, pd.DataFrame):
                            st.session_state.messages.append({"role": "assistant", "content": {"type": "table", "data": res}})
                        elif isinstance(res, pd.Series):
                            st.session_state.messages.append({"role": "assistant", "content": {"type": "table", "data": res.to_frame()}})
                        else:
                            st.session_state.messages.append({"role": "assistant", "content": str(res)})
                    except Exception as e:
                        st.session_state.messages.append({"role": "assistant", "content": f"Error evaluating pandas expression: {e}"})
                elif "<sql>" in raw:
                    query = raw.split("<sql>")[1].split("</sql>")[0]
                    try:
                        conn = sqlite3.connect(":memory:")
                        df.to_sql("data", conn, index=False)
                        res = pd.read_sql_query(query, conn)
                        st.session_state.messages.append({"role": "assistant", "content": {"type": "table", "data": res}})
                    except Exception as e:
                        st.session_state.messages.append({"role": "assistant", "content": f"SQL execution error: {e}"})
                else:
                    st.session_state.messages.append({"role": "assistant", "content": raw})
            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": f"Error handling model response: {e}"})
        st.rerun()

# ---------------------------
# DASHBOARD TAB — with graph picker and robust plotting
# ---------------------------
with tab_dashboard:
    if st.session_state.df is None:
        st.warning("Upload a dataset first using Chat tab.")
        st.stop()
    df = st.session_state.df.copy()

    st.markdown("## 📊 Interactive Dashboard")

    # Sidebar filters
    st.sidebar.header("Filters & Chart Options")
    selected_col = st.sidebar.selectbox("Select column to filter (optional)", [None] + list(df.columns))

    if selected_col:
        if pd.api.types.is_numeric_dtype(df[selected_col]):
            min_val = float(df[selected_col].min())
            max_val = float(df[selected_col].max())
            user_range = st.sidebar.slider("Range", min_val, max_val, (min_val, max_val))
            filtered_df = df[(df[selected_col] >= user_range[0]) & (df[selected_col] <= user_range[1])]
        else:
            choices = st.sidebar.multiselect("Select values", df[selected_col].dropna().unique())
            filtered_df = df[df[selected_col].isin(choices)] if choices else df
    else:
        filtered_df = df

    st.markdown("### 📄 Filtered Table")
    st.dataframe(filtered_df.reset_index(drop=True))

    # Chart builder — graph selection
    st.markdown("### 📈 Chart Builder")
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = filtered_df.select_dtypes(include=[object, "category"]).columns.tolist()

    chart_type = st.selectbox("Chart type", [
        "Scatter", "Line", "Bar", "Histogram", "Box", "Heatmap", "Count (categorical)", "Pie"
    ])

    x_col = st.selectbox("X-axis", [None] + list(filtered_df.columns), index=1 if len(filtered_df.columns) > 0 else 0)
    y_col = st.selectbox("Y-axis (optional)", [None] + list(filtered_df.columns))

    plot_btn = st.button("Plot")

    if plot_btn:
        try:
            if chart_type == "Scatter":
                if x_col not in numeric_cols or y_col not in numeric_cols:
                    st.error("Scatter requires numeric X and Y columns.")
                else:
                    fig, ax = plt.subplots()
                    ax.scatter(filtered_df[x_col], filtered_df[y_col], alpha=0.7)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    ax.set_title(f"{x_col} vs {y_col}")
                    st.pyplot(fig)

            elif chart_type == "Line":
                if x_col is None or y_col is None:
                    st.error("Line chart requires both X and Y.")
                else:
                    fig, ax = plt.subplots()
                    ax.plot(filtered_df[x_col], filtered_df[y_col])
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    st.pyplot(fig)

            elif chart_type == "Bar":
                if y_col is None:
                    st.error("Bar chart requires Y column (values) and X column (categories).")
                else:
                    fig, ax = plt.subplots()
                    if x_col is None:
                        # aggregate by index
                        filtered_df[y_col].plot(kind='bar', ax=ax)
                    else:
                        grouped = filtered_df.groupby(x_col)[y_col].mean().reset_index()
                        ax.bar(grouped[x_col].astype(str), grouped[y_col])
                        ax.set_xticklabels(grouped[x_col].astype(str), rotation=45, ha='right')
                    st.pyplot(fig)

            elif chart_type == "Histogram":
                if x_col not in numeric_cols:
                    st.error("Histogram requires a numeric X column.")
                else:
                    fig, ax = plt.subplots()
                    ax.hist(filtered_df[x_col].dropna(), bins=30)
                    ax.set_xlabel(x_col)
                    st.pyplot(fig)

            elif chart_type == "Box":
                if y_col not in numeric_cols:
                    st.error("Box plot requires a numeric Y column.")
                else:
                    fig, ax = plt.subplots()
                    ax.boxplot(filtered_df[y_col].dropna())
                    ax.set_title(f"Box plot of {y_col}")
                    st.pyplot(fig)

            elif chart_type == "Heatmap":
                if len(numeric_cols) < 2:
                    st.error("Heatmap requires at least two numeric columns.")
                else:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.heatmap(filtered_df[numeric_cols].corr(), annot=True, ax=ax)
                    st.pyplot(fig)

            elif chart_type == "Count (categorical)":
                if x_col not in cat_cols:
                    st.error("Select a categorical X column for counts.")
                else:
                    counts = filtered_df[x_col].value_counts()
                    fig, ax = plt.subplots()
                    ax.bar(counts.index.astype(str), counts.values)
                    ax.set_xticklabels(counts.index.astype(str), rotation=45, ha='right')
                    st.pyplot(fig)

            elif chart_type == "Pie":
                if x_col not in cat_cols:
                    st.error("Pie chart requires a categorical column.")
                else:
                    counts = filtered_df[x_col].value_counts()
                    fig, ax = plt.subplots()
                    ax.pie(counts.values, labels=counts.index.astype(str), autopct='%1.1f%%')
                    st.pyplot(fig)

            else:
                st.error("Chart type not implemented.")

        except Exception as e:
            st.exception(e)

# ---------------------------
# ANALYTICS TAB
# ---------------------------
with tab_analytics:
    if st.session_state.df is None:
        st.warning("Upload a dataset first.")
        st.stop()
    df = st.session_state.df
    st.markdown("## 📈 Automated Analytics")

    # Summary
    try:
        st.markdown("### Summary Statistics")
        st.dataframe(df.describe(include='all'))
    except Exception as e:
        st.error(f"Failed to compute summary statistics: {e}")

    # Missing values
    try:
        st.markdown("### Missing Values")
        st.dataframe(df.isnull().sum().rename("missing_count").to_frame())
    except Exception as e:
        st.error(f"Failed to compute missing values: {e}")

    # Correlation heatmap
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            st.markdown("### Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df[numeric_cols].corr(), annot=False, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.info("Not enough numeric columns for a correlation heatmap.")
    except Exception as e:
        st.error(f"Failed to draw correlation heatmap: {e}")

# ---------------------------
# End
# ---------------------------
