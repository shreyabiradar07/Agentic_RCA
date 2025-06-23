import streamlit as st
import random
from streamlit_autorefresh import st_autorefresh
from transformers import pipeline
import os

from agentic_rca import knowledge_graph_data, logging_data_simulated, RCAAgent

os.environ["STREAMLIT_FILE_WATCHER"] = "none"

st.set_page_config(page_title="Monitoring Dashboard", layout="wide")

def generate_monitoring_data():
    return {
        "WebFrontend": {
            "p99_latency": round(random.uniform(0.6, 0.9), 2),
            "cpu_utilization": round(random.uniform(0.4, 0.7), 2),
            "memory_usage": round(random.uniform(0.5, 1.1), 2)
        },
        "ServiceA": {
            "response_time": round(random.uniform(0.5, 0.8), 2),
            "cpu_utilization": round(random.uniform(0.4, 0.9), 2),
            "memory_usage": round(random.uniform(0.5, 1.2), 2),
            "gc_time": round(random.uniform(0.2, 0.6), 2)
        },
        "DatabaseX": {
            "query_execution_time": round(random.uniform(0.8, 1.3), 2),
            "cpu_utilization": round(random.uniform(0.5, 0.8), 2),
            "memory_usage": round(random.uniform(0.4, 1.0), 2),
            "disk_io": round(random.uniform(200, 500), 2),
            "cache_hit_ratio": round(random.uniform(0.6, 0.95), 2),
            "detailed": {
                "slowest_queries": random.sample([
                    "SELECT COUNT(*) FROM orders WHERE status = 'pending' (0.9s)",
                    "SELECT * FROM products WHERE price > 1000 (1.3s)",
                    "UPDATE inventory SET stock = stock - 1 WHERE id = ? (1.05s)",
                    "DELETE FROM sessions WHERE last_active < NOW() - INTERVAL '30 days' (1.2s)"
                    "SELECT * FROM products WHERE category = 'electronics' AND price > 500 ORDER BY created_at DESC (1.3s)",
                    "SELECT COUNT(*) FROM orders WHERE status = 'pending' AND created_at < NOW() - INTERVAL '1 day' (1.1s)",
                    "UPDATE inventory SET stock = stock - 1 WHERE product_id IN (SELECT id FROM products WHERE discontinued = false) (1.05s)",
                    "DELETE FROM user_sessions WHERE last_active < NOW() - INTERVAL '30 days' (1.2s)",
                    "SELECT u.id, u.name, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id (1.4s)"
                ], 3)
            }
        }
    }

# Auto-refresh dashboard every 3 seconds
st_autorefresh(interval=15000, limit=None, key="refresh")

st.title("📊 Intelligent RCA dashboard")

data = generate_monitoring_data()

# Split layout into 2 columns: left for metrics, right for chatbot
col1, col2 = st.columns([3, 1])

# === LEFT: Metrics and DB queries ===
with col1:
    st.subheader("📈 System Health Overview")

    metrics_table = []
    for service, metrics in data.items():
        row = {"Service": service}
        for k, v in metrics.items():
            if isinstance(v, dict): continue  # skip nested
            row[k] = v
        metrics_table.append(row)

    st.table(metrics_table)

    # st.subheader("🛢️ Slowest DB Queries")
    # for query in data["DatabaseX"]["detailed"]["slowest_queries"]:
    #     st.code(query)

# === RIGHT: Chatbot ===
with col2:
    st.subheader("🤖 Metrics Chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    def answer_bot(question):
        question = question.lower()
        if "latency" in question:
            return "Latency refers to the time taken to process a request. Values over 0.7s might indicate slowness."
        elif "gc" in question:
            return "GC (Garbage Collection) time is the duration spent reclaiming memory. High GC time can impact performance."
        elif "disk" in question:
            return "Disk I/O refers to read/write speed. High values could indicate DB bottlenecks."
        elif "query" in question:
            return "Slow queries are operations that take longer than expected to execute, usually over 1s."
        else:
            return "I can help explain metrics like latency, memory usage, disk I/O, or DB queries."

    user_input = st.text_input("Ask a question:", key="user_input")

    if user_input:
        if "last_user_input" not in st.session_state or st.session_state.last_user_input != user_input:
            response = answer_bot(user_input)
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Bot", response))
            st.session_state.last_user_input = user_input

    st.markdown("### 💬 Chat History")
    for sender, message in st.session_state.chat_history[-10:]:
        st.markdown(f"**{sender}:** {message}")


# Load LLM once
@st.cache_resource
def load_llm():
    return pipeline("text2text-generation", model="google/flan-t5-large")

llm_pipeline = load_llm()


# Slow query strings
raw_slow_queries = [
    "SELECT COUNT(*) FROM orders WHERE status = 'pending' (0.9s)",
    "SELECT * FROM products WHERE price > 1000 (1.3s)",
    "UPDATE inventory SET stock = stock - 1 WHERE id = ? (1.05s)"
    "SELECT * FROM products WHERE category = 'electronics' AND price > 500 ORDER BY created_at DESC (1.3s)",
    "SELECT COUNT(*) FROM orders WHERE status = 'pending' AND created_at < NOW() - INTERVAL '1 day' (1.1s)",
    "UPDATE inventory SET stock = stock - 1 WHERE product_id IN (SELECT id FROM products WHERE discontinued = false) (1.05s)",
    "DELETE FROM user_sessions WHERE last_active < NOW() - INTERVAL '30 days' (1.2s)",
    "SELECT u.id, u.name, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id (1.4s)"
]

# Transform into dictionary list
slow_queries = []
for q in raw_slow_queries:
    query_text, latency_str = q.rsplit("(", 1)
    latency = float(latency_str.strip("s)")) * 1000  # Convert to ms
    slow_queries.append({"query": query_text.strip(), "latency_ms": round(latency)})

# UI
st.subheader("🧠 Root Cause Analysis (RCA) Engine")

if "rca_output" not in st.session_state:
    st.session_state.rca_output = ""
if "show_slow_queries" not in st.session_state:
    st.session_state.show_slow_queries = False
if "selected_queries" not in st.session_state:
    st.session_state.selected_queries = []


if st.button("Run RCA Analysis"):
    rca_agent = RCAAgent(knowledge_graph_data, logging_data_simulated, generate_monitoring_data(), use_llm=llm_pipeline)
    report = rca_agent.run_rca()
    st.session_state.rca_output = report
    st.session_state.show_slow_queries = True
    st.session_state.selected_queries = random.sample(slow_queries, k=random.choice([2, 3]))


if st.session_state.rca_output:
    st.markdown("### 📄 RCA Report")
    st.text_area("Report Output", st.session_state.rca_output, height=150)

if st.session_state.show_slow_queries and st.session_state.selected_queries:
    st.markdown("##### 🐢 Slow DB Queries")
    for q in st.session_state.selected_queries:
        st.code(f"{q['query']}  -- {q['latency_ms']} ms", language="sql")
