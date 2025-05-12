# dashboard.py
import streamlit as st
import random
import time

def generate_monitoring_data():
    return {
        "WebFrontend": {"p99_latency": round(random.uniform(0.6, 0.9), 2)},
        "ServiceA": {"response_time": round(random.uniform(0.4, 0.8), 2)},
        "DatabaseX": {
            "query_execution_time": round(random.uniform(0.8, 1.3), 2),
            "cpu_utilization": round(random.uniform(0.5, 0.8), 2),
            "detailed": {
                "slowest_queries": [random.choice([
                    "SELECT * FROM users WHERE id = ? (1.1s)",
                    "SELECT COUNT(*) FROM orders WHERE status = 'pending' (0.9s)",
                    "SELECT * FROM products WHERE price > 1000 (1.3s)"
                ])]
            }
        }
    }

st.set_page_config(page_title="Latency Monitoring Dashboard", layout="wide")
st.title("📊 Service Latency Dashboard")

placeholder = st.empty()

while True:
    data = generate_monitoring_data()

    with placeholder.container():
        st.subheader("Latest Latencies")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("WebFrontend p99 Latency", f"{data['WebFrontend']['p99_latency']}s")
        with col2:
            st.metric("ServiceA Response Time", f"{data['ServiceA']['response_time']}s")
        with col3:
            st.metric("DatabaseX Query Time", f"{data['DatabaseX']['query_execution_time']}s")

        st.subheader("DatabaseX Slowest Query")
        st.code(data['DatabaseX']['detailed']['slowest_queries'][0])

    time.sleep(3)  # Refresh every 3 seconds
