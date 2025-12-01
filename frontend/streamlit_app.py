import requests
import streamlit as st
import pandas as pd

# ================== CONFIG ==================
API_BASE_URL = "http://127.0.0.1:8000"

# ================== UI LAYOUT ==================
st.set_page_config(
    page_title="Nifty50 LR Predictor",
    page_icon="📈",
    layout="centered",
)

st.title("📈 Nifty50 Next-Day Price Prediction")
st.caption("Powered by FastAPI + Linear Regression + PostgreSQL")

st.markdown("---")

# ================== HELPER: CALL API ==================
@st.cache_data(ttl=60)
def call_prediction_api():
    url = f"{API_BASE_URL}/predict/next-day"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.RequestException as e:
        return None, str(e)


# ================== MAIN CONTENT ==================
with st.spinner("Fetching latest prediction from API..."):
    data, error = call_prediction_api()

if error:
    st.error(f"❌ Could not fetch prediction from API.\n\n`{error}`")
    st.info(
        "Make sure FastAPI is running:\n\n"
        "```bash\n"
        "cd D:\\Nifty50_StopPrediction\\backend\n"
        ".\\venv\\Scripts\\activate\n"
        "uvicorn fastapi_app:app --reload\n"
        "```"
    )
else:
    last_date = data["last_date"]
    last_close = data["last_close"]
    predicted = data["predicted_next_day_close"]
    metrics = data["model_metrics"]

    st.subheader("📅 Latest Market Snapshot")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Last Date", last_date)
    with col2:
        st.metric("Last Close", f"{last_close:,.2f}")

    st.markdown("---")

    st.subheader("🔮 Predicted Next-Day Close")
    st.metric(
        "Predicted Close (Next Session)",
        f"{predicted:,.2f}",
        f"{predicted - last_close:+.2f} vs last close",
    )

    st.markdown("---")
    st.subheader("📊 Model Evaluation Metrics")

    # Turn metrics dict into a DataFrame for a nice table
    metrics_df = pd.DataFrame(
        [{"Metric": k, "Value": v} for k, v in metrics.items()]
    )

    st.dataframe(metrics_df, use_container_width=True)

    with st.expander("Raw API Response JSON"):
        st.json(data)

st.markdown("---")
st.caption("Tip: Refresh the page after your Airflow pipeline runs to see updated predictions.")
