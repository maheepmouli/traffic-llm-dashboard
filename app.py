# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Setup ---
st.set_page_config(page_title="Traffic LLM Dashboard", layout="wide")
st.title("🚦 Traffic Congestion Insights Dashboard")

# --- Load Data ---
@st.cache_data
def load_data():
    return pd.read_csv("city_data.csv")

df = load_data()

# --- Sidebar Input ---
st.sidebar.header("Ask a Question")
user_input = st.sidebar.text_input("Try questions like:", "Which city is most affected by precipitation?")

# --- Analysis Function ---
def handle_query(query):
    if "precipitation" in query.lower():
        feature = "prcp"
    elif "aqi" in query.lower():
        feature = "AQI_mean"
    elif "wind" in query.lower():
        feature = "wspd"
    elif "temperature" in query.lower() or "temp" in query.lower():
        feature = "tavg"
    else:
        return "Sorry, I couldn't match your question to a known feature.", None

    # Calculate correlations
    corr = df.groupby("CITY")[[feature, "congestion_index"]].corr().unstack().iloc[:,1]
    corr = corr.abs().sort_values(ascending=False)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=corr.values, y=corr.index, ax=ax)
    ax.set_title(f"Impact of {feature.upper()} on Congestion Index")
    ax.set_xlabel("|Correlation|")
    ax.set_ylabel("City")
    st.pyplot(fig)

    # Explanation
    top_city = corr.idxmax()
    return f"{top_city} is most affected by {feature.upper()} based on correlation with congestion index.", fig

# --- Output ---
if user_input:
    with st.spinner("Analyzing..."):
        answer, _ = handle_query(user_input)
        st.subheader("Answer")
        st.write(answer)
else:
    st.write("Enter a question like 'Which city is most affected by AQI?' or 'Where does precipitation impact congestion the most?'")
