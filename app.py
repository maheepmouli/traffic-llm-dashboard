# app.py – Smart Assistant with Dynamic Plots
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai

# --- Streamlit Setup ---
st.set_page_config(page_title="Traffic LLM Assistant", layout="wide")
st.title("🧠 Traffic Smart Assistant")

# --- Load Data ---
@st.cache_data
def load_data():
    return pd.read_csv("city_data.csv")

df = load_data()

# --- Sidebar Input ---
st.sidebar.header("Ask me anything")
user_query = st.sidebar.text_area("Try questions like:", "Show trend of congestion in Barcelona")
openai_api_key = st.sidebar.text_input("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", type="password")

# --- Core Functionality ---
def plot_and_answer(query):
    query_lower = query.lower()

    # Line Chart (Trends)
    if "trend" in query_lower or "over time" in query_lower:
        for city in df['CITY'].unique():
            if city.lower() in query_lower:
                city_data = df[df['CITY'].str.lower() == city.lower()].copy()
                city_data['date'] = pd.to_datetime(city_data['date'])
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(city_data['date'], city_data['congestion_index'])
                ax.set_title(f"Congestion Trend Over Time in {city}")
                ax.set_ylabel("Congestion Index")
                ax.set_xlabel("Date")
                st.pyplot(fig)
                return f"Trend of congestion in {city} plotted over time.", fig
        return "Please mention a specific city for the trend.", None

    # Scatter Plot (Relationship)
    elif "scatter" in query_lower or "relationship" in query_lower:
        for city in df['CITY'].unique():
            if city.lower() in query_lower:
                city_data = df[df['CITY'].str.lower() == city.lower()]
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(city_data['AQI_mean'], city_data['congestion_index'], alpha=0.6)
                ax.set_xlabel("AQI")
                ax.set_ylabel("Congestion Index")
                ax.set_title(f"AQI vs Congestion in {city}")
                st.pyplot(fig)
                return f"Scatter plot of AQI vs congestion for {city}.", fig
        return "Please specify a city for the scatter plot.", None

    # Box Plot (Distributions)
    elif "distribution" in query_lower or "boxplot" in query_lower:
        if "speed" in query_lower:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=df, x="MANAGEMENT_TYPE", y="SPEED", ax=ax)
            ax.set_title("Speed Distribution by Management Type")
            st.pyplot(fig)
            return "Boxplot showing SPEED distribution by traffic management type.", fig
        return "Try asking for 'boxplot of SPEED by management type' or similar.", None

    # Bar Chart (Correlation Ranking)
    elif "most affected by" in query_lower:
        if "precipitation" in query_lower:
            feature = "prcp"
        elif "aqi" in query_lower:
            feature = "AQI_mean"
        elif "wind" in query_lower:
            feature = "wspd"
        elif "temperature" in query_lower or "temp" in query_lower:
            feature = "tavg"
        else:
            return "I couldn't detect the factor to analyze. Try AQI, precipitation, wind, or temperature.", None

        corr = df.groupby("CITY")[[feature, "congestion_index"]].corr().unstack().iloc[:,1]
        corr = corr.abs().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=corr.values, y=corr.index, ax=ax)
        ax.set_title(f"Impact of {feature.upper()} on Congestion Index")
        st.pyplot(fig)

        top_city = corr.idxmax()
        return f"{top_city} is most affected by {feature.upper()} based on correlation with congestion index.", fig

    # Default fallback
    else:
        return "I'm still learning to understand this question. Try asking for trends, relationships, boxplots, or rankings.", None

# --- Optional OpenAI LLM Support ---
def get_llm_answer(query, context_summary):
    if not openai_api_key:
        return "LLM response not available. Please provide your OpenAI API key."

    openai.api_key = openai_api_key
    messages = [
        {"role": "system", "content": "You are a traffic analyst. Answer clearly based on the context."},
        {"role": "user", "content": f"{query}\n\nContext: {context_summary}"}
    ]
    response = openai.ChatCompletion.create(model="gpt-4", messages=messages)
    return response['choices'][0]['message']['content']

# --- Display Results ---
if user_query:
    with st.spinner("Analyzing..."):
        response_text, fig = plot_and_answer(user_query)
        st.subheader("Insight")
        st.write(response_text)

        if openai_api_key:
            context = df.describe(include='all').to_string()
            llm_response = get_llm_answer(user_query, context)
            st.subheader("LLM-enhanced Explanation")
            st.write(llm_response)
else:
    st.write("Type your question in the sidebar and hit enter!")
