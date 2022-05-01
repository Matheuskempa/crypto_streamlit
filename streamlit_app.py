import pandas as pd
import plotly.express as ps
import streamlit as st
from cripto_project import CriptoProject


def request_crypto(type):
    crypto = CriptoProject()
    df = crypto(type)
    st.title("Base de dados")
    st.dataframe(df)
    return df

def create_graph(df):
    st.title("Serie temporal")
    return ps.line(
        df,
        x="data_final",
        y="close"

    )

st.set_page_config(
    page_title="Crypto Dashboard",
    layout="wide"
)

# --- Left Menu ---
st.sidebar.header("Filtrar crypto")
currency = st.sidebar.selectbox(
    "Moeda",
    options=["-","BTC", "ETH", "DOGE"]
)
st.title("Explicando o projeto")
st.write("Aqui vai ficar o texto\nBem aqui mesmo")

if currency != "-":
    df = request_crypto(currency)
    fig = create_graph(df)
    st.plotly_chart(fig)

