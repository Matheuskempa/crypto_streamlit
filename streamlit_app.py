import plotly.express as ps
import streamlit as st
from cripto_project import CriptoProject, CriptoProjectAnalysis
import pandas as pd


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

def show_analysis(df):

    type = st.sidebar.selectbox(
        "Modelo",
        options=["-","Deep Learning", "Random Forest"]
    )
    
    percentual = st.sidebar.slider('Percentual de Treino', value=0.5, min_value=0.0, max_value=1.0, step=0.1)

    analysis = CriptoProjectAnalysis()
    df_ = analysis.get_real_analysed_dataframe(df)

    qtnd = round(len(df_)*percentual)

    # datas = df_[(qtnd):].reset_index()["data_final"].to_list()
    if type == "-":
        pass
    elif type == "Deep Learning":

        fig_1 = analysis.get_relational_graph(df_)  
        st.title("Período de Análise")
        st.write("No gráfico abaixo na curva azul temos o reço de fechamento e na curva vermelha temos a quantidade de notícias com sentimento negativo no dia anterior.")          
        st.plotly_chart(fig_1)
        # fig_2 = analysis.get_variation_graph(df_)
        # st.plotly_chart(fig_2)
        base = analysis._make_predictions_deep_learning(df_, qtnd)
        
        
        st.dataframe(df_)

        # base["data_final"] = pd.DataFrame(datas)
        
        a,p,r = analysis._get_metrics(base)

        st.title("Métricas de Avaliação")
        # st.subtitle("Acurácia")
        st.write(f"Acurácia de :{str(a)}")
        # st.subtitle("Precisão")
        st.write(f"Precisão de :{str(p)}")
        # st.subtitle("Recall")
        st.write(f"Recall de :{str(r)}")
        st.title("Resultados")
        # st.subtitle("Previsão")
        st.dataframe(base)


    else:

        fig_1 = analysis.get_relational_graph(df_)  
        st.title("Período de Análise")
        st.write("No gráfico abaixo na curva azul temos o reço de fechamento e na curva vermelha temos a quantidade de notícias com sentimento negativo no dia anterior.")          
        st.plotly_chart(fig_1)
        # fig_2 = analysis.get_variation_graph(df_)
        # st.plotly_chart(fig_2)
        base = analysis._make_predictions_random_forest(df_, qtnd)
        st.dataframe(df_)
        
        # base["data_final"] = pd.DataFrame(datas)
        
        a,p,r = analysis._get_metrics(base)

        st.title("Métricas de Avaliação")
        # st.subtitle("Acurácia")
        st.write(f"Acurácia de :{str(a)}")
        # st.subtitle("Precisão")
        st.write(f"Precisão de :{str(p)}")
        # st.subtitle("Recall")
        st.write(f"Recall de :{str(r)}")
        st.title("Resultados")
        # st.subtitle("Previsão")
        st.dataframe(base)
    
    

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

if currency != "-":
    df = request_crypto(currency)
    fig = create_graph(df)
    st.plotly_chart(fig)

    show_analysis(df)
