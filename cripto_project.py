import requests
import pandas as pd
from datetime import datetime, timedelta

class CriptoProject():
    
    def __init__(self):
        print("Started CriptoProject.")
    
    def get_history_df(self,ticker='BTC'):
        """
        Get history dataframe from the desired ticker
        
        Params
        --------
        ticker: string
            initials crypto ticker
        
        Returns
        --------
        df: pandas.DataFrame

        """

        request = requests.get(f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={ticker}&tsym=USD&limit=2000&api_key=b25c013fd0cba42a723f53b10986209dd496ef34ddb1aec41145e220b152d7f9")
        df = pd.DataFrame(request.json()["Data"]["Data"])
        df["datetime"] = [datetime.fromtimestamp(elemento_data) for elemento_data in df.time]#transform into datetime
        df["data_final"] = [elemento_data.date() for elemento_data in df.datetime]
        self.min_date = df["datetime"].min()
        print("History Dataframe completed.")
        
        return df

    def _request_news(self,ticker,page):
        """
        Get history news dataframe from the desired ticker
        
        Params
        --------
        ticker: string
            initials crypto ticker
        page: int
            number of the news's page
        
        Returns
        --------
        df: pandas.DataFrame

        """
        request = requests.get(f"https://cryptonews-api.com/api/v1?tickers={ticker}&items=50&page={str(page)}&token=h1gxccwbm4lyfsdwqxe7uyjczdgm09fvygtvlk0g")
        df = pd.DataFrame(request.json()['data'])
        df["date"] = pd.to_datetime(pd.DataFrame(request.json()['data']).date)

        return df

    def get_all_news_database(self, ticker='BTC'):
        """
        Get all history news dataframe from the desired ticker
        
        Params
        --------
        ticker: string
            initials crypto ticker

        Notes
        --------
        Limit off 200 pages on crypto news api.
        
        """
        n = 0
        data_page = 1
        dataframes_list = []
        while n==0:
            _df = self._request_news("BTC",data_page)
            date_1 = _df["date"].min().date()
            date_2 = self.min_date.date()
            if  date_1 > date_2 :
                dataframes_list.append(_df)
                if data_page>=10:
                    n = n + 1
                else:
                    data_page = data_page + 1
            else:
                n = n + 1 
        
        df = pd.concat(dataframes_list)

        df["data_final"] = [pd.to_datetime(elemento_data).date() for elemento_data in df.date] #transform into datetime
        print("News Dataframe completed.")
        return df

    def get_resume_news(self, new_df):
        """
        Agregate all news by date, type and sentiment.

        Params
        --------
        new_df: pandas.DataFrame
            news dataframe

        Returns
        --------
        df: pandas.DataFrame

        Notes
        --------
        Necessary to have the following collumns:
            -  data_final,type,sentiment,news_url.
        
        """

        df = new_df.groupby(["data_final","type","sentiment"])[['news_url']].count().reset_index()
        df = df.rename({"news_url":"values"},axis=1)
        print("Dataframe resume news completed.")

        return df

    def get_converted_dataframe(self,df_history,df_resume_news):
        """
        Creates new columns into history dataframe, this columns will be the sentiment analysis of the data. 

        Params
        --------
        df_history: pandas.DataFrame
            history cripto dataframe

        df_resume_news: pandas.DataFrame
            resume news dataframe

        Returns
        --------
        df: pandas.DataFrame

        """
        article_negative_list = []
        article_neutral_list = []
        article_positive_list = []
        video_negative_list = []
        video_neutral_list = []
        video_positive_list = []


        for tempo in df_history.data_final:
            data_filter = tempo-timedelta(days=1)
            data_filter_2 = tempo-timedelta(days=2)
            a = df_resume_news[(df_resume_news["data_final"]<=data_filter)&(df_resume_news["data_final"]>=data_filter_2)]
            try:
                article_negative = a[(a["type"]=="Article")&(a["sentiment"]=="Negative")].reset_index(drop=True)["values"][0]
            except Exception:
                article_negative = 0
            try:
                article_neutral = a[(a["type"]=="Article")&(a["sentiment"]=="Neutral")].reset_index(drop=True)["values"][0]
            except Exception:
                article_neutral = 0
            try:
                article_positive = a[(a["type"]=="Article")&(a["sentiment"]=="Positive")].reset_index(drop=True)["values"][0]
            except Exception:
                article_positive = 0 
            try:
                video_negative = a[(a["type"]=="Video")&(a["sentiment"]=="Negative")].reset_index(drop=True)["values"][0]
            except Exception:
                video_negative = 0 
            try:
                video_neutral = a[(a["type"]=="Video")&(a["sentiment"]=="Neutral")].reset_index(drop=True)["values"][0]
            except Exception:
                video_neutral = 0
            try:
                video_positive = a[(a["type"]=="Video")&(a["sentiment"]=="Positive")].reset_index(drop=True)["values"][0]
            except Exception:
                video_positive = 0
            
            article_negative_list.append(article_negative)
            article_neutral_list.append(article_neutral)
            article_positive_list.append(article_positive)
            video_negative_list.append(video_negative)
            video_neutral_list.append(video_neutral)
            video_positive_list.append(video_positive)
        
        df_history_final = df_history.copy()

        df_history_final["article_negative"] = pd.DataFrame(article_negative_list)
        df_history_final["article_neutral"] = pd.DataFrame(article_neutral_list)
        df_history_final["article_positive"] = pd.DataFrame(article_positive_list)
        df_history_final["video_negative"] = pd.DataFrame(video_negative_list)
        df_history_final["video_neutral"] = pd.DataFrame(video_neutral_list)
        df_history_final["video_positive"] = pd.DataFrame(video_positive_list)

        df_history_final["fechamento"] = df_history_final["close"]/df_history_final["open"]-1
        df_history_final["fechamento_binario"] = [1 if i>0 else 0 for i in df_history_final["fechamento"]]
        print("Dataframe with news adapted.")

        return df_history_final

    def __call__(self, ticker):
        """
        Execute all other functions created.

        """
        df_history = self.get_history_df(ticker)
        df_news = self.get_all_news_database(ticker)
        df_resume_news = self.get_resume_news(df_news)
        df = self.get_converted_dataframe(df_history,df_resume_news)

        return df



