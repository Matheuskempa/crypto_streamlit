import requests
import pandas as pd
from datetime import datetime, timedelta
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score,precision_score,recall_score


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

    def get_all_news_database(self,ticker='BTC'):
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
                if data_page>=190:
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

    def __call__(self,ticker):
        """
        Execute all other functions created.

        Params
        --------
        df_history: pandas.DataFrame
            history cripto dataframe
        
        Returns        
        --------
        df: pandas.DataFrame

        """
        df_history = self.get_history_df(ticker)
        df_news = self.get_all_news_database(ticker)
        df_resume_news = self.get_resume_news(df_news)
        df = self.get_converted_dataframe(df_history,df_resume_news)

        return df



class CriptoProjectAnalysis():
    
    def __init__(self):
        print("Started CriptoProjectAnalysis.")

    def get_real_analysed_dataframe(self,df):
        """
        Filter the first valid date and filter pandas dataframe.

        Params
        --------
        df: pandas.DataFrame
            history cripto dataframe
        
        Returns        
        --------
        df_final: pandas.DataFrame        

        """

        data = df[(df["article_negative"]>0)&
                    (df["article_neutral"]>0)&
                    (df["article_positive"]>0)&
                    (df["video_negative"]>0)&
                    (df["video_neutral"]>0)&
                    (df["video_positive"]>0)].reset_index(drop=True)["data_final"][0]

        df_final = df[df["data_final"]>=pd.to_datetime(str(data)).date()].reset_index(drop=True)
        return df_final
    
    def get_analysed_time_series_graph(self,df):
        """
        Generates the time series graph.

        Params
        --------
        df: pandas.DataFrame
        
        Returns        
        --------
        fig: graph        

        """
        fig = px.line(df.reset_index(), x="data_final", y="close")

        return fig.show()

    def get_variation_graph(self,df,coluna='fechamento_binario',valores="fechamento"):
        """
        Generates the time series graph.

        Params
        --------
        df: pandas.DataFrame

        coluna: string
            name of de column coloured

        valores: string 
            name of the collumn value
        
        Returns        
        --------
        fig: graph        

        """

        fig = px.bar(df, x=df.data_final, y=valores, color=coluna)
        fig.update_yaxes(ticklabelposition="inside top", title=None)
        return fig

    def get_relational_graph(self,df,category='article_negative'):
        """
        Generates the time series graph.

        Params
        --------
        df: pandas.DataFrame
        
        category: string
            wich variable analyse
        
        Returns        
        --------
        fig: graph        

        """

        fig = px.line(df, x="data_final", y="close")
        fig_2 = px.line(df, x="data_final", y=category)
        fig_2.update_traces(yaxis="y2")
        subfig = make_subplots(specs=[[{"secondary_y": True}]])
        subfig.add_traces(fig.data + fig_2.data)
        subfig.for_each_trace(lambda t: t.update(line=dict(color=t.marker.color)))
        # fig_2.update_layout(height=800, width=1000, title_text="Time Series")
        # subfig.show()
        return subfig

    def _make_predictions_random_forest(self,df,quantidade):
        """
        Generates the predictions with random forest.

        Params
        --------
        df: pandas.DataFrame

        quantidade: int
        
        Returns        
        --------
        df_final: pandas.DataFrame
        

        """
        base_x = df[['open', 'volumefrom','article_negative', 'article_neutral', 'article_positive',
                          'video_negative', 'video_neutral', 'video_positive']]
        base_y = df[['fechamento_binario']]
        
        clf = RandomForestClassifier(n_estimators=95,max_depth=2, random_state=0)

        real_list = []
        prediction_list = []

        for i in range(quantidade,len(base_x)):
            
            base_x_new = base_x[:i]
            base_y_new = base_y[:i]
            base_x_test = base_x[i:i+1]
            base_y_test = base_y[i:i+1]
            
            
            clf.fit(base_x_new.values, base_y_new.fechamento_binario.ravel())
            
            try:
                valor_predito = clf.predict(base_x_test)[0]
                prediction_list.append(valor_predito)
                real_list.append(base_y_test.values[0][0])
            except:
                break
        df_final = pd.DataFrame(zip(df.fechamento[quantidade:].reset_index(drop=True).to_list(),prediction_list,real_list),columns=["var","predito","real"])
        return df_final
    
    def _make_predictions_deep_learning(self,df,quantidade):
        """
        Generates the predictions with random forest.

        Params
        --------
        df: pandas.DataFrame

        quantidade: int
        
        Returns        
        --------
        df_final: pandas.DataFrame
          
        """
        base_x = df[['open', 'volumefrom','article_negative', 'article_neutral', 'article_positive',
                          'video_negative', 'video_neutral', 'video_positive']]
        base_y = df[['fechamento_binario']]
 
        model = Sequential() 
        model.add(Dense(50, activation='relu', input_dim=8))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
        model.summary()
        
        real_list = []
        prediction_list = []

        for i in range(quantidade,len(base_x)):
            
            base_x_new = base_x[:i]
            base_y_new = base_y[:i]
            base_x_test = base_x[i:i+1]
            base_y_test = base_y[i:i+1]
            
            model.fit(base_x_new, base_y_new, epochs=50, batch_size=30)
            
            try:
                valor_predito = model.predict(base_x)[0]
                print(valor_predito)
                print(base_y_test.values[0][0])
                prediction_list.append(valor_predito)
                real_list.append(base_y_test.values[0][0])
            except:
                break
        
        y_pred = []
        for i in prediction_list:
            y_pred.append(round(i[0],0))
        df_final = pd.DataFrame(zip(df.fechamento[quantidade:].reset_index(drop=True).to_list(),y_pred,real_list),columns=["var","predito","real"])
        
        return df_final

    def _get_metrics(self,df):
        """
        Get all Metrics of the predictions.

        Params
        --------
        df: pandas.DataFrame
        
        Returns        
        --------
        accuracy: float
        prediction: float
        recall: float        
        
        """
        "predito","real"
        acc = accuracy_score(df.real.to_list(), df.predito.to_list())
        precision = precision_score(df.real.to_list(), df.predito.to_list())
        recall = recall_score(df.real.to_list(), df.predito.to_list())
        return acc, precision, recall

    def __call__(self,df,type,percentual):
        """
        Execute all other functions created.

        Params
        --------
        df: pandas.DataFrame
            history cripto dataframe
        
        type: string
            Deep learning or Random Forest 

        percentual: float
        
        Returns        
        --------
        df: pandas.DataFrame
        
        """
        df_ = self.get_real_analysed_dataframe(df)

        qtnd = round(len(df_)*percentual)

        datas = df_[(qtnd+1):].reset_index()["data_final"].to_list()

        if type == "Deep Learning":

            self.get_analysed_time_series_graph(df_)

            self.get_relational_graph(df_)            
            
            self.get_variation_graph(df_)

            base = self._make_predictions_deep_learning(df_, qtnd)

            base["data_final"] = pd.DataFrame(datas)


        else:
            base = self._make_predictions_random_forest(df_, qtnd)

        a,p,r = self._get_metrics(base)
        
        return base,a,p,r
