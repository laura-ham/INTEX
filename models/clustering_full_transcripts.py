from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import NMF, non_negative_factorization
from matplotlib.ticker import AutoMinorLocator
from pyLDAvis import sklearn as sklearn_lda
import more_itertools
import pickle
import pyLDAvis
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import datetime
from sklearn.manifold import TSNE
import umap
import plotly.express as px
import gensim
import re
from itertools import combinations
import time
from functools import reduce
import warnings

# required libraries for bokeh plots
from callbacks_bokeh import input_callback, selected_code_videos, selected_code_articles  # file with customJS callbacks for bokeh
                                                      # github.com/MaksimEkin/COVID19-Literature-Clustering/blob/master/lib/call_backs.py
import bokeh
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, CustomJS, Slider, TapTool, TextInput
from bokeh.palettes import Category20
from bokeh.transform import linear_cmap, transform
from bokeh.io import output_file, show, output_notebook
from bokeh.plotting import figure
from bokeh.models import RadioButtonGroup, TextInput, Div, Paragraph
from bokeh.layouts import column, widgetbox, row, layout
from bokeh.layouts import column

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

sns.set_style('whitegrid')
matplotlib.use('agg')

sns.set()
sns.set_context("notebook",
    font_scale=1.25,
    rc={"lines.linewidth": 3})
sns.set_palette('husl')

DEFAULT_STOPWORDS = ["aiemmin","aika","aikaa","aikaan","aikaisemmin","aikaisin","aikajen","aikana","aikoina","aikoo","aikovat","aina","ainakaan","ainakin","ainoa","ainoat","aiomme","aion","aiotte","aist","aivan","ajan","alas","alemmas","alkuisin","alkuun","alla","alle","aloitamme","aloitan","aloitat","aloitatte","aloitattivat","aloitettava","aloitettevaksi","aloitettu","aloitimme","aloitin","aloitit","aloititte","aloittaa","aloittamatta","aloitti","aloittivat","alta","aluksi","alussa","alusta","annettavaksi","annetteva","annettu","ansiosta","antaa","antamatta","antoi","aoua","apu","asia","asiaa","asian","asiasta","asiat","asioiden","asioihin","asioita","asti","avuksi","avulla","avun","avutta","edelle","edelleen","edellä","edeltä","edemmäs","edes","edessä","edestä","ehkä","ei","eikä","eilen","eivät","eli","ellei","elleivät","ellemme","ellen","ellet","ellette","emme","en","enemmän","eniten","ennen","ensi","ensimmäinen","ensimmäiseksi","ensimmäisen","ensimmäisenä","ensimmäiset","ensimmäisiksi","ensimmäisinä","ensimmäisiä","ensimmäistä","ensin","entinen","entisen","entisiä","entisten","entistä","enää","eri","erittäin","erityisesti","eräiden","eräs","eräät","esi","esiin","esillä","esimerkiksi","et","eteen","etenkin","etessa","ette","ettei","että","haikki","halua","haluaa","haluamatta","haluamme","haluan","haluat","haluatte","haluavat","halunnut","halusi","halusimme","halusin","halusit","halusitte","halusivat","halutessa","haluton","he","hei","heidän","heidät","heihin","heille","heillä","heiltä","heissä","heistä","heitä","helposti","heti","hetkellä","hieman","hitaasti","hoikein","huolimatta","huomenna","hyvien","hyviin","hyviksi","hyville","hyviltä","hyvin","hyvinä","hyvissä","hyvistä","hyviä","hyvä","hyvät","hyvää","hän","häneen","hänelle","hänellä","häneltä","hänen","hänessä","hänestä","hänet","häntä","ihan","ilman","ilmeisesti","itse","itsensä","itseään","ja","jo","johon","joiden","joihin","joiksi","joilla","joille","joilta","joina","joissa","joista","joita","joka","jokainen","jokin","joko","joksi","joku","jolla","jolle","jolloin","jolta","jompikumpi","jona","jonka","jonkin","jonne","joo","jopa","jos","joskus","jossa","josta","jota","jotain","joten","jotenkin","jotenkuten","jotka","jotta","jouduimme","jouduin","jouduit","jouduitte","joudumme","joudun","joudutte","joukkoon","joukossa","joukosta","joutua","joutui","joutuivat","joutumaan","joutuu","joutuvat","juuri","jälkeen","jälleen","jää","kahdeksan","kahdeksannen","kahdella","kahdelle","kahdelta","kahden","kahdessa","kahdesta","kahta","kahteen","kai","kaiken","kaikille","kaikilta","kaikkea","kaikki","kaikkia","kaikkiaan","kaikkialla","kaikkialle","kaikkialta","kaikkien","kaikkin","kaksi","kannalta","kannattaa","kanssa","kanssaan","kanssamme","kanssani","kanssanne","kanssasi","kauan","kauemmas","kaukana","kautta","kehen","keiden","keihin","keiksi","keille","keillä","keiltä","keinä","keissä","keistä","keitten","keittä","keitä","keneen","keneksi","kenelle","kenellä","keneltä","kenen","kenenä","kenessä","kenestä","kenet","kenettä","kennessästä","kenties","kerran","kerta","kertaa","keskellä","kesken","keskimäärin","ketkä","ketä","kiitos","kohti","koko","kokonaan","kolmas","kolme","kolmen","kolmesti","koska","koskaan","kovin","kuin","kuinka","kuinkan","kuitenkaan","kuitenkin","kuka","kukaan","kukin","kukka","kumpainen","kumpainenkaan","kumpi","kumpikaan","kumpikin","kun","kuten","kuuden","kuusi","kuutta","kylliksi","kyllä","kymmenen","kyse","liian","liki","lisäksi","lisää","lla","luo","luona","lähekkäin","lähelle","lähellä","läheltä","lähemmäs","lähes","lähinnä","lähtien","läpi","mahdollisimman","mahdollista","me","meidän","meidät","meihin","meille","meillä","meiltä","meissä","meistä","meitä","melkein","melko","menee","meneet","menemme","menen","menet","menette","menevät","meni","menimme","menin","menit","menivät","mennessä","mennyt","menossa","mihin","mikin","miksi","mikä","mikäli","mikään","mille","milloin","milloinkan","millä","miltä","minkä","minne","minua","minulla","minulle","minulta","minun","minussa","minusta","minut","minuun","minä","missä","mistä","miten","mitkä","mitä","mitään","moi","molemmat","mones","monesti","monet","moni","moniaalla","moniaalle","moniaalta","monta","muassa","muiden","muita","muka","mukaan","mukaansa","mukana","mutta","muu","muualla","muualle","muualta","muuanne","muulloin","muun","muut","muuta","muutama","muutaman","muuten","myöhemmin","myös","myöskin","myöskään","myötä","ne","neljä","neljän","neljää","niiden","niihin","niiksi","niille","niillä","niiltä","niin","niinä","niissä","niistä","niitä","noiden","noihin","noiksi","noilla","noille","noilta","noin","noina","noissa","noista","noita","nopeammin","nopeasti","nopeiten","nro","nuo","nyt","näiden","näihin","näiksi","näille","näillä","näiltä","näin","näinä","näissä","näissähin","näissälle","näissältä","näissästä","näistä","näitä","nämä","ohi","oikea","oikealla","oikein","ole","olemme","olen","olet","olette","oleva","olevan","olevat","oli","olimme","olin","olisi","olisimme","olisin","olisit","olisitte","olisivat","olit","olitte","olivat","olla","olleet","olli","ollut","oma","omaa","omaan","omaksi","omalle","omalta","oman","omassa","omat","omia","omien","omiin","omiksi","omille","omilta","omissa","omista","on","onkin","onko","ovat","paikoittain","paitsi","pakosti","paljon","paremmin","parempi","parhaillaan","parhaiten","perusteella","peräti","pian","pieneen","pieneksi","pienelle","pienellä","pieneltä","pienempi","pienestä","pieni","pienin","poikki","puolesta","puolestaan","päälle","runsaasti","saakka","sadam","sama","samaa","samaan","samalla","samallalta","samallassa","samallasta","saman","samat","samoin","sata","sataa","satojen","se","seitsemän","sekä","sen","seuraavat","siellä","sieltä","siihen","siinä","siis","siitä","sijaan","siksi","sille","silloin","sillä","silti","siltä","sinne","sinua","sinulla","sinulle","sinulta","sinun","sinussa","sinusta","sinut","sinuun","sinä","sisäkkäin","sisällä","siten","sitten","sitä","ssa","sta","suoraan","suuntaan","suuren","suuret","suuri","suuria","suurin","suurten","taa","taas","taemmas","tahansa","tai","takaa","takaisin","takana","takia","tallä","tapauksessa","tarpeeksi","tavalla","tavoitteena","te","teidän","teidät","teihin","teille","teillä","teiltä","teissä","teistä","teitä","tietysti","todella","toinen","toisaalla","toisaalle","toisaalta","toiseen","toiseksi","toisella","toiselle","toiselta","toisemme","toisen","toisensa","toisessa","toisesta","toista","toistaiseksi","toki","tosin","tuhannen","tuhat","tule","tulee","tulemme","tulen","tulet","tulette","tulevat","tulimme","tulin","tulisi","tulisimme","tulisin","tulisit","tulisitte","tulisivat","tulit","tulitte","tulivat","tulla","tulleet","tullut","tuntuu","tuo","tuohon","tuoksi","tuolla","tuolle","tuolloin","tuolta","tuon","tuona","tuonne","tuossa","tuosta","tuota","tuotä","tuskin","tykö","tähän","täksi","tälle","tällä","tällöin","tältä","tämä","tämän","tänne","tänä","tänään","tässä","tästä","täten","tätä","täysin","täytyvät","täytyy","täällä","täältä","ulkopuolella","usea","useasti","useimmiten","usein","useita","uudeksi","uudelleen","uuden","uudet","uusi","uusia","uusien","uusinta","uuteen","uutta","vaan","vahemmän","vai","vaiheessa","vaikea","vaikean","vaikeat","vaikeilla","vaikeille","vaikeilta","vaikeissa","vaikeista","vaikka","vain","varmasti","varsin","varsinkin","varten","vasen","vasenmalla","vasta","vastaan","vastakkain","vastan","verran","vielä","vierekkäin","vieressä","vieri","viiden","viime","viimeinen","viimeisen","viimeksi","viisi","voi","voidaan","voimme","voin","voisi","voit","voitte","voivat","vuoden","vuoksi","vuosi","vuosien","vuosina","vuotta","vähemmän","vähintään","vähiten","vähän","välillä","yhdeksän","yhden","yhdessä","yhteen","yhteensä","yhteydessä","yhteyteen","yhtä","yhtäälle","yhtäällä","yhtäältä","yhtään","yhä","yksi","yksin","yksittäin","yleensä","ylemmäs","yli","ylös","ympäri","älköön","älä"]
regex1 = '[a-zA-Z\u00c0-\u017e]{3,30}'

class Pipeline:
    def __init__(self,
                 dataframe=None,
                 additional_stopwords=[],
                 min_df=0.0005,
                 max_df=1.0):
        self.original_df = dataframe
        self.additional_stopwords = DEFAULT_STOPWORDS
        self.additional_stopwords.extend(additional_stopwords)
        self.set_targets(dataframe)
        self.config = {
            "countvectorizer": {
                "max_df": max_df,
                "min_df": min_df,
                "max_features": 10000
            },
            "tfidf": {
                "max_df": max_df,
                "min_df": min_df,
                "max_features": 10000,
                "sublinear_tf": True,
                "ngram_range": (1,3),
                "token_pattern": regex1
                # "vocabulary": self.list_of_strings
            },
            "nmf": {
                "init": "nndsvd",
                "alpha": 0,
                "random_state": 2018
            },
        }

    def fit_transform(self, topics=8):
        self.nr_of_topics = topics
        self.vectorizer = TfidfVectorizer(**self.config["tfidf"],
                                          stop_words=self.additional_stopwords)
        self.vectorized_out = self.vectorizer.fit_transform(self.target)
        self.features = self.vectorizer.get_feature_names()
        self.nmf_model = NMF(**self.config["nmf"], n_components=topics)
        self.nmf_matrix = self.nmf_model.fit_transform(self.vectorized_out)
        self.nmf_components = self.nmf_model.components_

        self.doc_topic_dists = self.nmf_matrix / self.nmf_matrix.sum(
            axis=1)[:, None]
        self.doc_topic_dists = np.nan_to_num(self.doc_topic_dists, nan=1/self.nr_of_topics)
        self.top_words_map = self._top_words_map()
        self.doc_topic_matrix_df = self._doc_topic_matrix_df()

        return

    def get_residuals(self):
        A = self.vectorized_out.copy()
        H = self.nmf_matrix.copy()
        W = self.nmf_components.copy()

        # Get the residuals for each document
        r = np.zeros(A.shape[0])
        for row in range(A.shape[0]):
            r[row] = np.linalg.norm(A[row, :] - H[row, :].dot(W), 'fro')
        # Add the residuals to the df
        self.doc_topic_matrix_df_resid = self.doc_topic_matrix_df.copy()
        self.doc_topic_matrix_df_resid['resid'] = r

        return sum(np.sqrt(r)), self.nmf_model.reconstruction_err_, self.doc_topic_matrix_df_resid

    def get_residuals_per_topic(self):

        res_top_df = self.doc_topic_matrix_df_resid.copy()

        res_top_df = res_top_df.reset_index()
        res_top_df = res_top_df.melt(id_vars=["id", "date", "title", "resid"],
                                         var_name="cluster",
                                         value_name="value")

        res_top_df['relative_resid'] = res_top_df['resid'] * res_top_df['value']
        
        res_top_df.reset_index()
        res_top_df.drop(['id','date','title','resid','value'], axis=1)
        res_top_df = res_top_df.groupby('cluster')['relative_resid'].apply(lambda x: (x>0.0001).mean()).reset_index()
        res_top_df.columns = ['cluster', 'relative_resid']
        res_top_df = res_top_df.sort_values(by='relative_resid', ascending=False)
        res_top_df = res_top_df.round(2)
        #res_top_df['cluster'] = res_top_df['cluster'].apply(lambda t: t.split(':')[0]) # this leads to descending misbehaviour.. 

        # x = topic
        # y = average residual
        
        fig = px.bar(res_top_df, x="cluster", y="relative_resid", text='relative_resid', title='Average residual per topic')
        fig.update_layout(xaxis_title='Topic', yaxis_title='Average residual')
        fig.update_xaxes(categoryorder='total descending')
        fig.update_layout(
            xaxis = dict(
                tickmode = 'linear',
                tickfont = dict(
                    size = 10
                )
            ),
            yaxis = dict(
                tickfont = dict(
                    size = 10
                )
            )
        )
        fig.update_traces(marker_color='rgb(0, 180, 200)', textposition='inside', textfont_size=8)

        return fig

    def fit_transform_merge(self, feature_1, feature_2):
        W = np.copy(self.nmf_matrix)
        H = np.copy(self.nmf_components)

        # merge (addition) column values of W
        W[:, feature_1] = W[:, feature_1] + W[:, feature_2]
        self.W = np.delete(W, feature_2, 1)

        # merge (addition) row values of H
        H[feature_1, :] = H[feature_1, :] + H[feature_2, :]
        self.H = np.delete(H, feature_2, 0)

        # W becomes the new nmf_matrix
        # H becomes the new components

        self.nmf_matrix = np.copy(self.W)
        self.nmf_components = np.copy(self.H)
        self.nr_of_topics -= 1

        self.doc_topic_dists = self.nmf_matrix / self.nmf_matrix.sum(
            axis=1)[:, None]
        self.doc_topic_dists = np.nan_to_num(self.doc_topic_dists, nan=1/self.nr_of_topics)
        self.top_words_map = self._top_words_map()
        self.doc_topic_matrix_df = self._doc_topic_matrix_df()

    def fit_transform_split(self, topics, fixed_H, column):
        self.nr_of_topics = topics

        repeats = np.ones(len(self.nmf_matrix.T), dtype=int)
        repeats[column] = int(2)
        W = np.repeat(self.nmf_matrix.T, repeats, axis=0)
        self.W = W.T

        self.W = np.ascontiguousarray(self.W, dtype=np.float64)

        # self.W, self.H, n_iter = non_negative_factorization(self.vectorized_out, n_components=topics, init='custom', random_state=0, update_H=True, H=fixed_H, W=self.W)
        self.W, self.H, n_iter = non_negative_factorization(self.vectorized_out, n_components=topics, init='custom', random_state=0, update_H=False, H=fixed_H)

        self.nmf_matrix = np.copy(self.W)
        self.nmf_components = np.copy(self.H)

        self.doc_topic_dists = self.nmf_matrix / self.nmf_matrix.sum(
            axis=1)[:, None]
        self.doc_topic_dists = np.nan_to_num(self.doc_topic_dists, nan=1/self.nr_of_topics)
        self.top_words_map = self._top_words_map()
        self.doc_topic_matrix_df = self._doc_topic_matrix_df() 

        return
    
    def delete_word_from_topic(self, topic_to_delete_from, word_to_delete, top_words_in_topic):
        self.W = np.copy(self.nmf_matrix)
        H = np.copy(self.nmf_components)

        index_of_word_to_remove = self.features.index(word_to_delete.replace(' ', '_'))

        H[topic_to_delete_from][index_of_word_to_remove] = 0

        self.W, self.H, n_iter = non_negative_factorization(self.vectorized_out, n_components=self.nr_of_topics, init='custom', random_state=0, update_H=False, H=H)

        self.nmf_matrix = np.copy(self.W)
        self.nmf_components = np.copy(self.H)

        self.doc_topic_dists = self.nmf_matrix / self.nmf_matrix.sum(
            axis=1)[:, None]
        self.doc_topic_dists = np.nan_to_num(self.doc_topic_dists, nan=1/self.nr_of_topics)
        self.top_words_map = self._top_words_map()
        self.doc_topic_matrix_df = self._doc_topic_matrix_df()

    def rename_topic(self, topic, new_name):
        new_dict = {}
        for k, v in self.top_words_map.items():
            if k.split(':')[0] == str(topic):
                new = k.split(':')[0] + ': ' + new_name
                new_dict[new] = v
                continue
            new_dict[k] = v
        self.top_words_map = new_dict
        self.cluster_names = list(self.top_words_map.keys())
        self.doc_topic_matrix_df = self._doc_topic_matrix_df()

    def get_most_common_words(self, df, nr_of_words=10):
        stopwords = self.additional_stopwords
        self.count_vectorizer = CountVectorizer(**self.config["countvectorizer"],
            stop_words=stopwords,
            ngram_range=(1, 3))  # , vocabulary=self.list_of_strings)

        self.count_data = self.count_vectorizer.fit_transform(df['text'])
        words = self.count_vectorizer.get_feature_names()
        total_counts = np.zeros(len(words))
        for t in self.count_data:
            total_counts += t.toarray()[0]

        count_dict = (zip(words, total_counts))
        count_dict = sorted(count_dict, key=lambda x: x[1],
                            reverse=True)[0:nr_of_words]
        words = [w[0] for w in count_dict]
        return words, count_dict

    def plotly_visualize_most_common_words(self, df, nr_of_words, words, count_dict):
        if len(words) < 15:
            words, count_dict = self.get_most_common_words(df, nr_of_words)
        else:
            words = words[:15]
            count_dict = count_dict[:15]

        counts = [w[1] for w in count_dict]

        d = {'Word': words, 'Count': counts}
        df_show = pd.DataFrame(data=d)

        fig = px.bar(df_show, x='Word', y='Count', text='Count', title='Most common words')
        fig.update_layout(xaxis_title='Word', yaxis_title='Frequency')
        fig.update_traces(marker_color='rgb(0, 180, 200)')
        fig.update_traces(textposition='inside', textfont_size=10)
        fig.update_layout(
            xaxis = dict(
                tickfont = dict(
                    size = 10
                )
            ),
            yaxis = dict(
                tickfont = dict(
                    size = 10
                )
            )
        )

        return fig

        # x_pos = np.arange(len(words))
        # plt.figure(1, figsize=(15, 15 / 1.6180)) 
        # sns.barplot(x_pos, counts)
        # plt.xticks(x_pos, words, rotation=45)
        # plt.xlabel('Word', weight='bold', fontsize=20)
        # plt.ylabel('Count', weight='bold', fontsize=20)
        # plt.title('{} Most common words'.format(nr_of_words), weight='bold', fontsize=25)
        # plt.tight_layout()
        # plt.show()
        # return plt

    def dataframe_top_words(self, nr_of_words):
        top_words = {k: v.split(', ') for k, v in self.top_words_map.items()}
        top_words = {
            k: [word.replace("_", " ") for word in v]
            for k, v in top_words.items()
        }  # to get spaces back instead of underscores, for visualization purposes only
        df_top_words = pd.DataFrame.from_dict(top_words)
        return df_top_words[:nr_of_words]

    def generate_ldavis(self):        
        params = {"mds": "pcoa"}
        try:
            LDAvis_prepared = sklearn_lda.prepare(self.nmf_model,
                                                self.vectorized_out,
                                                self.vectorizer, **params)
        except:
            return "This visualization is currently not available."
        return LDAvis_prepared

    def set_targets(self, df):
        self.target = df["text"]

    def _top_words_map(self):
        words_map = {}
        self.cluster_names = []
        for cluster, comp_content_zip in enumerate(
                self.nmf_components):
            top_words = ", ".join([
                self.features[i] for i in comp_content_zip.argsort()[:-50 - 1:-1]
            ])
            key = str(cluster) + ': ' + top_words.split(
                ',')[0] + ',' + top_words.split(',')[1] + ',' + top_words.split(',')[2]
            words_map.update({key: top_words})
            self.cluster_names.append(key)
        return words_map

    def _doc_topic_matrix_df(self):
        doc_topic_matrix_df = pd.DataFrame(self.nmf_matrix)
        doc_topic_matrix_df = pd.DataFrame(
            self.doc_topic_dists)  # for sum up to 1
        doc_topic_matrix_df["id"] = [x for x in list(self.original_df.index)]
        doc_topic_matrix_df["title"] = [
            x for x in list(self.original_df.title)
        ]
        doc_topic_matrix_df["date"] = [
            x for x in list(self.original_df.published_time)
        ]

        # change order of columns for visualization purposes
        doc_topic_matrix_df = doc_topic_matrix_df.set_index("id")
        cols = list(doc_topic_matrix_df.columns.values)
        cols = cols[-1:] + cols[:-1]
        cols = cols[0:1] + cols[-1:] + cols[1:-1]
        df = doc_topic_matrix_df[cols]
        cols_with_names = cols[:2] + self.cluster_names
        df.columns = cols[:2] + self.cluster_names
        return df

    def document_comparison_plot(self, article_id_list):
        df = self.doc_topic_matrix_df.copy()
        df["title"] = df["title"].apply(
            lambda x: x.replace('\n', ' ').replace('\r', ''))
        df["text"] = [x for x in list(self.original_df.text)]
        df = df.reset_index()
        df = df.melt(
            id_vars=["id", "date", "title", "text"],
            var_name="Cluster",
            value_name="value")
        df = df.round(3)

        sub_df = df.loc[df['id'].isin(article_id_list)]

        fig = px.bar(sub_df, x="value", y="id", color='Cluster',
                    hover_data=["title", "Cluster", "value"],
                    height=400,
                    text='value',
                    title='NMF scores per content item', orientation='h').for_each_trace(lambda t: t.update(name=t.name.split(':')[0]))
        fig.update_layout(legend=dict(font_size=10), yaxis_title='Content ID', xaxis_title='NMF cluster value')
        fig.update_traces(textposition='inside', textfont_size=8)
        fig.update_layout(
            xaxis = dict(
                tickfont = dict(
                    size = 10
                )
            ),
            yaxis = dict(
                tickfont = dict(
                    size = 8
                )
            )
        )
        return fig

    def df_to_export_to_csv(self):
        df_to_return = self.doc_topic_matrix_df.copy()
        df_to_return["title"] = df_to_return["title"].apply(
            lambda x: x.replace('\n', ' ').replace('\r', ''))
        df_to_return = df_to_return.reset_index()
        df_to_return = df_to_return.melt(id_vars=["id", "date", "title"],
                                         var_name="cluster",
                                         value_name="value")
        df_to_return = df_to_return.set_index(['id', 'cluster'])

        return df_to_return

    def clusters_over_time_plots(self, start, end):
        doc_topic_matrix_df = self.doc_topic_matrix_df.copy() # for absolute production (sum up to 1 per article)
        doc_topic_matrix_df = self.df_to_export_to_csv()
        doc_topic_matrix_df = doc_topic_matrix_df.reset_index().set_index(pd.DatetimeIndex(doc_topic_matrix_df['date'])).groupby(['cluster',pd.Grouper(freq='MS')])['value'].sum().reset_index(name='value')

        fig_line = px.line(doc_topic_matrix_df, x="date", y="value", color='cluster', title='Content production over time').for_each_trace(lambda t: t.update(name=t.name.split(':')[0]))
        fig_line.update_traces(mode='lines+markers')
        fig_line.update_layout(legend=dict(font_size=10), xaxis_title='Month', yaxis_title='Production')
        fig_line.update_layout(
            xaxis = dict(
                tickfont = dict(
                    size = 10
                )
            ),
            yaxis = dict(
                tickfont = dict(
                    size = 10
                )
            )
        )

        fig_bar = px.bar(doc_topic_matrix_df, x="date", y="value", color='cluster', title='Content production over time (stacked bar chart)').for_each_trace(lambda t: t.update(name=t.name.split(':')[0]))
        fig_bar.update_layout(barmode='stack')
        fig_bar.update_layout(legend=dict(font_size=10), xaxis_title='Month', yaxis_title='Production')
        fig_bar.update_layout(
            xaxis = dict(
                tickfont = dict(
                    size = 10
                )
            ),
            yaxis = dict(
                tickfont = dict(
                    size = 10
                )
            )
        )

        return fig_line, fig_bar

        doc_topic_matrix_df = self.doc_topic_matrix_df.copy() # for absolute production (sum up to 1 per article)
        doc_topic_matrix_df = self.df_to_export_to_csv()
        doc_topic_matrix_df = doc_topic_matrix_df.reset_index().set_index(pd.DatetimeIndex(doc_topic_matrix_df['date'])).groupby(['cluster',pd.Grouper(freq='MS')])['value'].sum().reset_index(name='value')

        fig = px.bar(doc_topic_matrix_df, x="date", y="value", color='cluster', title='Content production over time (stacked bar chart)').for_each_trace(lambda t: t.update(name=t.name.split(':')[0]))
        #fig.update_traces(mode='lines+markers')
        fig.update_layout(barmode='stack')
        fig.update_layout(legend=dict(font_size=10), xaxis_title='Month', yaxis_title='Production')
        fig.update_layout(
            xaxis = dict(
                tickfont = dict(
                    size = 10
                )
            ),
            yaxis = dict(
                tickfont = dict(
                    size = 10
                )
            )
        )
        return fig

    def get_production_graph(self, consumption_df):
        doc_topic_matrix_df = self.doc_topic_matrix_df.copy() # for absolute production (sum up to 1 per article)
        consumption_df.index.name = 'id'

        #consumption_df = consumption_df.drop(['total_views', 'gender'], axis=1)
        consumption_df = consumption_df.groupby(['id', 'age_group']).sum().reset_index().set_index('id')

        df_0_44 = consumption_df[consumption_df['age_group'] == "0-44 (excl 15-29)"].drop(['age_group'], axis=1)
        df_15_29 = consumption_df[consumption_df['age_group'] == "15-29"].drop(['age_group'], axis=1)
        df_45 = consumption_df[consumption_df['age_group'] == "45+"].drop(['age_group'], axis=1)

        df_0_44 = df_0_44.merge(doc_topic_matrix_df, left_index=True, right_index=True)
        df_15_29 = df_15_29.merge(doc_topic_matrix_df, left_index=True, right_index=True)
        df_45 = df_45.merge(doc_topic_matrix_df, left_index=True, right_index=True)

        dfs = {'0-44': df_0_44, '15-29': df_15_29, '45+': df_45}
        sums_consumption_dfs = []
        for name, df in dfs.items():
            df = df.drop(['title', 'date'], axis=1)
            df = df.multiply(df['total_minutes'], axis=0)
            df = df.drop(['total_minutes'], axis=1)
            sums = df.sum(axis = 0, skipna = True, numeric_only=True)
            sums_consumption_df = pd.DataFrame(sums, columns=['consumption {}'.format(name)])
            sums_consumption_df['Total consumption {}'.format(name)] = float(sums_consumption_df.sum(axis = 0, skipna = True, numeric_only=True))
            sums_consumption_df['Relative consumption {}'.format(name)] = sums_consumption_df['consumption {}'.format(name)] / sums_consumption_df['Total consumption {}'.format(name)]
            sums_consumption_df.index.name = 'id'
            sums_consumption_dfs.append(sums_consumption_df)
        
        sums = doc_topic_matrix_df.sum(axis = 0, skipna = True, numeric_only=True)
        sums_df = pd.DataFrame(sums, columns=['production'])
        sums_df['Total production'] = float(sums_df.sum(axis = 0, skipna = True, numeric_only=True))
        sums_df['Relative production'] = sums_df['production'] / sums_df['Total production']
        sums_df.index.name = 'id'

        data_frames = sums_consumption_dfs
        data_frames.append(sums_df)

        df = reduce(lambda  left,right: pd.merge(left,right,on=['id'],
                                            how='outer'), data_frames)

        df = df.round(4)
        df = df.reset_index()

        fig = go.Figure(
            data=[
                go.Bar(
                    name="Production",
                    x=df.index,
                    y=df["Relative production"],
                    offsetgroup=0,
                    marker_color='rgb(0, 180, 200)',
                    text=df['id']
                ),
                go.Scatter(
                    name="Consumption 15-29",
                    x=df.index,
                    y=df["Relative consumption 15-29"],
                    mode='markers',
                    marker_size=10,
                    text=df["Relative consumption 15-29"]
                ),
                go.Scatter(
                    name="Consumption 0-44",
                    x=df.index,
                    y=df["Relative consumption 0-44"],
                    mode='markers',
                    marker_size=10,
                    text=df["Relative consumption 0-44"]
                ),
                go.Scatter(
                    name="Consumption 45+",
                    x=df.index,
                    y=df["Relative consumption 45+"],
                    mode='markers',
                    marker_size=10,
                    text=df["Relative consumption 45+"]
                ),
            ],
            layout=go.Layout(
                title="Content production and consumption per topic",
                xaxis=dict(
                    title='Topic',
                    tickmode='linear'
                        ),
                yaxis=dict(
                    title='Production and consumption percentage',
                    tickformat='.2%'
                        )
            )
        )

        fig.update_layout(legend=dict(font_size=10))
        fig.update_layout(
            xaxis = dict(
                tickfont = dict(
                    size = 10
                )
            ),
            yaxis = dict(
                tickfont = dict(
                    size = 10
                )
            )
        )

        return fig

    def get_columns_to_split(self, column):
        df = self.dataframe_top_words(20)
        return df.iloc[:, [column]]

    def split_topic(self, column, features_1, features_2):
        # get index values of features_1 and features_2
        H_columns_feature_1_keep = []
        for feature in features_1:
            H_columns_feature_1_keep.append(self.vectorizer.vocabulary_[feature])
        H_columns_feature_2_keep = []
        for feature in features_2:
            H_columns_feature_2_keep.append(self.vectorizer.vocabulary_[feature])

        H = np.copy(self.nmf_components)

        # replicate row
        repeats = np.ones(len(H), dtype=int)
        repeats[column] = int(2)
        H = np.repeat(H, repeats, axis=0)

        # modify rows
        row_1_idx = column
        row_2_idx = column + 1

        original_row = H[column]

        # row_1 = np.zeros(self.config['tfidf']['max_features'])
        # row_1 = [max(1.0, original_row[i]*10)  if i in H_columns_feature_1_keep else 0.0 for i in range(len(original_row))]
        # row_2 = [max(1.0, original_row[i]*10) if i in H_columns_feature_2_keep else 0.0 for i in range(len(original_row))]

        row_1 = [original_row[i] if i in H_columns_feature_1_keep else 0.0 for i in range(len(original_row))]
        row_2 = [original_row[i] if i in H_columns_feature_2_keep else 0.0 for i in range(len(original_row))]

        df_h = pd.DataFrame(H)

        df_h.loc[row_1_idx, :] = row_1
        df_h.loc[row_2_idx, :] = row_2

        return df_h

    def generate_bokeh_umap(self, media_type):
        output_notebook()

        topics = []
        labels = []
        for key, value in self.top_words_map.items():
            topics.append(value)
            labels.append(key)

        if len(labels) >= 20000:
            reducer = umap.UMAP(n_neighbors=100, metric='hellinger')
        if len(labels) >= 5000:
            reducer = umap.UMAP(n_neighbors=50, metric='hellinger')
        else:
            reducer = umap.UMAP(metric='hellinger')        
        
        X = self.vectorized_out.copy()
        X_embedded = reducer.fit_transform(X)

        # tsne = TSNE(verbose=1, perplexity=100, random_state=42)
        # X = self.vectorized_out
        # X_embedded = tsne.fit_transform(X.toarray())

        df_tmp = pd.DataFrame(self.doc_topic_dists)
        df_tmp['topic'] = df_tmp.idxmax(axis=1)
        y_labels = df_tmp['topic'].values
        y_labels_new = []
        for i in y_labels:
            y_labels_new.append(labels[i])
        
        df = self.original_df.copy()

        # data sources
        if media_type == 'videos':
            source = ColumnDataSource(data=dict(
                x= X_embedded[:,0], 
                y= X_embedded[:,1],
                x_backup = X_embedded[:,0],
                y_backup = X_embedded[:,1],
                desc= y_labels,
                ids= df['id'], 
                titles= df['title'],
                published_times = df['first_airing'],
                text = df['text'],
                publication_end_times = df['publication_end_time'],
                media_availables = df['media_available'],
                duration_minutes = df['duration_minutes'],
                finnpanel_genres = df['finnpanel_genre'],
                labels = ["Topic " + str(x) for x in y_labels_new],
                links = df['link']
                ))

            # hover over information
            hover = HoverTool(tooltips=[
                ("Id", "@ids{safe}"),
                ("Title", "@titles{safe}"),
                ("Published", "@published_times{safe}"),
                # ("Text", "@texts{safe}"),
                ("Publication ends", "@publication_end_times{safe}"),
                ("Currently available", "@media_availables{safe}"),
                ("Duration (minutes)", "@duration_minutes{safe}"),
                ("Finnpanel genres", "@finnpanel_genres{safe}"),
                ("Link", "@links")
            ],
            point_policy="follow_mouse")
        
        elif media_type == 'articles':
            source = ColumnDataSource(data=dict(
                x= X_embedded[:,0], 
                y= X_embedded[:,1],
                x_backup = X_embedded[:,0],
                y_backup = X_embedded[:,1],
                desc= y_labels,
                ids= df.index, 
                titles= df['title'],
                published_times = df['published_time'].dt.strftime('%Y-%m-%d %H:%M'),
                text = df['text'],
                labels = ["Topic " + str(x) for x in y_labels_new],
                links = df['link']
                ))

            # hover over information
            hover = HoverTool(tooltips=[
                ("Id", "@ids{safe}"),
                ("Title", "@titles{safe}"),
                ("Published", "@published_times{safe}"),
                # ("Text", "@texts{safe}"),
                ("Link", "@links")
            ],
            point_policy="follow_mouse")

        # map colors
        mapper = linear_cmap(field_name='desc', 
            palette=Category20[20],
            low=min(y_labels) ,high=max(y_labels))

        # prepare the figure
        plot = figure(plot_width=1200, plot_height=850, 
                tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset', 'save', 'tap'], 
                title="Clustering of the content with UMAP and NMF", 
                toolbar_location="above")

        # plot settings
        plot.scatter('x', 'y', size=5, 
                source=source,
                fill_color=mapper,
                line_alpha=0.3,
                line_color="black",
                legend = 'labels')
        plot.legend.background_fill_alpha = 0.6

        # Keywords
        text_banner = Paragraph(text= 'Keywords: Slide to specific cluster to see the keywords.', height=45)
        input_callback_1 = input_callback(plot, source, text_banner, topics, self.nr_of_topics)

        # currently selected article
        div_curr = Div(text="""Click on a plot to see the link to the article.""",height=150)
        if media_type == 'videos':
            callback_selected = CustomJS(args=dict(source=source, current_selection=div_curr), code=selected_code_videos())
        elif media_type == 'articles':
            callback_selected = CustomJS(args=dict(source=source, current_selection=div_curr), code=selected_code_articles())
        taptool = plot.select(type=TapTool)
        taptool.callback = callback_selected

        # WIDGETS
        slider = Slider(start=0, end=self.nr_of_topics, value=self.nr_of_topics, step=1, title="Topic #")#, js_event_callbacks=input_callback_1)
        slider.js_on_change("value", input_callback_1)
        keyword = TextInput(title="Search:")#, js_event_callbacks=input_callback_1)
        keyword.js_on_change("value", input_callback_1)

        # pass call back arguments
        input_callback_1.args["text"] = keyword
        input_callback_1.args["slider"] = slider

        # STYLE
        slider.sizing_mode = "stretch_width"
        slider.margin=15

        keyword.sizing_mode = "scale_both"
        keyword.margin=15

        div_curr.style={'color': '#BF0A30', 'font-family': 'Helvetica Neue, Helvetica, Arial, sans-serif;', 'font-size': '1.1em'}
        div_curr.sizing_mode = "scale_both"
        div_curr.margin = 20

        text_banner.style={'color': '#0269A4', 'font-family': 'Helvetica Neue, Helvetica, Arial, sans-serif;', 'font-size': '1.1em'}
        text_banner.sizing_mode = "scale_both"
        text_banner.margin = 20

        plot.sizing_mode = "scale_both"
        plot.margin = 5

        r = row(div_curr,text_banner)
        r.sizing_mode = "stretch_width"

        # LAYOUT OF THE PAGE
        l = layout([
            [slider, keyword],
            [text_banner],
            [div_curr],
            [plot],
        ])
        l.sizing_mode = "scale_both"

        # show
        output_file('t-sne_interactive_streamlit.html')
        show(l)

        return(l)

    def build_w2v(self):
        #docgen = TokenGenerator(self.original_df["text"], self.additional_stopwords )
        docgen = TokenGenerator(self.original_df["text"], self.additional_stopwords )
        self.w2v_model = gensim.models.Word2Vec(docgen, size=500, min_count=0.0005, sg=1)
        #self.w2v_model.save("data/w2v/w2v-model-articlefulltext1441.bin")
        return self.w2v_model

    def calculate_coherence(self, w2v_model, term_rankings ):
        overall_coherence = 0.0
        for topic_index in range(len(term_rankings)):
            # check each pair of terms
            pair_scores = []
            for pair in combinations( term_rankings[topic_index], 2 ):
                pair_scores.append( w2v_model.similarity(pair[0], pair[1]) )
            # get the mean for all pairs in this topic
            topic_score = sum(pair_scores) / len(pair_scores)
            overall_coherence += topic_score
        # get the mean score across all topics
        return overall_coherence / len(term_rankings)
        
    def get_descriptor( self, all_terms, H, topic_index, top ):
        # reverse sort the values to sort the indices
        top_indices = np.argsort( H[topic_index,:] )[::-1]
        # now get the terms corresponding to the top-ranked indices
        top_terms = []
        for term_index in top_indices[0:top]:
            term = all_terms[term_index].split(' ')
            top_terms.extend( term )
        return top_terms

    def get_coherence(self, k, w2v_model):
        term_rankings = []
        for topic_index in range(k):
            descriptor = self.get_descriptor(self.features, self.nmf_components, topic_index, 30)
            term_rankings.append(descriptor)
        # Now calculate the coherence based on our Word2vec model
        coherence = self.calculate_coherence( w2v_model, term_rankings ) 
        return coherence
    
    def get_coherence_per_topic(self, w2v_model):
        coherence_per_topic = []
        topic_numbers = []

        for topic_index in range(self.nr_of_topics):
            term_rankings = [self.get_descriptor(self.features, self.nmf_components, topic_index, 15)]
            # Now calculate the coherence based on our Word2vec model
            coherence = self.calculate_coherence( w2v_model, term_rankings )
            coherence_per_topic.append(coherence)
            topic_numbers.append(self.cluster_names[topic_index])

        d = {"topic": topic_numbers, "coherence": coherence_per_topic}
        df = pd.DataFrame(d)
        df = df.round(2)

        fig = px.bar(df, x="topic", y="coherence", text='coherence', title='Coherence per topic')
        #fig.update_traces(mode='lines+markers')
        fig.update_layout(xaxis_title='Topic', yaxis_title='Coherence')
        fig.update_xaxes(categoryorder='total ascending')
        fig.update_layout(
            xaxis = dict(
                tickmode = 'linear',
                tickfont = dict(
                    size = 10
                )
            ),
            yaxis = dict(
                tickfont = dict(
                    size = 10
                )
            )
        )
        fig.update_traces(marker_color='rgb(0, 180, 200)', textposition='inside', textfont_size=8)

        return df, fig

    def plot_df_help(self):
        stopwords = self.additional_stopwords
        self.vectorizer = TfidfVectorizer(**self.config["tfidf"],
                                          stop_words=stopwords)
        self.vectorized_out = self.vectorizer.fit_transform(self.target)
        words = self.vectorizer.get_feature_names()

        count_data = np.array(self.vectorized_out.toarray())
        total_doc_counts = []
        for wordcolumn in count_data.T:
            total_doc_counts.append(np.count_nonzero(wordcolumn)/self.vectorized_out.shape[0])

        count_doc_dict = (zip(words, total_doc_counts))
        count_doc_dict = sorted(count_doc_dict, key=lambda x: x[1],
                            reverse=True)
        
        df = pd.DataFrame(count_doc_dict, columns = ['Term', 'df'])
        #df.df = df.df.round(2)

        # counts = [w[1] for w in count_doc_dict][:20]
        # words = [w[0] for w in count_doc_dict][:20]

        # d = {'Word': words, 'Count': counts}
        # df_show = pd.DataFrame(data=d)

        # fig = px.bar(df_show, x='Word', y='Count', text='Count', title='Words with highest document frequency (df)')
        # fig.update_layout(xaxis_title='Word', yaxis_title='Document frequency (df)')
        # fig.update_traces(marker_color='rgb(0, 180, 200)')
        # fig.update_traces(textposition='inside', textfont_size=10)
        # fig.update_layout(
        #     xaxis = dict(
        #         tickfont = dict(
        #             size = 10
        #         )
        #     ),
        #     yaxis = dict(
        #         tickfont = dict(
        #             size = 10
        #         )
        #     )
        # )

        return df

class TokenGenerator:
    def __init__( self, documents, stopwords ):
        self.documents = documents
        self.stopwords = stopwords
        self.tokenizer = re.compile( r"(?u)\b\w\w+\b" )

    def __iter__( self ):
        print("Building Word2Vec model ...")
        for doc in self.documents:
            tokens = []
            for tok in self.tokenizer.findall( doc ):
                if tok in self.stopwords:
                    tokens.append( "<stopword>" )
                elif len(tok) >= 3:
                    tokens.append( tok )
            yield tokens
