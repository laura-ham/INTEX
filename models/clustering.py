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
from functools import reduce
import time
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

class Pipeline:
    def __init__(self,
                 dataframe=None,
                 additional_stopwords=[],
                 min_df=0.0005,
                 max_df=1.0):
        self.original_df = dataframe
        self.additional_stopwords = additional_stopwords
        self.set_targets(dataframe)  # , self.additional_stopwords)
        self.list_of_strings = []
        # for i in self.original_df.tags:
        #     if len(i) == 1:
        #         self.list_of_strings.append(i[0])
        self.list_of_strings.extend(
            list(set(np.concatenate(self.original_df.tags, axis=None))))
        # for item in self.list_of_strings:
        #     if ' ' in item:
        #         words = item.split()
        #         if len(words) > 1:
        #             self.list_of_strings.extend(words)
        #             for i in range(1,len(words)):
        #                 list_of_tuples = more_itertools.windowed(words, i)
        #                 for elem in list_of_tuples:
        #                     self.list_of_strings.extend(' '.join(list(elem)))
        # list_of_strings = list(set(np.concatenate(self.original_df.tags, axis=None)))
        self.list_of_strings = sorted(
            list(set(map(str.lower, self.list_of_strings))))

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
                "sublinear_tf": True  # ,
                # "ngram_range": (1,5),
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
        self.nr_of_topics -= 1

        self.nmf_matrix = np.copy(self.W)
        self.nmf_components = np.copy(self.H)

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
        self.count_vectorizer = CountVectorizer(**self.config["countvectorizer"],
            stop_words=self.additional_stopwords,
            ngram_range=(1, 1),
            vocabulary=self.list_of_strings)
        self.count_data = self.count_vectorizer.fit_transform(
            df['terms'])
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

        d = {'Tag': words, 'Count': counts}
        df_show = pd.DataFrame(data=d)

        fig = px.bar(df_show, x='Tag', y='Count', text='Count', title='Most common tags')
        fig.update_layout(xaxis_title='Tag', yaxis_title='Frequency')
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

    def dataframe_top_words(self, nr_of_words):
        top_words = {k: v.split(', ') for k, v in self.top_words_map.items()}
        top_words = {
            k: [word.replace("_", " ") for word in v]
            for k, v in top_words.items()
        }  # to get spaces back instead of underscores, for visualization purposes only
        df_top_words = pd.DataFrame.from_dict(top_words)
        return df_top_words[:nr_of_words]

    def generate_ldavis(self):
        # 0 = (nr_of_topics, total_nr_of_words)
        # 1 = (nr_of_articles, total_nr_of_words)
        # 2 = (nr_of_articles, nr_of_words_per_article)

        params = {"mds": "pcoa"}
        try:
            LDAvis_prepared = sklearn_lda.prepare(self.nmf_model,
                                                self.vectorized_out,
                                                self.vectorizer, **params)
        except:
            return "This visualization is currently not available."
        return LDAvis_prepared

    def set_targets(self, df):
        self.target = df["terms"]

    def _top_words_map(self):
        words_map = {}
        self.cluster_names = []
        for cluster, comp_content_zip in enumerate(
                self.nmf_components):
            top_words = ", ".join([
                self.features[i] for i in comp_content_zip.argsort()[:-50 - 1:-1]
            ])
            key = str(cluster) + ': ' + top_words.split(
                ',')[0] + ', ' + top_words.split(',')[1] + ',' + top_words.split(',')[2]
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
        df["tags"] = [x for x in list(self.original_df.tags)]
        df = df.reset_index()
        df = df.melt(
            id_vars=["id", "date", "title", "tags"],
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
        df_to_return["tags"] = [x for x in list(self.original_df.tags)]
        df_to_return = df_to_return.reset_index()
        df_to_return = df_to_return.melt(
            id_vars=["id", "date", "title", "tags"],
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
        
    def get_production_graph(self, consumption_df):
        doc_topic_matrix_df = self.doc_topic_matrix_df.copy() # for absolute production (sum up to 1 per article)
        consumption_df.index.name = 'yle_article_id'

        # consumption_df = consumption_df.drop(['total_views', 'gender'], axis=1)
        consumption_df = consumption_df.groupby(['yle_article_id', 'age_group']).sum().reset_index().set_index('yle_article_id')

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
            sums_consumption_df.index.name = 'yle_article_id'
            sums_consumption_dfs.append(sums_consumption_df)
        
        sums = doc_topic_matrix_df.sum(axis = 0, skipna = True, numeric_only=True)
        sums_df = pd.DataFrame(sums, columns=['production'])
        sums_df['Total production'] = float(sums_df.sum(axis = 0, skipna = True, numeric_only=True))
        sums_df['Relative production'] = sums_df['production'] / sums_df['Total production']
        sums_df.index.name = 'yle_article_id'

        data_frames = sums_consumption_dfs
        data_frames.append(sums_df)

        df = reduce(lambda  left,right: pd.merge(left,right,on=['yle_article_id'],
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
                    text=df['yle_article_id']
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
            H_columns_feature_1_keep.append(self.vectorizer.vocabulary_[feature.replace(" ", "_")])
        H_columns_feature_2_keep = []
        for feature in features_2:
            H_columns_feature_2_keep.append(self.vectorizer.vocabulary_[feature.replace(" ", "_")])

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

        X = self.vectorized_out
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
        df['text'] = df['terms']

        # data sources
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
        docgen = TokenGenerator(self.original_df["terms"], self.additional_stopwords )
        self.w2v_model = gensim.models.Word2Vec(docgen, size=500, min_count=0.0005, sg=1)
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
            top_terms.append( all_terms[term_index] )
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
                elif len(tok) >= 2:
                    tokens.append( tok )
            yield tokens
