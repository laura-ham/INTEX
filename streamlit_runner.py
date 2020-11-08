import sys
import streamlit as st
import json
import os
import re
import pandas as pd
import numpy as np
import argparse
import requests
import logging
import yaml
import glob
import datetime
import time
import boto3
from models import clustering
from models import clustering_full_transcripts
import pyLDAvis
import webbrowser
import base64
from utils import SessionState, dataset, s3
import warnings
import plotly.express as px
from gensim.models import KeyedVectors
import time
import seaborn as sns
from modules import custom_ui, visualizations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# set color for plots to Yle turquoise
cm = sns.light_palette("#00b4c8", as_cmap=True)

# suppress warnings to improve ux
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", FutureWarning)

# define stopwords
DEFAULT_STOPWORDS = []
DEFAULT_STOPWORDS_VIDEOS = []
CITIES = ['helsinki', 'tampere', 'turku', 'oulu', 'lahti', 'jyväskylä', 'kuopio', 'pori', 'lappeenranta', 'vaasa', 'joensuu', 'hämeenlinna', 'kokkola', 'mikkeli', 'rovaniemi', 'kouvola', 'savonlinna', 'kemi', 'porvoo', 'kirkonkylä', 'kemijärvi', 'sodankylä', 'kangasala', 'varkaus', 'kuusamo', 'teuva', 'saarijärvi', 'lemland', 'simpele', 'lumijoki', 'urjala', 'geta', 'laukaa', 'ingå', 'laitila', 'kitee', 'kaustinen', 'honkajoki', 'kauhajoki', 'lieksa', 'sonkajärvi', 'suonenjoki', 'enontekiö', 'nastola', 'kyyjärvi', 'vårdö', 'savukoski', 'sipoo', 'siilinjärvi', 'tuusniemi', 'masku', 'ranua', 'lapua', 'nivala', 'nummela', 'pelkosenniemi', 'brändö', 'pulkkila', 'pudasjärvi', 'haapajärvi', 'harjavalta', 'karstula', 'kauniainen', 'korsnäs', 'ruokolahti', 'jomala', 'loimaa', 'muhos', 'rautalampi', 'rautavaara', 'lohja', 'taipalsaari', 'imatra', 'jämijärvi', 'raseborg', 'karkkila', 'kalajoki', 'puumala', 'uusikaupunki', 'soini', 'pihtipudas', 'toijala', 'lemi', 'savitaipale', 'karvia', 'ilmajoki', 'orimattila', 'oripää', 'kankaanpää', 'vantaa', 'pielavesi', 'kronoby', 'kivijärvi', 'juva', 'polvijärvi', 'polvijärvi', 'lappträsk', 'luvia', 'taivalkoski', 'ypäjä', 'joutsa', 'malax', 'mynämäki', 'ikaalinen', 'pertunmaa', 'järvenpää', 'kannus', 'nurmijärvi', 'sottunga', 'hämeenkyrö', 'ödkarby', 'tammela', 'multia', 'vieremä', 'liminka', 'kiuruvesi', 'kauhava', 'hyvinkää', 'outokumpu', 'kristinestad', 'nousiainen', 'iisalmi', 'hammarland', 'pieksämäki', 'hyrynsalmi', 'lappajärvi', 'suomussalmi', 'tuusula', 'kaskinen', 'hankasalmi', 'muonio', 'maaninka', 'eckerö', 'heinävesi', 'kannonkoski', 'kyrö', 'kirkkonummi', 'larsmo', 'askola', 'forssa', 'mänttä', 'utsjoki', 'parola', 'äänekoski', 'halsua', 'ilomantsi', 'alavus', 'taivassalo', 'lestijärvi', 'haapavesi', 'punkalaidun', 'uurainen', 'pyhäjoki', 'köyliö', 'kittilä', 'bennäs', 'simo', 'huittinen', 'ruovesi', 'keitele', 'dalsbruk', 'loppi', 'pyhäranta', 'lieto', 'kempele', 'aura', 'riihimäki', 'kustavi', 'liperi', 'kokemäki', 'luhanka', 'veteli', 'konnevesi', 'perho', 'rusko', 'nakkila', 'reisjärvi', 'godby', 'jämsä', 'vörå', 'hollola', 'jokioinen', 'närpes', 'seinäjoki', 'petäjävesi', 'rauma', 'ii', 'säkylä', 'kangasniemi', 'siuntio', 'hamina', 'pälkäne', 'kärsämäki', 'kuortane', 'vimpeli', 'koski tl', 'kausala', 'björby', 'alavieska', 'sysmä', 'tervola', 'kajaani', 'rantasalmi', 'hanko', 'sastamala', 'mäntsälä', 'ivalo', 'parikkala', 'pargas', 'siikainen', 'hirvensalmi', 'kuhmoinen', 'jalasjärvi', 'turenki', 'kotka', 'evijärvi', 'isokyrö', 'hartola', 'vääksy', 'vaala', 'virrat', 'keuruu', 'tornio', 'pornainen', 'pukkila', 'isojoki', 'pello', 'kurikka', 'järvelä', 'merijärvi', 'tohmajärvi', 'toholampi', 'muurame', 'sievi', 'naantali', 'lumparland', 'leppävirta', 'oulainen', 'jakobstad', 'joroinen', 'salo', 'laihia', 'lovisa', 'myrskylä', 'kuhmo', 'kontiolahti', 'paltamo', 'pirkkala', 'karlby', 'hämeenkoski', 'nurmes', 'toivakka', 'tarvasjoki', 'nokia', 'korsholm', 'hailuoto', 'sulkava', 'humppila', 'ristijärvi', 'kumlinge', 'ruukki', 'eura', 'salla', 'eurajoki', 'föglö', 'kimito', 'kihniö', 'posio', 'lempäälä', 'kinnula', 'åva', 'utajärvi', 'valtimo', 'paimio', 'karijoki', 'keminmaa', 'kolari', 'kaavi', 'mariehamn', 'raisio', 'enonkoski', 'korkeakoski', 'kaarina', 'orivesi', 'mäntyharju', 'alajärvi', 'puolanka', 'virolahti', 'tyrnävä', 'heinola', 'ähtäri', 'siltakylä', 'oitti', 'marttila', 'vesanto', 'kerava', 'ulvila', 'padasjoki', 'juankoski', 'espoo', 'parkano', 'merikarvia', 'raahe', 'tervo', 'lapinlahti', 'viitasaari', 'vesilahti', 'sotkamo', 'sauvo', 'ylivieska', 'vinkkilä', 'pyhäntä', 'pyhäsalmi', 'miehikkälä', 'valkeakoski', 'ylöjärvi', 'juuka', 'ylitornio', 'pomarkku', 'somero', 'rääkkylä', 'nykarleby', 'lavia', 'taavetti']
REGIONS = ['lapland', 'northern_ostrobothnia', 'kainuu', 'north_karelia', 'northern_saviona', 'southern_saviona', 'south_karelia', 'central_finland', 'soutnern_ostrobothnia', 'ostrobothnia', 'central_ostrobothnia', 'prikanmaa', 'satakunta', 'päijät_häme', 'kanta_häme', 'kymenlaakso', 'uusimaa' ,'southwest_finland', 'åland']
REGIONS.extend([x.lower() for x in ['forssan_seutukunta', 'riihimäen_seutukunta', 'itälapin_seutukunta', 'yläsavon_seutukunta', 'haminan_seutukunta', 'yläpirkanmaan_seutukunta', 'savonlinnan_seutukunta', 'lahden_seutukunta', 'loimaan_seutukunta', 'turunmaan_seutukunta', 'pohjoislapin_seutukunta', 'oulun_seutukunta', 'kouvolan_seutukunta', 'hämeenlinnan_seutukunta', 'mikkelin_seutukunta', 'Ahvenanmaa', 'Ahvenanmaa', 'Etelä_Pohjanmaa', 'EteläPohjanmaa', 'Etelä_Savo','EteläSavo', 'Kainuu', 'Kanta_Häme', 'Keski_Pohjanmaa', 'Keski_Suomi', 'KantaHäme', 'KeskiPohjanmaa', 'KeskiSuomi','Kymenlaakso', 'Lappi', 'pikanmaa', 'phojanmaa', 'Pohjois_Karjala', 'Pohjois_Pohjanmaa', 'Pohjois_Savo', 'Päijät_Häme', 'PohjoisKarjala', 'PohjoisPohjanmaa', 'PohjoisSavo', 'PäijätHäme', 'Satakunta', 'Uusimaa', 'varsinais_suomi', 'jyväskylän_seutukunta', 'keskisuomi', 'viitasaaren_seutukunta', 'äänekosken_seutukunta', 'seutukunta', 'hankasalmen_seutu', 'seutu', 'joensuun_seutukunta', 'tampereen_seutukunta', 'sisäsavon_seutukunta', 'kuopion_seutukunta', 'koillissavon_seutukunta', 'seurakunnat', 'varkauden_seutukunta', 'pohjoiskarjala', 'pohjoissavo', 'eteläsuomi', 'seinäjoen_seutukunta', 'pietarsaaren_seutukunta', 'kokkolan_seutukunta', 'pietarsaari', 'vaasan_seutukunta', 'eteläpohjanmaa', 'pohjanmaa', 'keskipohjanmaa', 'pohjoissuomi', 'salon_seutukunta', 'eteläkarjala', 'suomen_keskusta', 'saarijärven', 'kemitornion_seutukunta', 'rovaniemen_seutukunta', 'lounaissuomi', 'varsinaissuomi', 'turun_seutukunta', 'vakkasuomen_seutukunta', 'helsingin_seutu']])
CITIES.extend(REGIONS)
ARTICLE_BRANDS_STOPWORDS = ["uutiset", "yle", "astudio", "sannikka__ukkola", "8_minuuttia", "yle_perjantai", "a_talk", "mot", "puoli_seitsemän", "sannikka", "ukkola", "8 minuuttia", "puoli", "a talk", "seitsemän"]
ARTICLE_DEPARTMENT_NAMES = ["politiikka", 'kotimaan', 'ulkomaan', 'uutiset', "kotimaan uutiset", "kotimaan_uutiset", "ulkomaat", "ulkomaan uutiset", "ulkomaan_uutiset", "talous", 'terveys', 'elämäntapa', 'kulttuuri', 'suomi']
ARTICLE_DEFAULT_STOPWORDS = ['yle', 'yle uutiset somenostot', 'yle_uutiset_somenostot', 'yleiset', 'yle uutiset 60 vuotta', 'yle_uutiset_60_vuotta', 'vuotta', 'uutiset', 'somenostot']

# UI dropdown menu options
POSSIBLE_MEDIA_TYPES = ['Articles', 'Videos'] #['Articles 2019 (tags) (1441_ajankohtaiset only)', 'Articles 2019 (full text) (1441_ajankohtaiset only)', 'All articles Q1 2020', 'Documentaries'] # , 'Video (not supported yet)', 'Audio (not supported yet)']
POSSIBLE_ARTICLE_ORGANIZATIONS = ['All']
POSSIBLE_VIDEO_GENRES = ['Documentaries']
POSSIBLE_TOPIC_COUNTS = range(5, 51)

# to save app status between user interactions
global session_state
session_state = SessionState.get(clustering_done_on=None, in_merging_process=False, in_splitting_process=False, in_deleting_process=False, clustering_test = None, cluster_amount=POSSIBLE_TOPIC_COUNTS[0], merged=False, splitted=False, w2v_tags=None, w2v_articlesfulltext=None, w2v_alldocs=None, in_removingkeyword_process=False, in_renaming_process=False, past_refinements=[])

def get_consumption_data(filename, idxname):
    path = './data/consumption/' + filename
    df_consumption = load_data_from_path(path, idxname)

    # keep indexes that was clustered on
    df_clusters = session_state.clustering_test.doc_topic_matrix_df

    df_consumption = df_consumption[df_consumption.index.isin(df_clusters.index)]
    df_consumption['total_minutes'] = df_consumption['total_minutes'].astype(float)

    if idxname == 'program_yle_id':
        df_consumption = df_consumption.drop(['total_views', 'gender'], axis=1)

    return df_consumption

@st.cache
def get_data_videos(start_date, end_date, exclude_ids, video_genre, streamable):
    # if docs_only_currentlystreamable:
    #     df_meta = load_data_from_path('./data/doc_subs_newnlp_with_metadata.json', 'program_yle_id')
    #     df_meta['title'] = df_meta['program_title']

    #     df_texts = load_data_from_path('./data/currentlystreamable.json', 'content_yle_id')

    # Get data from genre
    if video_genre == 'Documentaries':
        df_meta = load_data_from_path('./data/videos_docs_metadata.json', 'program_yle_id')
        df_meta['title'] = df_meta['program_title']

        df_texts = load_data_from_path('./data/videos_docs_subs.json', 'content_yle_id')

    df = df_meta.merge(df_texts, left_index=True, right_index=True)

    # AVAILABILITY FILTER
    today = datetime.date.today().strftime("%Y-%m-%d")
    if streamable == 'Include only currently streamable':
        df = df[(df['media_available'] == True)]
        df = df[(df['publication_start_time'] <= today)]

        # df = df.loc[~(df['publication_end_time'] <= today)]
        # df.drop(ids_to_drop, inplace=True)
        # df = df[(df['publication_start_time'] <= today)]
    elif streamable == 'Include only streamable in past':
        # df = df[~(df['media_available'] == True)]
        # df = df[~(df['publication_end_time'] >= today)]
        # ids_to_drop = df[(df['publication_end_time'] >= today)].index
        # df.drop(ids_to_drop, inplace=True)
        ids_to_drop = df[(df['media_available'] == True)].index
        df.drop(ids_to_drop, inplace=True)

    df['ratio'] = df['text'].str.len() / df['duration'] * 100
    df['len_subs'] = df['text'].str.len()
    df = df[df['ratio'] >= 0.02]

    df['id'] = df.index 

    df["published_time"] = pd.to_datetime(df['first_airing'],
                                            format='%Y-%m-%d %H:%M:%S.%f',errors='coerce').dt.tz_localize('UTC', nonexistent='shift_forward').dt.tz_convert('EET')

    #df = df.drop(['Unnamed: 0', 'index', 'duration', 'areena_genres', 'program_title', 'duration_minutes', 'ratio', 'len_subs', 'first_airing', 'finnpanel_genre'], axis=1)
    df = df.drop(['Unnamed: 0', 'index', 'areena_genres', 'program_title', 'ratio', 'len_subs'], axis=1)

    df = df[(df['published_time'] >= pd.to_datetime(start_date).tz_localize('EET'))
            & (df['published_time'] <= pd.to_datetime(end_date).tz_localize('EET'))]
        
    df['link'] = df['id'].apply(lambda x: 'http://areena.yle.fi/{0}'.format(x))

    df = df.drop(exclude_ids)
    df.drop_duplicates(subset='id', keep='first', inplace=True)
    
    df['Number of words'] = [len(terms.split()) for terms in df['text'].values]

    df["text"] = df['text'].str.lower()

    # drop duplicate indices
    df = df.loc[~df.index.duplicated(keep='first')]

    return df

@st.cache
def get_data_videos_s3(start_date, end_date, exclude_ids, s3_agent, dataset_name, config):
    directory = '{}/{}/{}'.format(config['processed_output_bucket_path'], 'Videos', dataset_name)

    # get ids
    filename = 'ids'
    extension = 'json'
    if s3_agent.is_object_available(directory, filename, extension):
        ids_json = s3_agent.load_object(directory, filename, extension, use_json=True)
        df_meta = pd.DataFrame(ids_json)
        df_meta.set_index('program_yle_id', inplace=True)
        df_meta.drop_duplicates(inplace=True)
        df_meta['title'] = df_meta['program_title']
    else:
        return st.error('File with ids not found in S3')

    # get full texts
    filename = 'text_preprocessed'
    extension = 'pkl'
    if s3_agent.is_object_available(directory, filename, extension):
        df_texts = s3_agent.load_object(directory, filename, extension, pickled=True)
        df_texts.set_index('id', inplace=True)
    else:
        return st.error('File with texts not found in S3')

    df = df_meta.merge(df_texts, left_index=True, right_index=True)

    df['ratio'] = df['text'].str.len() / df['duration'] * 100
    df['len_subs'] = df['text'].str.len()
    df = df[df['ratio'] >= 0.02]

    df['id'] = df.index 

    df["published_time"] = pd.to_datetime(df['first_airing'], errors='coerce', unit='ms').dt.tz_localize('UTC', nonexistent='shift_forward').dt.tz_convert('EET')
    df["publication_start_time"] = pd.to_datetime(df['publication_start_time'],errors='coerce', unit='ms').dt.tz_localize('UTC', nonexistent='shift_forward').dt.tz_convert('EET')
    df["publication_end_time"] = pd.to_datetime(df['publication_end_time'],errors='coerce', unit='ms').dt.tz_localize('UTC', nonexistent='shift_forward').dt.tz_convert('EET')

    #df = df.drop(['Unnamed: 0', 'index', 'duration', 'areena_genres', 'program_title', 'duration_minutes', 'ratio', 'len_subs', 'first_airing', 'finnpanel_genre'], axis=1)
    df = df.drop(['type', 'program_title', 'ratio', 'len_subs'], axis=1)

    df = df[(df['published_time'] >= pd.to_datetime(start_date).tz_localize('EET'))
            & (df['published_time'] <= pd.to_datetime(end_date).tz_localize('EET'))]
        
    df['link'] = df['id'].apply(lambda x: 'http://areena.yle.fi/{0}'.format(x))

    df = df.drop(exclude_ids)
    df.drop_duplicates(subset='id', keep='first', inplace=True)
    
    # df['Number of words'] = [len(terms.split()) for terms in df['text'].values]

    df["text"] = df['text'].str.lower()

    # drop duplicate indices
    df = df.loc[~df.index.duplicated(keep='first')]

    return df

@st.cache
def get_data_article_fulltext(organizations, start_date, end_date, exclude_ids):
    df_titles = load_data_from_path('./data/articles_ids.json', 'article_yle_id')
    df_titles.drop_duplicates(inplace=True)

    df_texts = pd.read_pickle(
        './data/articles_full_text_preprocessed.pkl')
    df_texts.set_index('id', inplace=True)

    df = df_titles.merge(df_texts, how='right', left_index=True, right_index=True)

    df["published_time"] = pd.to_datetime(df['published_time'], format='%Y-%m-%d %H:%M:%S.%f',errors='coerce').dt.tz_localize('UTC', nonexistent='shift_forward').dt.tz_convert('EET')

    df = df[(df['published_time'] >= pd.to_datetime(start_date).tz_localize('EET'))
            & (df['published_time'] <= pd.to_datetime(end_date).tz_localize('EET'))]

    df['link'] = df.index
    df['link'] = df['link'].apply(lambda x: 'http://yle.fi/uutiset/{0}'.format(x) if (x.split('-')[0] == '3') else 'No link available')

    df = df.drop(exclude_ids)

    df['Number of words'] = [len(terms.split()) for terms in df['text'].values]

    df["text"] = df['text'].str.lower()

    # drop duplicate indices
    df = df.loc[~df.index.duplicated(keep='first')]

    return df

@st.cache
def get_data_article_fulltext_s3(organizations, start_date, end_date, exclude_ids, s3_agent, dataset_name, config):
    directory = '{}/{}/{}'.format(config['processed_output_bucket_path'], 'Articles', dataset_name)

    # get ids
    filename = 'ids'
    extension = 'json'
    if s3_agent.is_object_available(directory, filename, extension):
        ids_json = s3_agent.load_object(directory, filename, extension, use_json=True)
        df_titles = pd.DataFrame(ids_json)
        df_titles.set_index('article_yle_id', inplace=True)
        df_titles.drop_duplicates(inplace=True)
        df_titles = df_titles.drop(['organization'], axis=1)
    else:
        return st.error('File with ids not found in S3')

    # get full texts
    filename = 'text_preprocessed'
    extension = 'pkl'
    if s3_agent.is_object_available(directory, filename, extension):
        df_texts = s3_agent.load_object(directory, filename, extension, pickled=True)
        df_texts.set_index('id', inplace=True)
    else:
        return st.error('File with texts not found in S3')

    df = df_titles.merge(df_texts, how='right', left_index=True, right_index=True)

    df["published_time"] = pd.to_datetime(df['published_time'], errors='coerce', unit='ms').dt.tz_localize('UTC', nonexistent='shift_forward').dt.tz_convert('EET')

    df = df[(df['published_time'] >= pd.to_datetime(start_date).tz_localize('EET'))
            & (df['published_time'] <= pd.to_datetime(end_date).tz_localize('EET'))]

    df['link'] = df.index
    df['link'] = df['link'].apply(lambda x: 'http://yle.fi/uutiset/{0}'.format(x) if (x.split('-')[0] == '3') else 'No link available')

    df = df.drop(exclude_ids)

    df['Number of words'] = [len(terms.split()) for terms in df['text'].values]

    df["text"] = df['text'].str.lower()

    # drop duplicate indices
    df = df.loc[~df.index.duplicated(keep='first')]

    return df

@st.cache
def get_data_article_tags(organizations, start_date, end_date, exclude_ids):
    if organizations == '1441_ajankohtaiset':
        df_titles = load_data_from_path('./data/articles_ajankohtaiset_201820_ids.json', 'article_yle_id')
        df_titles.drop_duplicates(inplace=True)
        df_tags = load_data_from_path('./data/articles_ajankohtaiset_201820_tags_formatted.json', 'id')
    elif organizations == 'All':
        df_titles = load_data_from_path('./data/articles_ids.json', 'article_yle_id')
        #df_titles = df_titles.drop(['section_name', 'organization', 'homesection_name'], axis=1)
        df_titles.drop_duplicates(inplace=True)

        df_tags = load_data_from_path('./data/articles_tags.json', 'id')

    # df_tags['tags'] = df_tags['tags'].map(
    #     replace_spaces
    # )  # to make one token out of one tag consisting of multiple words or containing regex
    
    df = df_titles.merge(df_tags, left_index=True, right_index=True)

    df["published_time"] = pd.to_datetime(df['published_time'], format='%Y-%m-%d %H:%M:%S.%f',errors='coerce').dt.tz_localize('UTC', nonexistent='shift_forward').dt.tz_convert('EET')

    df = df[(df['published_time'] >= pd.to_datetime(start_date).tz_localize('EET'))
            & (df['published_time'] <= pd.to_datetime(end_date).tz_localize('EET'))]

    df['terms'] = [" ".join(tag) for tag in df['tags'].values]
    df['terms'] = df['terms'].str.lower()
    df = df[df.terms != ' ']

    df['link'] = df.index
    df['link'] = df['link'].apply(lambda x: 'http://yle.fi/uutiset/{0}'.format(x) if (x.split('-')[0] == '3') else 'No link available')

    df['Number of tags'] = [len(terms.split()) for terms in df['terms'].values]

    df = df.drop(exclude_ids)

    # drop duplicate indices
    df = df.loc[~df.index.duplicated(keep='first')]

    return df

@st.cache
def get_data_article_tags_s3(organizations, start_date, end_date, exclude_ids, s3_agent, dataset_name, config):
    directory = '{}/{}/{}'.format(config['processed_output_bucket_path'], 'Articles', dataset_name)

    # get ids
    filename = 'ids'
    extension = 'json'
    if s3_agent.is_object_available(directory, filename, extension):
        ids_json = s3_agent.load_object(directory, filename, extension, use_json=True)
        df_titles = pd.DataFrame(ids_json)
        df_titles.set_index('article_yle_id', inplace=True)
        df_titles.drop_duplicates(inplace=True)
        df_titles = df_titles.drop(['organization'], axis=1)
    else:
        return st.error('File with ids not found in S3')

    # get tags
    filename = 'tags'
    extension = 'json'
    if s3_agent.is_object_available(directory, filename, extension):
        ids_json = s3_agent.load_object(directory, filename, extension, use_json=True)
        df_tags = pd.DataFrame(ids_json)
        df_tags.set_index('id', inplace=True)
    else:
        return st.error('File with ids not found in S3')

    df = df_titles.merge(df_tags, left_index=True, right_index=True)

    df["published_time"] = pd.to_datetime(df['published_time'], errors='coerce', unit='ms').dt.tz_localize('UTC', nonexistent='shift_forward').dt.tz_convert('EET')

    df = df[(df['published_time'] >= pd.to_datetime(start_date).tz_localize('EET'))
            & (df['published_time'] <= pd.to_datetime(end_date).tz_localize('EET'))]

    df['terms'] = [" ".join(tag) for tag in df['tags'].values]
    df['terms'] = df['terms'].astype(str).str.lower()
    df = df[df.terms != ' ']

    df['link'] = df.index
    df['link'] = df['link'].apply(lambda x: 'http://yle.fi/uutiset/{0}'.format(x) if (x.split('-')[0] == '3') else 'No link available')

    df['Number of tags'] = [len(terms.split()) for terms in df['terms'].values]

    df = df.drop(exclude_ids)

    # drop duplicate indices
    df = df.loc[~df.index.duplicated(keep='first')]

    return df

def load_data_from_path(path, idx_name):
    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df.set_index(idx_name, inplace=True)
    return df

@st.cache
def replace_spaces(stringlist):
    return [
        re.sub("[!@#$--&'()]", '', str(word)).replace(" ", "_").replace("-", "_")
        for word in stringlist
    ]
    # return [word.replace(" ", "_").replace("-", "_") for word in stringlist]

def rename_cluster(clustering_pipeline):
    placeholder_topic_to_rename = st.sidebar.empty()
    placeholder_new_name = st.sidebar.empty()
    placeholder_button = st.sidebar.empty()
    placeholder_cancelbutton = st.sidebar.empty()

    topic_to_rename = placeholder_topic_to_rename.selectbox("Select the topic you want to rename", range(0, clustering_pipeline.nr_of_topics))
    new_name = placeholder_new_name.text_input("Write the new name")

    if placeholder_button.button("Apply rename cluster"):
        clustering_pipeline.rename_topic(topic_to_rename, new_name)

        st.sidebar.markdown('Topic {topic} was renamed to "{name}". See the results on the right.'.format(topic=str(topic_to_rename), name=new_name))

        placeholder_topic_to_rename.empty()
        placeholder_new_name.empty()
        session_state.in_renaming_process = False
        session_state.past_refinements.append('Renamed cluster {} to {}'.format(topic_to_rename, new_name))
        placeholder_button.empty()
        placeholder_cancelbutton.empty()

    return clustering_pipeline, session_state

def remove_keyword(clustering_pipeline):
    placeholder_topic_to_delete_from = st.sidebar.empty()
    placeholder_word_to_delete = st.sidebar.empty()
    placeholder_button = st.sidebar.empty()
    placeholder_cancelbutton = st.sidebar.empty()
    session_state.in_removingkeyword_process = True

    topic_to_delete_from = placeholder_topic_to_delete_from.selectbox("Select the topic you want to remove a word from", range(0, clustering_pipeline.nr_of_topics))

    df_column = clustering_pipeline.get_columns_to_split(topic_to_delete_from)

    top_words_in_topic = []
    for value in df_column.values.tolist():
        top_words_in_topic.append(value[0])

    word_to_delete = placeholder_word_to_delete.selectbox("The keyword you want to remove from the topic", top_words_in_topic)

    if placeholder_button.button("Apply remove keyword"):
        clustering_pipeline.delete_word_from_topic(topic_to_delete_from, word_to_delete, top_words_in_topic)

        placeholder_topic_to_delete_from.empty()
        placeholder_word_to_delete.empty()
        session_state.in_removingkeyword_process = False
        placeholder_button.empty()
        placeholder_cancelbutton.empty()
        session_state.past_refinements.append('Removed keyword {} from topic {}'.format(word_to_delete, topic_to_delete_from))

        st.sidebar.markdown('Word "{word}" was removed from topic {topic}. See the results on the right.'.format(word=word_to_delete, topic=str(topic_to_delete_from)))
        return clustering_pipeline, session_state
        
    if placeholder_cancelbutton.button('Cancel removing keyword'):
        session_state.in_removingkeyword_process = False
        placeholder_topic_to_delete_from.empty()
        placeholder_word_to_delete.empty()
        placeholder_button.empty()
        placeholder_cancelbutton.empty()

    return clustering_pipeline, session_state

def merge_clusters(clustering_pipeline, cluster_amount):
    global session_state
    placeholder_instructions = st.sidebar.empty()
    placeholder_merge_1 = st.sidebar.empty()
    placeholder_merge_2 = st.sidebar.empty()
    to_merge_1 = placeholder_merge_1.selectbox("Select first topic cluster to merge", range(0, cluster_amount))
    to_merge_2 = placeholder_merge_2.selectbox("Select second topic cluster to merge", range(0, cluster_amount), index=1)
    session_state.in_merging_process = True

    placeholder_button = st.sidebar.empty()
    placeholder_cancelbutton = st.sidebar.empty()

    if to_merge_1 != to_merge_2:

        if placeholder_button.button('Apply merge clusters {} and {}'.format(to_merge_1, to_merge_2)):
            session_state.in_merging_process = False

            clustering_pipeline.fit_transform_merge(to_merge_1, to_merge_2)
            # st.subheader(
            #     'NEW cluster preview: most frequent words (rows) per cluster (columns)')
            # display table with top words per clusters
            # df = clustering_pipeline.dataframe_top_words(20)

            st.sidebar.markdown('See the results on the right')
            placeholder_merge_1.empty()
            placeholder_merge_2.empty()
            placeholder_instructions.empty()
            placeholder_button.empty()
            placeholder_cancelbutton.empty()

            session_state.past_refinements.append('Merged clusters {} and {}'.format(to_merge_1, to_merge_2))

            session_state.merged = True
            return clustering_pipeline, cluster_amount-1, session_state
    else:
        placeholder_instructions.markdown('Please select two different clusters')
    
    if placeholder_cancelbutton.button('Cancel merging'):
        session_state.in_merging_process = False
        placeholder_merge_1.empty()
        placeholder_merge_2.empty()
        placeholder_button.empty()
        placeholder_cancelbutton.empty()
        placeholder_instructions.empty()

    return clustering_pipeline, cluster_amount, session_state

def split_clusters(clustering_pipeline, cluster_amount):
    global session_state
    placeholder_to_split = st.sidebar.empty()
    to_split = placeholder_to_split.selectbox("Select the cluster number to split", range(0, cluster_amount))

    # dataframe
    df_column = clustering_pipeline.get_columns_to_split(to_split)
    # st.dataframe(df_column)

    top_words_in_topic = []
    for value in df_column.values.tolist():
        top_words_in_topic.append(value[0])

    # select words to keep in topic 1
    placeholder_to_keep_1 = st.sidebar.empty()
    topic_1_keep = placeholder_to_keep_1.multiselect('Select words to keep in one topic:', top_words_in_topic, default=top_words_in_topic)

    # select words to keep in topic 2
    placeholder_to_keep_2 = st.sidebar.empty()
    topic_2_keep = placeholder_to_keep_2.multiselect('Select words to keep in the other topic:', top_words_in_topic, default=top_words_in_topic)

    placeholder_button = st.sidebar.empty()
    placeholder_cancelbutton = st.sidebar.empty()
    
    if placeholder_button.button('Apply split cluster {} and show results'.format(to_split)):
        clustering_pipeline.fit_transform_split(cluster_amount+1, clustering_pipeline.split_topic(to_split, topic_1_keep, topic_2_keep), to_split)
        
        # st.subheader(
        #     'NEW Cluster preview: most frequent words (rows) per cluster (columns)')
        # display table with top words per clusters
        # st.dataframe(clustering_pipeline.dataframe_top_words(20))

        placeholder_to_split.empty()
        placeholder_to_keep_1.empty()
        placeholder_to_keep_2.empty()
        placeholder_button.empty()
        placeholder_cancelbutton.empty()
        session_state.in_splitting_process = False

        st.sidebar.markdown('See the results on the right')
        session_state.splitted = True
        session_state.past_refinements.append('Splitted topic {} into topics with words {} and {}'.format(to_split, str(topic_1_keep).strip('[]'), str(topic_2_keep).strip('[]')))
        return clustering_pipeline, cluster_amount+1, session_state

    if placeholder_cancelbutton.button('Cancel splitting'):
        placeholder_to_split.empty()
        placeholder_to_keep_1.empty()
        placeholder_to_keep_2.empty()
        placeholder_button.empty()
        placeholder_cancelbutton.empty()
        session_state.in_splitting_process = False

    
    return clustering_pipeline, cluster_amount, session_state

def generate_csv(clustering_pipeline):
    df_tmp = clustering_pipeline.dataframe_top_words(50)
    df_tmp = df_tmp.transpose()
    df_tmp = df_tmp.apply(', '.join, axis=1)
    csv = df_tmp.to_csv()
    b64 = base64.b64encode(csv.encode()).decode(
    )  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
    return href

def run_videos(session_state, start_date, end_date, exclude_ids, video_genre=None, streamable=None, use_s3=False, s3_agent=None, dataset=None, config=None):
    if use_s3:
        df = get_data_videos_s3(start_date, end_date, exclude_ids, s3_agent, dataset, config)
    else:
        df = get_data_videos(start_date, end_date, exclude_ids, video_genre, streamable)

    st.header('1. Data Input')
    if st.checkbox("Show dataset", value=1):
        #st.dataframe(df)
        st.dataframe(df[['id', 'published_time', 'title', 'text', 'publication_end_time', 'media_available']])

    st.subheader('Information on data input quality')
    st.write('Number of data items: ', df.shape[0])
    if df.shape[0] <= 250:
        st.warning('WARNING: You have selected a small number of data, so clustering results might be poor. Filter out less data or select a bigger time window for better clustering results.')
    if st.checkbox("Show distribution plot of number of words per video", value=0):
        fig = px.histogram(df, x="Number of words", hover_data=df[['published_time', 'title', 'text', 'Number of words']], title='Distribution - Video transcript word count')
        fig.update_layout(xaxis_title='Word count', yaxis_title='Frequency')
        fig.update_traces(marker_color='rgb(0, 180, 200)')
        st.plotly_chart(fig, use_container_width=True)


    # INTERACTIVE MODELING

    # Model Configuration

    st.sidebar.header('2. Model Configuration')
    st.sidebar.subheader('2.1. Excluding Words')

    clustering_pipeline = clustering_full_transcripts.Pipeline(df)

    # STOPWORDS
    # for performance, stopwords can only be inputted manually with big data
    # most_common_words, count_dict = clustering_pipeline.get_most_common_words(df, 20)
    # most_common_words.extend(DEFAULT_STOPWORDS_VIDEOS)
    # most_common_words.sort()

    additional_stopwords = []
    # manual_stopwords = st.sidebar.multiselect(
    #     'Use the bar chart on the right to inform about the most common words in the dataset influencing the cluster results. Select words to exclude:',
    #     most_common_words,
    #     default=DEFAULT_STOPWORDS)
    # additional_stopwords.extend(manual_stopwords)
    
    free_stopwords = st.sidebar.text_input(
        'Use the bar chart on the right to inform about the most common words in the dataset influencing the cluster results. Write stopwords you want to exclude (separate terms by comma):')
    if "," in free_stopwords:
        additional_stopwords.extend(free_stopwords.replace(" ", "").split(','))
    else:
        additional_stopwords.extend(free_stopwords.split())

    clustering_pipeline.additional_stopwords = additional_stopwords

    st.sidebar.subheader('2.2. Model Hyperparameters')
    max_df = st.sidebar.slider(
        "Set the maximum df (ignore terms that have a document frequency higher than the given threshold, as proportion of documents):",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.01,
        format='%.4f')
    min_df = st.sidebar.slider(
        "Set the minimum df (ignore terms that have a document frequency lower than the given threshold, as proportion of documents):",
        min_value=0.0,
        max_value=0.01,
        value=0.0005,
        step=0.0001,
        format='%.4f')
    clustering_pipeline.config["tfidf"]["min_df"] = min_df
    clustering_pipeline.config["tfidf"]["max_df"] = max_df
    clustering_pipeline.config["countvectorizer"]["min_df"] = min_df
    clustering_pipeline.config["countvectorizer"]["max_df"] = max_df


    st.header('2. Model Configuration Figures')
    st.write("The cluster model generates clusters or topics from words in the documents. Words that occur very often in documents, have a big influence on the model. If these words are general words that have nothing to do with the actual topic or meaning of the document, they might have a negative influence on the clusters that are automatically derived. Therefore, you are given the option to exclude words from the model. Words that occur most often, and words that occur in the most documents, can be analysed here. Based on these visualizations, decide whether you want to exclude some of these words. In the left panel you can select or type words to exclude, or change the df (document frequency) parameter to exclude words that occur in many documents.")
    st.subheader('2.1. Most common words')

    # bar chart with most common words (goal is to see if there is any outliers,
    #   words that should be taken into stopword list)
    placeholder_checkbox_show_common_words_graph = st.empty()
    placeholder_dictionery_text = st.empty()
    placeholder_common_words_graph = st.empty()

    show_mostcommonwordsbarchart = True
    if session_state.clustering_done_on == 'videos_docs':
        show_mostcommonwordsbarchart = False
    
    if placeholder_checkbox_show_common_words_graph.checkbox('Show bar chart with most common words', value=show_mostcommonwordsbarchart):
        placeholder_dictionery_text.markdown(
            'Use the following figure to decide if you want to exclude words from influencing the topic clusters. Examples are words like and "uutiset" and "Yle". You can write the words you want to exclude in the input field in the sidebar.'
        )
        most_common_words, count_dict = clustering_pipeline.get_most_common_words(df, 20) # for performance, this is only run here
        words_to_plot = [item for item in most_common_words if item not in additional_stopwords]
        count_dict = [(item, count) for item, count in count_dict if item in words_to_plot]
        placeholder_common_words_graph.plotly_chart(clustering_pipeline.plotly_visualize_most_common_words(df, 15, words_to_plot, count_dict), use_container_width=True)

    # print df help plot
    st.subheader('2.2. Document Frequency (df) optimalization help')
    placeholder_checkbox_show_df_graph = st.empty()
    placeholder_df_text = st.empty()
    placeholder_df_df = st.empty()

    if placeholder_checkbox_show_df_graph.checkbox('Show df parameter selection help', value=False):
        placeholder_df_text.markdown(
            'This table shows the document frequency (df) of the words in the dataset. This is a value between 0 and 1, a ratio in how many documents the word is present. Words that have a high df value (occur in a lot of documents), are potentially "stopwords", and can be filtered out (in the left panel).'
        )
        df_df = clustering_pipeline.plot_df_help()
        placeholder_df_df.dataframe(df_df.style.format({'df': '{:.4f}'}))

    # if session_state.w2v_alldocs == None:
    #     cwd = os.getcwd()
    #     path = cwd + '/data/w2v/w2v-model-alldocs.bin'
    #     session_state.w2v_alldocs = KeyedVectors.load(path)

    # if st.sidebar.button('Help on choosing the optimal amount of topics (calculation takes some time...)'):
    #     st.sidebar.markdown('Results appear on the right.')
    #     logger.info("Calculating optimal number of clusters...")

    #     k = []
    #     coherences = []
    #     for i in range(5, 25):
    #         clustering_pipeline.fit_transform(topics=i)
    #         coherence = clustering_pipeline.get_coherence(i, session_state.w2v_alldocs)
    #         coherences.append(coherence)
    #         k.append(i)

    #     d = {'k':k, 'coherence': coherences}
    #     df_coherence = pd.DataFrame(data=d)

    #     st.subheader('2.2 Mean coherence of the desciptors per amount of topics')
    #     # st.dataframe(df_coherence)

    #     st.write('This graph shows the coherence per number of topics. Generally, higher the coherence, the better the topic describes one group of documents. A low coherence means that the extracted topics might be too general. It is recommended to choose a number of topics with a high coherence measure.')
    #     fig = px.line(df_coherence, x="k", y="coherence", title='Topic coherence per number of topics')
    #     fig.update_layout(xaxis_title='Number of topics', yaxis_title='Coherence')
    #     fig.update_layout(xaxis = dict(tickfont = dict(size = 10)), yaxis = dict(tickfont = dict(size = 10)))
    #     st.plotly_chart(fig, use_container_width=True)

    topics_amount = st.sidebar.number_input("Select how many topics you want to extract", 2, 50, 5, 1)



    # if st.sidebar.button('Suggest number of clusters (may take some time)'):
    #     logger.info("Calculating optimal number of clusters...")

    #     w2v_model = clustering_pipeline.build_w2v()
        
    #     k = []
    #     coherences = []
    #     for i in range(5, 25):
    #         clustering_pipeline.fit_transform(topics=i)
    #         coherence = clustering_pipeline.get_coherence(i)
    #         coherences.append(coherence)
    #         k.append(i)

    #     d = {'k':k, 'coherence': coherences}
    #     session_state.df_coherence = pd.DataFrame(data=d)

    #     session_state.suggest = True

    # if session_state.suggest:
    #     st.subheader('2.2 Mean coherence of the desciptors per k')
    #     st.dataframe(session_state.df_coherence)

    #     fig = px.line(session_state.df_coherence, x="k", y="coherence", title='Coherence')
    #     st.plotly_chart(fig, use_container_width=True)


        # k = []
        # ssrs = []
        # rec_ers = []
        # for i in range(5, 25):
        #     clustering_pipeline.fit_transform(topics=i)
        #     ssr, rec_er, residuals_per_doc = clustering_pipeline.get_residuals()
        #     k.append(i)
        #     ssrs.append(ssr)
        #     rec_ers.append(rec_er)
        # d = {'k':k, 'ssr': ssrs, 'reconstruction error': rec_ers}
        # df_ssr = pd.DataFrame(data=d)

        # session_state.suggest = True

    # if session_state.suggest:
    #     st.subheader('2.2 Sum of Squared Residuals per k')
    #     st.dataframe(df_ssr)

    #     fig = px.line(df_ssr, x="k", y="ssr", title='SSR')
    #     st.plotly_chart(fig, use_container_width=True)
    #     fig = px.line(df_ssr, x="k", y="reconstruction error", title='reconstruction error')
    #     st.plotly_chart(fig, use_container_width=True)


    if st.sidebar.button('RUN'):
        logger.info("New clustering initialized...")
        clustering_pipeline.fit_transform(topics=topics_amount)
        session_state.cluster_amount = topics_amount
        session_state.clustering_test = clustering_pipeline
        session_state.clustering_done_on = 'videos_docs'
        session_state.merged = False
        session_state.splitted = False
    
    if session_state.clustering_done_on != 'videos_docs':
        return

    # Topic Refinement

    st.sidebar.header('3. Topic Refinement')

    # # REMOVING KEYWORD
    want_to_remove_keyword = st.sidebar.button("I want to remove a keyword from a topic")
    if want_to_remove_keyword:
        session_state.in_removingkeyword_process = True
    if session_state.in_removingkeyword_process:
        session_state.clustering_test, session_state = remove_keyword(session_state.clustering_test)
    
    # MERGING
    if not session_state.in_removingkeyword_process:
        want_to_merge = st.sidebar.button("I want to merge two clusters")
        if want_to_merge: 
            session_state.in_merging_process = True
        if session_state.in_merging_process:
            session_state.clustering_test, session_state.cluster_amount, session_state = merge_clusters(session_state.clustering_test, session_state.cluster_amount)

    # SPLITTING
    if not session_state.in_merging_process and not session_state.in_removingkeyword_process:
        want_to_split = st.sidebar.button("I want to split a cluster")
        if want_to_split:
            session_state.in_splitting_process = True
        if session_state.in_splitting_process:
            session_state.clustering_test, session_state.cluster_amount, session_state = split_clusters(session_state.clustering_test, session_state.cluster_amount)

    # RENAME TOPIC
    if not session_state.in_merging_process and not session_state.in_removingkeyword_process and not session_state.in_splitting_process:
        want_to_rename = st.sidebar.button("I want to rename a cluster")
        if want_to_rename:
            session_state.in_renaming_process = True
        if session_state.in_renaming_process:
            session_state.clustering_test, session_state = rename_cluster(session_state.clustering_test)

    st.header('3. Model Output Analysis Figures')
    st.subheader('3.1. Topic-Term Perspective')

    if st.checkbox("Show Topic-Term Table: Most frequent words (rows) per cluster (columns)", value=1):
        placeholder_topic_term_table = st.empty()
        topic_term_table = session_state.clustering_test.dataframe_top_words(20)  
        placeholder_topic_term_table.dataframe(topic_term_table)

    if st.button("Generate interactive cluster visualization (topic point of view, to explore topic and keyword relations) (warning: this takes some time)"):
        visualizations.get_lda_vis(session_state.clustering_test)

    # document-topic matrix
    st.subheader('3.2. Document-Topic Perspective')

    if st.checkbox("Show document-topic matrix: comparison of individual content", value=1):
        doc_topic_matrix_df = session_state.clustering_test.doc_topic_matrix_df
        # doc_topic_matrix_df['id'] = df.index
        # cols = doc_topic_matrix_df.columns.tolist()
        # cols = cols[-1:] + cols[:-1]
        # st.dataframe(doc_topic_matrix_df[cols])
        # st.dataframe(doc_topic_matrix_df.style.background_gradient(cmap=cm)) #coloring costs too much time
        st.dataframe(doc_topic_matrix_df)

    if st.button("Generate interactive document-cluster visualization (individual document point of view) (warning: this takes some time)"):
        visualizations.get_doctop_vis(session_state.clustering_test, media='videos')

    if st.checkbox("Show interactive visualization: compare distribution over topics of individual documents", value=0):
        article_id_list = df.index.values.tolist()
        article_title_list = df['title'].tolist()
        show = [
            str(m) + ' - ' + n for m, n in zip(article_id_list, article_title_list)
        ]
        content_to_display_with_title = st.multiselect(
            "Select articles to compare and show in bar chart", show, default=show[:10])
        content_to_display = []

        placeholder_contentbarchart = st.empty()
        placeholder_contentbarcharttable = st.empty()

        if len(content_to_display_with_title) < 1 or len(content_to_display_with_title) > 20:
            placeholder_contentbarchart.text('Select up to 20 content items to compare')
        else:
            for i in content_to_display_with_title:
                content_to_display.append(i.split()[0])

            plot = session_state.clustering_test.document_comparison_plot(content_to_display)
            placeholder_contentbarchart.plotly_chart(plot, use_container_width=True)

            # sub_df = doc_topic_matrix_df.loc[content_to_display]
            # sub_df["tags"] = [x for x in list(df.tags[content_to_display])]
            # cols = list(sub_df.columns.values)
            # cols = cols[0:2] + cols[-1:] + cols[2:-1]

            # placeholder_contentbarcharttable.dataframe(sub_df[cols])
    
    # Topic quality measure
    st.subheader('3.3 Performance per topic (indication of which topics to refine)')
    if st.checkbox('Show performance per topic', value=False):
        st.write('The performance per topic gives information on how well each topic able to reflecting the documents that belong to it. The graph below shows average residual per topic. This number is an estimate of the topic\'s performance. Keep in mind that this is just an estimation; studies show that topics are best to be interpreted by humans.')
        st.write('The HIGHER the residual of a topic, the LOWER the performance. Usually, topics with a high residual value are too general. Thus, the topics on the left side of the graph are reflecting the data the worst. The overall performance of the model is likely to be improved if you pay attention to refining the worst topics (with highest residuals), for example by splitting a topic.')
        ssr, rec_er, residuals_per_doc = session_state.clustering_test.get_residuals()
        residuals_per_topic = session_state.clustering_test.get_residuals_per_topic()
        st.plotly_chart(residuals_per_topic, use_container_width=True)

    # EDA
    st.header('4. Additional Exploratory Data Analysis')

    # EDA Time
    st.subheader('4.1. Cluster development over time')
    default_show_timeplot = False
    default_timeplot_text = "Show visualization: clusters over time"
    timewindow = end_date-start_date
    timewindow_days = timewindow.days
    if timewindow_days > 1000 or session_state.clustering_test.nr_of_topics > 15:
        default_show_timeplot = False
        default_timeplot_text = "Show visualization: clusters over time (warning: high number of topics and/or large time window makes the plot messy)"
    if st.checkbox(default_timeplot_text, value=default_show_timeplot):
        time_line, time_bar = session_state.clustering_test.clusters_over_time_plots(start_date, end_date)
        st.plotly_chart(time_line, use_container_width=True)
        st.plotly_chart(time_bar, use_container_width=True)

    # EDA consumption
    st.subheader('4.2 Content production vs. consumption per topic')
    if st.checkbox('Show plot', value=False):
        consumption_df = get_consumption_data('videos_docs_consumption.json', 'program_yle_id')

        prodconschart = session_state.clustering_test.get_production_graph(consumption_df)
        st.plotly_chart(prodconschart, use_container_width=True)
    
    # Exports Sidebar
    st.sidebar.header("5. Export to CSV")
    if st.sidebar.button('Topic-Term Matrix: Most common words per cluster'):
        st.sidebar.markdown(generate_csv(session_state.clustering_test), unsafe_allow_html=True)

    if st.sidebar.button("Document-Topic Matrix: Document distribution per cluster"):
        df_export = session_state.clustering_test.df_to_export_to_csv()
        # if sys.getsizeof(df_export) <= 45000000: # Streamlit can't handle files bigger than 50 mb, so limit to 45 mb for export
        #     csv = df_export.to_csv()
        #     b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        #     href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
        #     st.sidebar.markdown(href, unsafe_allow_html=True)
        # else: # if the matrix is larger than 45 mb, export the matrix with cluster values in columns, with one row per article
        #     st.sidebar.markdown('The matrix was too large, now it will give you the matrix formatted as in 3.2 (with cluster values in columns)')
        #     csv = doc_topic_matrix_df.to_csv()
        #     b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        #     href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
        #     st.sidebar.markdown(href, unsafe_allow_html=True)
            
        st.sidebar.markdown('The matrix was too large, now it will give you the matrix formatted as in 3.2 (with cluster values in columns)')
        csv = doc_topic_matrix_df.to_csv()
        b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
        st.sidebar.markdown(href, unsafe_allow_html=True)

    if st.sidebar.button('Consumption data'):
        consumption_df = get_consumption_data('videos_docs_consumption.json', 'program_yle_id')
        csv = consumption_df.to_csv()
        b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
        st.sidebar.markdown(href, unsafe_allow_html=True)

    if st.sidebar.button('Cluster input settings and manipulations'):
        data = {
            'Variable': ['Media Type', 'Organizations', 'Clustering based on Tags or full text', 'Individual content IDs excluded', 'Start date', 'End date', 'Excluded words from list', 'Excluded other stopwords (free field)', 'Number of clusters', 'max_df', 'min_df', 'Topic refinement actions'],
            'Value': ['Articles', organizations, 'Full text', exclude_ids, start_date, end_date, manual_stopwords, free_stopwords, session_state.cluster_amount, max_df, min_df, session_state.past_refinements]
        }
        df = pd.DataFrame(data, columns = ['Variable', 'Value'])
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
        st.sidebar.markdown(href, unsafe_allow_html=True)

def run_fulltext(session_state, organizations, start_date, end_date, exclude_ids, use_s3=False, s3_agent=None, dataset=None, config=None):
    if use_s3:
        df = get_data_article_fulltext_s3(organizations, start_date, end_date, exclude_ids, s3_agent, dataset, config)
    else:
        df = get_data_article_fulltext(organizations, start_date, end_date, exclude_ids)
    
    st.header('1. Data Input')
    if st.checkbox("Show dataset", value=1):
        st.dataframe(df[['published_time', 'title', 'text']])

    st.subheader('Information on data input quality')
    st.write('Number of data items: ', df.shape[0])
    if df.shape[0] <= 250:
        st.warning('WARNING: You have selected a small number of data, so clustering results might be poor. Filter out less data or select a bigger time window for better clustering results.')
    if st.checkbox("Show distribution plot of number of tags per article", value=0):
        fig = px.histogram(df, x="Number of words", hover_data=df[['published_time', 'title', 'text', 'Number of words']], title='Distribution - Article text word count')
        fig.update_layout(xaxis_title='Word count', yaxis_title='Frequency')
        fig.update_traces(marker_color='rgb(0, 180, 200)')
        st.plotly_chart(fig, use_container_width=True)

    # INTERACTIVE MODELING

    # Model Configuration

    st.sidebar.header('2. Model Configuration')
    st.sidebar.subheader('2.1. Excluding Words')
    #st.sidebar.markdown('Use the bar chart on the right to inform about the most common words in the dataset inlfuencing the cluster results.')

    clustering_pipeline = clustering_full_transcripts.Pipeline(df)

    # STOPWORDS
    most_common_words, count_dict = clustering_pipeline.get_most_common_words(df, 20)
    most_common_words.extend(DEFAULT_STOPWORDS)

    most_common_words.sort()

    additional_stopwords = []
    manual_stopwords = st.sidebar.multiselect(
        'Use the bar chart on the right to inform about the most common words in the dataset influencing the cluster results. Select words to exclude:',
        most_common_words,
        default=DEFAULT_STOPWORDS)

    exclude_brands = False
    exclude_brands = st.sidebar.checkbox('Exclude article publisher brands', value=True)
    if exclude_brands:
        additional_stopwords.extend(ARTICLE_BRANDS_STOPWORDS)
    exclude_departments = False
    exclude_departments = st.sidebar.checkbox('Exclude article publishing departments', value=True)
    if exclude_brands:
        additional_stopwords.extend(ARTICLE_DEPARTMENT_NAMES)
    exclude_cities = st.sidebar.checkbox('Exclude Finnish cities and regions', value=True)
    if exclude_cities:
        additional_stopwords.extend(CITIES)

    additional_stopwords.extend(manual_stopwords)
    additional_stopwords.extend(ARTICLE_DEFAULT_STOPWORDS)
    
    free_stopwords = st.sidebar.text_input(
        'Write any other stopwords you want to include (separate terms by comma):')
    if "," in free_stopwords:
        additional_stopwords.extend(free_stopwords.replace(" ", "").split(','))
    else:
        additional_stopwords.extend(free_stopwords.split())

    clustering_pipeline.additional_stopwords = additional_stopwords

    st.sidebar.subheader('2.2. Model Hyperparameters')
    max_df = st.sidebar.slider(
        "Set the maximum df (ignore terms that have a document frequency higher than the given threshold, as proportion of documents):",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.01,
        format='%.4f')
    min_df = st.sidebar.slider(
        "Set the minimum df (ignore terms that have a document frequency lower than the given threshold, as proportion of documents):",
        min_value=0.0,
        max_value=0.01,
        value=0.0005,
        step=0.0001,
        format='%.4f')
    clustering_pipeline.config["tfidf"]["min_df"] = min_df
    clustering_pipeline.config["tfidf"]["max_df"] = max_df
    clustering_pipeline.config["countvectorizer"]["min_df"] = min_df
    clustering_pipeline.config["countvectorizer"]["max_df"] = max_df


    st.header('2. Model Configuration Figures')
    st.write("The cluster model generates clusters or topics from words in the documents. Words that occur very often in documents, have a big influence on the model. If these words are general words that have nothing to do with the actual topic or meaning of the document, they might have a negative influence on the clusters that are automatically derived. Therefore, you are given the option to exclude words from the model. Words that occur most often, and words that occur in the most documents, can be analysed here. Based on these visualizations, decide whether you want to exclude some of these words. In the left panel you can select or type words to exclude, or change the df (document frequency) parameter to exclude words that occur in many documents.")
    st.subheader('2.1. Most common words')

    # bar chart with most common words (goal is to see if there is any outliers,
    #   words that should be taken into stopword list)
    placeholder_checkbox_show_common_words_graph = st.empty()
    placeholder_dictionery_text = st.empty()
    placeholder_common_words_graph = st.empty()

    show_mostcommonwordsbarchart = True
    if session_state.clustering_done_on == 'article_1441_fulltext':
        show_mostcommonwordsbarchart = False
    
    if placeholder_checkbox_show_common_words_graph.checkbox('Show bar chart with most common words', value=show_mostcommonwordsbarchart):
        placeholder_dictionery_text.markdown(
            'Use the following figure to decide if you want to exclude words from influencing the topic clusters. Examples are words like and "uutiset" and "Yle". You can write the words you want to exclude in the input field in the sidebar.'
        )
        words_to_plot = [item for item in most_common_words if item not in additional_stopwords]
        count_dict = [(item, count) for item, count in count_dict if item in words_to_plot]
        placeholder_common_words_graph.plotly_chart(clustering_pipeline.plotly_visualize_most_common_words(df, 15, words_to_plot, count_dict), use_container_width=True)

    # print df help plot
    st.subheader('2.2. Document Frequency (df) optimalization help')
    placeholder_checkbox_show_df_graph = st.empty()
    placeholder_df_text = st.empty()
    placeholder_df_df = st.empty()

    if placeholder_checkbox_show_df_graph.checkbox('Show df parameter selection help', value=False):
        placeholder_df_text.markdown(
            'This table shows the document frequency (df) of the words in the dataset. This is a value between 0 and 1, a ratio in how many documents the word is present. Words that have a high df value (occur in a lot of documents), are potentially "stopwords", and can be filtered out (in the left panel).'
        )
        df_df = clustering_pipeline.plot_df_help()
        placeholder_df_df.dataframe(df_df.style.format({'df': '{:.4f}'}))

    # if st.sidebar.button('Help on choosing the optimal amount of topics (calculation takes some time...)'):
    #     st.sidebar.markdown('Results appear on the right.')
    #     logger.info("Calculating optimal number of clusters...")

    #     #clustering_pipeline.build_w2v()

    #     if session_state.w2v_articlesfulltext == None:
    #         cwd = os.getcwd()
    #         path = cwd + '/data/w2v/w2v-model-articlefulltext1441.bin'
    #         session_state.w2v_articlesfulltext = KeyedVectors.load(path)

    #     k = []
    #     coherences = []
    #     for i in range(5, 25):
    #         clustering_pipeline.fit_transform(topics=i)
    #         coherence = clustering_pipeline.get_coherence(i, session_state.w2v_articlesfulltext)
    #         coherences.append(coherence)
    #         k.append(i)

    #     d = {'k':k, 'coherence': coherences}
    #     df_coherence = pd.DataFrame(data=d)

    #     st.subheader('2.3 Mean coherence of the desciptors per amount of topics')
    #     # st.dataframe(df_coherence)

    #     st.write('This graph shows the coherence per number of topics. Generally, higher the coherence, the better the topic describes one group of documents. A low coherence means that the extracted topics might be too general. It is recommended to choose a number of topics with a high coherence measure.')
    #     fig = px.line(df_coherence, x="k", y="coherence", title='Topic coherence per number of topics')
    #     fig.update_layout(xaxis_title='Number of topics', yaxis_title='Coherence')
    #     fig.update_layout(xaxis = dict(tickfont = dict(size = 10)), yaxis = dict(tickfont = dict(size = 10)))
    #     st.plotly_chart(fig, use_container_width=True)

    topics_amount = st.sidebar.number_input("Select how many topics you want to extract", 2, 50, 5, 1)

    # with st.spinner("Loading clustering..."):
    if st.sidebar.button('RUN'):
        logger.info("New clustering initialized...")
        clustering_pipeline.fit_transform(topics=topics_amount)
        session_state.cluster_amount = topics_amount
        session_state.clustering_test = clustering_pipeline
        session_state.clustering_done_on = dataset
        session_state.merged = False
        session_state.splitted = False

        # st.dataframe(clustering_pipeline.nmf_matrix)
        # st.dataframe(clustering_pipeline.nmf_components)
    
    if session_state.clustering_done_on != dataset:
        return

    # Topic Refinement

    st.sidebar.header('3. Topic Refinement')

    # # REMOVING KEYWORD
    want_to_remove_keyword = st.sidebar.button("I want to remove a keyword from a topic")
    if want_to_remove_keyword:
        session_state.in_removingkeyword_process = True
    if session_state.in_removingkeyword_process:
        session_state.clustering_test, session_state = remove_keyword(session_state.clustering_test)
    
    # MERGING
    if not session_state.in_removingkeyword_process:
        want_to_merge = st.sidebar.button("I want to merge two clusters")
        if want_to_merge: 
            session_state.in_merging_process = True
        if session_state.in_merging_process:
            session_state.clustering_test, session_state.cluster_amount, session_state = merge_clusters(session_state.clustering_test, session_state.cluster_amount)

    # SPLITTING
    if not session_state.in_merging_process and not session_state.in_removingkeyword_process:
        want_to_split = st.sidebar.button("I want to split a cluster")
        if want_to_split:
            session_state.in_splitting_process = True
        if session_state.in_splitting_process:
            session_state.clustering_test, session_state.cluster_amount, session_state = split_clusters(session_state.clustering_test, session_state.cluster_amount)

    # RENAME TOPIC
    if not session_state.in_merging_process and not session_state.in_removingkeyword_process and not session_state.in_splitting_process:
        want_to_rename = st.sidebar.button("I want to rename a cluster")
        if want_to_rename:
            session_state.in_renaming_process = True
        if session_state.in_renaming_process:
            session_state.clustering_test, session_state = rename_cluster(session_state.clustering_test)

    # # DELETION
    # if not session_state.in_merging_process:
    #     want_to_delete = st.sidebar.button("I want to delete a cluster")
    #     if want_to_delete:
    #         session_state.in_deleting_process = True
    #     if session_state.in_deleting_process:
    #         session_state.clustering_test, session_state.cluster_amount, session_state = delete_cluster(session_state.clustering_test, session_state.cluster_amount)


    st.header('3. Model Output Analysis Figures')

    st.subheader('3.1. Topic-Term Perspective')

    placeholder_topic_term_table = st.empty()
    if st.checkbox("Show Topic-Term Table: Most frequent words (rows) per cluster (columns)", value=1):
        topic_term_table = session_state.clustering_test.dataframe_top_words(20)
        #placeholder_topic_term_table.dataframe(topic_term_table)
        st.dataframe(topic_term_table)

    if st.button("Generate interactive cluster visualization (topic point of view, to explore topic and keyword relations) (warning: this takes some time)"):
        visualizations.get_lda_vis(session_state.clustering_test)

    # if st.button("Open the latest created visualization"):
    #     list_of_files = glob.glob(os.getcwd()+'/visualizations/*.html')
    #     if len(list_of_files) < 1:
    #         st.write('There are no visualizations to show')
    #     else:
    #         url = max(list_of_files, key=os.path.getctime)
    #         url = 'file://'+os.path.abspath(url)
    #         webbrowser.open_new_tab(url)

    # document-topic matrix
    st.subheader('3.2. Document-Topic Perspective')

    if st.checkbox("Show document-topic matrix: comparison of individual content", value=1):
        doc_topic_matrix_df = session_state.clustering_test.doc_topic_matrix_df
        # doc_topic_matrix_df['id'] = df.index
        # cols = doc_topic_matrix_df.columns.tolist()
        # cols = cols[-1:] + cols[:-1]
        # st.dataframe(doc_topic_matrix_df[cols])
        # st.dataframe(doc_topic_matrix_df.style.background_gradient(cmap=cm)) #coloring costs too much time
        st.dataframe(doc_topic_matrix_df)

    if st.button("Generate interactive document-cluster visualization (individual document point of view) (warning: this takes some time)"):
        visualizations.get_doctop_vis(session_state.clustering_test, media='articles')

    if st.checkbox("Show interactive visualization: compare distribution over topics of individual documents", value=0):
        article_id_list = df.index.values.tolist()
        article_title_list = df['title'].tolist()
        show = [
            str(m) + ' - ' + n for m, n in zip(article_id_list, article_title_list)
        ]
        content_to_display_with_title = st.multiselect(
            "Select articles to compare and show in bar chart", show, default=show[:10])
        content_to_display = []

        placeholder_contentbarchart = st.empty()
        placeholder_contentbarcharttable = st.empty()

        if len(content_to_display_with_title) < 1 or len(content_to_display_with_title) > 20:
            placeholder_contentbarchart.text('Select up to 20 content items to compare')
        else:
            for i in content_to_display_with_title:
                content_to_display.append(i.split()[0])

            plot = session_state.clustering_test.document_comparison_plot(content_to_display)
            placeholder_contentbarchart.plotly_chart(plot, use_container_width=True)

            # sub_df = doc_topic_matrix_df.loc[content_to_display]
            # sub_df["tags"] = [x for x in list(df.tags[content_to_display])]
            # cols = list(sub_df.columns.values)
            # cols = cols[0:2] + cols[-1:] + cols[2:-1]

            # placeholder_contentbarcharttable.dataframe(sub_df[cols])

    # Topic quality measure
    st.subheader('3.3 Performance per topic (indication of which topics to refine)')
    if st.checkbox('Show performance per topic', value=False):
        st.write('The performance per topic gives information on how well each topic able to reflecting the documents that belong to it. The graph below shows average residual per topic. This number is an estimate of the topic\'s performance. Keep in mind that this is just an estimation; studies show that topics are best to be interpreted by humans.')
        st.write('The HIGHER the residual of a topic, the LOWER the performance. Usually, topics with a high residual value are too general. Thus, the topics on the left side of the graph are reflecting the data the worst. The overall performance of the model is likely to be improved if you pay attention to refining the worst topics (with highest residuals), for example by splitting a topic.')
        ssr, rec_er, residuals_per_doc = session_state.clustering_test.get_residuals()
        residuals_per_topic = session_state.clustering_test.get_residuals_per_topic()
        st.plotly_chart(residuals_per_topic, use_container_width=True)

    # EDA
    st.header('4. Additional Exploratory Data Analysis')

    # EDA Time
    st.subheader('4.1. Cluster development over time')
    default_show_timeplot = False
    default_timeplot_text = "Show visualization: clusters over time"
    timewindow = end_date-start_date
    timewindow_days = timewindow.days
    if timewindow_days > 1000 or session_state.clustering_test.nr_of_topics > 15:
        default_show_timeplot = False
        default_timeplot_text = "Show visualization: clusters over time (warning: high number of topics and/or large time window makes the plot messy)"
    if st.checkbox(default_timeplot_text, value=default_show_timeplot):
        time_line, time_bar = session_state.clustering_test.clusters_over_time_plots(start_date, end_date)
        st.plotly_chart(time_line, use_container_width=True)
        st.plotly_chart(time_bar, use_container_width=True)

    # EDA consumption
    st.subheader('4.2 Content production vs. consumption per topic')
    if st.checkbox('Show plot', value=False):
        consumption_df = get_consumption_data('articles_consumption.json', 'article_yle_id')

        prodconschart = session_state.clustering_test.get_production_graph(consumption_df)
        st.plotly_chart(prodconschart, use_container_width=True)

    # Exports Sidebar
    st.sidebar.header("5. Export to CSV")
    if st.sidebar.button('Topic-Term Matrix: Most common words per cluster'):
        st.sidebar.markdown(generate_csv(session_state.clustering_test), unsafe_allow_html=True)

    if st.sidebar.button("Document-Topic Matrix: Document distribution per cluster"):
        df_export = session_state.clustering_test.df_to_export_to_csv()
        # if sys.getsizeof(df_export) <= 45000000: # Streamlit can't handle files bigger than 50 mb, so limit to 45 mb for export
        #     csv = df_export.to_csv()
        #     b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        #     href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
        #     st.sidebar.markdown(href, unsafe_allow_html=True)
        # else: # if the matrix is larger than 45 mb, export the matrix with cluster values in columns, with one row per article
        #     st.sidebar.markdown('The matrix was too large, now it will give you the matrix formatted as in 3.2 (with cluster values in columns)')
        #     csv = doc_topic_matrix_df.to_csv()
        #     b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        #     href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
        #     st.sidebar.markdown(href, unsafe_allow_html=True)

        st.sidebar.markdown('The matrix was too large, now it will give you the matrix formatted as in 3.2 (with cluster values in columns)')
        csv = doc_topic_matrix_df.to_csv()
        b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
        st.sidebar.markdown(href, unsafe_allow_html=True)    
            
    if st.sidebar.button('Consumption data'):
        consumption_df = get_consumption_data('articles_consumption.json', 'article_yle_id')
        csv = consumption_df.to_csv()
        b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
        st.sidebar.markdown(href, unsafe_allow_html=True)

    if st.sidebar.button('Cluster input settings and manipulations'):
        data = {
            'Variable': ['Media Type', 'Organizations', 'Clustering based on Tags or full text', 'Individual content IDs excluded', 'Start date', 'End date', 'Excluded words from list', 'Excluded other stopwords (free field)', 'Number of clusters', 'max_df', 'min_df', 'Topic refinement actions'],
            'Value': ['Articles', organizations, 'Full text', exclude_ids, start_date, end_date, manual_stopwords, free_stopwords, session_state.cluster_amount, max_df, min_df, session_state.past_refinements]
        }
        df = pd.DataFrame(data, columns = ['Variable', 'Value'])
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
        st.sidebar.markdown(href, unsafe_allow_html=True)

def run_article_tags(session_state, include_article_organizations, start_date, end_date, exclude_ids, use_s3=False, s3_agent=None, dataset=None, config=None):
    if use_s3:
        df = get_data_article_tags_s3(include_article_organizations, start_date, end_date, exclude_ids, s3_agent, dataset, config)
    else:
        df = get_data_article_tags(include_article_organizations, start_date, end_date, exclude_ids)

    st.header('1. Data Input')
    if st.checkbox("Show dataset", value=1):
        st.dataframe(df[['published_time', 'title', 'terms']])

    st.subheader('Information on data input quality')
    st.write('Number of data items: ', df.shape[0])
    if df.shape[0] <= 250:
        st.warning('WARNING: You have selected a small number of data, so clustering results might be poor. Filter out less data or select a bigger time window for better clustering results.')
    if st.checkbox("Show distribution plot of number of tags per article", value=0):
        fig = px.histogram(df, x="Number of tags", hover_data=df[['published_time', 'title', 'terms', 'Number of tags']], title='Distribution - Article tag count')
        fig.update_layout(xaxis_title='Tag count', yaxis_title='Frequency')
        fig.update_traces(marker_color='rgb(0, 180, 200)')
        st.plotly_chart(fig, use_container_width=True)

    # INTERACTIVE MODELING

    # Model Configuration

    st.sidebar.header('2. Model Configuration')
    st.sidebar.subheader('2.1. Excluding Words')

    clustering_pipeline = clustering.Pipeline(df)

    # STOPWORDS
    most_common_words, count_dict = clustering_pipeline.get_most_common_words(df, 40)

    additional_stopwords = []
    manual_stopwords = st.sidebar.multiselect(
        'Use the bar chart on the right to inform about the most common words in the dataset influencing the cluster results. Select words to exclude:',
        most_common_words[:20],
        default=DEFAULT_STOPWORDS)
    additional_stopwords.extend(manual_stopwords)
    
    free_stopwords = st.sidebar.text_input(
        'Write any other stopwords you want to include (separate terms by comma):')
    if "," in free_stopwords:
        additional_stopwords.extend(free_stopwords.replace(" ", "_").split(','))
    else:
        additional_stopwords.extend(free_stopwords.replace(" ", "_").split())

    # clustering_pipeline.additional_stopwords = additional_stopwords

    # additional_stopwords = []
    # additional stopword selection from all possible tags (1) or from most common tags (2)?
    # 1: all possible tags
    # manual_stopwords = st.sidebar.multiselect(
    #     'Use the bar chart on the right to inform about the most common words in the dataset influencing the cluster results. Select words to exclude:',
    #     clustering_pipeline.list_of_strings,
    #     default=DEFAULT_STOPWORDS)
    # additional_stopwords.extend(manual_stopwords)
    # 2: only most common tags
    # most_common_words, _ = clustering_pipeline.get_most_common_words(df, 10)
    # most_common_words.extend(DEFAULT_STOPWORDS)
    # most_common_words.sort()
    # additional_stopwords = st.sidebar.multiselect(
    #     'Select words to exclude',
    #     most_common_words,
    #     default=DEFAULT_STOPWORDS)

    exclude_brands = False
    exclude_brands = st.sidebar.checkbox('Exclude article publisher brands', value=True)
    if exclude_brands:
        additional_stopwords.extend(ARTICLE_BRANDS_STOPWORDS)
    exclude_departments = False
    exclude_departments = st.sidebar.checkbox('Exclude article publishing departments', value=True)
    if exclude_brands:
        additional_stopwords.extend(ARTICLE_DEPARTMENT_NAMES)
    exclude_cities = st.sidebar.checkbox('Exclude Finnish cities and regions', value=True)
    if exclude_cities:
        additional_stopwords.extend(CITIES)

    additional_stopwords.extend(ARTICLE_DEFAULT_STOPWORDS)

    clustering_pipeline.additional_stopwords = additional_stopwords
    st.header('2. Model Configuration Figures')
    st.subheader('2.1. Most common words')

    # bar chart with most common words (goal is to see if there is any outliers,
    #   words that should be taken into stopword list)
    placeholder_checkbox_show_common_words_graph = st.empty()
    placeholder_dictionery_text = st.empty()
    placeholder_common_words_graph = st.empty()

    st.sidebar.subheader('2.2. Model Hyperparameters')

    show_mostcommonwordsbarchart = False
    if include_article_organizations == 'All':
        if session_state.clustering_done_on != 'article_all_tags':
            show_mostcommonwordsbarchart = True
    else:
        if session_state.clustering_done_on != 'article_1441_tags':
            show_mostcommonwordsbarchart = True
    
    if placeholder_checkbox_show_common_words_graph.checkbox('Show bar chart with most common words', value=show_mostcommonwordsbarchart):
        placeholder_dictionery_text.markdown(
            'Use the following figure to decide if you want to exclude words from influencing the topic clusters. Examples are words like and "uutiset" and "Yle". You can write the words you want to exclude in the input field in the sidebar.'
        )
        words_to_plot = [item for item in most_common_words if item not in additional_stopwords]
        count_dict = [(item, count) for item, count in count_dict if item in words_to_plot]
        placeholder_common_words_graph.plotly_chart(clustering_pipeline.plotly_visualize_most_common_words(df, 15, words_to_plot, count_dict), use_container_width=True)

    # if session_state.w2v_tags == None:
    #     cwd = os.getcwd()
    #     path = cwd + '/data/w2v/w2v-model-tags1441.bin'
    #     session_state.w2v_tags = KeyedVectors.load(path)

    # if st.sidebar.button('Help on choosing the optimal amount of topics (calculation takes some time...)'):
    #     st.sidebar.markdown('Results appear on the right.')
    #     logger.info("Calculating optimal number of clusters...")

    #     k = []
    #     coherences = []
    #     for i in range(5, 25):
    #         clustering_pipeline.fit_transform(topics=i)
    #         coherence = clustering_pipeline.get_coherence(i, session_state.w2v_tags)
    #         coherences.append(coherence)
    #         k.append(i)

    #     d = {'k':k, 'coherence': coherences}
    #     df_coherence = pd.DataFrame(data=d)

    #     st.subheader('2.2 Mean coherence of the desciptors per amount of topics')
    #     # st.dataframe(df_coherence)

    #     st.write('This graph shows the coherence per number of topics. Generally, higher the coherence, the better the topic describes one group of documents. A low coherence means that the extracted topics might be too general. It is recommended to choose a number of topics with a high coherence measure.')
    #     fig = px.line(df_coherence, x="k", y="coherence", title='Topic coherence per number of topics')
        # fig.update_layout(xaxis_title='Number of topics', yaxis_title='Coherence')
        # fig.update_layout(xaxis = dict(tickfont = dict(size = 10)), yaxis = dict(tickfont = dict(size = 10)))
    #     st.plotly_chart(fig, use_container_width=True)

    topics_amount = st.sidebar.number_input("Select how many topics you want to extract", 2, 50, 5, 1)

        # k = []
        # ssrs = []
        # rec_ers = []
        # for i in range(5, 25):
        #     clustering_pipeline.fit_transform(topics=i)
        #     ssr, rec_er, residuals_per_doc = clustering_pipeline.get_residuals()
        #     k.append(i)
        #     ssrs.append(ssr)
        #     rec_ers.append(rec_er)
        # d = {'k':k, 'ssr': ssrs, 'reconstruction error': rec_ers}
        # df_ssr = pd.DataFrame(data=d)

        # session_state.suggest = True

    # if session_state.suggest:
    #     st.subheader('2.2 Sum of Squared Residuals per k')
    #     st.dataframe(df_ssr)

    #     fig = px.line(df_ssr, x="k", y="ssr", title='SSR')
    #     st.plotly_chart(fig, use_container_width=True)
    #     fig = px.line(df_ssr, x="k", y="reconstruction error", title='reconstruction error')
    #     st.plotly_chart(fig, use_container_width=True)


    # with st.spinner("Loading clustering..."):
    if st.sidebar.button('RUN'):
        logger.info("New clustering initialized...")
        clustering_pipeline.fit_transform(topics=topics_amount)
        session_state.cluster_amount = topics_amount
        session_state.clustering_test = clustering_pipeline
        session_state.merged = False
        session_state.splitted = False

        if include_article_organizations == 'All':
            session_state.clustering_done_on = 'article_all_tags'
        else:
            session_state.clustering_done_on = 'article_1441_tags'

    if include_article_organizations == 'All':
        if session_state.clustering_done_on != 'article_all_tags':
            return
    else:
        if session_state.clustering_done_on != 'article_1441_tags':
            return

    # Topic Refinement

    st.sidebar.header('3. Topic Refinement')

    # # REMOVING KEYWORD
    want_to_remove_keyword = st.sidebar.button("I want to remove a keyword from a topic")
    if want_to_remove_keyword:
        session_state.in_removingkeyword_process = True
    if session_state.in_removingkeyword_process:
        session_state.clustering_test, session_state = remove_keyword(session_state.clustering_test)
    
    # MERGING
    if not session_state.in_removingkeyword_process:
        want_to_merge = st.sidebar.button("I want to merge two clusters")
        if want_to_merge: 
            session_state.in_merging_process = True
        if session_state.in_merging_process:
            session_state.clustering_test, session_state.cluster_amount, session_state = merge_clusters(session_state.clustering_test, session_state.cluster_amount)

    # SPLITTING
    if not session_state.in_merging_process and not session_state.in_removingkeyword_process:
        want_to_split = st.sidebar.button("I want to split a cluster")
        if want_to_split:
            session_state.in_splitting_process = True
        if session_state.in_splitting_process:
            session_state.clustering_test, session_state.cluster_amount, session_state = split_clusters(session_state.clustering_test, session_state.cluster_amount)

    # RENAME TOPIC
    if not session_state.in_merging_process and not session_state.in_removingkeyword_process and not session_state.in_splitting_process:
        want_to_rename = st.sidebar.button("I want to rename a cluster")
        if want_to_rename:
            session_state.in_renaming_process = True
        if session_state.in_renaming_process:
            session_state.clustering_test, session_state = rename_cluster(session_state.clustering_test)

    st.header('3. Model Output Analysis Figures')

    st.subheader('3.1. Topic-Term Perspective')

    placeholder_topic_term_table = st.empty()
    if st.checkbox("Show Topic-Term Table: Most frequent words (rows) per cluster (columns)", value=1):
        topic_term_table = session_state.clustering_test.dataframe_top_words(20)
        #placeholder_topic_term_table.dataframe(topic_term_table)
        st.dataframe(topic_term_table)

    if st.button("Generate interactive cluster visualization (topic point of view, to explore topic and keyword relations) (warning: this takes some time)"):
        visualizations.get_lda_vis(session_state.clustering_test)

    # if st.button("Open the latest created visualization"):
    #     list_of_files = glob.glob(os.getcwd()+'/visualizations/*.html')
    #     if len(list_of_files) < 1:
    #         st.write('There are no visualizations to show')
    #     else:
    #         url = max(list_of_files, key=os.path.getctime)
    #         url = 'file://'+os.path.abspath(url)
    #         webbrowser.open_new_tab(url)

    # document-topic matrix
    st.subheader('3.2. Document-Topic Perspective')

    if st.checkbox("Show document-topic matrix: comparison of individual content", value=1):
        doc_topic_matrix_df = session_state.clustering_test.doc_topic_matrix_df
        st.dataframe(doc_topic_matrix_df)
        #t.dataframe(doc_topic_matrix_df.style.background_gradient(cmap=cm)) #coloring the graph costs too much time.. 

    if st.button("Generate interactive document-cluster visualization (individual document point of view) (warning: this takes some time)"):
        visualizations.get_doctop_vis(session_state.clustering_test, media='articles')

    if st.checkbox("Show interactive visualization: compare distribution over topics of individual documents", value=0):
        article_id_list = df.index.values.tolist()
        article_title_list = df['title'].tolist()
        show = [
            str(m) + ' - ' + n for m, n in zip(article_id_list, article_title_list)
        ]
        content_to_display_with_title = st.multiselect(
            "Select articles to compare and show in bar chart", show, default=show[:10])
        content_to_display = []

        placeholder_contentbarchart = st.empty()
        placeholder_contentbarcharttable = st.empty()

        if len(content_to_display_with_title) < 1 or len(content_to_display_with_title) > 20:
            placeholder_contentbarchart.text('Select up to 20 content items to compare')
        else:
            for i in content_to_display_with_title:
                content_to_display.append(i.split()[0])

            plot = session_state.clustering_test.document_comparison_plot(content_to_display)
            placeholder_contentbarchart.plotly_chart(plot, use_container_width=True)

            # sub_df = doc_topic_matrix_df.loc[content_to_display]
            # sub_df["tags"] = [x for x in list(df.tags[content_to_display])]
            # cols = list(sub_df.columns.values)
            # cols = cols[0:2] + cols[-1:] + cols[2:-1]

            # placeholder_contentbarcharttable.dataframe(sub_df[cols])

    # EDA
    st.header('4. Additional Exploratory Data Analysis')

    # EDA Time
    st.subheader('4.1. Cluster development over time')
    default_show_timeplot = False
    default_timeplot_text = "Show visualization: clusters over time"
    timewindow = end_date-start_date
    timewindow_days = timewindow.days
    if timewindow_days > 1000 or session_state.clustering_test.nr_of_topics > 15:
        default_show_timeplot = False
        default_timeplot_text = "Show visualization: clusters over time (warning: high number of topics and/or large time window makes the plot messy)"
    
    if st.checkbox(default_timeplot_text, value=default_show_timeplot):
        time_line, time_bar = session_state.clustering_test.clusters_over_time_plots(start_date, end_date)
        st.plotly_chart(time_line, use_container_width=True)
        st.plotly_chart(time_bar, use_container_width=True)

    # EDA Consumption
    st.subheader('4.2 Content production vs. consumption per topic')
    if st.checkbox('Show plot', value=False):
        if include_article_organizations == '1441_ajankohtaiset':
            consumption_df = get_consumption_data('articles_consumption.json', 'article_yle_id')
        else: # all
            consumption_df = get_consumption_data('articles_consumption.json', 'article_yle_id')

        prodconschart = session_state.clustering_test.get_production_graph(consumption_df)
        st.plotly_chart(prodconschart, use_container_width=True)

    # Data export sidebar
    st.sidebar.header("5. Export to CSV")
    if st.sidebar.button('Topic-Term Matrix: Most common words per cluster'):
        st.sidebar.markdown(generate_csv(session_state.clustering_test), unsafe_allow_html=True)

    if st.sidebar.button("Document-Topic Matrix: Document distribution per cluster"):
        df_export = session_state.clustering_test.df_to_export_to_csv()
        # if sys.getsizeof(df_export) <= 45000000: # Streamlit can't handle files bigger than 50 mb, so limit to 45 mb for export
        #     csv = df_export.to_csv()
        #     b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        #     href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
        #     st.sidebar.markdown(href, unsafe_allow_html=True)
        # else: # if the matrix is larger than 45 mb, export the matrix with cluster values in columns, with one row per article
        #     st.sidebar.markdown('The matrix was too large, now it will give you the matrix formatted as in 3.2 (with cluster values in columns)')
        #     csv = doc_topic_matrix_df.to_csv()
        #     b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        #     href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
        #     st.sidebar.markdown(href, unsafe_allow_html=True)
            
        st.sidebar.markdown('The matrix was too large, now it will give you the matrix formatted as in 3.2 (with cluster values in columns)')
        csv = doc_topic_matrix_df.to_csv()
        b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
        st.sidebar.markdown(href, unsafe_allow_html=True)

    if st.sidebar.button('Consumption data'):
        if include_article_organizations == '1441_ajankohtaiset':
            consumption_df = get_consumption_data('articles_consumption.json', 'article_yle_id')
        else: # all
            consumption_df = get_consumption_data('articles_consumption.json', 'article_yle_id')
        csv = consumption_df.to_csv()
        b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
        st.sidebar.markdown(href, unsafe_allow_html=True)

    if st.sidebar.button('Cluster input settings and manipulations'):
        data = {
            'Variable': ['Media Type', 'Organizations', 'Clustering based on Tags or full text', 'Individual content IDs excluded', 'Start date', 'End date', 'Excluded words from list', 'Excluded other stopwords (free field)', 'Number of clusters', 'Topic refinement actions'],
            'Value': ['Articles', include_article_organizations, 'Full text', exclude_ids, start_date, end_date, manual_stopwords, free_stopwords, session_state.cluster_amount, session_state.past_refinements]
        }
        df = pd.DataFrame(data, columns = ['Variable', 'Value'])
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
        st.sidebar.markdown(href, unsafe_allow_html=True)

def print_how_to_use_text():
    st.markdown('This application has four sections, each representing one step in the modeling process.')
    st.markdown('1. **Select the data input**. The first step is to select the data you want to model. Use the selection boxes in the left panel to select the source, filter on metadata and time window. You may want to use the tables and figure in this main panel to confirm you have chosen the data that you want. A warning will be given when you have selected data of poor quality (when there is not enough data to model on). You can also make an estimate on the quality yourself by checking the number of contents and a distribution figure of the text in the content.')
    st.markdown('2. **Optimize the data**. The model that derives topics from content is highly influenced by words that occur very frequently in the dataset. If these words are general words that don\'t tell anything about the meaning of a document, then this might have a negative impact on the result. Most of these *\'stopwords\'* are filtered out automatically, but you can filter out more manually. Use the figures about the most frequent words in the entire dataset (2.1) and in how many documents each words occurs (*document frequency*, *df*, 2.2, only for non-tag modeling).  \n Once you are satisfied with the data input, **select the number of topics** you would like to have, and **run the model**. If you are not sure, you can ask the model for calculating the optimal amount of topics (only for non-tags). This calculation will take some time, since it calculates the performance for 5-25 amount of topics. This performance measure (topic coherence) is based on how well the top words describe the actual topic. Thus, the more general the topic, the lower the score. In practice, this will mean that deriving a larger number of topics will often have a better performance, because the topics are more specific. Thus, it is best to determine the right amount of topics yourself, based on your modeling goal.')
    st.markdown('3. **Output analysis**  \n \
        Once you have run the model, output analysis figures will appear in the main panel. Use these figures to determine if you are satisfied with the topics. The quality of the topics derived from the data is represented in three ways:  \n \
            **3.1 Topic-Term Perspective**. This tells you the most about the topics themselves. The words that occur most frequently in each topic are given in a table, allowing you to make a quick eye-balling estimate of the topic quality. If you want to explore the relations between topics and its terms, then you can generate an additional visualization, which will be shown in the main panel only after pressing the button. Make sure to save the picture if you want to use it later on, since it takes some time to generate this again.  \n \
            **3.2 Document-Topic Perspective**. A second way of validating whether the modeling was successful, is by analysing how individual pieces of content are classified right. For each document (video or article), a normalized distribution over the topics is calculated (how much the topic is represented in that document). Again, in this panel, only the distribution of individual documents is visualized, and you can analyze the relations between documents by an extra visualization that will be rendered by pressing a button.  \n \
            **3.3 Topic quality measure** (not for tags). The quality of individual topics is estimated by the residuals (the difference between the actual words in a document versus the predicted values by the topic model). The higher the residual, the worse a topic is thus representing the data. Try to focus on refining topics with a high residual, instead of the topics with a lower value.  \n \
        \
        **  \n Model refinement**  \n \
            You can make changes to the model in two ways:  \n \
            **- Model hyperparameters**. Exclude or include words and change the number of topics to derive, and run the model again. Keep in mind that a the previous model will be discarded when you change anything and press the \'RUN\' button again.  \n \
            **- Topic refinement**. You can make changes to the individual topics in section 3 of the left panel. You can remove specific words from a topic, merge two topics, split a topic, and rename topics. The model automatically runs again once you apply the changes.')
    st.markdown('4. **Additional exploratory data analysis**  \n \
        When you are happy with the results, you can do some preliminary data analysis within this application. This includes visualizations of content production by topics over time, and comparison with content consumption by audience by different age groups.')
    st.markdown('5. **Data and settings export**  \n \
        If you want to do further data analysis, you can export the topics with the top terms, the distribution of topics over the contents and the audience consumption data as csv files. Additionally, you can export the final settings as well as all your manual operations you made during the modeling as csv file.')
    st.subheader('Tips')
    st.markdown('1. Press the \'RUN\' button if you changed anything in the data input (step 1 and 2), to see the new modeling results. Keep in mind that this discards your previous results.')
    st.markdown('2. Every plot is interactive, and can be downloaded as PNG or HTML directly in the upper left corner of the plot. The best quality is saved when you first enlarge the plot, and then save it.')
    st.markdown('3. Calculated performance measures (coherence and residuals) are *estimates*. They are presented in this application to assist you in the modeling process, but keep in mind that your own judgement of the topics is always the best measure. (We are modeling natural language data, which is made for human understanding, not machine understanding. Therefore, the best interpretation is also done by humans, not by machines).')
    st.markdown('4. Only interact with the application is not \'*running*\'. When the model is running, this is shown in the upper right corner. Interrupting this might cause an automatic restart of the application, meaning that you lose the current modeling process.')

def run(config):
    """ Builds the UI"""

    global session_state

    custom_ui.sidebar_settings()

    # Some textual explanation for the user about the application
    st.title('Customer Profile Clustering Application')
    st.markdown('THIS APPLICATION IS UNDER DEVELOPMENT, PLEASE USE THIS VERSION FOR TESTING PURPOSES ONLY! For questions or issues, ping [me](mailto:laura.ham@yle.fi), or let me know on [Github](https://github.com/Yleisradio/customer-profile-api/issues).')
    st.header('About this application')
    st.markdown('This is a tool to automatically derive topics from a selection of production content. This may be a subset of all videos or articles, by filtering on metadata and time of production. You can use this tool to explore themes in that appear in the selected content, and use these themes for further data analysis, like trend discovery or comparison with consumption data.')
    st.header('How to use')
    if st.checkbox('Show How to use and Tips', value=False):
        print_how_to_use_text()
        
    # DATASET INPUT
    st.sidebar.header("1. Data input")
    st.sidebar.subheader('1.1 Select data source')

    media_type = st.sidebar.selectbox('Select what media to include in the clustering:', POSSIBLE_MEDIA_TYPES, index=0)

    if media_type == 'Articles':
        # Filters for articles
        
        # Publishing organization
        include_article_organizations = st.sidebar.selectbox(
            'Select what article organizations to include in the clustering',
            POSSIBLE_ARTICLE_ORGANIZATIONS,
            index=0)

        # Tags or full text
        use_tags = True
        if include_article_organizations == '1441_ajankohtaiset':
            if st.sidebar.selectbox('I want to cluster based on:', ('Tags', 'Full text'), index=0) != 'Tags':
                use_tags = False
        else:
            if st.sidebar.selectbox('I want to cluster based on:', (['Tags']), index=0) != 'Tags':
                use_tags = False

        # exclude individual document
        exclude_ids = st.sidebar.text_input(
            'ID of individual content to exclude (separated by comma):')
        if "," in exclude_ids:
            exclude_ids = exclude_ids.replace(" ", "").split(',')
        else:
            exclude_ids = exclude_ids.split()
        
        # Time window
        st.sidebar.subheader('1.2 Select time window')
        placeholder_time_explanation = st.sidebar.empty()
        if include_article_organizations == 'All':
            placeholder_time_explanation.markdown('Possible date range for selected data is 2020/01/01 - 2020/05/11')
            default_startdate = datetime.date(2020, 1, 1)
            default_enddate = datetime.date(2020, 5, 11)
        elif include_article_organizations == '1441_ajankohtaiset':
            if use_tags == True:
                placeholder_time_explanation.markdown('Possible date range for selected data is 2018/01/01 - 2020/05/11')
            else:
                placeholder_time_explanation.markdown('Possible date range for selected data is 2019/01/01 - 2020/04/30')
            default_startdate = datetime.date(2019, 1, 1)
            default_enddate = datetime.date(2020, 4, 30)

        start_date = st.sidebar.date_input(
            "What is the start date for the data in the clustering?",
            default_startdate)
        end_date = st.sidebar.date_input(
            "What is the end date for the data in the clustering?",
            default_enddate)

    else: #media_type == 'Videos':
        # Filters for videos

        # Genre
        video_genre = st.sidebar.selectbox(
            'Select what video genre to cluster on:',
            POSSIBLE_VIDEO_GENRES,
            index=0)

        # Streamable
        streamable = st.sidebar.selectbox(
            'Filter on current availability on Yle Areena',
            ('Include all content', 'Include only currently streamable', 'Include only streamable in past'))

            # exclude individual document
        
        exclude_ids = st.sidebar.text_input(
            'ID of individual content to exclude (separated by comma):')
        if "," in exclude_ids:
            exclude_ids = exclude_ids.replace(" ", "").split(',')
        else:
            exclude_ids = exclude_ids.split()

        # Time window
        st.sidebar.subheader('1.2 Select time window')
        st.sidebar.markdown('Possible date range for selected data is 2016/01/01 - 2020/04/15')
        start_date = st.sidebar.date_input(
            "What is the start date for the data in the clustering?",
            datetime.date(2016, 1, 1))
        end_date = st.sidebar.date_input(
            "What is the end date for the data in the clustering?",
            datetime.date(2020, 4, 15))

    # start Model Configuration run
    if media_type == 'Articles':
        if use_tags:
            run_article_tags(session_state, include_article_organizations, start_date, end_date, exclude_ids)
        else:
            run_fulltext(session_state, include_article_organizations, start_date, end_date, exclude_ids)
    else: #media_type == 'Videos':
        run_videos(session_state, start_date, end_date, exclude_ids, video_genre, streamable)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="Environment to use", default="test")
    return parser.parse_args()

def load_config(env):
    with open("config.yml") as f:
        return yaml.load(f)[env]

if __name__ == "__main__":
    logger.info("Starting..")
    args = parse_args()
    env = args.env
    logger.info(f"Loading application for {env}...")
    run(load_config(env))
