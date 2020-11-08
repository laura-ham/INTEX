import json
import pandas as pd
from utils.redshift import sql_connect
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import logging
import boto3
import botocore

logger = logging.getLogger(__name__)

ARTICLE_IDS_QRY = """
SELECT DISTINCT article_yle_id
FROM pub_yle.v_article_v2
WHERE date_published >= '{min_date}'
AND language = 'fi'
"""

ARTICLE_TEXT_DATE_QRY = """
SELECT 'article' AS type,
       a.article_yle_id AS id,
       t.text_content AS text
FROM pub_yle.v_article_v2 a
LEFT JOIN pub_yle.v_article_content_v2 t
ON a.article_yle_id = t.article_yle_id
WHERE a.date_published >= '{min_date}'
AND a.language = 'fi'
AND t.n_words > 10
"""

ARTICLE_TEXT_ID_QRY = """
SELECT 'article' AS type,
       article_yle_id AS id,
       text_content AS text
FROM pub_yle.v_article_content_v2
WHERE article_yle_id in %s
"""


def get_article_ids_from_redshift_by_date(articles_min_date, config):
    qry_articles = ARTICLE_IDS_QRY.format(min_date=articles_min_date)
    with sql_connect(config) as conn:
        with conn.cursor() as cur:
            cur.execute(qry_articles)
            article_ids_list = [item[0] for item in cur.fetchall()]
    logger.info(f'Found {len(article_ids_list)} article_ids in redshift')
    return article_ids_list


def get_articles_df_from_redshift_by_date(articles_min_date, config):
    qry_articles = ARTICLE_TEXT_DATE_QRY.format(min_date=articles_min_date)
    with sql_connect(config) as conn:
        with conn.cursor() as cur:
            cur.execute(qry_articles)
            return pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])


def get_articles_df_from_redshift_by_id_list(article_ids_list, config):
    with sql_connect(config) as conn:
        with conn.cursor() as cur:
            cur.execute(ARTICLE_TEXT_ID_QRY, (tuple(article_ids_list), ))
            return pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
