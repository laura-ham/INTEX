import psycopg2
import logging

logger = logging.getLogger(__name__)


def sql_connect(config):
    rs_config = config["redshift"]
    return psycopg2.connect(
        dbname=rs_config["db"],
        host=rs_config["host"],
        port=rs_config["port"],
        user=rs_config["user"],
        password=rs_config["password"],
    )
