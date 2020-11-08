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
from models import clustering
import pyLDAvis
import webbrowser
import base64

import pytest

cluster_amount = 8
additional_stopwords = ['astudio', 'uutiset', 'kotimaan_uutiset']
min_df = 0.0005
max_df = 1

with open("config.yml") as f:
    config = yaml.load(f)["local"]


def load_data(path, idx_name):
    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df.set_index(idx_name, inplace=True)

    return df


def replace_spaces(stringlist):
    return [
        re.sub("[!@#$--&'()]", '', word).replace(" ", "_").replace("-", "_")
        for word in stringlist
    ]

def test_run():
    return