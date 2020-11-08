import yaml
import json
import logging
import requests

POS_KEEPS = ["NOUN", "VERB", "ADJ"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_texts(config, texts):
    processed_texts = [None for x in texts]
    url = config["core_nlp_api"]

    # note: we could do this in parallel but since
    # there are few texts, not worth the trouble
    for ix, t in enumerate(texts):
        processed_texts[ix] = _process_single_text(url, t)
    return processed_texts


def _shoutout_count(text):
    return text.count("[")


def _shoutouts(text):
    return [x.replace("]", " | ") for x in text.split("[") if "]" in x]


def _clean(doc):
    return " ".join([
        x["lemma"].replace("#", "") for x in doc
        if not x["stopword"] and not x["like_num"] and x["pos"] in POS_KEEPS
    ])


def _process_single_text(url, t):
    response = requests.post(url,
                             timeout=10,
                             json={"data": {
                                 "text": t,
                                 "forced": True
                             }})
    if (response is not None and response.status_code == 200
            and response.text is not None):
        return _clean(json.loads(response.text))
    else:
        logger.error(response.status_code)
        logger.error(response.text)
        raise ValueError(t)
