import requests
#from utils.text_fallback import process_single_text as process_text_fallback
import logging
import json
import pandas as pd
from markdown import Markdown
from io import StringIO

def unmarkdown_element(element, stream=None):
    if stream is None:
        stream = StringIO()
    if element.text:
        stream.write(element.text)
    for sub in element:
        unmarkdown_element(sub, stream)
    if element.tail:
        stream.write(element.tail)
    return stream.getvalue()

POS_KEEPS = set(["PROPN", "ADJ", "NOUN"])
logger = logging.getLogger(__name__)

# patching Markdown
Markdown.output_formats["plain"] = unmarkdown_element
__md = Markdown(output_format="plain")
__md.stripTopLevelTags = False


def _fetch_and_process_text(id_text_tuple, config=None):
    url = config["nlp_api"]
    text_id = id_text_tuple[0]
    text_body = id_text_tuple[1]
    try:
        response = requests.post(
            f"{url}/{text_id}",
            timeout=200,
            json={"data": {"text": text_body, "forced": True}},
        )
        if (
            response is not None
            and response.status_code == 200
            and response.text is not None
        ):
            return _parse_doc(json.loads(response.text))
        else:
            logger.warning(
                f"NLP API returned: status: {response.status_code}, text: {response.text}. Returning empty string"
            )
            return ""
            # return process_text_fallback(text_body)
    except (json.decoder.JSONDecodeError, requests.exceptions.RequestException) as e:
        logger.warning(
            f"Ran into problems reading nlp api: {e}, {text_body} returning empty string."
        )
        return ""
        # return process_text_fallback(text_body)


def _parse_doc(doc):
    parsed = []
    for token in doc:
        if (
            token["pos"] in POS_KEEPS
            and not token["stopword"]
            and not token["like_num"]
        ):
            # hashes are used to represent "yhdys#sana" and we want the full lemmatized word
            parsed.append(token["lemma"].replace("#", ""))
    return " ".join(parsed)


def _unmarkdown(text):
    return __md.convert(text)


def _process_text(config, id_text_tuples):
    processed = [
        str(_fetch_and_process_text(x, config)) for x in id_text_tuples if x is not None
    ]
    return processed


def process_text_df(config, df, text_col, id_col):
    # Remove very short texts
    df = df[df[text_col].str.len() > 5]
    df[text_col] = df[text_col].apply(_unmarkdown)

    # Preprocess text, return '' if returned Doc object was None
    df[text_col] = _process_text(config, zip(df[id_col], df[text_col]))
    return df[pd.notnull(df[text_col])]
