import nltk
import pandas as pd
try:
    import libvoikko
except ModuleNotFoundError:
    from voikko import libvoikko
import logging
from nltk.corpus import stopwords


logger = logging.getLogger(__name__)
nltk.download('stopwords')

EMAIL_REGEX = (
    r"(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)"
    r"*|\"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\"
    r"[\x01-\x09\x0b\x0c\x0e-\x7f])*\")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])"
    r"?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])"
    r"?|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.)"
    r"{3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]"
    r":(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]"
    r"|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])")

URL_REGEX = (r"(https?:\/\/(?:www\.|(?!www))"
             r"[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]"
             r"\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]"
             r"\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))"
             r"[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})")

URL_EMAIL_REGEX = '|'.join([EMAIL_REGEX, URL_REGEX])

FIN_STOPS = set(stopwords.words('finnish'))
FIN_STOPS.update([
    'hei', 'moi', 'moikka', 'moro', 'tervehdys', 'terve', 'terveisin', 'siis',
    'myös', 'kiitos', 'yle', 'uutisikkuna', 'kiitoksia', 'kiitosta', 'ok',
    'eli', 'okei', 'no', 'sitten', 'jo', 'vielä', 'aina', 'jotta'
])
DEL_FROM_FINSTOPS = ['en', 'et', 'ei', 'emme', 'ette', 'eivät']
for word in DEL_FROM_FINSTOPS:
    FIN_STOPS.remove(word)

SWE_STOPS = set(stopwords.words('swedish'))
SWE_STOPS.remove('min')
SWE_STOPS.update(['swe', 'svenska', 'dag', 'buu', 'klubben', 'fråga', 'veckans', 'jag'])

EN_STOPS = set(stopwords.words('english'))
EN_STOPS.update(['of', 'and', 'you', 'for', 'what', 'have', 'can'])
DEL_FROM_ENSTOPS = ['on', 'as', 'a', 'd', 'm', 'o', 's', 't', 'me', 'no', 'y']
for word in DEL_FROM_ENSTOPS:
    EN_STOPS.remove(word)

OTHER_STOPS = set([
    'mailto', 'subject', 'from', 'to', 'vs', 'message', 'original', 'date',
    're', 'terv', 'sent', 'from', 'kello', 'fin', 'swe', 'uutisikkuna'
])

FINAL_STOPS = FIN_STOPS | OTHER_STOPS

voikko = libvoikko.Voikko('fi')
voikko.setIgnoreDot(True)


def _fin_lemmatize_word(string):
    voikkofied = voikko.analyze(string)
    if len(voikkofied) > 0 and voikkofied[0].get('BASEFORM') is not None:
        return voikkofied[0]['BASEFORM']
    else:
        return string


def _finnish_detector(text):
    token_set = set(text.split())
    n_fi = len(token_set.intersection(FIN_STOPS))
    n_swe = len(token_set.intersection(SWE_STOPS))
    n_en = len(token_set.intersection(EN_STOPS))
    return (n_fi > n_en) & (n_fi > n_swe)


def process_and_filter(words):
    return [_fin_lemmatize_word(word) for word in words.split()
            if len(word) > 1 and word not in FINAL_STOPS]


def _process_text(text_series):
    # Remove emails and URLs
    text_series = text_series.str.replace(URL_EMAIL_REGEX, ' ').lower()

    # Remove all except letters and whitespace
    text_series = text_series.str.replace(':', '').replace('[^A-Za-zÄÖÅäöå]', ' ')

    # Remove stop words and 1 char tokens, then lemmatize with Voikko
    # Return as lowercased string as Voikko uppercases some words
    return text_series.apply(process_and_filter).str.join(' ').str.lower()


def process_single_text(text):
    text = text.replace(URL_EMAIL_REGEX, ' ')
    text = ''.join([c if c.isalpha() else ' ' for c in text.lower()])
    return ' '.join(process_and_filter(text)).lower()


def process_text_df(df, text_col):
    # Remove very short texts
    df = df[df[text_col].str.len() > 5]

    # Remove non-Finnish documents
    df = df[df[text_col].apply(_finnish_detector)]

    # Preprocess text
    df[text_col] = _process_text(df[text_col])
