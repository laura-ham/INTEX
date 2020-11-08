# INTEX - INteractive Topic EXplorer
This repository contains the code for the INTEX. This human-in-the-loop topic modeling application is developed for interactive topic exploration, especially for people not familiar with data science. INTEX is developed according to the HCD framework, considering non-technical media experts as the main stakeholders of the application. This application is built and tested with Finnish media (article and video content).

## Getting Started

- `pip install -r requirements` 

### Data requirements
Data is not included in this code. The application can be run with json and pkl files placed in the `/data` folder. Data options and requirements:
- Articles full text: 
  - `/data/articles_full_text_preprocessed.pkl`: pkl file with `id` and `text` of all articles to include.
  - `/data/articles_ids.json`: json file with article ids, published time and title. Like:
    ```json
    [{"article_yle_id":"3-10590777", "published_time":"2019-01-11 19:53:05", "title":"Puoli seitsemän on yleisön mielestä vuoden 2018 paras tv-ohjelma – \"Positiivisuudella on selvästi tilausta\""}, {...}, ..., {...}]
    ```

- Article clustering by given tags:
  - `/data/articles_tags.json`: pkl file with `id` and `tag` of all articles to include. Like:
    ```json
    {"id":{"0":"3-11312437", ...}, "tags":{"0":["koronavirus","Ulkomaat","palvelutalot","laitoshuolto","asuntolat","kuolema","sosiaalipalvelut","Age_UK_Group","The_Daily_Telegraph","kuolleisuus","lahjoitukset"], ...}}
    ```
  - `/data/articles_ids.json`: json file with article ids, published time and title. Like:
    ```json
    [{"article_yle_id":"3-10590777", "published_time":"2019-01-11 19:53:05", "title":"Puoli seitsemän on yleisön mielestä vuoden 2018 paras tv-ohjelma – \"Positiivisuudella on selvästi tilausta\""}, {...}, ..., {...}]
    ```

- Videos:
  - Documentaries:
    - `/data/videos_docs_metadata.json`: 
    ```json
    [{"program_yle_id": "1-2219790", "first_airing": "2017-03-26 17:00:00", "program_title": "Dok: Heijastuksia", "duration": 3595000, "duration_minutes": "59.9166", "areena_genres": null, "finnpanel_genre": "Kulttuuriohjelmat", "publication_start_time": "2019-04-01 06:00:00", "publication_end_time": "2019-04-28 20:59:00"}, ...]
    ```
    - `/data/videos_docs_subs.json`: 
    ```json
    {"Unnamed: 0":{"0":0, ...}, "text":{"0":"toimittaja Olli haapakangas ....", ...}}
    ```

- Consumption data:
- `/data/consumption/videos_docs_consumption.json`: 
  ```json 
  [{"article_yle_id":"20-152550", "total_minutes":1232.3862, "age_group":"0-44 (excl 15-29)"}, ... ]
  ```
- `/data/consumption/articles_consumption.json`: 
  ```json 
  [{"program_yle_id": "1-2282759", "total_minutes": "7862.8499", "total_views": "633.5455", "age_group": "45+", "gender": "M"}, ... ]
  ```

### Test the Interactive Topic Model application
- `streamlit run streamlit_runner.py`
