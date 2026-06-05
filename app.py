import os
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import Tuple, Dict, Any
from handle import upload_data, check_columns, handling_language_field, delete_file_if_exists
from handle import color_to_excel
import time
import sqlite3
from openai import OpenAI
import pandas as pd
from gen_func import log, clean_log
from LLM import define_relation, check_country_in_db_v2, define_country
from LLM import define_sentiment, define_emotions, define_message
from LLM import define_officials, check_patterns_relevance, check_patterns_country
from LLM import check_patterns_sentiments, check_patterns_emotions
from LLM import check_patterns_messages, check_patterns_officials, check_patterns_country_by_name
from LLM import check_region_in_db_v1, check_city_in_db_v1, check_messages

system_messages = {
    'relevance': 'Does the text of this social media post relate to environmental issues, climate change, ecology, green energy, pollution, biodiversity, or sustainable development? Write only 1 if yes, and 0 if no.',
    'country': 'In what country does the author of this social media post live? Write only the country name in English.',
    'message': ('You are an analyst who selects the most appropriate topic label from a list for this '
                'social media post about environmental or climate issues. Answer only with the exact label name.'),
    'sentiment': ("Evaluate the sentiment of the following text regarding environmental protection and climate action. "
                  "Determine whether the sentiment is 'positive', 'negative', or 'unknown'. Answer with 1 word only."),
    'officials': ("You are an analyst who determines whether the text contains a comment by an official, government agency, "
                  "or recognized environmental organization (e.g. IPCC, UN, national ministry, Greenpeace, WWF). "
                  "Consider a direct quote or paraphrase with reference to a specific person or institution. "
                  "If not found, write '0'. If yes, write the institution or position of the person."),
    'check_region': ('You are an analyst classifying social media authors by geographic location. '
                     'Based on available data about the author (name, language, group name, context), determine which country the author is from. '
                     "Write only the country name in English, or 'unknown' if insufficient data."),
    'emotions': ("Identify the single dominant emotion expressed in this social media post about environmental "
                 "or climate issues. Choose exactly one English label from: anxiety, fear, anger, indignation, "
                 "sadness, guilt, helplessness, disappointment, hope, optimism, determination, inspiration, "
                 "gratitude, admiration, interest, joy, relief, indifference. If no clear emotion, answer 0. Answer with one word only."),
}

correct_answers = ['тривога', 'страх', 'гнів', 'обурення', 'смуток', 'провина', 'безпорадність',
                'розчарування', 'надія', 'оптимізм', 'рішучість', 'натхнення', 'вдячність',
                'захоплення', 'зацікавленість', 'радість', 'полегшення', 'байдужість',
                'nan', '', 'positive', 'negative', 'unknown', '1', '0']

gpt_fine_tunning_models = {
    'relevance':       os.getenv('MODEL_RELEVANCE'),
    'eco_message':     os.getenv('MODEL_ECO_MESSAGE'),
    'officials':       os.getenv('MODEL_OFFICIALS'),
    'emotions':        os.getenv('MODEL_EMOTIONS'),
    'region_check':    os.getenv('MODEL_REGION_CHECK'),
}
emotions_dict = {'anxiety': 'тривога', 'fear': 'страх', 'anger': 'гнів', 'indignation': 'обурення',
                 'sadness': 'смуток', 'guilt': 'провина', 'helplessness': 'безпорадність',
                 'disappointment': 'розчарування', 'hope': 'надія', 'optimism': 'оптимізм',
                 'determination': 'рішучість', 'inspiration': 'натхнення', 'gratitude': 'вдячність',
                 'admiration': 'захоплення', 'interest': 'зацікавленість', 'joy': 'радість',
                 'relief': 'полегшення', 'indifference': 'байдужість', '0': 'nan'}

gpt_models = {'3.5': 'gpt-3.5-turbo', '4': 'gpt-4-turbo', '4o': 'gpt-4o', '4o-mini': 'gpt-4o-mini'}

conn = sqlite3.connect('sentiment_v2.db', check_same_thread=False)
cursor = conn.cursor()

secondary_limit=0
data_limit = 1000000

# Store processed file paths
global_processed_file = None

# user_api = st.text_input("Введіть API:").strip()


gpt_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


app = FastAPI()

# Initialize shared state in app.state
template_state = {
    "if_stop": None,
    "details": "",
    "relevance": "",
    "country": "",
    "emotions": "",
    "sentiments": "",
    "messages": "",
    "officials": ""
}
app.state.state = template_state.copy()

state = app.state.state
state["status"] = "done"

# Mapping of numeric codes to analysis keys
ANALYSIS_TYPES: Dict[int, str] = {
    1: "relevance",
    2: "country_pattern",
    3: "country_bd",
    4: "country_AI",
    5: "region_bd",
    6: "region_patterns",
    7: "emotions",
    8: "sentiments",
    9: "messages",
    10: "officials"
}

class FilePayload(BaseModel):
    """
    Payload for uploading a file path and a tuple of numeric codes for analyses.
    """
    file_path: str
    codes: Tuple[int, ...]


def analyze(df: pd.DataFrame, codes: Tuple[int, ...]) -> Dict[str, Any]:
    """
    Internal function to analyze DataFrame based on requested numeric codes.
    Processes each code using ANALYSIS_TYPES and returns results dict.
    Handles invalid codes here.
    """

    # Clear all ANALYSIS_TYPES keys in state
    for key in template_state:
        state[key] = ""

    size = len(df)
    log(df.Мова.value_counts())
    
    # Check for invalid codes
    invalid_codes = [c for c in codes if c not in ANALYSIS_TYPES]
    if invalid_codes:
        state["status"] = "error"
        state["details"] = f"invalid analysis codes: {invalid_codes}"
        print(f"Invalid analysis codes: {invalid_codes}")
        print(state)
        return
        # removed return to allow proper error handling upstream

    for code in codes:
        analysis_key = ANALYSIS_TYPES.get(code)
        log(f"Processing analysis type: {analysis_key.upper()}")
        if not analysis_key:
            continue
        # region RELEVANCE
        if analysis_key == "relevance":
            current_row = 1
            try:
                df['relevance_mask'] = ""
                
                state["relevance"] = "0"

                log('Relevance analysis started.')

                for idx, row in df.iterrows():
                    if not state['if_stop']:
                        log('ROW:', str(idx))

                        res = None
                        if row['Текст'] and str(row['Текст'])!='nan' or \
                                    row['Заголовок'] and str(row['Заголовок'])!='nan':
                            res = check_patterns_relevance(str(row['Текст']), str(row['Заголовок']))
                            if res:
                                df.at[idx, 'relevance_mask'] = 'green'
                            log(f'check patterns (rel): {res}')
                            if not res:
                                res = define_relation(text=str(row['Текст']), 
                                                    model='4o-mini', 
                                                    gpt_client=gpt_client,
                                                    system_message=system_messages['relevance'])
                                df.at[idx, 'relevance_mask'] = 'blue'
                            df.at[idx, 'marker'] = res
                            log(f'Result {repr(res)} was added.')
                        else:
                            log(f'No text! {repr(res)} was added.')
                            df.at[idx, 'marker'] = '0'
                        current_row = idx+1
                        state["relevance"] = str(idx+1 % size) + ' / ' + str(size)
                        log("Chunk:", str(idx+1 % size), '/', str(size))
                    else:
                        state['status'] = 'stopped'
                        state['if_stop'] = None
                        log('Analysis STOPPED by user.')
                        return
                log('Relevance analysis completed.')

                while 'index' in df.columns:
                    print('drop index')
                    df.drop(columns=['index'], inplace=True)

                color_to_excel(df, "relevance.xlsx")
                color_to_excel(df, "Result.xlsx")
            except Exception as e:
                log(f"Relevance analysis error: {e}")
                state["status"] = "error"
                state["details"] = f"Current row: {current_row}. Relevance analysis failed: {e}"
                return
        # endregion

        # region COUNTRY
        if analysis_key == "country_bd" or analysis_key == "country_pattern" or \
                analysis_key == "country_AI" or analysis_key == "region_bd" or \
                    analysis_key == "region_patterns":
            current_row = 1
            try:
                df = pd.read_excel("Result.xlsx")
                if 'country_mask' not in df.columns:
                    df['country_mask'] = ""
                if 'region_mask' not in df.columns:
                    df['region_mask'] = ""

                state["country"] = "0"

                log('Placement analysis started.')

                for idx, row in df.iterrows():
                    if not state['if_stop']:
                        log('ROW:', str(idx))

                        country = str(row['Країна']).strip()
                        region = str(row['Регіон']).strip()

                        country_mask = None
                        region_mask = None

                        if not country or country == 'nan':
                            log('Country doesn`t exist')

                            res = None

                            if analysis_key == "country_pattern":
                                log('Check country patterns')
                                res = check_patterns_country_by_name(row['Автор'], 
                                                                    row['Місце публікації'])
                                res = res[0] if res[0] else ''
                                if res:
                                    country = res
                                    country_mask = 'green'
                            
                            if analysis_key == "country_bd" and (not country or country == 'nan'):
                                log('Check country in DB.')
                                res = check_country_in_db_v2(author=row['Автор'], 
                                                            cursor=cursor,
                                                            table='author').strip()
                                if not res:
                                    res = check_country_in_db_v2(author=row['Місце публікації'], 
                                        cursor=cursor,
                                        table='place').strip()
                                if not res:
                                    res = check_country_in_db_v2(author=row['Автор'], 
                                        cursor=cursor,
                                        table='secondary_authors').strip()
                                if not res:
                                    res = check_country_in_db_v2(author=row['Місце публікації'], 
                                        cursor=cursor,
                                        table='secondary_places').strip()
                                if res:
                                    country = res
                                    country_mask = 'gray'
                                    log(f'DB result: {country}, color: {country_mask}')
                            
                            if analysis_key == "country_AI" and (not country or country == 'nan'):
                                res = define_country(text=row['Текст'], 
                                                    author=row['Автор'], 
                                                    post_place=row['Місце публікації'],
                                                    model='4o-mini', 
                                                    gpt_client=gpt_client,
                                                    system_message=system_messages['country'])
                                country_mask = 'blue'
                                country = res 
                            log('New Country:', country, 'Color:', country_mask)

                            df.at[idx, 'country_mask'] = country_mask
                        else:
                            log('Country exists:', row['Країна'])

                            country_aliases = {
                                'USA': 'США', 'United States': 'США', 'United States of America': 'США', 'US': 'США',
                                'Соединённые Штаты': 'США', 'Сполучені Штати': 'США',
                                'Germany': 'Німеччина', 'Deutschland': 'Німеччина', 'Германия': 'Німеччина',
                                'France': 'Франція', 'Франция': 'Франція',
                                'United Kingdom': 'Велика Британія', 'UK': 'Велика Британія',
                                'Great Britain': 'Велика Британія', 'Britain': 'Велика Британія',
                                'Великобритания': 'Велика Британія',
                                'Poland': 'Польща', 'Polska': 'Польща', 'Польша': 'Польща',
                                'Sweden': 'Швеція', 'Sverige': 'Швеція', 'Швеция': 'Швеція',
                                'Norway': 'Норвегія', 'Norge': 'Норвегія', 'Норвегия': 'Норвегія',
                                'Netherlands': 'Нідерланди', 'Nederland': 'Нідерланди',
                                'Holland': 'Нідерланди', 'Нидерланды': 'Нідерланди',
                                'Canada': 'Канада', 'Australia': 'Австралія', 'Австралия': 'Австралія',
                                'China': 'Китай', '中国': 'Китай',
                                'India': 'Індія', 'Индия': 'Індія',
                                'Brazil': 'Бразилія', 'Brasil': 'Бразилія', 'Бразилия': 'Бразилія',
                                'Japan': 'Японія', '日本': 'Японія', 'Япония': 'Японія',
                                'Finland': 'Фінляндія', 'Suomi': 'Фінляндія', 'Финляндия': 'Фінляндія',
                                'Denmark': 'Данія', 'Danmark': 'Данія', 'Дания': 'Данія',
                                'Spain': 'Іспанія', 'España': 'Іспанія', 'Испания': 'Іспанія',
                                'Italy': 'Італія', 'Italia': 'Італія', 'Италия': 'Італія',
                                'Austria': 'Австрія', 'Österreich': 'Австрія', 'Австрия': 'Австрія',
                                'Switzerland': 'Швейцарія', 'Schweiz': 'Швейцарія',
                                'Suisse': 'Швейцарія', 'Svizzera': 'Швейцарія', 'Швейцария': 'Швейцарія',
                            }
                            country = country_aliases.get(country, country)

                        if country == 'nan':
                            country = ''

                        df.at[idx, 'Країна'] = country
                        

                        if not region or region == 'nan':
                            res = None
                            if analysis_key== "region_patterns":
                                log('Region doesn`t exist')
                                res = check_patterns_country_by_name(row['Автор'], 
                                                                    row['Місце публікації'])
                                if res[1]:
                                    region = res[1]
                                    region_mask = 'green'
                            if analysis_key== "region_bd" and (not region or region == 'nan'):
                                log('Check region in DB. Result:', res)
                                res = check_region_in_db_v1(author=row['Автор'], 
                                                            cursor=cursor,
                                                            table='author').strip()
                                if not res:
                                    res = check_region_in_db_v1(author=row['Місце публікації'], 
                                                                cursor=cursor,
                                                                table='place').strip()
                                if not res:
                                    res = check_region_in_db_v1(author=row['Місце публікації'], 
                                                                cursor=cursor,
                                                                table='secondary_places').strip()
                                if not res:
                                    res = check_region_in_db_v1(author=row['Автор'], 
                                                                cursor=cursor,
                                                                table='secondary_authors').strip()
                                if res:
                                    region = res
                                    region_mask = 'gray'
                            df.at[idx, 'region_mask'] = region_mask
                        
                        if region == 'nan':
                            region = ''

                        df.at[idx, 'Регіон'] = region 
                        
                        current_row = idx + 1
                        state["country"] = str(idx+1 % size) + '/' + str(size)
                        log(str(idx+1 % size) + '/' + str(size))
                    else:
                        state['status'] = 'stopped'
                        state['if_stop'] = None
                        log('Analysis STOPPED by user.')
                        return
                while 'index' in df.columns:
                    df.drop(columns=['index'], inplace=True)

                color_to_excel(df, "country.xlsx")
                color_to_excel(df, "Result.xlsx")

                log('Country analysis completed.')
            except Exception as e:
                log(f"Country analysis error: {e}")
                state["status"] = "error"
                state["details"] = f"Current_row: {current_row}. Country analysis failed: {e}"
                return
        # endregion

        # region SENTIMENTS    
        if analysis_key == "sentiments":
            current_row = 1
            try:
                df = pd.read_excel("Result.xlsx")            

                df['sentiment_mask'] = ""
                
                state["sentiments"] = "0"

                log('Sentiments analysis started.')

                for idx, row in df.iterrows():
                    if not state['if_stop']:
                        log('ROW:', str(idx))

                        if row['Текст'] or str(row["Текст"]) != 'nan':
                            if not row['Настрій'] or str(row['Настрій']) == 'nan':
                                color = None
                                log('Sentiment doesn`t exist')

                                res = check_patterns_sentiments(row['Текст'])
                                log('Check sentiment patterns:', res)
                                if res:
                                    color = 'green'

                                if not res:
                                    res = define_sentiment(text=row['Текст'],
                                                            model=gpt_fine_tunning_models['relevance'],
                                                            gpt_client=gpt_client,
                                                            system_message=system_messages['sentiment'])
                                    color = 'blue'
                                df.at[idx, 'Настрій'] = res

                                if color:
                                    df.at[idx, 'sentiment_mask'] = color
                                    log('Sentiment saves.')
                            else:
                                log('Sentiment exists:', row['Настрій'])
                        else:
                            log(f'No text!')
                        
                        current_row = idx + 1
                        state["sentiments"] = str(idx+1 % size) + '/' + str(size)
                        log(str(idx+1 % size), '/', str(size))
                    else:
                        state['status'] = 'stopped'
                        state['if_stop'] = None
                        log('Analysis STOPPED by user.')
                        return
                    
                while 'index' in df.columns:
                    print('drop index')
                    df.drop(columns=['index'], inplace=True)

                color_to_excel(df, "sentiments.xlsx")
                color_to_excel(df, "Result.xlsx")

                log('Sentiments analysis completed.')
            except Exception as e:
                log(f"Sentiments analysis error: {e}")
                state["status"] = "error"
                state["details"] = f"Current row: {current_row}. Sentiments analysis failed: {e}"
                return
        # endregion
        
        # region EMOTIONS
        if analysis_key == "emotions":
            current_row = 1
            try:
                df = pd.read_excel("Result.xlsx")            

                df['emotion_mask'] = ""
                
                state["emotions"] = "0"

                log('Emotions analysis started.')

                for idx, row in df.iterrows():
                    if not state['if_stop']:
                        log('ROW:', str(idx))

                        if row['Текст'] or str(row["Текст"]) != 'nan':
                            if not row['Емоції'] or str(row['Емоції']) == 'nan':
                                log('Emotion doesn`t exist')

                                color = None

                                res = check_patterns_emotions(row['Текст'])
                                log('Check emotion patterns:', res)
                                if res:
                                    color = 'green'

                                if not res:
                                    res = define_emotions(text=row['Текст'],
                                                        model=gpt_fine_tunning_models['emotions'],
                                                        gpt_client=gpt_client,
                                                        system_message=system_messages['emotions'])

                                    color = 'blue'

                                df.at[idx, 'Емоції'] = res
                                if color:
                                    df.at[idx, 'emotion_mask'] = color
                                log('Emotion saved.')
                            else:
                                log('Emotion exists:', row['Емоції'])
                        else:
                            log(f'No text! {repr(res)} was added.')
                        
                        current_row = idx + 1
                        state["emotions"] = str(idx+1 % size) + '/' + str(size)
                        log(str(idx+1 % size), '/', str(size))
                    else:
                        state['status'] = 'stopped'
                        state['if_stop'] = None
                        log('Analysis STOPPED by user.')
                        return
                    
                df['Емоції'] = df['Емоції'].apply(lambda x: '' if str(x) == 'nan' or x == ' ' else x)
                    
                while 'index' in df.columns:
                    print('drop index')
                    df.drop(columns=['index'], inplace=True)

                color_to_excel(df, "emotions.xlsx")
                color_to_excel(df, "Result.xlsx")

                log('Emotions analysis completed.')
            except Exception as e:
                log(f"Emotions analysis error: {e}")
                state["status"] = "error"
                state["details"] = f"Current row: {current_row}. Emotions analysis failed: {e}"
                return
        # endregion

        # region MESSAGES
        if analysis_key == "messages":
            current_row = 1
            try:
                df = pd.read_excel("Result.xlsx")            

                df['message_mask'] = ""
                
                state["messages"] = "0"

                log('Messages analysis started.')

                for idx, row in df.iterrows():
                    if not state['if_stop']:
                        log('ROW:', str(idx))


                        if str(row["Текст"]) != 'nan' and row['Текст']:
                            if not row['Тема'] or str(row['Тема']) == 'nan':
                                log('Message doesn`t exist')
                                color = None

                                # check regular expressions
                                res = check_patterns_messages(row['Текст'])
                                log('Check message patterns:', res)
                                if res:
                                    color = 'green'

                                if not res:
                                    log('Try to define topic by AI')
                                    rez = define_message(text=row['Текст'],
                                                        model=gpt_fine_tunning_models['eco_message'],
                                                        gpt_client=gpt_client,
                                                        system_message=system_messages['message'])
                                    res, color = check_messages(rez, 'eco')
                                df.at[idx, 'Тема'] = res
                                df.at[idx, 'message_mask'] = color
                                if color == 'red':
                                    df.at[idx, 'uMessage'] = rez
                                log('Message saved.')
                            else:
                                log('Message exists:', row['Тема'])
                        else:
                            log(f'No text!')



                        current_row = idx + 1        
                        state["messages"] = str(idx+1 % size) + '/' + str(size)
                        log(str(idx+1 % size), '/', str(size))
                    else:
                        state['status'] = 'stopped'
                        state['if_stop'] = None
                        log('Analysis STOPPED by user.')
                        return
                            
                while 'index' in df.columns:
                    print('drop index')
                    df.drop(columns=['index'], inplace=True)

                color_to_excel(df, "messages.xlsx")
                color_to_excel(df, "Result.xlsx")

                log('Messages analysis completed.')
            except Exception as e:
                log(f"Messages analysis error: {e}")
                state["status"] = "error"
                state["details"] = f"Current row: {current_row}. Messages analysis failed: {e}"
                return
        # endregion

        # region OFFICIALS
        if analysis_key == "officials":
            current_row = 1
            try:
                df = pd.read_excel("Result.xlsx")            

                df['official_mask'] = ""
                df['source_mask'] = ""
                
                state["officials"] = "0"

                log('Officials analysis started.')

                for idx, row in df.iterrows():
                    if not state['if_stop']:
                        log('ROW:', str(idx))


                        if row['Текст'] or str(row["Текст"]) != 'nan':
                            if not row['Офіційність'] or str(row['Офіційність']) == 'nan' or str(row['Офіційність']) == '-':
                                log('Official doesn`t exist')
                                color = None

                                res = check_patterns_officials(row['Текст'])
                                log('Check official patterns:', res)
                                if res:
                                    color = 'green'

                                if not res:
                                    res = define_officials(text=row['Текст'],
                                                        author=row['Автор'],
                                                        model=gpt_fine_tunning_models['officials'],
                                                        gpt_client=gpt_client,
                                                        system_message=system_messages['officials'])
                                    color = 'blue'

                                if res:
                                    df.at[idx, 'Офіційність'] = '1'
                                    df.at[idx, 'джерело'] = res

                                    df.at[idx, 'source_mask'] = color
                                else:
                                    df.at[idx, 'Офіційність'] = '0'

                                df.at[idx, 'official_mask'] = color
                            else:
                                log('Official has already existed.')
                        else:
                            log(f'No text! {repr(res)} was added.')
                                
                        current_row = idx + 1
                        state["officials"] = str(idx+1 % size) + '/' + str(size)
                        log(str(idx+1 % size), '/', str(size))
                    else:
                        state['status'] = 'stopped'
                        state['if_stop'] = None
                        log('Analysis STOPPED by user.')
                        return
                            
                df['джерело'] = df['джерело'].astype(str).str.replace(r"^(.)", lambda x: x.group(1).upper(), regex=True)
                df['джерело'] = df['джерело'].apply(lambda x: '' if str(x) == 'Nan' or x == ' ' else x)

                while 'index' in df.columns:
                    print('drop index')
                    df.drop(columns=['index'], inplace=True)

                color_to_excel(df, "officials.xlsx")
                color_to_excel(df, "Result.xlsx")

                log('Officials analysis completed.')
            except Exception as e:
                log(f"Officials analysis error: {e}")
                state["status"] = "error"
                state["details"] = f"Current row: {current_row}. Officials analysis failed: {e}"
                return
        # endregion
    # Mark state as done
    state["status"] = "done"

# region files
@app.post("/files")
def new_file(payload: FilePayload, background_tasks: BackgroundTasks):
    """
    Endpoint to receive the Excel file path and a tuple of numeric codes.
    Checks service availability, creates DataFrame, and schedules analysis.
    """
    # Check if already busy
    if state.get("status") == "busy":
        return {"status": "busy"}
    # Read Excel into DataFrame
    try:
        df = upload_data(file_path=payload.file_path)
    except Exception as e:
        state["status"] = "error"
        state["details"] = f"Failed to read Excel: {e}"
        return {"status": "error", "details": state["details"]}
    
    df = handling_language_field(df=df)
    df = df.reset_index()

    color_to_excel(df, "Result.xlsx")
        
    # Start analysis
    state["status"] = "busy"
    background_tasks.add_task(analyze, df, payload.codes)
    return {"status": "busy"}

# region status
@app.get("/status")
def get_status():
    """
    Endpoint to retrieve the current state.
    """
    status_val = state.get("status")

    if status_val == "done":
        return {"status": "done"}

    if status_val == "stopped":
        return {"status": "stopped"}

    if status_val == "busy":
        response = {"status": "busy"}
        for key, val in state.items():
            if key in ("status", "details", "if_stop"):
                continue
            if val:
                response[key] = val
        return response

    if status_val == "error":
        return {"status": "error", "details": state.get("details")}

    return {"status": "error", "details": f"Unknown status: {status_val}"}

# region stop
@app.post("/stop")
def stop():
    """
    Endpoint to stop processing.
    Waits up to 30 seconds, checking every 10 seconds.
    """
    status_val = state.get("status")

    # If not busy, return error
    if status_val != "busy":
        state["status"] = "error"
        state["details"] = "Бекенд не зайнятий, навантажте його!)"
        return

    # Signal stop
    state["if_stop"] = True

    # retries = 3
    # for _ in range(retries):
    #     time.sleep(10)
    #     status_val = state.get("status")
    #     if status_val == "stopped":
    #         state["if_stop"] = None
    #         return {"status": "stopped"}
    #     if status_val not in ("busy", "stopped"):
    #         return {"status": "error", "details": "process is not busy"}

    # # Still busy after retries
    # state["status"] = "error"
    # state["details"] = "can't stop"
    # return {"status": "error", "details": state.get("details")}
