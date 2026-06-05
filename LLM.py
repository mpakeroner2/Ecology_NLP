import os
import sqlite3
import openai
import time
import pandas as pd
import numpy as np
from openai import OpenAI
from gen_func import log, clean_log
import re
from rapidfuzz import fuzz, process
from openai import APITimeoutError, APIConnectionError, RateLimitError
from dotenv import load_dotenv
load_dotenv()

system_messages = {
    'relevance': (
        "Does the text of this social media post relate to environmental issues, climate change, "
        "ecology, green energy, pollution, biodiversity, or sustainable development? "
        "Write only 1 if yes, and 0 if no."
    ),
    'country': "In what country does the author of this social media post live? Write only the country name in English.",
    'message': (
        "You are an analyst who selects the most appropriate topic label from a list "
        "for this social media post about environmental or climate issues. "
        "Answer only with the exact label name. "
        "List of labels: 'Вчені фіксують новий температурний рекорд', 'Крижані шапки та льодовики продовжують танути', 'Рівень моря загрозливо підвищується', 'Екстремальні погодні явища частішають', 'Вуглецеві викиди досягли рекордного рівня', 'Кліматичні цілі країн не виконуються', 'Вчені попереджають про кліматичні точки неповернення', 'Посухи та спекотні хвилі стають інтенсивнішими', 'Повені завдають дедалі більших збитків', 'Відновлювана енергетика стрімко розвивається', 'Сонячна та вітрова енергетика стає дешевшою', 'Електромобілі набувають масової популярності', 'Зелені технології залучають рекордні інвестиції', 'Ядерна енергетика обговорюється як кліматичне рішення', 'Зелений водень - паливо майбутнього', 'Енергоефективність будівель набуває пріоритету', 'Зелена енергетика стикається з проблемами мережі', 'Країни підписують нові кліматичні угоди', 'Кліматичні переговори зайшли в глухий кут', 'Міжнародна кліматична співпраця розвивається', 'Уряди приймають екологічне законодавство', 'Вуглецевий ринок та торгівля квотами розвиваються', 'Кліматичне фінансування для бідних країн недостатнє', 'Молоді активісти вимагають кліматичних дій', 'Екологічних активістів переслідують', 'Масові кліматичні страйки охопили міста', 'Люди закликають бойкотувати компанії-забруднювачі', 'Компанії беруть зобовязання досягти net-zero', 'Гринвошинг компаній викрито', 'Зелена економіка створює нові робочі місця', 'Нафтові компанії звинувачують у прихованні кліматичних ризиків', 'Субсидії викопному паливу продовжуються попри кризу', 'Зелені інвестиції та кліматичне фінансування зростають', 'Біорізноманіття скорочується загрозливими темпами', 'Ліси масово вирубуються та горять', 'Океани закислюються через викиди CO2', 'Коралові рифи гинуть через потепління', 'Тварини опиняються під загрозою зникнення', 'Пластикове забруднення охопило океани та суходіл', 'Якість повітря у містах погіршується', 'Питна вода стає дедалі більшим дефіцитом', 'Кліматичні біженці - нова глобальна проблема', 'Корінні народи захищають природні екосистеми', 'Кліматична тривога зростає серед молоді', 'Люди змінюють споживчу поведінку заради довкілля', 'Екологічна справедливість: бідні країни страждають найбільше', 'Науковці дослідили нові аспекти змін клімату', 'Викиди метану виявилися більшими ніж повідомлялося', 'Агробізнес чинить значний тиск на довкілля', 'Уряди не виконують кліматичних зобовязань', 'Медіа недостатньо висвітлюють кліматичні проблеми', 'Відновлення екосистем та посадка лісів набирає обертів', 'Сталий розвиток та кругова економіка впроваджуються', 'Міста впроваджують програми зеленої інфраструктури', 'Зелена дипломатія стає частиною зовнішньої політики'"
    ),
    'sentiment': (
        "Evaluate the sentiment of the following text regarding environmental protection and climate action. "
        "Determine whether the sentiment is positive, negative, or unknown. Answer with 1 word only."
    ),
    'officials': (
        "You are an analyst who determines whether the text contains a comment by an official, "
        "government agency, or recognized environmental organization (e.g. IPCC, UN, national ministry, Greenpeace, WWF). "
        "Consider a direct quote or paraphrase with reference to a specific person or institution. "
        "If not found, write 0. If yes, write the institution or position of the person."
    ),
    'check_region': (
        "You are an analyst classifying social media authors by geographic location. "
        "Based on available data about the author (name, language, group name, context), "
        "determine which country the author is from. "
        "Write only the country name in English, or unknown if insufficient data."
    ),
    'emotions': (
        "Identify the single dominant emotion expressed in this social media post about "
        "environmental or climate issues. Choose exactly one English label from: anxiety, fear, "
        "anger, indignation, sadness, guilt, helplessness, disappointment, hope, optimism, "
        "determination, inspiration, gratitude, admiration, interest, joy, relief, indifference. "
        "If no clear emotion, answer 0. Answer with one word only."
    ),
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

secondary_limit=0
data_limit = 1000000

eco_titles_list = [
    # Клімат / температура
    'Вчені фіксують новий температурний рекорд',
    'Крижані шапки та льодовики продовжують танути',
    'Рівень моря загрозливо підвищується',
    'Екстремальні погодні явища частішають',
    'Вуглецеві викиди досягли рекордного рівня',
    'Кліматичні цілі країн не виконуються',
    'Вчені попереджають про кліматичні точки неповернення',
    'Посухи та спекотні хвилі стають інтенсивнішими',
    'Повені завдають дедалі більших збитків',
    # Зелена енергетика
    'Відновлювана енергетика стрімко розвивається',
    'Сонячна та вітрова енергетика стає дешевшою',
    'Електромобілі набувають масової популярності',
    'Зелені технології залучають рекордні інвестиції',
    'Ядерна енергетика обговорюється як кліматичне рішення',
    'Зелений водень — паливо майбутнього',
    'Енергоефективність будівель набуває пріоритету',
    'Зелена енергетика стикається з проблемами мережі',
    # Міжнародна політика
    'Країни підписують нові кліматичні угоди',
    'Кліматичні переговори зайшли в глухий кут',
    'Міжнародна кліматична співпраця розвивається',
    'Уряди приймають екологічне законодавство',
    'Вуглецевий ринок та торгівля квотами розвиваються',
    'Кліматичне фінансування для бідних країн недостатнє',
    # Активізм
    'Молоді активісти вимагають кліматичних дій',
    'Екологічних активістів переслідують',
    'Масові кліматичні страйки охопили міста',
    'Люди закликають бойкотувати компанії-забруднювачі',
    # Бізнес та економіка
    "Компанії беруть зобов'язання досягти net-zero",
    'Гринвошинг компаній викрито',
    'Зелена економіка створює нові робочі місця',
    'Нафтові компанії звинувачують у прихованні кліматичних ризиків',
    'Субсидії викопному паливу продовжуються попри кризу',
    'Зелені інвестиції та кліматичне фінансування зростають',
    # Природа / біорізноманіття
    'Біорізноманіття скорочується загрозливими темпами',
    'Ліси масово вирубуються та горять',
    'Океани закислюються через викиди CO2',
    'Коралові рифи гинуть через потепління',
    'Тварини опиняються під загрозою зникнення',
    'Пластикове забруднення охопило океани та суходіл',
    'Якість повітря у містах погіршується',
    'Питна вода стає дедалі більшим дефіцитом',
    # Люди та суспільство
    'Кліматичні біженці — нова глобальна проблема',
    'Корінні народи захищають природні екосистеми',
    'Кліматична тривога зростає серед молоді',
    'Люди змінюють споживчу поведінку заради довкілля',
    'Екологічна справедливість: бідні країни страждають найбільше',
    'Науковці дослідили нові аспекти змін клімату',
    # Критика бездіяльності
    'Викиди метану виявилися більшими ніж повідомлялося',
    'Агробізнес чинить значний тиск на довкілля',
    "Уряди не виконують кліматичних зобов'язань",
    'Медіа недостатньо висвітлюють кліматичні проблеми',
    # Позитивні дії
    'Відновлення екосистем та посадка лісів набирає обертів',
    'Сталий розвиток та кругова економіка впроваджуються',
    'Міста впроваджують програми зеленої інфраструктури',
    'Зелена дипломатія стає частиною зовнішньої політики',
]

cleaned_eco_mes = {}

eco_titles = {}
for i in eco_titles_list:
    eco_titles[i.lower()] = i


def gpt_chat(system_message, text, gpt_client, version='4o-mini', fine_tunning=None, temprature = 0, top_p=None):
    log('FUNCTION: GPT_CHAT')

    log('Defining model')
    # Fall back to base gpt-4o-mini when no fine-tuned model / version is given
    if not version or version not in gpt_models:
        version = '4o-mini'
    cur_model = fine_tunning if fine_tunning else gpt_models[version]
    log(f'current model: {cur_model}')

    log(f'Params: tempreture: {temprature}, top_p: {top_p}')
    log('system:', system_message)

    params = {
        "model": cur_model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": text.strip()}
        ]
    }
    if temprature is not None:
        params["temperature"] = temprature
    if top_p is not None:
        params["top_p"] = top_p

    try:
        retries = 3
        for attempt in range(retries):
            try:
                response = gpt_client.chat.completions.create(timeout=10, **params)
                break
            except (APITimeoutError, APIConnectionError, RateLimitError) as e:
                log(f"Retry {attempt+1}/{retries} after error: {e}")
                if attempt == retries - 1:
                    raise e
            time.sleep(2)
    except Exception as e:
        log(str(e))
        log(str(text))
        raise e

    log('Asking model...')
    answer = response.choices[0].message.content.lower().strip()
    log('MODEL ANSWER:', answer)
    return answer

def define_relation(text, model, system_message, gpt_client):
    """ Returns 1 if the post is related to environmental/climate topics, 0 - otherwise
    
    text - text of the post.
            non-empty string
    """
    log('FUNCTION DEFINE RELATIONS STARTED.')

    log(f'Arguments:\nText:{str(text)}')

    version = None
    ft_model = None

    if model in gpt_models.keys():
        version = model
    else:
        ft_model = model
    log('New model. version:', version, '/ fine_tunning model:', ft_model)

    log('Asking  model')
    answer = gpt_chat(system_message=system_message,
             text=text,
             version=version,
             gpt_client=gpt_client,
             fine_tunning=ft_model) 
    
    log('Checking AI answer')
    if answer not in ['0', '1']:
        log(f'ERROR WITH RELATIONS.\nTEXT: {text}.\nANSWER: {answer}')
        # raise Exception(f'ERROR WITH RELATIONS.\nTEXT: {text}.\nANSWER: {answer}')
        return ""
    else:
        log('FUNCTION END. Returned answer:', answer)
        return answer

# # copy to main function
def check_dublicates_db(db='sentiment.db', table='main'):
    log('FUNCTION CHECK_DUBLICATES STARTED')
    log(f'Arguments:\ndb={db}\ntable={table}')
    conn = sqlite3.connect(db)
    log('DB connected.')
    cursor = conn.cursor()
    log('Cursor created.')

    sql = f"""SELECT author, COUNT(*)
                FROM {table}
                GROUP BY author
                HAVING COUNT(*) > 1;
                """
    log('New request:',sql)
    res = cursor.execute(sql).fetchall()
    log('Result:', 'No dublicates' if not res else res)
    conn.close()
    log('FUNCTION ENDED. DB CLOSED')
    return res

# COUNTRY
def check_country_in_db(author, db='sentiment.db'):
    log('FUNCTION CHECK COUNTRY IN DB STARTED')
    log(f'Arguments:\nauthor={author}\ndb={db}')
    conn = sqlite3.connect(db)
    log('DB connected.')
    cursor = conn.cursor()
    log('Cursor created.')

    author = str(author).replace("'", '"')
    log('New author:', author)

    sql = f"""SELECT author, country FROM main WHERE author = '{author}'"""
    log('New request:', sql)

    res = cursor.execute(sql).fetchone()
    log('Result:', res)

    if res and res[1] and res[1].strip():
        country = res[1].strip()
        log('Country from DB:', country)
    else:
        log('No country in DB')
        country = ''
    conn.close()
    log('FUNCTION ENDED. DB CLOSED')
    return country

def check_country_in_db_v2(author, cursor, table):
    log('FUNCTION CHECK COUNTRY IN DB STARTED')

    author = str(author).replace("'", '"').strip()
    log('New author:', author)

    sql = f"""SELECT Автор, Країна FROM {table} WHERE Автор = '{author}'"""
    log('New request:', sql)

    res = cursor.execute(sql).fetchone()
    log('Result:', res)

    if res and res[1] and res[1].strip():
        country = res[1].strip()
        log('Country from DB:', country)
    else:
        log('No country in DB')
        country = ''
    log('FUNCTION ENDED. DB CLOSED')
    return country

def check_region_in_db_v1(author, cursor, table):
    log('FUNCTION CHECK REGION IN DB STARTED')

    author = str(author).replace("'", '"')
    log('New author:', author)

    sql = f"""SELECT Автор, Регіон FROM {table} WHERE Автор = '{author}'"""
    log('New request:', sql)

    res = cursor.execute(sql).fetchone()
    log('Result:', res)

    if res and res[1]:
        region = res[1].strip()
        log('New region:', region)
    else:
        log('No region in DB')
        region = ''

    log('FUNCTION ENDED. DB CLOSED')
    return region

def check_city_in_db_v1(author, cursor, table):
    log('FUNCTION CHECK CITY IN DB STARTED')

    author = str(author).replace("'", '"')
    log('New author:', author)

    sql = f"""SELECT Автор, Місто FROM {table} WHERE Автор = '{author}'"""
    log('New request:', sql)

    res = cursor.execute(sql).fetchone()
    log('Result:', res)

    if res and res[1]:
        region = res[1].strip()
        log('New region:', region)
    else:
        log('No region in DB')
        region = ''

    log('FUNCTION ENDED. DB CLOSED')
    return region

def define_country(text, author, post_place, model, system_message, gpt_client):
    log('FUNCTION DEFINE COUNTRY STARTED.')
    log(f'Arguments:\nText:{str(text)}\nAuthor:{str(author)}')

    country_aliases = {
        'germany': 'Німеччина', 'deutschland': 'Німеччина',
        'france': 'Франція', 'франция': 'Франція',
        'united kingdom': 'Велика Британія', 'uk': 'Велика Британія',
        'poland': 'Польща', 'polska': 'Польща', 'польша': 'Польща',
        'sweden': 'Швеція', 'sverige': 'Швеція',
        'norway': 'Норвегія', 'norge': 'Норвегія',
        'finland': 'Фінляндія', 'suomi': 'Фінляндія',
        'denmark': 'Данія', 'danmark': 'Данія',
        'usa': 'США', 'united states': 'США', 'us': 'США',
        'canada': 'Канада', 'australia': 'Австралія',
        'china': 'Китай', 'india': 'Індія',
        'brazil': 'Бразилія', 'brasil': 'Бразилія',
        'japan': 'Японія',
    }

    text = f"Author:{author}. Post: {text}"
    log('Asking model')
    answer = gpt_chat(system_message=system_message,
                      text=text,
                      version='4o-mini',
                      gpt_client=gpt_client)

    answer = country_aliases.get(answer.lower().strip(), answer.strip())
    log('FUNCTION END. Returned answer:', answer)
    return answer


def define_sentiment(text, model, system_message, gpt_client):
    log('FUNCTION DEFINE MESSAGE STARTED.')
    log(f'Arguments:\nText:{str(text)}')

    text = f"Post: {text}" 

    gpt_models = {'3.5': 'gpt-3.5-turbo', '4': 'gpt-4-turbo', '4o': 'gpt-4o', 
                  '4o-mini': 'gpt-4o-mini'}

    version = None
    ft_model = None

    if model in gpt_models.keys():
        version = model
    else:
        ft_model = model
    log('New model. version:', version, '/ fine_tunning model:', ft_model)

    log('Asking  model')
    answer = gpt_chat(system_message=system_message,
             text=text,
             version=version,
             gpt_client=gpt_client,
             fine_tunning=ft_model) 

    log('answer:', answer, '- translating')
    if answer.lower().strip() in ['positive', "позитивний", "позитивне", "позитивна", "позитивный"]:
        answer = 'позитивний'
    elif answer.lower().strip() in ['negative', 'негативний', "негативне", "негативна",
                                    "негатив", "негативно"]:
        answer = 'негативний'
    elif answer.lower().strip() in ['uknown', 'нейтральне', 'neutral', "нейтральна", "нейтральне",
                                    "нейтральний", "нейтральный", 'mixed', "0"]:
        answer = 'нейтральний'
    
    log('FUNCTION END. Returned answer:', answer)
    return answer

def define_emotions(text, model, system_message, gpt_client, temprature=None, top_p=None):
    log('FUNCTION DEFINE EMOTIONS STARTED.')
    log(f'Arguments:\nText:{str(text)}')

    gpt_models = {'3.5': 'gpt-3.5-turbo', '4': 'gpt-4-turbo', '4o': 'gpt-4o', 
                  '4o-mini': 'gpt-4o-mini'}

    emotions_dict = {'anxiety': 'тривога', 'fear': 'страх', 'anger': 'гнів', 'indignation': 'обурення',
                 'sadness': 'смуток', 'guilt': 'провина', 'helplessness': 'безпорадність',
                 'disappointment': 'розчарування', 'hope': 'надія', 'optimism': 'оптимізм',
                 'determination': 'рішучість', 'inspiration': 'натхнення', 'gratitude': 'вдячність',
                 'admiration': 'захоплення', 'interest': 'зацікавленість', 'joy': 'радість',
                 'relief': 'полегшення', 'indifference': 'байдужість', '0': 'відсутні'}

    text = f"Post: {text}" 

    version = None
    ft_model = None

    if model in gpt_models.keys():
        version = model
    else:
        ft_model = model
    log('New model. version:', version, '/ fine_tunning model:', ft_model)

    log('Asking  model')
    answer = gpt_chat(system_message=system_message,
             text=text,
             version=version,
             gpt_client=gpt_client,
             fine_tunning=ft_model, 
             temprature=temprature, 
             top_p=top_p) 
    
    log('Model Answer:', answer, '- use dictionary.')
    try:
        answer = emotions_dict[answer.lower().strip()]
    except Exception as e:
        answer = '?'
        log('e')
        print(e)
            
    log('FUNCTION END. Returned answer:', answer)
    return answer

def define_message(text, model, system_message, gpt_client):
    log('FUNCTION DEFINE MESSAGE STARTED.')
    log(f'Arguments:\nText:{str(text)}')

    text = f"Post: {text}"

    version = None
    ft_model = None

    if model in gpt_models.keys():
        version = model
    else:
        ft_model = model
    log('New model. version:', version, '/ fine_tunning model:', ft_model)

    log('Asking  model')
    answer = gpt_chat(system_message=system_message,
             text=text,
             version=version,
             fine_tunning=ft_model,
             gpt_client=gpt_client) 
    
    log('FUNCTION END. Returned answer:', answer)
    return answer

def define_officials(text, author, model, system_message, gpt_client):
    log('FUNCTION DEFINE OFFICIALS STARTED.')
    log(f'Arguments:\nText:{str(text)}\nAuthor:{str(author)}')

    text = f"Author: {author}.Post: {text}" 

    version = None
    ft_model = None

    if model in gpt_models.keys():
        version = model
    else:
        ft_model = model
    log('New model. version:', version, '/ fine_tunning model:', ft_model)

    log('Asking  model')
    answer = gpt_chat(system_message=system_message,
             text=text,
             version=version,
             fine_tunning=ft_model,
             gpt_client=gpt_client) 
    
    if answer == '0' or answer == 0:
        answer = ""
        
    log('FUNCTION END. Returned answer:', answer)
    return answer

def check_patterns_relevance(*args):
    text = ""
    for i in args:
        text += str(i) + ' '
    text = text.strip().lower()

    patterns = [
        ("1", r"\bклімат\w*\b"),
        ("1", r"\bекологі\w*\b"),
        ("1", r"\bглобальне потепління\b"),
        ("1", r"\bco2\b"),
        ("1", r"\bвикид\w*\b"),
        ("1", r"\bвідновлюван\w* енергетик\w*\b"),
        ("1", r"\bбіорізноманіт\w*\b"),
        ("1", r"\bгрінвошинг\b"),
        ("1", r"\bnet.zero\b"),
        ("1", r"\bfridays for future\b"),
        ("1", r"\bзелен\w* перехід\b"),
    ]

    for name, pattern in patterns:
        if re.search(pattern, text):
            return name
    return None

def check_patterns_country(*args):
    text = ""
    for i in args:
        text += str(i) + ' '
    text = text.strip().lower()

    patterns = [
        ("Німеччина", r"\w*deutsch\w*"),
        ("Франція",   r"\bfranc\w*"),
        ("Велика Британія", r"\w*british\w*|\buk\b|\bgb\b"),
        ("Польща",   r"\w*pols[ck]\w*"),
        ("Швеція",   r"\w*sverig\w*"),
        ("Норвегія", r"\w*norg\w*|\bnorway\b"),
        ("Фінляндія", r"\w*suomi\w*|\bfinlan\w*"),
        ("Данія",    r"\w*danm\w*|\bdenmar\w*"),
        ("США",      r"\busa\b|\bus\b|\bamerica\b"),
        ("Канада",   r"\w*canad\w*"),
        ("Австралія", r"\w*australi\w*"),
        ("Китай",    r"\w*chin[ae]\w*"),
        ("Індія",    r"\w*indi[ae]\w*"),
        ("Бразилія", r"\w*brasil\w*|\bbrazil\w*"),
        ("Японія",   r"\w*japan\w*"),
    ]

    for name, pattern in patterns:
        if re.search(pattern, text):
            return name
    return None

def check_patterns_sentiments(*args):
    text = ""
    for i in args:
        text += str(i) + ' '
    text = text.strip().lower()

    patterns = [
        ("негативний", r"\bкліматична катастрофа\b"),
        ("негативний", r"\bекологічна катастрофа\b"),
        ("позитивний", r"\bзелена революція\b"),
        ("позитивний", r"\bекологічний прогрес\b"),
    ]

    for name, pattern in patterns:
        if re.search(pattern, text):
            return name
    return None

def check_patterns_emotions(*args):
    text = ""
    for i in args:
        text += str(i) + ' '
    text = text.strip().lower()

    patterns = [
        ("тривога", r"\bкліматична тривога\b"),
        ("гнів",   r"\bкліматичний гнів\b"),
        ("надія",  r"\bзелене майбутнє\b"),
    ]

    for name, pattern in patterns:
        if re.search(pattern, text):
            return name
    return None

def check_patterns_messages(*args):
    text = ""
    for i in args:
        text += str(i) + ' '
    text = text.strip().lower()

    patterns = [
        ("Молоді активісти вимагають кліматичних дій", r"\bfridays for future\b"),
        ("Екстремальні погодні явища частішають",       r"\bстихійне лихо\b"),
        ("Вуглецеві викиди досягли рекордного рівня",   r"\bco2\b.*?рекорд"),
    ]

    for name, pattern in patterns:
        if re.search(pattern, text):
            return name
    return None

def check_patterns_officials(*args):
    text = ""
    for i in args:
        text += str(i) + ' '
    text = text.strip().lower()

    patterns = [
        ("IPCC",         r"\bipcc\b"),
        ("ООН / UNEP",   r"\bunep\b"),
        ("Greenpeace",   r"\bgreenpeace\b"),
        ("WWF",          r"\bwwf\b"),
        ("Міністерство екології", r"\bмінекологі\w*\b"),
    ]

    for name, pattern in patterns:
        if re.search(pattern, text):
            return name
    return None

def check_patterns_country_by_name(*args):
    """Detect author country/region/city by name or group via simple regex.

    Returns a (country, region, city) tuple on first match, else ['', '', ''].
    Neutral, climate-discourse oriented country set.
    """
    text = ""
    for i in args:
        text += str(i) + ' '
    text = text.strip().lower()

    patterns = [
        # Німеччина
        (("Німеччина", "Берлін", "Берлін"), r"\bberlin\b|\bберлін\b|\bберлин\b"),
        (("Німеччина", "Баварія", "Мюнхен"), r"\bmunich\b|\bmunchen\b|\bмюнхен\b"),
        (("Німеччина", "Гамбург", "Гамбург"), r"\bhamburg\b|\bгамбург\b"),
        (("Німеччина", "", ""), r"\bgermany\b|\bdeutschland\b|\bнімеччин\w*|\bгермани\w*"),
        # Франція
        (("Франція", "Іль-де-Франс", "Париж"), r"\bparis\b|\bпариж\b"),
        (("Франція", "Овернь-Рона-Альпи", "Ліон"), r"\blyon\b|\bліон\b|\bлион\b"),
        (("Франція", "", ""), r"\bfrance\b|\bфранці\w*|\bфранци\w*"),
        # Велика Британія
        (("Велика Британія", "Англія", "Лондон"), r"\blondon\b|\bлондон\b"),
        (("Велика Британія", "Шотландія", "Единбург"), r"\bedinburgh\b|\bединбург\b"),
        (("Велика Британія", "Англія", "Манчестер"), r"\bmanchester\b|\bманчестер\b"),
        (("Велика Британія", "", ""), r"\bunited kingdom\b|\bbritain\b|\bбритан\w*|\buk\b|\bgb\b"),
        # Іспанія
        (("Іспанія", "Мадрид", "Мадрид"), r"\bmadrid\b|\bмадрид\b"),
        (("Іспанія", "Каталонія", "Барселона"), r"\bbarcelona\b|\bбарселон\w*"),
        (("Іспанія", "", ""), r"\bspain\b|\bespana\b|\bіспані\w*|\bиспани\w*"),
        # Італія
        (("Італія", "Лаціо", "Рим"), r"\brome\b|\broma\b|\bрим\b"),
        (("Італія", "Ломбардія", "Мілан"), r"\bmilan\b|\bmilano\b|\bмілан\b|\bмилан\b"),
        (("Італія", "", ""), r"\bitaly\b|\bitalia\b|\bіталі\w*|\bитали\w*"),
        # Польща
        (("Польща", "Мазовецьке воєводство", "Варшава"), r"\bwarsaw\b|\bwarszawa\b|\bваршав\w*"),
        (("Польща", "Малопольське воєводство", "Краків"), r"\bkrakow\b|\bcracow\b|\bкраків\b|\bкраков\b"),
        (("Польща", "", ""), r"\bpoland\b|\bpolska\b|\bпольщ\w*|\bпольш\w*"),
        # Швеція
        (("Швеція", "Стокгольм", "Стокгольм"), r"\bstockholm\b|\bстокгольм\b"),
        (("Швеція", "", ""), r"\bsweden\b|\bsverige\b|\bшвеці\w*|\bшвеци\w*"),
        # Норвегія
        (("Норвегія", "Осло", "Осло"), r"\boslo\b|\bосло\b"),
        (("Норвегія", "", ""), r"\bnorway\b|\bnorge\b|\bнорвегі\w*|\bнорвеги\w*"),
        # Фінляндія
        (("Фінляндія", "Гельсінкі", "Гельсінкі"), r"\bhelsinki\b|\bгельсінкі\b|\bхельсинки\b"),
        (("Фінляндія", "", ""), r"\bfinland\b|\bsuomi\b|\bфінлянді\w*|\bфинлянди\w*"),
        # Нідерланди
        (("Нідерланди", "Північна Голландія", "Амстердам"), r"\bamsterdam\b|\bамстердам\b"),
        (("Нідерланди", "", ""), r"\bnetherlands\b|\bholland\b|\bнідерланд\w*|\bнидерланд\w*"),
        # США
        (("США", "Нью-Йорк", "Нью-Йорк"), r"\bnew york\b|\bнью-йорк\w*|\bнью йорк\w*"),
        (("США", "Каліфорнія", "Лос-Анджелес"), r"\blos angeles\b|\bcalifornia\b|\bкаліфорні\w*|\bлос-анджелес\w*"),
        (("США", "Вашингтон", "Вашингтон"), r"\bwashington\b|\bвашингтон\w*"),
        (("США", "", ""), r"\busa\b|\bunited states\b|\bamerica\b|\bсша\b|\bамерик\w*"),
        # Канада
        (("Канада", "Онтаріо", "Торонто"), r"\btoronto\b|\bторонто\b"),
        (("Канада", "Британська Колумбія", "Ванкувер"), r"\bvancouver\b|\bванкувер\b"),
        (("Канада", "", ""), r"\bcanada\b|\bканад\w*"),
        # Австралія
        (("Австралія", "Новий Південний Уельс", "Сідней"), r"\bsydney\b|\bсідней\b|\bсидней\b"),
        (("Австралія", "Вікторія", "Мельбурн"), r"\bmelbourne\b|\bмельбурн\b"),
        (("Австралія", "", ""), r"\baustralia\b|\bавстралі\w*|\bавстрали\w*"),
        # Бразилія
        (("Бразилія", "Сан-Паулу", "Сан-Паулу"), r"\bsao paulo\b|\bсан-паулу\b"),
        (("Бразилія", "Ріо-де-Жанейро", "Ріо-де-Жанейро"), r"\brio de janeiro\b|\bріо-де-жанейро\b|\bрио-де-жанейро\b"),
        (("Бразилія", "", ""), r"\bbrazil\b|\bbrasil\b|\bбразилі\w*|\bбразили\w*"),
        # Японія
        (("Японія", "Токіо", "Токіо"), r"\btokyo\b|\bтокіо\b|\bтокио\b"),
        (("Японія", "Осака", "Осака"), r"\bosaka\b|\bосака\b"),
        (("Японія", "", ""), r"\bjapan\b|\bяпоні\w*|\bяпони\w*"),
        # Індія
        (("Індія", "Делі", "Делі"), r"\bdelhi\b|\bделі\b|\bдели\b"),
        (("Індія", "Махараштра", "Мумбаї"), r"\bmumbai\b|\bмумбаї\b|\bмумбаи\b"),
        (("Індія", "", ""), r"\bindia\b|\bінді\w*|\bинди\w*"),
        # Китай
        (("Китай", "Пекін", "Пекін"), r"\bbeijing\b|\bпекін\b|\bпекин\b"),
        (("Китай", "Шанхай", "Шанхай"), r"\bshanghai\b|\bшанхай\b"),
        (("Китай", "", ""), r"\bchina\b|\bкита\w*"),
    ]

    for name, pattern in patterns:
        if re.search(pattern, text):
            return name
    return ['', '', '']

def check_messages(mes, lang='eco'):
    val_t = eco_titles

    if mes in val_t.keys():
        return (val_t[mes], 'blue')

    for score_threshold in range(95,19,-5):
        best_match = process.extractOne(
            mes,
            val_t.keys(),
            scorer=fuzz.token_sort_ratio  # Використовуємо token_sort_ratio, щоб уникнути впливу порядку слів
        )
        
        if best_match:
            match_title, score, _ = best_match
            # Якщо найкращий варіант має схожість вище або рівну порогу, повертаємо його
            if score >= score_threshold:
                return (val_t[match_title], 'red')
    return ("", 'red')
