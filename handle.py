from gen_func import log
from langdetect import detect
import pandas as pd
import streamlit as st
import os

def check_columns(df):
    log(f'Function CHECK_COLUMNS started.')
    cols = ['Дата', 'Час', 'Збережено', 'Заголовок', 'Текст', 'Тип посту',
            'URL', 'Тональність', 'Автор', 'Профіль', 'Підписники', 'Демографія',
            'Вік', 'Платформа', "Місце публікації", "Профіль місця публікації",
            "Підписники місця публікації", "Тип джерела", "Мова", "Країна", "Регіон",
            "Місто", "Настрій", "Офіційність", "Емоції", "Тема",
            "джерело", 'uMessage']

    dfcols = df.columns

    # Keep optional marker / colour-mask columns if they are already present.
    if 'marker' in dfcols:
        cols.append('marker')
    for col in dfcols:
        if '_mask' in col:
            cols.append(col)
    for col in dfcols:
        if '_val' in col:
            cols.append(col)

    if 'uMessage' not in dfcols:
        df['uMessage'] = ''

    # Normalise spelling / case variants of the source export to the canonical
    # column names used throughout the pipeline.
    rename_map = {
        'дата': 'Дата',
        'Дайджест тексту': 'Текст',
        'Джерело': 'Платформа',
        'мова': 'Мова',
        'країна': 'Країна',
        'настрій': 'Настрій',
        'офіційність': 'Офіційність',
        'офіційніть': 'Офіційність',
        'офіційнісь': 'Офіційність',
        'офційність': 'Офіційність',
        'емоції': 'Емоції',
        'емоція': 'Емоції',
        'Емоція': 'Емоції',
        'меседж': 'Тема',
    }
    to_rename = {old: new for old, new in rename_map.items() if old in df.columns}
    if to_rename:
        log(f'Renaming columns to canonical names: {to_rename}')
        df = df.rename(columns=to_rename)

    # Add the analysis columns the pipeline fills in later, if absent.
    for col in ('Настрій', 'Офіційність', 'Емоції', 'Тема', 'джерело'):
        if col not in df.columns:
            log(f'{col} added.')
            df[col] = ""

    df = df.loc[:, ~df.columns.duplicated()]
    log(f'Duplicates deleted.')
    print(df.columns)
    try:
        df = df[cols]
    except Exception as e:
        st.error(f'Error: {e}.')
        raise Exception(f'Error: {e}.')
    log(f'Names of columns changed and RETURNED. Columns: {list(df.columns)}.')
    return df

def upload_data(file_path=None, folder_path=None):
    
    log(f'Function UPLOAD_DATA STARTED!')
    if folder_path and file_path or not folder_path and not file_path:
        log(f'upload_data function work with 1 argument')
        raise Exception('upload_data function work with 1 argument')
    elif file_path:
        log(f'Path for file: {file_path}.')
        f_name = file_path if type(file_path)==str else file_path.name
        if f_name.endswith(('.xlsx', '.xls', '.xlsm', '.xlsb')):
            log(f'File type is excel.')
            df = pd.read_excel(file_path)
            log(f"Successfully read: {file_path}")
            
            df = check_columns(df)
            log(f'DataFrame RETURNED.')
            return df
        else:
            log(f'File {file_path} has wrong type.')
            raise Exception(f'File {file_path} has wrong type. Please, upload excel file.')
    elif folder_path:
        log(f'Path of folder: {folder_path}.')
        dataframes = []
        log(f'LOOP for files started.')
        for file in os.listdir(folder_path):
            log(f'New file: {file}.')
            path = os.path.join(folder_path, file)
            log(f'Path: {path}.')
            if file.endswith(('.xlsx', '.xls', '.xlsm', '.xlsb')):
                log(f'File type is excel.')
                df = pd.read_excel(path)
                df = check_columns(df)
                dataframes.append(df)
                log(f"Successfully read: {file}")
        
        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
            log(f'DataFrames concateneted and RETURNED.')
            return combined_df

def handling_language_field(df):
    log(f'Function HANDLING_LANGUAGE_FIELD STARTED.')
    log(f'Loop by rows started.')
    for idx, row in df.iterrows():
        text = str(row["Текст"]).strip().replace("\n", " ")
        language = str(row["Мова"]).strip()
        log(f'New text: {text}.\nPredefined language: {language}')
        if text:
            # Detect language only when the source did not provide a usable value.
            # Stores the raw ISO 639-1 code returned by langdetect (e.g. 'en', 'de', 'fr').
            if not language or language.lower() in ('nan', '', 'unknown', 'na'):
                try:
                    code = detect(text)
                    log(f'Lang detected: {code}.')
                except Exception:
                    code = 'unknown'
                    log(f'Lang is not determined.')
                df.at[idx, 'Мова'] = code
                log(f"Language set to {code}.")
            else:
                log(f'Language already provided: {language}.')
        else:
            log(f'No signs in text.')
    log(f'DataFrame language field normalised and RETURNED.')
    return df

def delete_file_if_exists(file_name, folder_path=''):

    file_path = os.path.join(folder_path, file_name)
    
    if os.path.isfile(file_path):  # Перевірка, чи файл існує
        os.remove(file_path)  # Видалення файлу

def color_to_excel(df: pd.DataFrame, filename: str, index: bool = False, finish: bool = False) -> None:
    """
    Функція приймає:
      - df: вхідний DataFrame, який містить дані для відображення та стовпці-маски для зафарбовування.
      - filename: назва Excel-файлу, у який буде збережено результат.
      - index: логічне значення. Якщо False (за замовчуванням), то індекс буде скинутий (не збережений).
      - finish: якщо True, то перед збереженням видаляються колонки-маски.
    
    У вхідному DataFrame маємо такі колонки-маски:
      - 'country_mask' (для колонки 'Країна')
      - 'region_mask' (для колонки 'Регіон')
      - 'city_mask'   (для колонки 'Місто')
      - 'relevance_mask' (для колонки 'marker')
    
    Доступні (напівпрозорі) кольори задаються наступним чином:
      - 'green' -> '#C8E6C9'
      - 'blue'  -> '#BBDEFB'
      - 'gray'  -> '#E0E0E0'
      - 'red'   -> '#FFCDD2'
    """
    # Створюємо копію датафрейму, щоб не змінити оригінал
    df_display = df.copy()
   
    # Якщо finish=True, спочатку видаляємо стовпці-маски з датафрейму,
    # щоб вони не відображались у фінальному файлі.
    if finish:
        # Видаляємо всі колонки, у назві яких міститься '_mask' або '_val'
        cols_to_drop = [col for col in df_display.columns if '_mask' not in col and '_val' not in col]
        df_display = df_display[cols_to_drop]
    
    # Визначаємо відповідність між назвами колонок для відображення та їхніми масками.
    # Зверніть увагу: якщо finish=True, то стовпці-маски вже видалені,
    # тому умовно стилізацію застосовуємо лише до колонок, для яких існують маски в оригінальному df.
    mask_mapping = {
        'Країна': 'country_mask',
        'Регіон': 'region_mask',
        'Місто': 'city_mask',
        'marker': 'relevance_mask',
        'Настрій': 'sentiment_mask',
        'Емоції': 'emotion_mask',
        'Тема': 'message_mask',
        'Офіційність': 'official_mask',
        'джерело': 'source_mask',
    }
    
    # Словник для перетворення заданих кольорів у напівпрозорі значення (rgba)
    color_mapping = {
        'green': '#C8E6C9',
        'blue': '#BBDEFB',
        'gray': '#E0E0E0',
        'red': '#FFCDD2'
    }
    
    def highlight_row(row):
        """
        Функція для застосування стилю до кожного рядка.
        Для кожної клітинки перевіряється, чи потрібно її фарбувати, і формується список CSS-стилів.
        """
        styles = [''] * len(row)
        # Отримуємо індекс поточного рядка, щоб з нього брати значення з масок
        row_idx = row.name
        
        # Зверніть увагу: якщо finish=True, то у df_display вже немає стовпців-масок.
        # Тому, щоб зберегти логіку фарбування, можна звертатись до оригінального df,
        # або попередньо зберегти інформацію про маски.
        # Тут для прикладу я звертатимусь до оригінального df, якщо такий доступний.
        for i, col in enumerate(row.index):
            mask_col = mask_mapping.get(col)
            if mask_col and mask_col in df.columns:  # звертаємось до оригінального df
                mask_val = df.loc[row_idx, mask_col]
                if pd.notnull(mask_val) and mask_val in color_mapping:
                    styles[i] = f'background-color: {color_mapping[mask_val]}'
                else:
                    styles[i] = ''
            else:
                styles[i] = ''
        return styles
        
    # Створюємо styler на основі оновленого DataFrame
    styler = df_display.style.apply(highlight_row, axis=1)
    
    # Додаткове виведення колонок для перевірки
    print("Колонки фінального DataFrame:", df_display.columns.tolist())

    styler.data['Емоції'] = styler.data['Емоції'].apply(lambda x: '' if str(x).lower()=='відсутні' else x)
    
    # Зберігаємо результат у Excel-файл
    styler.to_excel(filename, engine='openpyxl', index=index)
    print(f"Файл успішно збережено: {filename}")
