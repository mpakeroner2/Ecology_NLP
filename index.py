import streamlit as st
st.set_page_config(page_title="Climate Discourse Analyzer", layout="centered")
from gen_func import log, clean_log
import time
from typing import Dict, Any
import pandas as pd
import requests
from streamlit_cookies_manager import EncryptedCookieManager
import uuid

# ============================
# Configuration
# ============================
FASTAPI_BASE_URL = "http://localhost:8000"  # <-- change if server runs elsewhere
POLL_INTERVAL = 1  # seconds between /status requests

# Initialize storage before any st. commands
storage = EncryptedCookieManager(
        prefix="senti-test2",
        password="123sdasdfsdfsdfsdfsdfsdfsdfsdfsdf"
    )

def get_unique_key(prefix="download"):
    return f"{prefix}_{uuid.uuid4()}"

# region poll status
def poll_status(poll_container) -> str:
    """Poll /status until a terminal state is reached, rendering updates each cycle."""
    while True:
        try:
            r = requests.get(f"{FASTAPI_BASE_URL}/status", timeout=10)
            payload = r.json()
        except Exception:
            payload = {"status": "error", "details": "Бекенд не відповідає. Зверніться до розробника."}

        show_request(poll_container, payload)

        current = payload.get("status", "unknown")
        if current in {"done", "error", "stopped"}:
            return payload

        time.sleep(POLL_INTERVAL)

# region show request
def show_request(obj, response: Dict[str, Any]) -> None:
    """Render the FastAPI /status payload into the Block3 container."""
    # Prepare timestamp (без секунд)
    timestamp = time.strftime('%Y-%m-%d %H:%M')
    status = response.get("status", "unknown")
    details = response.get("details", "")

    # Build message lines
    lines = [timestamp]
    # Determine status line and color
    if status == "done":
        status_line = "СТАТУС: РОБОТА ЗАКІНЧЕНА"
        color = "green"
    elif status == "error":
        status_line = "СТАТУС: ПОМИЛКА. ГОТОВИЙ ПРАЦЮВАТИ."
        color = "red"
    elif status == "busy":
        status_line = "СТАТУС: ПРАЦЮЄ."
        color = "orange"
    elif status == "stopped":
        status_line = "СТАТУС: ЗУПИНЕНО КОРИСТУВАЧЕМ"
        color = "yellow"
    elif status == 'info':
        status_line = "СТАТУС: ІНФОРМАЦІЯ"
        color = "blue"
    else:
        # unknown or missing status
        status_line = "СТАТУС: НЕВІДОМИЙ СТАН"
        color = "red"
    lines.append(status_line)

    # Third line: details or progress
    if status == "busy":
        prog = []
        for k, v in response.items():
            if k != "status":
                if isinstance(v, (float, int)):
                    prog.append(f"{k}: {int(v*100)}%")
                else:
                    prog.append(f"{k}: {v}")
        if prog:
            lines.append(", ".join(prog))
    else:
        # for error or unknown, show details or developer prompt
        if (status == "error" or status == 'info') and details:
            lines.append(details)
        elif status not in {"done", "busy", "stopped"}:
            lines.append("Зверніться до розробника")

    # Render combined message in Block3
    message = "<br>".join(lines)

    log(f'status = {status}')
    # Download button for Excel (only for done, error, stopped)
    if status in {"done", "error", "stopped"} and details != "Бекенд не відповідає. Зверніться до розробника.":
        log('Готуємо файл до завантаження', file='log1.txt')
        try:
            cont = obj.container()
            # Показуємо повідомлення про статус
            cont.markdown(
                f"<span style='color:{color}; font-size:14px;'>" + message + "</span>",
                unsafe_allow_html=True,
            )
            # Додаємо кнопку завантаження Excel
            with open("Result.xlsx", "rb") as f:
                data = f.read()
            log('Файл знайдений', file='log1.txt')
            cont.download_button(
                label="Завантажити останній Excel",
                data=data,
                file_name="Result.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=get_unique_key()
            )
            log('Кнопка підготовлена', file='log1.txt')
        except Exception as e:
            log(f'Помилка зчитування файла: {e}', file='log1.txt')
            cont.error("Не вдалося завантажити Excel файл.")
    else:
        # Файл не готовий. Сервер ще працює
        obj.markdown(
        f"<span style='color:{color}; font-size:14px;'>" + message + "</span>",
        unsafe_allow_html=True,
    )

# region Trace button
if st.button("Відслідковувати статус"):
    # Start polling loop
    poll_container = st.empty()
    poll_status(poll_container)
        
# ============================
# Reserve containers for subsequent blocks
# ============================
st.write("### Параметри аналізу дискурсу")
# region PARAMETERS
time.sleep(1)
if not list(storage.values()):
    relevance_enabled = True
    country_bd_enabled = True
    country_patterns_enabled = True
    country_ai_enabled = True
    region_bd_enabled = True
    region_patterns_enabled = True
    sentiments_enabled = True
    emotions_enabled = True
    messages_enabled = True
    officials_enabled = True
else:
    relevance_enabled = True if storage.get("relevance_enabled")=='True' else False
    country_bd_enabled = True if storage.get("country_bd_enabled")=='True' else False
    country_ai_enabled = True if storage.get("country_ai_enabled")=='True' else False
    country_patterns_enabled = True if storage.get("country_patterns_enabled")=='True' else False
    region_bd_enabled = True if storage.get("region_bd_enabled")=='True' else False
    region_patterns_enabled = True if storage.get("region_patterns_enabled")=='True' else False
    sentiments_enabled = True if storage.get("sentiments_enabled")=='True' else False
    emotions_enabled = True if storage.get("emotions_enabled")=='True' else False
    messages_enabled = True if storage.get("messages_enabled")=='True' else False
    officials_enabled = True if storage.get("officials_enabled")=='True' else False
    
col1, col2 = st.columns(2)

with col1:
    relevance_enabled = st.toggle("Релевантність", value=relevance_enabled)
    country_bd_enabled = st.toggle("Країна БД", value=country_bd_enabled)
    country_patterns_enabled = st.toggle("Країна Патерни", value=country_patterns_enabled)
    country_ai_enabled = st.toggle("Країна ШІ", value=country_ai_enabled)
    region_bd_enabled = st.toggle("Регіон БД", value=region_bd_enabled)
    region_patterns_enabled = st.toggle("Регіон патерни", value=region_patterns_enabled)

with col2:
    emotions_enabled = st.toggle("Емоції", value=emotions_enabled)
    sentiments_enabled = st.toggle("Настрій", value=sentiments_enabled) 
    messages_enabled = st.toggle("Теми", value=messages_enabled)   
    officials_enabled = st.toggle("Офіційні заяви", value=officials_enabled)

# Зберігаємо налаштування
if st.button("Зберегти"):
    storage["relevance_enabled"] = str(relevance_enabled)
    storage["country_bd_enabled"] = str(country_bd_enabled)
    storage["country_ai_enabled"] = str(country_ai_enabled)
    storage["country_patterns_enabled"] = str(country_patterns_enabled)
    storage["region_bd_enabled"] = str(region_bd_enabled)
    storage["region_patterns_enabled"] = str(region_patterns_enabled)
    storage["sentiments_enabled"] = str(sentiments_enabled)
    storage["emotions_enabled"] = str(emotions_enabled)
    storage["messages_enabled"] = str(messages_enabled)
    storage["officials_enabled"] = str(officials_enabled)
    storage.save()
    st.write('Зміни збережено!')

# endregion

# region UPLOAD FILE 1
st.write("### Завантаження файлів для аналізу релевантності та географії")

current_file_1 = st.file_uploader("Оберіть файл для аналізу", key='analysis_1')

if current_file_1 and 'file1' not in st.session_state.keys():
    
    # Перевіряємо статус сервера перед обробкою файлу
    poll_container = st.empty()
    try:
        status_resp = requests.get(f"{FASTAPI_BASE_URL}/status", timeout=7)
        status_payload = status_resp.json()
    except Exception:
        st.error("Не вдалося отримати статус сервера.")
        st.stop()
    
    if status_payload.get("status") == "busy":
        poll_status(poll_container)
    else:
        # Зчитуємо Excel-файл і зберігаємо під назвою "current.xlsx"
        df = pd.read_excel(current_file_1)
        df.to_excel("current.xlsx", index=False)

        # Формуємо кортеж згідно з параметрами у storage
        option_map = [
            ("relevance_enabled", 1),
            ("country_patterns_enabled", 2),
            ("country_bd_enabled", 3),
            ("country_ai_enabled", 4),
            ("region_bd_enabled", 5),
            ("region_patterns_enabled", 6),
        ]
        selected_options = tuple(
            num for key, num in option_map if storage.get(key) == 'True'
        )
        
        # Відправляємо запит до FastAPI
        try:
            payload = {
                "file_path": "current.xlsx",
                "codes": selected_options
            }
            response = requests.post(
                f"{FASTAPI_BASE_URL}/files",
                json=payload,
                timeout=7
            )
            if response.ok:
                resp_json = response.json()
                status = resp_json.get("status", "")
                details = resp_json.get("details", "")
                if status == "busy":
                    show_request(poll_container, {'status': 'busy'})
                elif status == "error":
                    show_request(poll_container, {"status": "error", "details": details})
                poll_status(poll_container)
            else:
                st.error(f"Помилка при надсиланні файлу: {response.text}")
        except Exception as e:
            st.error(f"Виникла помилка: {e}")
        st.session_state['file1'] = True
# endregion

# region UPLOAD FILE 2
st.write("### Завантаження файлів для аналізу позиції, емоцій та меседжів")

current_file_2 = st.file_uploader("Оберіть файл для аналізу", key='analysis_2')

if current_file_2 and 'file2' not in st.session_state.keys():
    # Перевіряємо статус сервера перед обробкою файлу
    poll_container = st.empty()
    try:
        status_resp = requests.get(f"{FASTAPI_BASE_URL}/status", timeout=7)
        status_payload = status_resp.json()
    except Exception:
        st.error("Не вдалося отримати статус сервера.")
        st.stop()
    if status_payload.get("status") == "busy":
        poll_status(poll_container)
    else:
        # Зчитуємо Excel-файл і зберігаємо під назвою "current.xlsx"
        df = pd.read_excel(current_file_2)
        df.to_excel("current.xlsx", index=False)

        # Формуємо кортеж згідно з параметрами у storage
        option_map = [
            ("emotions_enabled", 7),
            ("sentiments_enabled", 8),
            ("messages_enabled", 9),
            ("officials_enabled", 10),
        ]
        selected_options = tuple(
            num for key, num in option_map if storage.get(key) == 'True'
        )
        # Відправляємо запит до FastAPI
        try:
            payload = {
                "file_path": "current.xlsx",
                "codes": selected_options
            }
            response = requests.post(
                f"{FASTAPI_BASE_URL}/files",
                json=payload,
                timeout=10
            )
            if response.ok:
                resp_json = response.json()
                status = resp_json.get("status", "")
                details = resp_json.get("details", "")
                if status == "busy":
                    show_request(poll_container, {'status': 'busy'})
                elif status == "error":
                    show_request(poll_container, {"status": "error", "details": details})
                poll_status(poll_container)
            else:
                st.error(f"Помилка при надсиланні файлу: {response.text}")
        except Exception as e:
            st.error(f"Виникла помилка: {e}")
        st.session_state['file2'] = True
# endregion

# region STOP ANALYSIS
if "stop_clicked" not in st.session_state:
    st.session_state.stop_clicked = False

if st.button("Зупинити аналіз"):
    st.session_state.stop_clicked = True

if st.session_state.stop_clicked:
    confirm = st.checkbox("Підтвердіть зупинку аналізу", key="confirm_action")
    if confirm:
        try:
            r = requests.post(f"{FASTAPI_BASE_URL}/stop", timeout=10)
            payload = r.json()
        except Exception:
            payload = {"status": "error", "details": "Бекенд не відповідає. Зверніться до розробника."}
        poll_container = st.empty()
        show_request(poll_container, {"status": "info", "details": "Зупиняємо роботу."})
        poll_status(poll_container)
        st.session_state.stop_clicked = False
        
        