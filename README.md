# 🌍 Climate Discourse Analyzer

An educational project: automated analysis of **how society discusses climate change and the environment on social media**.

It is a complete example of a **multi-stage Natural Language Processing (NLP) pipeline** that combines classic rules, a database, and modern Large Language Models (LLMs). The project shows not only *how to build it*, but **how to do it cheaply, quickly, and at scale**.

---

## 💡 The Idea

Thousands of posts about climate, green energy, pollution, and climate policy appear on social media every day. Reading and classifying them by hand is impossible. The questions a researcher wants to answer are:

- What is this post about — climate/environment or something else?
- Where is the author from (country, region)?
- What emotions and sentiment does the post carry?
- Which topic does it belong to (record heat, green energy, greenwashing…)?
- Is it a personal opinion or an **official statement** from an institution/expert (IPCC, UN, Greenpeace…)?

This tool answers all of these questions **automatically** — for thousands of posts in a row.

---

## 🧠 The Core Engineering Idea: Three Tiers (cheap → expensive)

The most important lesson of the project: **you don't need to call an expensive model for every post.** For each question the system tries three methods in turn and stops at the first one that returns an answer:

| Tier | Method | Speed | Cost |
|------|--------|-------|------|
| 1️⃣ | **Regular expressions** (keywords/patterns) | instant | free |
| 2️⃣ | **SQLite lookup** (already-known authors/places) | fast | free |
| 3️⃣ | **LLM call** (OpenAI GPT) | slow | costs tokens |

The LLM runs **only when** the simple methods fail. This saves money and time at scale. Every filled cell is traceable to its source:

> 🟩 green = pattern · ⬜ grey = database · 🟦 blue = LLM

---

## 🏗️ Architecture

```
┌────────────────────────┐     POST /files        ┌────────────────────────┐
│  Streamlit (index.py)  │ ──── GET /status ────► │    FastAPI (app.py)    │
│  file upload,          │ ◄─── POST /stop ─────  │  analysis orchestration│
│  toggles, status       │                        │  (background execution)│
└────────────────────────┘                        └───────────┬────────────┘
                                                              │
                          ┌───────────────────────────────────┼───────────────────────┐
                          ▼                                   ▼                       ▼
                  check_patterns_*()                 check_*_in_db_*()          define_*() / GPT
                  (regex, LLM.py)                  (SQLite, sentiment_v2.db)   (OpenAI API, LLM.py)
                                                              │
                                                              ▼
                                                   Result.xlsx (color-coded output)
```

- **Backend — FastAPI** ([app.py](app.py)): accepts an Excel file, runs the analysis in the background, exposes status via a REST API.
- **Frontend — Streamlit** ([index.py](index.py)): UI for uploading files, choosing analysis types, and tracking progress.
- **Logic — [LLM.py](LLM.py)**: all classification functions (patterns, DB queries, GPT calls, topic taxonomies).
- **Utilities — [handle.py](handle.py)**: Excel read/write, column normalization, color highlighting.

---

## 📊 10 Types of Analysis

| Code | What it detects | Possible values |
|------|-----------------|-----------------|
| 1 | **Relevance** | `1` / `0` — climate/environment or not |
| 2 | **Country** (patterns) | country name by author name/group |
| 3 | **Country** (database) | country name from SQLite |
| 4 | **Country** (LLM) | country name via GPT |
| 5 | **Region** (database) | region from SQLite |
| 6 | **Region** (patterns) | region by patterns |
| 7 | **Emotions** | one of 18 emotions (anxiety, hope, anger, optimism…) |
| 8 | **Sentiment** | `positive` / `negative` / `unknown` |
| 9 | **Topic** | label from the climate-topic taxonomy |
| 10 | **Officiality** | `0` or the name of an institution/role |

The analysis has several layers: first relevance filtering and geography, then emotions, sentiment, topic, and source officiality. It is a clear example of a **multi-stage pipeline** where each step adds a new slice of information about a post.

---

## 🚀 Getting Started

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create the sample database (synthetic data)
python init_db.py

# 3. Configure your OpenAI key
cp .env.example .env          # open .env and set OPENAI_API_KEY

# 4. Start the backend (terminal 1)
uvicorn app:app --port 8000

# 5. Start the frontend (terminal 2)
streamlit run index.py
```

Open the URL shown by Streamlit (usually `http://localhost:8501`), enable the analysis types you need, and upload an Excel file.

> **Without an OpenAI key**, tiers 1–2 (patterns and database) work. Tier 3 (LLM) requires `OPENAI_API_KEY`. If the `MODEL_*` variables in `.env` are empty, the default model `gpt-4o-mini` is used.

---

## 📥 Input Format

The system accepts an Excel file (`.xlsx`) where each row is one post. Key columns:

| Column | Description |
|--------|-------------|
| `Текст` | post text |
| `Заголовок` | title |
| `Автор` | author name/handle |
| `Місце публікації` | group / page / channel |
| `Мова` | post language (auto-detected) |

The other columns (`Країна`, `Регіон`, `Емоції`, `Настрій`, `Тема`, `Офіційність`) are filled by the analysis itself. If a column is missing, the program adds it automatically ([handle.py](handle.py) normalizes names and structure).

> Note: column headers are in Ukrainian because the tool is used in Ukrainian-language courses; [handle.py](handle.py) normalizes them on load.

---

## 🎓 What This Project Teaches

- **Multi-stage NLP pipelines** — how to break a complex task into sequential steps.
- **Hybrid approach** — combining rules (regex), a database, and an LLM instead of "everything through a neural network."
- **Cost optimization** — a cheap→expensive cascade so tokens aren't wasted.
- **Prompt engineering** — how to write system messages for classification.
- **Web development** — a REST API with FastAPI + an interactive UI with Streamlit.
- **Background tasks and state management** — starting/stopping a long analysis, polling status.
- **Data wrangling** — pandas, Excel read/write, table normalization.

---

## 🛠️ Tech Stack

`Python` · `FastAPI` · `Streamlit` · `OpenAI API` · `SQLite` · `pandas` · `openpyxl` · `langdetect` · `rapidfuzz`

---

## 📂 Project Structure

```
app.py            FastAPI: analysis orchestration, REST endpoints
LLM.py            Classification: GPT, regex, DB queries, topic taxonomies
index.py          Streamlit: UI
handle.py         Excel I/O, column normalization, highlighting
gen_func.py       Logging
sentiment_v2.db   SQLite: author/place reference — NOT in the repo, created by init_db.py
requirements.txt  Dependencies
```

---

## ⚠️ Notes

- The `sentiment_v2.db` database is **not included in the repository**. Create it locally.
- The `.env` file with your key is **not committed to Git** (see `.gitignore`).
- This project is built for educational purposes; production use requires additional validation and error handling.

---

## 🎯 About

This is a research and teaching project from the **State University of Trade and Economics** (Department of Digital Economy and System Analysis), Kyiv, Ukraine. It is used to teach students NLP and applied data analysis and to support research into public sentiment analysis.


# 🌍 Climate Discourse Analyzer

Навчальний проєкт: автоматизований аналіз того, **як суспільство обговорює зміни клімату та екологію в соціальних мережах**.

Це приклад повноцінного **багатоетапного конвеєра обробки природної мови (NLP)**, що поєднує класичні правила, базу даних і сучасні великі мовні моделі (LLM). Проєкт показує не лише «як зробити», а й **як зробити дешево, швидко й масштабовано**.

---

## 💡 Ідея

Щодня в соцмережах з'являються тисячі дописів про клімат, зелену енергетику, забруднення, кліматичну політику. Прочитати й класифікувати їх вручну неможливо. Питання, на які хоче відповісти дослідник:

- Про що цей допис — взагалі про клімат/екологію чи ні?
- Звідки автор (країна, регіон)?
- Які емоції та настрій у дописі?
- До якої теми він належить (рекордна спека, зелена енергетика, гринвошинг…)?
- Це особиста думка чи **офіційна заява** установи/експерта (IPCC, ООН, Greenpeace…)?

Цей інструмент відповідає на всі ці питання **автоматично** — і робить це для тисяч дописів поспіль.

---

## 🧠 Ключова інженерна ідея: три рівні (дешево → дорого)

Найважливіше, чого вчить проєкт: **не треба викликати дорогу модель для кожного допису**. Для кожного питання система пробує по черзі три способи й зупиняється на першому, що дав відповідь:

| Рівень | Метод | Швидкість | Вартість |
|--------|-------|-----------|----------|
| 1️⃣ | **Регулярні вирази** (ключові слова/патерни) | миттєво | безкоштовно |
| 2️⃣ | **Пошук у базі SQLite** (уже відомі автори/місця) | швидко | безкоштовно |
| 3️⃣ | **Виклик LLM** (OpenAI GPT) | повільно | коштує токени |

LLM запускається **лише тоді**, коли прості методи не спрацювали. Це економить гроші й час на великих обсягах. У результаті кожну заповнену клітинку видно за джерелом:

> 🟩 зелений = патерн · ⬜ сірий = база · 🟦 синій = LLM

---

## 🏗️ Архітектура

```
┌────────────────────────┐     POST /files        ┌────────────────────────┐
│   Streamlit (index.py) │ ──── GET /status ────► │    FastAPI (app.py)    │
│  завантаження файлів,  │ ◄─── POST /stop ─────  │  оркестрація аналізу   │
│  перемикачі, статус    │                        │  (фонове виконання)    │
└────────────────────────┘                        └───────────┬────────────┘
                                                              │
                          ┌───────────────────────────────────┼───────────────────────┐
                          ▼                                   ▼                       ▼
                  check_patterns_*()                 check_*_in_db_*()          define_*() / GPT
                  (регулярки, LLM.py)              (SQLite, sentiment_v2.db)   (OpenAI API, LLM.py)
                                                              │
                                                              ▼
                                                   Result.xlsx (кольоровий результат)
```

- **Backend — FastAPI** ([app.py](app.py)): приймає Excel-файл, запускає аналіз у фоні, віддає статус через REST API.
- **Frontend — Streamlit** ([index.py](index.py)): інтерфейс для завантаження файлів, вибору видів аналізу та відстеження прогресу.
- **Логіка — [LLM.py](LLM.py)**: усі функції класифікації (патерни, запити до БД, виклики GPT, таксономії тем).
- **Утиліти — [handle.py](handle.py)**: читання/запис Excel, нормалізація колонок, кольорове підсвічування.

---

## 📊 10 видів аналізу

| Код | Що визначає | Можливі значення |
|-----|-------------|------------------|
| 1 | **Релевантність** | `1` / `0` — чи про клімат/екологію |
| 2 | **Країна** (патерни) | назва країни за іменем/групою |
| 3 | **Країна** (база) | назва країни з SQLite |
| 4 | **Країна** (LLM) | назва країни за допомогою GPT |
| 5 | **Регіон** (база) | регіон з SQLite |
| 6 | **Регіон** (патерни) | регіон за патернами |
| 7 | **Емоції** | одна з 18 емоцій (тривога, надія, гнів, оптимізм…) |
| 8 | **Настрій** | `positive` / `negative` / `unknown` |
| 9 | **Тема** | мітка з таксономії кліматичних тем |
| 10 | **Офіційність** | `0` або назва установи/посади |

Аналіз має кілька рівнів: спочатку відбір релевантних дописів і географія, далі — емоції, настрій, тема та офіційність джерела. Це гарний приклад **багатоетапного конвеєра**, де кожен крок додає новий зріз інформації про допис.

---

## 🚀 Як запустити

```bash
# 1. Встановити залежності
pip install -r requirements.txt

# 2. Створити базу-приклад (синтетичні дані)
python init_db.py

# 3. Налаштувати ключ OpenAI
cp .env.example .env          # відкрийте .env і впишіть OPENAI_API_KEY

# 4. Запустити backend (термінал 1)
uvicorn app:app --port 8000

# 5. Запустити frontend (термінал 2)
streamlit run index.py
```

Відкрийте браузер за адресою, яку покаже Streamlit (зазвичай `http://localhost:8501`), увімкніть потрібні види аналізу та завантажте Excel-файл.

> **Без ключа OpenAI** працюють рівні 1–2 (патерни й база). Для рівня 3 (LLM) потрібен `OPENAI_API_KEY`. Якщо змінні `MODEL_*` у `.env` порожні — використовується базова модель `gpt-4o-mini`.

---

## 📥 Формат вхідних даних

Система приймає Excel-файл (`.xlsx`), де кожен рядок — один допис. Ключові колонки:

| Колонка | Опис |
|---------|------|
| `Текст` | текст допису |
| `Заголовок` | заголовок |
| `Автор` | ім'я/нік автора |
| `Місце публікації` | група / сторінка / канал |
| `Мова` | мова допису (визначається автоматично) |

Інші колонки (`Країна`, `Регіон`, `Емоції`, `Настрій`, `Тема`, `Офіційність`) заповнює сам аналіз. Якщо якоїсь колонки бракує — програма додасть її автоматично ([handle.py](handle.py) нормалізує назви та структуру).

---

## 🎓 Чого вчить цей проєкт

- **Багатоетапні NLP-конвеєри** — як розбити складну задачу на послідовні кроки.
- **Гібридний підхід** — поєднання правил (regex), бази даних і LLM замість «усе через нейромережу».
- **Оптимізація вартості** — каскад «дешево → дорого», щоб не витрачати токени даремно.
- **Промпт-інжиніринг** — як писати системні повідомлення для класифікації.
- **Веброзробка** — REST API на FastAPI + інтерактивний інтерфейс на Streamlit.
- **Фонові задачі та керування станом** — запуск/зупинка довгого аналізу, опитування статусу.
- **Робота з даними** — pandas, читання/запис Excel, нормалізація таблиць.

---

## 🛠️ Технології

`Python` · `FastAPI` · `Streamlit` · `OpenAI API` · `SQLite` · `pandas` · `openpyxl` · `langdetect` · `rapidfuzz`

---

## 📂 Структура проєкту

```
app.py            FastAPI: оркестрація аналізу, REST-ендпоінти
LLM.py            Класифікація: GPT, регулярки, запити до БД, таксономії тем
index.py          Streamlit: інтерфейс
handle.py         Excel I/O, нормалізація колонок, підсвічування
gen_func.py       Логування
sentiment_v2.db   SQLite: довідник авторів/місць — НЕ в репозиторії, створюється init_db.py
requirements.txt  Залежності
```

---

## ⚠️ Примітки

- База `sentiment_v2.db` **не входить до репозиторію**. Створіть її локально.
- Файл `.env` із вашим ключем **не потрапляє в Git** (див. `.gitignore`).
- Проєкт створено в навчальних цілях; для продакшену потрібні додаткові перевірки та обробка помилок.
