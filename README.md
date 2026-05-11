# 👩‍🏫 ClimaVAR API
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📖 About
ClimaVAR is an AI-powered tool launching ahead of COP30 that helps anyone spot and understand climate misinformation. Like VAR in football helps referees review decisions, ClimaVAR reviews climate claims — giving communities the tools to fight back against false narratives that delay action.

**Helpful links**:
* [ClimaVAR website](https://climavar.com/)

## 🏗️ Architecture
ClimaVAR uses a two-model pipeline:
- **ClimateGPT (Erasmus.AI)** — specialist LLM trained on 4.2B climate tokens, used for scientific answer retrieval and verdict signal
- **GPT-4o-mini (OpenAI)** — used for climate relevance checking, verdict classification, and football-style answer generation

Supported languages: English, Portuguese, Spanish.

## 👨‍💻 Branching

| Branch | Description |
|---|---|
| `speed-optimization` | v1 — RAG + ChromaDB + CARDS pipeline |
| `climategpt-integration` | v2 — ClimateGPT integration |
| `climategpt-backup` | v2.1 — ClimateGPT with GPT-4o-mini fallback |
| `climavar-v3` | v3 — Current production version |

Branch naming convention:

    feat/small-feature-description
    infr/small-step-message
    hotfix/small-task-message

## 🔧 Setup

### Prerequisites
1. Clone the repository
2. Create a `.env` file in `/backend/` using the template below
3. Fill in your own API keys — never commit real values

### Environment variables

Create a `.env` file in `/backend/` with the following keys. All values must be blank — fill in your own:

    OPENAI_API_KEY=
    CLIMATEGPT_API_KEY=
    CLIMAVAR_TOKEN=
    PGDATABASE=
    PGUSER=
    PGPASSWORD=
    PGHOST=
    PGPORT=
    APP_HOST_NAMES=
    APP_DEVELOPMENT=True
    APP_DEBUG=True

Never commit real API keys or tokens. All sensitive values must be set as environment variables in Railway (production) or in your local .env file (development). The .env file is listed in .gitignore and must never be pushed to GitHub.

### MacOS — Backend setup
1. Clone the repository and navigate to the root
2. Run `cd backend`
3. Run `python3 -m venv .venv`
4. Run `. .venv/bin/activate`
5. Run `export APP_DEVELOPMENT=True`
6. Run `pip install -r requirements.txt`
7. Run `python3 manage.py migrate`
8. Run `python3 manage.py createsuperuser` — create your admin account
9. Run `python3 manage.py runserver`

### Windows — Backend setup
1. Clone the repository and navigate to the root
2. Run `cd backend`
3. Run `python -m venv .venv`
4. Run `.venv/Scripts/activate`
5. Run `$env:APP_DEVELOPMENT="true"`
6. Run `pip install -r requirements.txt`
7. Run `python manage.py migrate`
8. Run `python manage.py createsuperuser`
9. Run `python manage.py runserver`

### Post-setup
* Use `127.0.0.1` instead of `localhost` to ensure cookies are handled correctly
* API: http://127.0.0.1:8000/api/
* Admin panel: http://127.0.0.1:8000/api/admin/
* API docs: http://127.0.0.1:8000/api/docs/

## 📡 API Usage

### Check a climate claim

POST /api/misclassifications/check-misclassification/

Headers:

    Authorization: Token <your_token>
    Content-Type: application/json

Request body:

    {
        "text": "Your climate claim or question here (10-300 characters)",
        "language": "english"
    }

Accepted language values: "english", "portuguese", "spanish"

If "language" is omitted, defaults to "english".

Response:

    {
        "llm_response": "GOAL! Sea levels are indeed rising...",
        "misinformation": 0,
        "references": [],
        "source": "climategpt"
    }

misinformation values:
- 0 = accurate
- 1 = misinformation
- 2 = partial / needs context

source values:
- "climategpt" = primary pipeline active
- "backup" = GPT-4o-mini fallback (ClimateGPT temporarily offline)

## 🧪 Development notes
* Please write API documentation and tests
* Follow best practices — reusable, modular, testable code
* Run `black` before pushing
* Use `127.0.0.1` not `localhost` for local development
