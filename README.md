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
