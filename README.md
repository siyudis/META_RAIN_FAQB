# LearnUp Academy — FAQ Chatbot Prototype

A basic hybrid RAG FAQ chatbot for an online training institute, powered by **GPT-OSS 120B via Groq API** with a lightweight TF-IDF FAQ retriever.

It answers typical student questions about courses, fees, enrollment, certificates, placements, and support — grounded in `data/faqs.json` so it doesn't hallucinate prices or policies.

---

## Features

- **Hybrid RAG**: retrieves the top 3 relevant FAQs from `data/faqs.json`, then lets the LLM rephrase the answer
- **Fallback mode**: works perfectly without an API key (returns direct FAQ text)
- **Web chat UI**: simple HTML/CSS/JS chat window with quick-question chips
- **Health check**: `GET /api/health` shows whether the LLM is live or in fallback
- **No vector DB needed**: TF-IDF keyword search keeps the prototype dependency-light

## Project structure

```
FAQ chatbot/
├── main.py               # FastAPI app + /api/chat endpoint
├── config.py             # Environment config (Groq key, model, branding)
├── requirements.txt
├── .env.example          # Copy to .env and add GROQ_API_KEY
├── data/
│   └── faqs.json         # 24 typical FAQs for an online training institute
└── static/
    ├── index.html
    ├── style.css
    └── app.js
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure the LLM (Groq)

Get a free Groq API key at https://console.groq.com/

```bash
copy .env.example .env
# Edit .env and set GROQ_API_KEY=gsk_....
```

> Using a different provider? See **Configuration** below — the model name and key are env-based.

### 3. Run

```bash
python main.py
```

Open http://localhost:8000 in your browser.

To run a different port:
```bash
uvicorn main:app --reload --port 8001
```

## API

### `POST /api/chat`

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"How much do courses cost?\"}"
```

Response:
```json
{
  "answer": "Course fees typically range from ₹4,999 …",
  "source": "llm | faq_fallback | no_context",
  "matched_faqs": ["How much do courses cost and what payment methods do you accept?"]
}
```

`source` tells you which path produced the answer:
- `llm` — LLM (Groq) answered using retrieved FAQ context
- `faq_fallback` — no Groq key / API error → direct FAQ text returned
- `no_context` — nothing matched; user is asked to contact support

### `GET /api/health`

```json
{ "status": "ok", "model": "openai/gpt-oss-120b", "groq_configured": true }
```

## Configuring & changing providers

All settings are environment variables (see `.env.example` and `config.py`):

| Variable | Default | What it does |
|---|---|---|
| `GROQ_API_KEY` | *(empty)* | Groq API key — leave empty to use fallback mode |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Model on Groq (change via your Groq account's available models) |
| `INSTITUTE_NAME` | `LearnUp Academy` | Branding shown in the system prompt |
| `SUPPORT_EMAIL` / `SUPPORT_PHONE` | `support@learnup.ai` / `1800-123-4567` | Fallback contact info |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Server bind |

**Switching to Together AI or Ollama (local)?** The Groq client in `main.py` (`call_llama`) is OpenAI-compatible. Point `base_url` / API key / model at the other provider and keep the same prompt.

## Editing the FAQs

Edit `data/faqs.json`. Each entry:

```json
{
  "id": "fees-payment",
  "category": "Fees & Payments",
  "question": "How much do courses cost?",
  "keywords": ["fees", "cost", "price", "emi"],
  "answer": "Full answer text here…"
}
```

- `question` — canonical question shown in `matched_faqs`
- `keywords` — synonyms that improve retrieval
- `answer` — exact text used as context for the LLM (and the fallback answer)
- No restart tricks needed — the file is re-read on every request during development.

## Pushing to GitHub

```bash
git init
git add .
git commit -m "Add FAQ chatbot prototype with GPT-OSS 120B + Groq"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

`.env` is already in `.gitignore` — never commit your Groq key.

## Suggested next steps

- [ ] Replace TF-IDF with a vector store (e.g., `sentence-transformers` + ChromaDB) as the FAQ list grows
- [ ] Add conversation history so follow-ups (“what about EMI?”) work
- [ ] Add per-category quick chips in the UI
- [ ] Stream responses (SSE) for a faster chat feel
- [ ] Add tests for retrieval quality
