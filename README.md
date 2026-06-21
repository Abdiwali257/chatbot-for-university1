# SIMAD University AI Assistant

This project is a Retrieval-Augmented Generation (RAG) chatbot for answering
questions from verified SIMAD University documents.

It reads PDF, Word, and Excel files; stores searchable embeddings in ChromaDB;
retrieves relevant passages; and optionally uses a Hugging Face model to write a
concise answer.

## Current data

The ingestion script scans:

- `data/` recursively
- Supported documents placed in the project root

Supported formats are `.pdf`, `.docx`, `.xlsx`, and `.xls`.

## Setup

The included virtual environment is currently ready to use:

```powershell
.\venv\Scripts\Activate.ps1
```

If it later breaks or the project moves to another computer, install Python
3.12 and recreate it:

```powershell
Remove-Item -Recurse -Force venv
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a **fine-grained** Hugging Face access token and enable:

`Make calls to Inference Providers`

Without that permission, Hugging Face returns `403 Forbidden`. Never commit or
share the token. Configure it interactively:

```powershell
.\venv\Scripts\python.exe configure_huggingface.py
```

The default generation model is `Qwen/Qwen2.5-7B-Instruct` with provider
selection set to `auto`. Change `HF_MODEL` or `HF_PROVIDER` in `.env` when using
a different Hugging Face model or inference provider.

The default embedding model for retrieval is `all-MiniLM-L6-v2`. Change
`EMBEDDING_MODEL` in `.env` only if you also rebuild the knowledge base, because
Chroma collections must use embeddings from the same model at training time and
chat time. For slower computers, keep `EMBEDDING_BATCH_SIZE` low, such as `8`;
larger values can use much more memory with BGE-M3.

The chatbot still works without a token, but it returns retrieved excerpts
instead of a conversational generated answer.

## Build the knowledge base

Run this whenever documents are added or changed:

```powershell
python train_data.py
```

The script rebuilds `chroma_db/simad_knowledge_base` and prints the number of
files and chunks indexed.

## Run the Django web application

The simplest command works even when the virtual environment is not activated:

```powershell
.\runserver.cmd
```

The launcher prevents multiple Django processes from competing on port 8000.

Alternatively, call the project's Python executable directly:

```powershell
.\venv\Scripts\python.exe manage.py migrate
.\venv\Scripts\python.exe manage.py runserver
```

Open `http://127.0.0.1:8000/` in a browser.

The Django application provides:

- A responsive browser chat interface
- `POST /api/chat/` for JSON chatbot requests
- Session-backed conversation history that remains visible after page reload
- `POST /api/clear/` to clear conversation history
- A shared cached RAG service so the embedding model loads once
- Structured course answers with codes, credits, theory, and practical hours
- Deterministic tuition comparisons calculated from the official fee document

Example API request:

```json
{"question": "What is FOC?"}
```

The original terminal chatbot remains available:

```powershell
.\runchatbot.cmd
```

## Django tests

```powershell
.\venv\Scripts\python.exe manage.py check
.\venv\Scripts\python.exe manage.py test
.\venv\Scripts\python.exe -m unittest test_chatbot_provider.py
```

## Evaluate retrieval

Before changing the embedding model, chunking, or threshold, run:

```powershell
.\venv\Scripts\python.exe evaluate_retrieval.py
```

After changing the embedding model, rebuild the knowledge base:

```powershell
.\venv\Scripts\python.exe train_data.py
```

Add real student questions and their expected source documents to
`evaluation_questions.json`. Treat this score as a basic regression test.

## Recommended development order

1. Review and clean source documents, especially placeholders and spelling.
2. Create a test-question set with expected answers and sources.
3. Measure retrieval and answer accuracy before changing models or chunk sizes.
4. Add a web or mobile interface only after the evaluation results are reliable.
5. Add an update process so university staff can keep official information current.

## Accuracy rules

- The chatbot should answer only from verified SIMAD documents.
- User-facing answers should not expose document names, page numbers, or internal retrieval wording.
- Fees, dates, requirements, policies, and course details must never be guessed.
- When the dataset does not contain an answer, the chatbot should say so.
