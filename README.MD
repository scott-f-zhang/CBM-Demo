CBM Demo - Quick Start

Prerequisites
- Python 3.9+ recommended
- Ensure models exist under `saved_models/` (e.g., `saved_models/essay/...`)

Backend (FastAPI)
1) Create and activate a virtual environment (optional)
   - macOS/Linux:
     - `python3 -m venv .venv && source .venv/bin/activate`
   - Windows (PowerShell):
     - `python -m venv .venv; .\.venv\Scripts\Activate.ps1`
2) Install dependencies:
   - `cd backend`
   - `pip install -r requirements.txt`
3) Start the API:
   - `uvicorn main:app --reload`  (recommended), or
   - `python main.py`
4) Open API docs: `http://localhost:8000/docs`

Frontend (Streamlit)
1) In a separate shell, create/activate venv if needed (or reuse):
2) Install dependencies:
   - `cd frontend`
   - `pip install -r requirements.txt`
3) Start the UI:
   - `streamlit run app.py`
4) Open UI: `http://localhost:8501`

Notes
- The frontend expects the backend at `http://localhost:8000`.
- If you change ports, update the frontend code accordingly.

