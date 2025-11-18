# CBM Demo — Quick Start

## Prerequisites
- Python 3.9+ recommended
- Models available under `saved_models/` (e.g., `saved_models/essay/...`)

## Backend (FastAPI)
macOS/Linux:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload  # or: python main.py
```

Windows (PowerShell):

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload  # or: python main.py
```

Open API docs: `http://localhost:8000/docs`

## Frontend (Streamlit)

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

Open UI: `http://localhost:8501`

## Notes
- Frontend expects the backend at `http://localhost:8000`.
- Ensure `saved_models/` exists; adjust dataset selection in `backend/main.py` if needed.