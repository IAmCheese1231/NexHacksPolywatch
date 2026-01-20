# NexHacksPolywatch (Polywatch)

Polywatch is a Polymarket risk dashboard that connects anomaly alerts → correlation simulation → portfolio impact.

## Run Locally

### 1) Python backend (portfolio + job runner)

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

# Run on 8002 (this repo defaults to 8002 in dev)
PORT=8002 python backend/main.py
```

Backend should be available at:

```text
http://localhost:8002/docs
```

### 2) Vite portfolio dashboard (embedded)

In a new terminal:

```bash
cd frontend
npm install
npm run dev
```

Vite dashboard:

```text
http://localhost:5173
```

### 3) Next.js app (main UI + API routes)

In a new terminal from the repo root:

```bash
npm install
npm run dev
```

Next app:

```text
http://localhost:3000
```

## Notes

- The Next.js Simulation “Add to Portfolio” calls `/api/portfolio/add`, which forwards to the Python backend.
	- Default dev backend: `http://127.0.0.1:8002`
- The Vite dashboard uses a dev proxy for `/api/*` to reach the backend.
	- Configured in `frontend/.env.local` via `VITE_API_PROXY_TARGET`.
