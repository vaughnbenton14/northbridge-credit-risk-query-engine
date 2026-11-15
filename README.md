# Northbridge Bank Credit Risk Query Engine

Academic Streamlit proof of concept using the **sanitized** Northbridge portfolio dataset.

## Repository files

- `app.py` — Streamlit user interface
- `query_engine.py` — routing, SQL generation, validation, retry, execution, response, and audit trace
- `credit_risk_portfolio_sanitized.db` — sanitized SQLite database
- `test_queries_sanitized.csv` — five supplied ground-truth test cases
- `requirements.txt` — Python dependencies
- `.streamlit/secrets.toml.example` — names of optional Streamlit secrets; contains no live credential
- `.gitignore` — prevents accidental commitment of local secret files

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Without an API key, the supplied five test cases run in deterministic development mode. When approved credentials are supplied through Streamlit Secrets, the same pipeline can use separate generation and evaluation LLMs.

## Streamlit Community Cloud

1. Upload these repository files to a GitHub repository.
2. In Streamlit Community Cloud, create an app from the repository.
3. Choose branch `main` and main file path `app.py`.
4. Set Python to 3.12 if the deployment interface exposes a Python-version setting.
5. Add approved API credentials in Advanced settings -> Secrets. Never place a live API key in GitHub.
6. Deploy and run TC-01 through TC-05, capturing the screenshots required by the project rubric.

## Governance safeguards

- Database connection uses SQLite read-only mode.
- Only one `SELECT`/`WITH` statement is allowed.
- Mutating/administrative SQL keywords are blocked.
- SQLite `EXPLAIN` validates parse/schema planning before execution.
- Generated SQL gets one retry after a failed gate, then escalates to a human analyst.
- The audit log preserves the initial SQL and initial gate result, plus the retry SQL and retry gate result when a retry occurs; the Streamlit UI displays both attempts and the final decision.
- Every answer exposes route, query ID/generated SQL, validation outcome, executed SQL, raw data, confidence, and timestamp.
- NPA/impaired is consistently defined as `Substandard`, `Doubtful`, or `Loss`; `Special Mention` is excluded.
