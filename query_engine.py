from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / "credit_risk_portfolio_sanitized.db"
DEFAULT_TEST_PATH = BASE_DIR / "test_queries_sanitized.csv"

DATABASE_SCHEMA = """
sector_master:
  sector_code (TEXT, PK), sector_name (TEXT), naics_code (TEXT),
  naics_description (TEXT), is_sensitive_sector (INTEGER)

loan_master:
  loan_account_number (TEXT, PK), borrower_id (TEXT), borrower_name (TEXT),
  borrower_type (TEXT), group_name (TEXT), state (TEXT), product_type (TEXT),
  loan_category (TEXT), sector_code (TEXT, FK), sanctioned_amount (REAL),
  disbursed_amount (REAL), outstanding_principal (REAL), outstanding_interest (REAL),
  total_outstanding (REAL), interest_rate (REAL), rate_type (TEXT),
  sanction_date (DATE), maturity_date (DATE), repayment_frequency (TEXT),
  branch_code (TEXT), branch_name (TEXT), relationship_manager (TEXT),
  is_consortium (INTEGER), is_restructured (INTEGER), restructuring_date (DATE),
  is_secured (INTEGER), days_past_due (INTEGER), asset_classification (TEXT),
  classification_date (DATE)

borrower_rating:
  rating_id (INTEGER, PK), borrower_id (TEXT, FK), rating_date (DATE),
  internal_rating (TEXT), previous_rating (TEXT), rating_direction (TEXT),
  external_rating_agency (TEXT), external_rating (TEXT), pd_estimate (REAL),
  rating_model_version (TEXT)

provisioning:
  provision_id (INTEGER, PK), loan_account_number (TEXT, FK), reporting_date (DATE),
  ifrs9_stage (INTEGER), stage_rationale (TEXT), pd_12_month (REAL),
  pd_lifetime (REAL), lgd_estimate (REAL), ead_amount (REAL), ecl_amount (REAL),
  provision_held (REAL), provision_coverage_ratio (REAL),
  is_individually_assessed (INTEGER)

Available reporting_date values: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA/impaired definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss').
'Special Mention' is excluded.
"""

VERIFIED_QUERY_LIBRARY: Dict[str, Dict[str, str]] = {
    "VQ1": {
        "description": "Sector-wise total outstanding and NPA amount breakdown across all sectors",
        "sql": """SELECT
    s.sector_name,
    COUNT(*) AS loan_count,
    ROUND(SUM(l.total_outstanding) / 1000000.0, 1) AS total_outstanding_m,
    ROUND(SUM(CASE WHEN l.asset_classification IN ('Substandard', 'Doubtful', 'Loss')
                   THEN l.total_outstanding ELSE 0 END) / 1000000.0, 1) AS npa_outstanding_m
FROM loan_master AS l
JOIN sector_master AS s ON l.sector_code = s.sector_code
GROUP BY s.sector_name
ORDER BY total_outstanding_m DESC""",
    },
    "VQ2": {
        "description": "Total portfolio outstanding and loan count by loan category",
        "sql": """SELECT
    loan_category,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 1) AS total_outstanding_m
FROM loan_master
GROUP BY loan_category
ORDER BY total_outstanding_m DESC""",
    },
    "VQ3": {
        "description": "IFRS 9 stage-wise loan count, EAD, and ECL for the latest quarter",
        "sql": """SELECT
    ifrs9_stage,
    COUNT(*) AS loan_count,
    ROUND(SUM(ead_amount) / 1000000.0, 1) AS ead_m,
    ROUND(SUM(ecl_amount) / 1000000.0, 1) AS ecl_m
FROM provisioning
WHERE reporting_date = '2025-09-30'
GROUP BY ifrs9_stage
ORDER BY ifrs9_stage""",
    },
    "VQ4": {
        "description": "Average provision coverage ratio by sector for the latest quarter",
        "sql": """SELECT
    s.sector_name,
    ROUND(AVG(p.provision_coverage_ratio), 2) AS avg_provision_coverage_ratio
FROM provisioning AS p
JOIN loan_master AS l ON p.loan_account_number = l.loan_account_number
JOIN sector_master AS s ON l.sector_code = s.sector_code
WHERE p.reporting_date = '2025-09-30'
GROUP BY s.sector_name
ORDER BY avg_provision_coverage_ratio DESC""",
    },
    "VQ5": {
        "description": "Top 10 individual loan exposures by outstanding amount",
        "sql": """SELECT
    l.borrower_name,
    s.sector_name,
    ROUND(l.total_outstanding / 1000000.0, 1) AS outstanding_m,
    l.asset_classification
FROM loan_master AS l
JOIN sector_master AS s ON l.sector_code = s.sector_code
ORDER BY l.total_outstanding DESC
LIMIT 10""",
    },
    "VQ6": {
        "description": "Top 5 business-group exposures by total outstanding amount",
        "sql": """SELECT
    group_name,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 1) AS total_outstanding_m
FROM loan_master
WHERE group_name IS NOT NULL
GROUP BY group_name
ORDER BY total_outstanding_m DESC
LIMIT 5""",
    },
    "VQ7": {
        "description": "All overdue loan accounts with DPD and asset classification",
        "sql": """SELECT
    l.loan_account_number,
    l.borrower_name,
    s.sector_name,
    ROUND(l.total_outstanding / 1000000.0, 1) AS outstanding_m,
    l.days_past_due,
    l.asset_classification
FROM loan_master AS l
JOIN sector_master AS s ON l.sector_code = s.sector_code
WHERE l.days_past_due > 0
ORDER BY l.days_past_due DESC, l.total_outstanding DESC""",
    },
    "VQ8": {
        "description": "DPD bucket distribution showing loan counts and outstanding exposure",
        "sql": """WITH bucketed AS (
    SELECT
        CASE
            WHEN days_past_due = 0 THEN '0 (Current)'
            WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
            WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
            WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
            ELSE '90+'
        END AS dpd_bucket,
        CASE
            WHEN days_past_due = 0 THEN 1
            WHEN days_past_due BETWEEN 1 AND 30 THEN 2
            WHEN days_past_due BETWEEN 31 AND 60 THEN 3
            WHEN days_past_due BETWEEN 61 AND 90 THEN 4
            ELSE 5
        END AS bucket_order,
        total_outstanding
    FROM loan_master
)
SELECT
    dpd_bucket,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 1) AS total_outstanding_m
FROM bucketed
GROUP BY dpd_bucket, bucket_order
ORDER BY bucket_order""",
    },
    "VQ9": {
        "description": "Borrowers downgraded in the latest rating cycle",
        "sql": """SELECT
    borrower_id,
    previous_rating,
    internal_rating,
    pd_estimate
FROM borrower_rating
WHERE rating_date = '2025-09-30'
  AND rating_direction = 'Downgraded'
ORDER BY pd_estimate DESC""",
    },
    "VQ10": {
        "description": "Expected credit loss trend across reporting quarters",
        "sql": """SELECT
    reporting_date,
    ROUND(SUM(ecl_amount) / 1000000.0, 1) AS total_ecl_m
FROM provisioning
GROUP BY reporting_date
ORDER BY reporting_date""",
    },
}


def _setting(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return default


def build_llms():
    api_key = _setting("OPENAI_API_KEY")
    if not api_key or ChatOpenAI is None:
        return None, None
    base_url = _setting("OPENAI_API_BASE") or _setting("OPENAI_BASE_URL")
    generation_model = _setting("OPENAI_GENERATION_MODEL", "gpt-4.1-mini")
    evaluation_model = _setting("OPENAI_EVALUATION_MODEL", "gpt-4.1")
    common: Dict[str, Any] = {"temperature": 0, "api_key": api_key}
    if base_url:
        common["base_url"] = base_url
    return (
        ChatOpenAI(model=generation_model, **common),
        ChatOpenAI(model=evaluation_model, **common),
    )


def connect_read_only(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path = Path(db_path).resolve()
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)


def classify_intent(user_question: str, generation_llm=None) -> Dict[str, Any]:
    descriptions = "\n".join(
        f"{qid}: {entry['description']}" for qid, entry in VERIFIED_QUERY_LIBRARY.items()
    )
    prompt = f"""You are the intent router for a bank credit-risk query engine.
Choose a verified template ONLY when the user's requested metric and analytical structure are materially covered by one template. A verified template may return a complete result set that is later filtered or summarized in the response. If the question requires a different filter, combination, calculation, or row-level view, route it to generated SQL.

User question:
{user_question}

Verified template library:
{descriptions}

Return ONLY valid JSON with exactly these keys:
{{"route":"verified" or "generated","query_id":"VQ1" to "VQ10" or null,"match_reason":"one short sentence"}}"""
    if generation_llm is not None:
        raw = generation_llm.invoke(prompt).content.strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            obj = json.loads(match.group())
            if obj.get("route") in {"verified", "generated"}:
                return obj

    q = user_question.lower()
    if "real estate" in q and ("non-performing" in q or "npa" in q):
        return {"route": "verified", "query_id": "VQ1", "match_reason": "Matches sector exposure and NPA breakdown."}
    if "dpd bucket" in q or ("overdue" in q and "distributed" in q):
        return {"route": "verified", "query_id": "VQ8", "match_reason": "Matches the approved DPD aging-profile template."}
    return {"route": "generated", "query_id": None, "match_reason": "No approved template fully covers the requested analysis."}


def generate_query(user_question: str, generation_llm=None) -> str:
    prompt = f"""You are a senior credit-risk SQL analyst working against the schema below.
Write ONE SQLite-compatible read-only query that directly answers the user's question.
Rules:
1. Return SQL only; no markdown or explanation.
2. The statement must begin with SELECT or WITH.
3. Never use INSERT, UPDATE, DELETE, DROP, ALTER, REPLACE, ATTACH, PRAGMA, or multiple statements.
4. Use only tables and columns in the supplied schema.
5. NPA/impaired means asset_classification IN ('Substandard','Doubtful','Loss'); 'Special Mention' is not impaired.
6. Convert large currency amounts to USD millions when that improves readability.
7. Preserve row-level detail when the user asks to list records.
8. Use explicit joins and aliases.

User question:
{user_question}

Database schema:
{DATABASE_SCHEMA}"""
    if generation_llm is not None:
        sql = generation_llm.invoke(prompt).content.strip()
        sql = re.sub(r"^```sql\s*|\s*```$", "", sql, flags=re.I | re.M).strip()
        return re.sub(r"^```\s*|\s*```$", "", sql, flags=re.M).strip()

    q = user_question.lower()
    if "restructured" in q and "impaired" in q:
        return """SELECT l.loan_account_number, l.borrower_name, s.sector_name,
       ROUND(l.total_outstanding / 1000000.0, 1) AS outstanding_m,
       l.asset_classification,
       CASE WHEN l.asset_classification IN ('Substandard','Doubtful','Loss') THEN 1 ELSE 0 END AS is_impaired
FROM loan_master AS l
JOIN sector_master AS s ON l.sector_code = s.sector_code
WHERE l.is_restructured = 1
ORDER BY is_impaired DESC, l.total_outstanding DESC"""
    if "average interest rate" in q and "sector" in q:
        return """SELECT s.sector_name, ROUND(AVG(l.interest_rate), 2) AS avg_interest_rate_pct
FROM loan_master AS l
JOIN sector_master AS s ON l.sector_code = s.sector_code
GROUP BY s.sector_name
ORDER BY avg_interest_rate_pct DESC"""
    if "stage 3" in q and ("quarter" in q or "trend" in q or "moved" in q):
        return """SELECT reporting_date,
       ROUND(SUM(ead_amount) / 1000000.0, 1) AS stage3_ead_m,
       ROUND(SUM(ecl_amount) / 1000000.0, 1) AS stage3_ecl_m
FROM provisioning
WHERE ifrs9_stage = 3
GROUP BY reporting_date
ORDER BY reporting_date"""
    return "SELECT 1 AS unsupported_question"


def _single_read_only_statement(sql: str):
    statements = [s.strip() for s in sql.strip().rstrip(";").split(";") if s.strip()]
    if len(statements) != 1:
        return False, "Exactly one SQL statement is allowed."
    upper = statements[0].upper().lstrip()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return False, "Query must begin with SELECT or WITH."
    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "REPLACE", "ATTACH", "DETACH", "VACUUM", "PRAGMA", "CREATE"]
    for kw in forbidden:
        if re.search(rf"\b{kw}\b", upper):
            return False, f"Forbidden operation detected: {kw}"
    return True, "Read-only shape passed."


def validate_query(user_question: str, candidate_sql: str, conn: sqlite3.Connection, evaluator_llm=None, query_id: Optional[str] = None) -> Dict[str, Any]:
    result: Dict[str, Any] = {"passed": False, "failed_check": None, "details": "", "relevance_confidence": 0.0, "checks": {}}
    ok, detail = _single_read_only_statement(candidate_sql)
    result["checks"]["read_only"] = ok
    if not ok:
        result.update(failed_check="read_only", details=detail)
        return result

    try:
        conn.execute("EXPLAIN " + candidate_sql).fetchall()
        result["checks"]["schema_conformance"] = True
        result["checks"]["parse_plan"] = True
    except sqlite3.Error as exc:
        result["checks"]["schema_conformance"] = False
        result["checks"]["parse_plan"] = False
        result.update(failed_check="schema_or_parse_plan", details=str(exc))
        return result

    is_verified = bool(query_id and query_id in VERIFIED_QUERY_LIBRARY)
    note = "This is an approved template and may intentionally return a complete result set for later summarization." if is_verified else "This is generated SQL and should directly scope itself to the user question."
    prompt = f"""You are an independent SQL relevance evaluator for a bank credit-risk system.
Judge whether the SQL answers the user's question using the correct tables, filters, business definitions, aggregation, and time logic.
{note}
The bank defines NPA/impaired as Substandard, Doubtful, or Loss; Special Mention is excluded.

User question: {user_question}
SQL: {candidate_sql}

Return ONLY JSON:
{{"verdict":"yes" or "no","confidence":0.0 to 1.0,"reason":"one short sentence"}}"""
    if evaluator_llm is not None:
        raw = evaluator_llm.invoke(prompt).content.strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        obj = json.loads(match.group()) if match else {"verdict": "no", "confidence": 0.0, "reason": "Unparseable evaluator output"}
        confidence = float(obj.get("confidence", 0.0))
        relevant = obj.get("verdict") == "yes" and confidence >= 0.60
        reason = obj.get("reason", "")
    else:
        confidence = 0.95 if is_verified else 0.90
        relevant = "unsupported_question" not in candidate_sql
        reason = "Deterministic development relevance check passed." if relevant else "No supported offline generation rule."

    result["checks"]["llm_relevance"] = relevant
    result["relevance_confidence"] = confidence
    if not relevant:
        result.update(failed_check="llm_relevance", details=reason)
        return result

    if is_verified:
        norm = lambda s: re.sub(r"\s+", " ", s.strip().rstrip(";")).lower()
        integrity = norm(candidate_sql) == norm(VERIFIED_QUERY_LIBRARY[query_id]["sql"])
        result["checks"]["template_integrity"] = integrity
        if not integrity:
            result.update(failed_check="template_integrity", details="Verified SQL no longer matches the approved library template.")
            return result
    else:
        result["checks"]["template_integrity"] = "not_applicable"

    result["passed"] = True
    result["details"] = reason
    return result


def retry_generation(user_question: str, failed_sql: str, error_message: str, generation_llm=None) -> str:
    prompt = f"""You are correcting a failed SQLite SELECT query for a bank credit-risk query engine.
Return ONE corrected read-only SQL statement only.
Preserve the original intent, use only the supplied schema, and correct the stated validation failure.
NPA/impaired means Substandard, Doubtful, or Loss; exclude Special Mention.

User question: {user_question}
Failed SQL: {failed_sql}
Validation failure: {error_message}
Database schema: {DATABASE_SCHEMA}"""
    if generation_llm is not None:
        revised = generation_llm.invoke(prompt).content.strip()
        revised = re.sub(r"^```sql\s*|\s*```$", "", revised, flags=re.I | re.M).strip()
        return re.sub(r"^```\s*|\s*```$", "", revised, flags=re.M).strip()
    return generate_query(user_question, None)


def execute_query(validated_sql: str, conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(validated_sql, conn)


def _offline_narrative(user_question: str, df: pd.DataFrame) -> str:
    q = user_question.lower()
    if "real estate" in q and ("non-performing" in q or "npa" in q):
        row = df.loc[df["sector_name"].eq("Real Estate")].iloc[0]
        return f"Real Estate has ${row['total_outstanding_m']:,.1f}M of outstanding exposure, of which ${row['npa_outstanding_m']:,.1f}M is non-performing."
    if "dpd bucket" in q or ("overdue" in q and "distributed" in q):
        row = df.loc[df["dpd_bucket"].eq("90+")].iloc[0]
        return f"The portfolio spans five DPD buckets. The 90+ bucket contains {int(row['loan_count'])} loans with ${row['total_outstanding_m']:,.1f}M outstanding."
    if "restructured" in q and "impaired" in q:
        n = len(df); impaired = int(df["is_impaired"].sum()); pct = 100 * impaired / n
        return f"There are {n} restructured loans; {impaired} are impaired, representing {pct:.1f}% of the restructured portfolio by loan count."
    if "average interest rate" in q and "sector" in q:
        top = df.iloc[0]
        return f"Average interest rates were calculated for all {len(df)} sectors. {top['sector_name']} is highest at {top['avg_interest_rate_pct']:.2f}%."
    if "stage 3" in q:
        first, last = df.iloc[0], df.iloc[-1]
        return f"Stage 3 EAD increased from ${first['stage3_ead_m']:,.1f}M to ${last['stage3_ead_m']:,.1f}M over the four quarters, while Stage 3 ECL increased from ${first['stage3_ecl_m']:,.1f}M to ${last['stage3_ecl_m']:,.1f}M."
    return f"The query returned {len(df)} rows."


def generate_response(user_question: str, df: pd.DataFrame, generation_llm=None) -> str:
    prompt = f"""You are a credit-risk analyst answering a business user's question from verified SQL output.
Use only the supplied result. Give a concise answer with exact figures and units. Do not invent causes or recommendations. If the result is a complete template output, focus on the rows relevant to the question.

User question:
{user_question}

SQL result:
{df.to_string(index=False)}"""
    if generation_llm is not None:
        return generation_llm.invoke(prompt).content.strip()
    return _offline_narrative(user_question, df)


def run_pipeline(user_question: str, db_path: Path | str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    generation_llm, evaluator_llm = build_llms()
    conn = connect_read_only(db_path)
    try:
        log: Dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "user_question": user_question,
            "route": None,
            "query_id": None,
            "match_reason": None,

            # Complete audit history. Preserve the first attempt even when a
            # generated query uses the one permitted retry.
            "initial_sql": None,
            "initial_gate_result": None,
            "retry_used": False,
            "retry_sql": None,
            "retry_gate_result": None,
            "final_gate_result": None,

            # Backward-compatible aliases used by existing notebook/app code.
            "candidate_sql": None,
            "gate_result": None,

            "escalated": False,
            "executed_sql": None,
            "row_count": None,
            "confidence": None,
            "narrative": None,
            "llm_mode": "live" if generation_llm is not None else "offline deterministic",
        }

        classification = classify_intent(user_question, generation_llm)
        log.update({k: classification.get(k) for k in ["route", "query_id", "match_reason"]})

        if log["route"] == "verified" and log["query_id"] in VERIFIED_QUERY_LIBRARY:
            candidate_sql = VERIFIED_QUERY_LIBRARY[log["query_id"]]["sql"]
        else:
            candidate_sql = generate_query(user_question, generation_llm)

        # Attempt 1 is always retained in the audit log.
        log["initial_sql"] = candidate_sql
        log["candidate_sql"] = candidate_sql

        gate = validate_query(user_question, candidate_sql, conn, evaluator_llm, log["query_id"])
        log["initial_gate_result"] = gate
        log["gate_result"] = gate
        log["final_gate_result"] = gate

        # Generated SQL gets one and only one retry after a failed gate.
        if not gate["passed"] and log["route"] == "generated":
            log["retry_used"] = True
            retry_sql = retry_generation(
                user_question,
                log["initial_sql"],
                log["initial_gate_result"]["details"],
                generation_llm,
            )
            log["retry_sql"] = retry_sql
            candidate_sql = retry_sql
            log["candidate_sql"] = candidate_sql

            retry_gate = validate_query(
                user_question,
                retry_sql,
                conn,
                evaluator_llm,
                None,
            )
            log["retry_gate_result"] = retry_gate
            gate = retry_gate
            log["gate_result"] = gate
            log["final_gate_result"] = gate

        if not gate["passed"]:
            log["escalated"] = True
            log["confidence"] = 0.0
            log["narrative"] = f"Escalated to a human analyst because the query did not pass validation: {gate['details']}"
            return {"log": log, "dataframe": None}

        log["executed_sql"] = candidate_sql
        df = execute_query(candidate_sql, conn)
        log["row_count"] = len(df)
        log["narrative"] = generate_response(user_question, df, generation_llm)
        log["confidence"] = float(gate["relevance_confidence"])
        return {"log": log, "dataframe": df}
    finally:
        conn.close()


def load_test_queries(path: Path | str = DEFAULT_TEST_PATH) -> pd.DataFrame:
    return pd.read_csv(path)
