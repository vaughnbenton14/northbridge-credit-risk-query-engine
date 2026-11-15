import streamlit as st

from query_engine import DEFAULT_TEST_PATH, load_test_queries, run_pipeline

st.set_page_config(page_title="Northbridge Credit Risk Query Engine", page_icon="🏦", layout="wide")

st.title("Northbridge Bank — Credit Risk Query Engine")
st.caption("Natural-language portfolio analysis with verified SQL, validation gates, traceability, and human escalation.")

with st.sidebar:
    st.header("About")
    st.write("Academic proof of concept using the sanitized Northbridge portfolio dataset.")
    st.info("The database connection is read-only. Generated SQL must pass the safety and relevance gate before execution.")
    st.caption("NPA / impaired = Substandard, Doubtful, or Loss. Special Mention is excluded.")

try:
    tests = load_test_queries(DEFAULT_TEST_PATH)
    test_labels = ["Custom question"] + [f"{r['Test Case']}: {r['User Query']}" for _, r in tests.iterrows()]
except Exception:
    tests = None
    test_labels = ["Custom question"]

selection = st.selectbox("Choose a supplied test case or enter a custom question", test_labels)
if selection == "Custom question":
    question = st.text_area("Business question", placeholder="Ask a routine commercial lending portfolio question...", height=100)
else:
    tc = selection.split(":", 1)[0]
    row = tests.loc[tests["Test Case"].eq(tc)].iloc[0]
    question = row["User Query"]
    st.text_area("Business question", value=question, height=100, disabled=True)

run = st.button("Run query", type="primary", use_container_width=False)

if run:
    if not question or not question.strip():
        st.warning("Enter a business question first.")
    else:
        with st.spinner("Routing, validating, and executing the request..."):
            result = run_pipeline(question.strip())
        log = result["log"]
        df = result["dataframe"]

        st.subheader("Answer")
        if log["escalated"]:
            st.error(log["narrative"])
        else:
            st.success(log["narrative"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Route", str(log["route"]).title())
        c2.metric("Query ID", log["query_id"] or "Generated")
        c3.metric("Confidence", f"{float(log['confidence'] or 0):.0%}")
        c4.metric("Rows returned", log["row_count"] if log["row_count"] is not None else 0)

        st.subheader("Raw result")
        if df is not None:
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.write("No query was executed.")

        with st.expander("Audit trace", expanded=True):
            st.write(f"**LLM mode:** {log['llm_mode']}")
            st.write(f"**Route decision:** {log['route']}")
            st.write(f"**Matched query ID:** {log['query_id'] or 'None — generated SQL'}")
            st.write(f"**Route rationale:** {log['match_reason']}")
            st.write(f"**Retry used:** {log['retry_used']}")
            st.write(f"**Escalated:** {log['escalated']}")

            st.markdown("#### Attempt 1 — Initial SQL")
            st.code(log.get("initial_sql") or "No SQL generated", language="sql")
            st.write("**Initial validation gate:**")
            st.json(log.get("initial_gate_result") or {})

            if log["retry_used"]:
                st.markdown("#### Attempt 2 — One permitted retry")
                st.code(log.get("retry_sql") or "No retry SQL generated", language="sql")
                st.write("**Retry validation gate:**")
                st.json(log.get("retry_gate_result") or {})

            st.markdown("#### Final decision")
            st.write("**Final validation gate:**")
            st.json(log.get("final_gate_result") or log.get("gate_result") or {})
            st.write("**Executed SQL:**")
            st.code(log["executed_sql"] or "No SQL executed", language="sql")
            st.caption(f"Audit timestamp (UTC): {log['timestamp_utc']}")

        if selection != "Custom question" and tests is not None:
            st.subheader("Ground-truth reference")
            st.write(f"**Expected route:** {row['Expected Route']}")
            expected_qid = row.get("Expected Query ID")
            st.write(f"**Expected query ID:** {expected_qid if str(expected_qid) != 'nan' else 'Generated'}")
            st.write(f"**Expected answer:** {row['Expected Answer']}")
