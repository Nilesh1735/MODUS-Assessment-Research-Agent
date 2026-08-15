"""
nodes/synthesizer.py
--------------------
Generate the final, traceable Markdown report.

Findings and contradictions are read back from SQLite (the source of truth), so this node
produces a correct report regardless of which path reached it — including the "no relevant
evidence" path routed straight from the grader.

Traceability is enforced deterministically: sources are numbered here in code, the LLM is
instructed to cite them inline as ``[n]``, and the authoritative ``## References`` section is
appended by code (not the model) so citation numbers always resolve to a real URL.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from backend import config
from backend.database import (
    get_contradictions,
    get_findings,
    insert_report,
    update_session_status,
)
from backend.llm import SynthesizedReport, structured_llm
from backend.state import ResearchState

logger = logging.getLogger(__name__)

_NO_SOURCE = "(source unavailable)"

_SYSTEM_PROMPT = (
    "You are a senior research analyst. Write a clear, professional Markdown report that answers "
    "the RESEARCH QUESTION using ONLY the provided findings. Requirements:\n"
    "- Begin with a '# ' title, then a '## Executive Summary'.\n"
    "- Organise the body into thematic '## ' sections with genuine analysis.\n"
    "- Support every claim with an inline citation like [n] using the given source numbers; you "
    "may combine sources like [1][3].\n"
    "- If contradictions are listed, add a '## Conflicting Evidence' section discussing them.\n"
    "- Do NOT invent information beyond the findings. Do NOT write a References section — it is "
    "appended automatically.\n"
    "- Be concise, factual, and well-structured."
)


def _insufficient_evidence_report(question: str) -> str:
    return (
        f"# Research Report: {question}\n\n"
        "## Summary\n\n"
        "No relevant sources with usable findings were identified for this question. The pipeline "
        "searched the web and graded the retrieved documents, but none passed the relevance and "
        "extraction stages. Consider rephrasing the question or broadening its scope.\n"
    )


def _fallback_body(question: str, numbered_findings: list[tuple[int, str]]) -> str:
    lines = [
        f"# Research Report: {question}",
        "",
        "> Automated fallback synthesis (LLM report generation was unavailable).",
        "",
        "## Key Findings",
        "",
    ]
    lines.extend(f"- {fact} [{num}]" for num, fact in numbered_findings)
    return "\n".join(lines)


def synthesizer(state: ResearchState) -> dict:
    """Synthesis node: persisted findings -> final Markdown report with citations."""
    session_id = state["session_id"]
    question = state["question"]

    findings = get_findings(session_id)
    contradictions = get_contradictions(session_id)

    if not findings:
        report = _insufficient_evidence_report(question)
        insert_report(session_id, report)
        update_session_status(session_id, "completed")
        logger.info("[synthesizer] session=%s: no findings; wrote insufficient-evidence report", session_id)
        return {"report": report}

    # Number sources by first appearance so inline [n] citations map to a stable reference list.
    url_to_num: dict[str, int] = {}
    ordered_sources: list[tuple[str, str]] = []  # (title, url) in citation order
    numbered_findings: list[tuple[int, str]] = []  # (citation_number, fact)

    findings_lines: list[str] = []
    for finding in findings:
        url = finding["url"] or _NO_SOURCE
        title = finding["title"] or url
        if url not in url_to_num:
            url_to_num[url] = len(ordered_sources) + 1
            ordered_sources.append((title, url))
        num = url_to_num[url]
        numbered_findings.append((num, finding["fact"]))
        findings_lines.append(
            f"- [{num}] ({finding['classification']}, confidence {finding['confidence']:.2f}) {finding['fact']}"
        )
    findings_block = "\n".join(findings_lines)
    if contradictions:
        contradictions_block = "\n".join(
            f"- {row['explanation']} (similarity {row['similarity']:.2f})" for row in contradictions
        )
    else:
        contradictions_block = "None detected."

    references = "\n".join(
        f"{num}. [{title}]({url})" for num, (title, url) in enumerate(ordered_sources, start=1)
    )

    try:
        result = structured_llm(SynthesizedReport, max_tokens=config.SYNTHESIZER_MAX_TOKENS).invoke(
            [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"RESEARCH QUESTION:\n{question}\n\n"
                        f"FINDINGS (each prefixed with its source citation number):\n{findings_block}\n\n"
                        f"CONTRADICTIONS:\n{contradictions_block}\n\n"
                        "Write the report now using inline [n] citations."
                    )
                ),
            ]
        )
        body = (result.report_markdown or "").strip()
        if not body:
            raise ValueError("empty report body")
    except Exception as exc:  # noqa: BLE001 - always produce a report
        logger.warning("[synthesizer] LLM synthesis failed (%s); using fallback report", exc)
        body = _fallback_body(question, numbered_findings)

    report = f"{body}\n\n## References\n\n{references}\n"
    insert_report(session_id, report)
    update_session_status(session_id, "completed")
    logger.info(
        "[synthesizer] session=%s: report generated (%d findings, %d sources, %d contradictions)",
        session_id,
        len(findings),
        len(ordered_sources),
        len(contradictions),
    )
    return {"report": report}
