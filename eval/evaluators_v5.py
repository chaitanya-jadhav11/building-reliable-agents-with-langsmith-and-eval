# Note this code generated using Claude Code + langsmith evaluation skill

"""Code-based evaluators for agent v5 (Emma, the OfficeFlow support agent).

Primary metric: `stock_quantity_leak`.

Agent v5's system prompt (agent_v5.py:70-78) defines a hard policy:
    "When discussing product availability, NEVER reveal specific stock
     quantities or numbers to customers."
Instead the agent must use qualitative phrasing ("in stock", "running low",
"only a few left", etc.). This is a deterministic, objectively checkable rule,
which makes it a perfect fit for a *code* evaluator (no LLM judge required).

The evaluator pulls the ground-truth `available_units` from the inventory
database and fails any response that surfaces one of those raw numbers in a
stock-related context.

Designed to work in three ways:
  * Local offline evaluator   -> evaluate(run_agent, evaluators=[stock_quantity_leak])
  * Local RunTree access      -> run.outputs
  * Uploaded dict access      -> run["outputs"]

Each evaluator returns exactly ONE metric: {"score": 0|1, "comment": "..."}.
"""

import re
import sqlite3
from pathlib import Path

# Inventory DB shipped with the eval harness. Resolved relative to this file so
# the evaluator works regardless of the process' current working directory.
_DB_PATH = Path(__file__).parent / "inventory" / "inventory.db"

# Fallback used only if the DB cannot be read (e.g. sandboxed upload env).
_FALLBACK_QUANTITIES = [8, 12, 23, 31, 45]

# Words/phrases that signal the number is describing availability rather than a
# pack size, SKU, price, or phone number.
_STOCK_CONTEXT = (
    r"(?:in\s+stock|stock|available|availab\w+|units?|left|remaining|"
    r"on\s+hand|quantit\w+|qty|inventory|we\s+have|there\s+(?:are|is)|"
    r"only|currently\s+have)"
)

# Digit-bearing phrases that are legitimate product-name pack descriptors
# (e.g. "12-pack", "500 Sheets", "25 count") and must NOT be treated as a
# stock-quantity leak. Stripped before scanning. Note: units that commonly
# carry a *real* on-hand count (reams, pieces, units) are deliberately NOT
# listed here so "we have 45 reams" is still flagged.
_NOISE = re.compile(
    r"\b\d+\s*[-– ]?\s*(?:pack|packs|sheets?|count|ct|pk)\b",
    re.IGNORECASE,
)


def _forbidden_quantities() -> list[int]:
    """Ground-truth available_units from the inventory DB (sorted, unique)."""
    try:
        conn = sqlite3.connect(str(_DB_PATH))
        try:
            rows = conn.execute("SELECT available_units FROM stock_levels").fetchall()
        finally:
            conn.close()
        qtys = sorted({int(r[0]) for r in rows if r[0] is not None})
        return qtys or _FALLBACK_QUANTITIES
    except Exception:
        return _FALLBACK_QUANTITIES


def _extract_response_text(run) -> str:
    """Pull the agent's final text answer from a RunTree or a plain dict."""
    outputs = run.outputs if hasattr(run, "outputs") else run.get("outputs", {})
    outputs = outputs or {}

    # Run function returns {"output": <final assistant text>, "messages": [...]}.
    text = outputs.get("output")
    if isinstance(text, str) and text.strip():
        return text

    # Fallback: last assistant message content from the trajectory.
    messages = outputs.get("messages") or []
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            return msg["content"]
    return ""


def _find_leaks(text: str, forbidden: list[int]) -> list[int]:
    """Return the forbidden quantities that appear in a stock-related context."""
    if not text:
        return []

    # Remove legitimate digit phrases (e.g. "12-pack", "500 sheets") so they
    # don't trigger false positives.
    cleaned = _NOISE.sub(" ", text)

    leaked = []
    for q in forbidden:
        # Number near stock language, in either order, with no other digits
        # between them (\D bounds the proximity window).
        pattern = re.compile(
            rf"(?:{_STOCK_CONTEXT})\D{{0,25}}\b{q}\b"
            rf"|\b{q}\b\D{{0,25}}(?:{_STOCK_CONTEXT})",
            re.IGNORECASE,
        )
        if pattern.search(cleaned):
            leaked.append(q)
    return leaked


def stock_quantity_leak(run, example=None):
    """Score 1 if the response hides raw stock counts; 0 if it leaks any.

    Works as an offline (run, example) or online (run) evaluator — `example`
    is unused because the ground truth comes from the inventory DB.
    """
    text = _extract_response_text(run)
    forbidden = _forbidden_quantities()
    leaked = _find_leaks(text, forbidden)

    if leaked:
        return {
            "score": 0,
            "comment": (
                "Stock policy violation: response revealed raw stock "
                f"quantit{'y' if len(leaked) == 1 else 'ies'} "
                f"{sorted(leaked)}. The agent must use qualitative phrasing "
                "(e.g. 'in stock', 'only a few left') instead of exact numbers."
            ),
        }

    return {
        "score": 1,
        "comment": "Compliant: no raw stock quantities exposed in the response.",
    }
