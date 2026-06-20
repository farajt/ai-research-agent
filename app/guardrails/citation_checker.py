import re
import json

from langchain_groq import ChatGroq

from app.config import settings

_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(api_key=settings.GROQ_API_KEY, model=settings.LLM_MODEL, temperature=0.0)
    return _llm


CITATION_PATTERN = re.compile(r"\[S(\d+)\]")


def extract_claims(answer: str) -> list[dict]:
    """Splits the answer into sentences and pairs each sentence with the
    citation markers it contains. Sentences with no citation markers are
    skipped - there's nothing to verify them against."""
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())

    claims = []
    for sentence in sentences:
        citation_ids = CITATION_PATTERN.findall(sentence)
        if citation_ids:
            claims.append(
                {
                    "sentence": sentence.strip(),
                    "citation_ids": [f"S{cid}" for cid in citation_ids],
                }
            )
    return claims


GUARDRAIL_SYSTEM_PROMPT = """You are a strict fact-checker. You will be given a list of
claims, each paired with one or more source citation IDs, and the full text of each cited
source. For each claim, determine whether the cited source(s) actually support the claim.

A claim is SUPPORTED only if the cited source contains information that directly backs it
up. A claim is UNSUPPORTED if the source doesn't mention it, contradicts it, or only loosely
relates to it without actually confirming the specific claim made.

Respond with ONLY a JSON array, no markdown, no explanation. Each item must look like:
{"claim": "<the claim text>", "supported": true or false, "reason": "<one short sentence>"}
"""


def _build_verification_prompt(claims: list[dict], chunks: list[dict]) -> str:
    source_lookup = {f"S{i}": c for i, c in enumerate(chunks, start=1)}

    lines = ["Claims to verify:\n"]
    for claim in claims:
        lines.append(f"Claim: {claim['sentence']}")
        for cid in claim["citation_ids"]:
            source = source_lookup.get(cid)
            if source:
                lines.append(f"  Cited source {cid}: {source['content']}")
        lines.append("")
    return "\n".join(lines)


def check_groundedness(answer: str, chunks: list[dict]) -> dict:
    """Runs a second LLM pass to verify every cited claim in the answer is
    actually backed up by its source. This catches a failure mode the first
    synthesis pass can't catch on its own: the model citing a source that
    sounds related but doesn't actually confirm the specific thing claimed.
    Same idea as NLI (entailment checking), implemented as a direct LLM call
    rather than a dedicated NLI model - simpler to wire up, same purpose.
    """
    claims = extract_claims(answer)
    if not claims:
        return {"checked_claims": 0, "flagged_claims": [], "is_fully_grounded": True}

    llm = get_llm()
    prompt = _build_verification_prompt(claims, chunks)

    response = llm.invoke(
        [
            {"role": "system", "content": GUARDRAIL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )

    raw = response.content.strip().replace("```json", "").replace("```", "").strip()

    try:
        verdicts = json.loads(raw)
    except json.JSONDecodeError:
        # Fail safe: if the guardrail's own output can't be parsed, don't
        # block the answer - just surface that the check couldn't complete,
        # so this is visible rather than silently swallowed.
        return {
            "checked_claims": len(claims),
            "flagged_claims": [],
            "is_fully_grounded": None,
            "guardrail_error": "Could not parse guardrail response",
        }

    flagged = [v for v in verdicts if isinstance(v, dict) and v.get("supported") is False]

    return {
        "checked_claims": len(claims),
        "flagged_claims": flagged,
        "is_fully_grounded": len(flagged) == 0,
    }
