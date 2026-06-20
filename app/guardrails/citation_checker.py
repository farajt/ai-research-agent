import re
import json

from langchain_groq import ChatGroq
from langfuse import observe

from app.config import settings
from app.utils.retry import with_retry

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
    """Lists each referenced source's full text exactly ONCE, then references
    it by ID in each claim line. The earlier version repeated full source
    text for every claim that cited it - with several claims often citing
    the same 2-3 sources, that duplicated large chunks of text 10-20x over
    and blew past the LLM provider's per-request token limit. This version
    sends each source's content a single time regardless of how many claims
    cite it.
    """
    source_lookup = {f"S{i}": c for i, c in enumerate(chunks, start=1)}

    # Only include sources actually cited by at least one claim, and only once each
    referenced_ids = sorted(
        {cid for claim in claims for cid in claim["citation_ids"]},
        key=lambda x: int(x[1:]),
    )

    lines = ["Sources:\n"]
    for cid in referenced_ids:
        source = source_lookup.get(cid)
        if source:
            lines.append(f"{cid}: {source['content']}")
    lines.append("")

    lines.append("Claims to verify:\n")
    for claim in claims:
        cited = ", ".join(claim["citation_ids"])
        lines.append(f"Claim: {claim['sentence']} (cited sources: {cited})")

    return "\n".join(lines)


@with_retry()
def _invoke_guardrail_llm(llm, messages):
    return llm.invoke(messages)


@observe(as_type="generation")
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

    try:
        response = _invoke_guardrail_llm(
            llm,
            [
                {"role": "system", "content": GUARDRAIL_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
    except Exception as e:
        # The guardrail is a safety net, not the main path - if the call
        # still fails after retries (provider outage, persistent rate limit),
        # don't take the whole /query endpoint down with it. Surface the
        # failure so it's visible, but let the answer through
        # ungrounded-unverified rather than erroring out entirely.
        return {
            "checked_claims": len(claims),
            "flagged_claims": [],
            "is_fully_grounded": None,
            "guardrail_error": f"Guardrail LLM call failed: {str(e)}",
        }

    raw = response.content.strip().replace("```json", "").replace("```", "").strip()

    # Models sometimes add a sentence before/after the JSON despite instructions
    # not to. Extract just the [...] portion rather than requiring the whole
    # response to be clean JSON - more forgiving, fewer false "parse failed" results.
    array_match = re.search(r"\[.*\]", raw, re.DOTALL)
    json_candidate = array_match.group(0) if array_match else raw

    try:
        verdicts = json.loads(json_candidate)
    except json.JSONDecodeError:
        # Fail safe: if the guardrail's own output still can't be parsed,
        # don't block the answer - just surface what came back so it's
        # visible and debuggable, rather than a generic unhelpful message.
        return {
            "checked_claims": len(claims),
            "flagged_claims": [],
            "is_fully_grounded": None,
            "guardrail_error": "Could not parse guardrail response",
            "guardrail_raw_output": raw[:500],
        }

    flagged = [v for v in verdicts if isinstance(v, dict) and v.get("supported") is False]

    return {
        "checked_claims": len(claims),
        "flagged_claims": flagged,
        "is_fully_grounded": len(flagged) == 0,
    }
