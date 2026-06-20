import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.guardrails.citation_checker import check_groundedness

# A fake source chunk that says nothing about Mars
fake_chunks = [
    {
        "content": "Bi-encoders encode text independently into vectors and compare them using cosine similarity.",
        "source": "test_source.txt",
    }
]

# A deliberately false claim citing that source - this should get flagged
fake_answer = (
    "Bi-encoders were first used to compare distances between cities on Mars [S1]. "
    "Bi-encoders encode text independently into vectors and compare them using cosine similarity [S1]."
)

result = check_groundedness(fake_answer, fake_chunks)

print("Checked claims:", result["checked_claims"])
print("Is fully grounded:", result["is_fully_grounded"])
print("\nFlagged claims:")
for f in result["flagged_claims"]:
    print(f"  - {f}")

if result["flagged_claims"]:
    print("\nPASS: guardrail correctly caught the false claim")
else:
    print("\nFAIL: guardrail did not catch the false claim - needs investigation")
