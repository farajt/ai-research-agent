import json

with open("data/eval_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"{'Question':<55}{'Naive KW':<12}{'Hybrid KW':<12}")
print("-" * 79)

for naive_q, hybrid_q in zip(data["naive"], data["hybrid"]):
    question = naive_q["question"]
    short_q = (question[:50] + "...") if len(question) > 50 else question
    print(f"{short_q:<55}{naive_q['keyword_coverage']:<12.2f}{hybrid_q['keyword_coverage']:<12.2f}")
