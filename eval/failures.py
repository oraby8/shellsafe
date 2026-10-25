"""Print a sample of failed commands from a scored results file."""
import json
import random
import sys

rows = [json.loads(l) for l in open(sys.argv[1])]
bad = [r for r in rows if not r["score"]]
print("failed:", len(bad), "empty:", sum(not r["cmd"] for r in bad),
      "multi-line/fenced raw:", sum("```" in r["raw"] or len(r["raw"].splitlines()) > 1 for r in bad))
random.seed(1)
for r in random.sample(bad, min(10, len(bad))):
    print(f"- {r['nl'][:75]}\n    got:  {r['cmd'][:90]}\n    gold: {r['gold'][:90]}")
