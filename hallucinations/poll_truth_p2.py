import argparse
import gzip
import json
import os
import re
import string
import sys
import time
from typing import Tuple, Set

from datasets import load_dataset
from openai import OpenAI
from utils import append_jsonl, normalise_logprobs

# load dataset for adding categories
truthfulqa = load_dataset("truthful_qa", "generation")["validation"]
qid_to_category = {i: q["category"] for i, q in enumerate(truthfulqa)}

POLL_INTERVAL_SEC = 30

# load processed evaluations
def load_done(path: str) -> Set[Tuple[str, float, str]]:
    """Return set of (Model, Temperature, QID) already written."""
    if not os.path.exists(path):
        return set()

    open_func = gzip.open if path.endswith(".gz") else open
    done = set()
    with open_func(path, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["Model"], rec["Temperature"], str(rec["Question_ind"]))
            done.add(key)
    return done

# extract label from evaluation text
def extract_label(eval_txt: str) -> str | None:
    try:
        maybe_json = json.loads(eval_txt)
        if isinstance(maybe_json, dict) and "label" in maybe_json:
            return maybe_json["label"].strip().lower()
    except json.JSONDecodeError:
        pass

    label_line = re.search(
        r"\blabel\s*[:=]\s*['\"`]*\s*(truthful|hallucinated)\b",
        eval_txt,
        flags=re.I,
    )
    if label_line:
        return label_line.group(1).lower()

    arrow = re.search(
        r"→\s*[*_`~]*\s*(truthful|hallucinated)\b",
        eval_txt,
        flags=re.I,
    )
    if arrow:
        return arrow.group(1).lower()

    return None

# parse arguments from command line
parser = argparse.ArgumentParser(description="Merge completed Batch jobs.")
parser.add_argument("batch_id", help="OpenAI Batch ID")
parser.add_argument(
    "--answers-file",
    default="results_truthqa.jsonl.gz",
    help="Gzipped JSONL of the model answers",
)
parser.add_argument(
    "--output-file",
    default="truthfulqa_evaluation_long.jsonl.gz",
    help="Where to append the evaluations JSONL",
)
args = parser.parse_args()

api_key = os.getenv("OPEN_API_KEY")
if not api_key:
    sys.exit("Please set the OPEN_API_KEY environment variable.")

client = OpenAI(api_key=api_key)


# load previously generated answers
with gzip.open(args.answers_file, "rt", encoding="utf-8") as f:
    answers_index = {
        (row["qid"], row["model"], row["setting"]): row for row in map(json.loads, f)
    }

already_done = load_done(args.output_file)

# poll openai batch api until batch completion
print(f"Polling Batch job {args.batch_id} …")
while True:
    job = client.batches.retrieve(args.batch_id)
    if job.status in {"completed", "failed", "expired", "canceled"}:
        print(f"Batch ended with status: {job.status}")
        break
    print(
        f"Status: {job.status:>11} | "
        f"{job.request_counts.get('completed',0):,}/{job.request_counts.get('total',0):,} tasks done"
    )
    time.sleep(POLL_INTERVAL_SEC)

if job.status != "completed":
    sys.exit(f"No output to process (status was {job.status}).")

# retrieve and decode results
blob = client.files.content(job.output_file_id).content
text = (
    gzip.decompress(blob).decode("utf-8")
    if blob.startswith(b"\x1f\x8b")
    else blob.decode("utf-8-sig")
)
lines = text.splitlines()

# process and append new evaluations
added = 0
for line in lines:
    rec = json.loads(line)
    qid, model, setting = rec["custom_id"].split("|")
    row = answers_index[(int(qid), model, setting)]
    key = (model, row["temperature"], qid)

    if key in already_done:
        continue

    
    choice = rec["response"]["body"]["choices"][0]

    eval_txt = choice["message"]["content"].lstrip("\ufeff").strip()
    judge_lp = normalise_logprobs(choice["logprobs"])
    label = extract_label(eval_txt)

    if label is None: 
        continue
    
    # append the evaluation results to the output file
    append_jsonl(
        {
            "Question_ind": int(qid),
            "Evaluation": eval_txt,
            "Model": model,
            "Temperature": row["temperature"],
            "Candidate Answer": row["answer"],
            "judge_tokens": judge_lp,
            "classification": label,
            "Category": qid_to_category.get(int(qid)),
            
        },
        args.output_file,
    )
    added += 1

print(f"Appended {added} new evaluations to {args.output_file}")
