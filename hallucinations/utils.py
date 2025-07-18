import gzip
import json

def normalise_logprobs(logprobs_obj):
    """Convert the SDK’s logprobs object into pure-dict form."""
    if not "content" in logprobs_obj:
        return []
    cleaned = []
    for entry in logprobs_obj["content"]:
        cleaned.append({
            "token":      entry["token"],
            "logprob":    entry["logprob"],
            "top_alts": [
                {"token": alt["token"], "logprob": ["logprob"]}
                for alt in entry["top_logprobs"]
            ],
        })
    return cleaned

def append_jsonl(record: dict, path: str) -> None:
    open_func = gzip.open if path.endswith(".gz") else open
    mode      = "at" if path.endswith(".gz") else "a"
    with open_func(path, mode, encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

