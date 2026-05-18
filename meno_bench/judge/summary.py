from meno_bench.models import TestCasesFileFull, TestOut, TestMetricsResults
from typing import TypedDict
from collections import defaultdict
from pathlib import Path
import json


class Summary(defaultdict[str, float]):
    pass


def get_summary(results: list[TestOut]) -> Summary:
    s = Summary(float)
    for result in results:
        s["length"] += len(result["case"]["model_answer"])
        for metric, score in result["result"].items():
            if isinstance(score, (dict, list, tuple, set)):
                continue
            s[metric] += score
        if "time_s" in result["case"]:
            s["rpm"] += result["case"]["time_s"]
    le = len(results)
    for metric in s.keys():
        if metric != "rpm":
            s[metric] /= le
        else:
            s["rpm"] = 60 / s["rpm"]
    return s


def summarize_to_file(in_file: Path, out_path: Path | None = None):
    with open(in_file, "r") as f:
        results: TestCasesFileFull = json.load(f)
    summary = get_summary(results)
    if out_path is None:
        sum_file = in_file.with_stem(f"{in_file.stem}_summary")
    else:
        sum_file = out_path
    with open(sum_file, "w") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
