from meno_bench.models import TestCasesFileFull, TestOut, TestMetricsResults
from typing import TypedDict
from collections import defaultdict
from pathlib import Path
import json


class Summary(TypedDict):
    length: float = 0.0
    correctness: float = 0.0
    clarity: float = 0.0
    # correctness2: float = 0.0
    # clarity2: float = 0.0
    correctness_rubrics: float = 0.0
    time_per_request: float = 0.0
    rouge: dict


def get_summary(results: list[TestOut]) -> Summary:
    s = Summary(
        length=0,
        correctness=0.0,
        clarity=0.0,
        # correctness2=0.0,
        # clarity2=0.0,
        correctness_rubrics=0.0,
        rpm=0,
        rouge=defaultdict(list),
    )
    for result in results:
        s["length"] += len(result["case"]["model_answer"])
        s["correctness"] += result["result"]["correctness"]["score"]
        s["clarity"] += result["result"]["clarity"]["score"]
        # s["correctness2"] += (result["result"]["correctness2"]["score"] or 0)
        # s["clarity2"] += (result["result"]["correctness2"]["score"] or 0)
        s["correctness_rubrics"] += (result["result"]["correctness_rubrics"]["score"] or 0)
        if "time_s" in result["case"]:
            s["rpm"] += result["case"]["time_s"]
        k: str
        v: list[float]
        for k, v in result["result"]["rouge"].items():
            if s["rouge"][k]:
                for i in range(len(s["rouge"][k])):
                    s["rouge"][k][i] += v[i]
            else:
                s["rouge"][k] = list(v)
    le = len(results)
    s["length"] /= le
    s["correctness"] /= le
    s["clarity"] /= le
    # s["correctness2"] /= le
    # s["clarity2"] /= le
    s["correctness_rubrics"] /= le
    s["rpm"] /= le
    s["rpm"] = 60 / s["rpm"]
    for k, v in s["rouge"].items():
        for i in range(len(v)):
            s["rouge"][k][i] /= le
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
