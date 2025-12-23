from meno_bench.models import TestCasesFileFull, TestOut, TestMetricsResults
from typing import TypedDict


class Summary(TypedDict):
    length: float = 0.0
    correctness: float = 0.0
    clarity: float = 0.0
    time_per_request: float = 0.0


def get_summary(results: list[TestOut]) -> Summary:
    s = Summary(length=0, correctness=0.0, clarity=0.0, time_per_request=0)
    for result in results:
        s["length"] += len(result["case"]["model_answer"])
        s["correctness"] += result["result"]["correctness"]["score"]
        s["clarity"] += result["result"]["clarity"]["score"]
        s["time_per_request"] += result["case"]["time_s"]
    le = len(results)
    s["length"] /= le
    s["correctness"] /= le
    s["clarity"] /= le
    s["time_per_request"] /= le

    return s
