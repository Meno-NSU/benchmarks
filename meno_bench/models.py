from typing import TypedDict, Any


class TestCaseFromFile(TypedDict):
    question: str
    ground_truth: str


class TestCaseFromFileFull(TypedDict):
    question: str
    ground_truth: str
    model_answer: str
    time_s: float


type TestCasesFile = list[TestCaseFromFile]

type TestCasesFileFull = list[TestCaseFromFileFull]


class TestResult(TypedDict):
    score: float | None
    reason: str | None


class TestMetricsResults(TypedDict):
    correctness: TestResult
    clarity: TestResult
    correctness2: TestResult
    clarity2: TestResult
    rouge: dict[Any, Any]


class TestOut(TypedDict):
    result: TestMetricsResults
    case: TestCaseFromFileFull

class Out(TypedDict):
    summary: dict
    cases: list[TestOut]
