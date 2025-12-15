from typing import TypedDict


class TestCaseFromFile(TypedDict):
    question: str
    ground_truth: str


class TestCaseFromFileFull(TypedDict):
    question: str
    ground_truth: str
    model_answer: str


type TestCasesFile = list[TestCaseFromFile]

type TestCasesFileFull = list[TestCaseFromFileFull]


class TestResult(TypedDict):
    score: float | None
    reason: str | None


class TestMetricsResults(TypedDict):
    correctness: TestResult
    clarity: TestResult


class TestOut(TypedDict):
    result: TestMetricsResults
    question: str
    model_answer: str
    ground_truth: str | None


class Out(TypedDict):
    summary: dict
    cases: list[TestOut]
