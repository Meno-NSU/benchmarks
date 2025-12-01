from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
import os
import json
from typing import TypedDict
import asyncio
import json
import os

import asyncio
import tqdm
from deepeval.models import DeepEvalBaseLLM


class TestCaseFromFile(TypedDict):
    question: str
    ground_truth: str

TestCasesFile = list[TestCaseFromFile]

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
    ref_answer: str | None

class Out(TypedDict):
    summary: dict
    cases: list[TestOut]


class NetworkModel(DeepEvalBaseLLM):

    def __init__(self):
        pass

    def load_model(self):
        return self
    
    def generate(self, prompt: str) -> str:
        pass

    def get_model_name(self):
        return "meno-model"


async def main():

    correctness = GEval(
        name="Correctness",
        evaluation_steps=[
            "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
            "You should also heavily penalize omission of detail",
            "Vague language, or contradicting OPINIONS, are OK"
        ],
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    )
    clarity = GEval(
        name="Clarity",
        evaluation_steps=[
            "Evaluate whether the response uses clear and direct language.",
            "Check if the explanation avoids jargon or explains it when used.",
            "Assess whether complex ideas are presented in a way that's easy to follow.",
            "Identify any vague or confusing parts that reduce understanding."
        ],
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    )
    # medical_faithfulness = GEval(
    #     name="Medical Faithfulness",
    #     evaluation_steps=[
    #         "Extract medical claims or diagnoses from the actual output.",
    #         "Verify each medical claim against the retrieved contextual information, such as clinical guidelines or medical literature.",
    #         "Identify any contradictions or unsupported medical claims that could lead to misdiagnosis.",
    #         "Heavily penalize hallucinations, especially those that could result in incorrect medical advice.",
    #         "Provide reasons for the faithfulness score, emphasizing the importance of clinical accuracy and patient safety."
    #     ],
    #     evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
    # )
    out: Out = {
        "summary": {},
        "cases": [],
    }

    with open(os.environ["TEST_CASES_JSON"], "r") as f:
        jsoned_tests: TestCasesFile = json.load(f)
    print("Preparing tests")
    for test_case in tqdm.tqdm(jsoned_tests):
        
        llm_test_case = LLMTestCase(
            input=test_case["question"],
            actual_output=None,
            expected_output=test_case["ground_truth"]
        )
        
        correctness.measure(llm_test_case)
        print(correctness.score, correctness.reason)
        
        clarity.measure(llm_test_case)
        print(clarity.score, clarity.reason)
        out["cases"].append(
            TestOut(
                result=TestMetricsResults(
                    correctness=TestResult(
                        score=correctness.score,
                        reason=correctness.reason
                    ),
                    clarity=TestResult(
                        score=clarity.score,
                        reason=clarity.reason
                    ),
                ),
                question=test_case["question"],
                model_answer="none",
                ref_answer=test_case["ground_truth"]
            )
        )
    print("Done with cases")
    with open("result.json", "w") as f:
        json.dump(out, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    asyncio.run(main())
