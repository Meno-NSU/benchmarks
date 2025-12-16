from meno_bench.settings import JudgeSettings
from meno_bench.models import TestCasesFileFull, TestOut, TestMetricsResults
from deepeval.test_case import LLMTestCase
from meno_bench.judge.google import get_model as get_google_model
from meno_bench.judge.openai_api import get_model as get_api_model
from meno_bench.geval import GEvalStandardJudge
import json
import tqdm
import traceback


def judge_cases(settings: JudgeSettings, test_cases: TestCasesFileFull) -> list[TestOut]:

    if settings.use_gemini:
        model = get_google_model(settings)
    else:
        model = get_api_model(settings)
    geval = GEvalStandardJudge(model)

    results: list[TestOut] = []
    try:
        for case in tqdm.tqdm(test_cases):
            llm_test_case = LLMTestCase(
                input=case["question"],
                actual_output=case["model_answer"],
                expected_output=case["ground_truth"],
            )

            results.append(
                TestOut(
                    result=TestMetricsResults(**geval.eval(llm_test_case)),
                    question=case["question"],
                    model_answer=case["model_answer"],
                    ground_truth=case["ground_truth"],
                )
            )
    except Exception as e:
        print(e)
        print(traceback.print_exc())
    return results


def judge(settings: JudgeSettings):
    with open(settings.file, "r") as f:
        test_cases: TestCasesFileFull = json.load(f)
    
    results = judge_cases(settings, test_cases)

    out_file = settings.file.with_stem(f"{settings.file.stem}_judged")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
