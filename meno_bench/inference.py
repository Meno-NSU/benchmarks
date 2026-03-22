from meno_bench.settings import InferenceSettings
from meno_bench.models import TestCasesFile, TestCasesFileFull, TestCaseFromFileFull
import json
import tqdm
import requests
from typing import Callable
from datetime import timedelta, datetime


def retrieve(
    address: str, model: str, text: str
) -> tuple[str, timedelta] | tuple[None, None]:
    messages = [
        {"role": "user", "content": text},
    ]

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "user": "test",
    }

    t = datetime.now()
    response = requests.post(address, json=payload)
    t = datetime.now() - t
    if response.status_code != 200:
        return None, None
    return response.json()["choices"][0]["message"]["content"], t


def inference_cases(
    interence_f: Callable[[str], tuple[str, timedelta]], cases: TestCasesFile
) -> TestCasesFileFull:
    results: TestCasesFileFull = []
    try:
        for jsoned_test in tqdm.tqdm(cases):
            if "model_answer" not in jsoned_test or jsoned_test["model_answer"] is None:
                model_answer, t = interence_f(jsoned_test["question"])
                if model_answer is None:
                    raise Exception("Could not retrieve model answer")
                results.append(
                    TestCaseFromFileFull(
                        question=jsoned_test["question"],
                        ground_truth=jsoned_test["ground_truth"],
                        model_answer=model_answer,
                        time_s=t.total_seconds(),
                    )
                )
    except Exception as e:
        print(e)
    return results


def inference(settings: InferenceSettings):
    with open(settings.file, "r") as f:
        jsoned_tests: TestCasesFile = json.load(f)

    result = inference_cases(
        lambda text: retrieve(settings.address, settings.model, text), jsoned_tests
    )

    out_file = settings.file.with_stem(f"{settings.file.stem}_out")
    with open(out_file, "w") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
