from geval_test.settings import InferenceSettings
from geval_test.models import TestCasesFile, TestCasesFileFull, TestCaseFromFileFull
import json
import tqdm
import requests


def retrieve(address: str, text: str) -> str | None:
    messages = [
        {"role": "user", "content": text},
    ]

    payload = {
        "model": "menon-1",
        "messages": messages,
        "stream": False,
        "user": "test",
    }

    response = requests.post(address, json=payload)
    if response.status_code != 200:
        return None
    return response.json()["choices"][0]["message"]["content"]


def inference_cases(
    settings: InferenceSettings, cases: TestCasesFile
) -> TestCasesFileFull:
    for jsoned_test in tqdm.tqdm(cases):
        if "model_answer" not in jsoned_test or jsoned_test["model_answer"] is None:
            jsoned_test["model_answer"] = retrieve(
                settings.address, jsoned_test["question"]
            )
    return cases


def inference(settings: InferenceSettings):
    with open(settings.file, "r") as f:
        jsoned_tests: TestCasesFile = json.load(f)

    jsoned_tests = inference_cases(settings, jsoned_tests)

    out_file = settings.file.with_stem(f"{settings.file.stem}_out")
    with open(out_file, "w") as f:
        json.dump(jsoned_tests, f, indent=4, ensure_ascii=False)
