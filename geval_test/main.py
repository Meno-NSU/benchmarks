from geval_test.settings import settings, InferenceSettings
from geval_test.inference import inference
from geval_test.judge import judge


def main():
    if isinstance(settings, InferenceSettings):
        inference(settings)
    else:
        judge(settings)


if __name__ == "__main__":
    main()
