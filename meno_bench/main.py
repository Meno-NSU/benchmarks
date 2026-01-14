from meno_bench.settings import get_settings, InferenceSettings, SummarySettings
from meno_bench.inference import inference
from meno_bench.judge.summary import summarize_to_file
from meno_bench.judge import judge


def main():
    # settings = get_settings("inference.example.env")
    settings = get_settings()
    if settings:
        if isinstance(settings, InferenceSettings):
            inference(settings)
        elif isinstance(settings, SummarySettings):
            summarize_to_file(settings.file, settings.out_file)
        else:
            judge(settings)


if __name__ == "__main__":
    main()
