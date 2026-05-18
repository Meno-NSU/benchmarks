from meno_bench.settings import (
    get_settings,
    InferenceSettings,
    SummarySettings,
    PlotSettings,
    MakeXLSettings,
    OpenAIJudgeSettings
)
from meno_bench.inference import inference
from meno_bench.judge.summary import summarize_to_file
from meno_bench.judge import judge
from meno_bench.plotting import plot_sums
from meno_bench.xl import make_report
from pathlib import Path


def main():
    # settings = get_settings("report.env")
    # settings = get_settings()
    settings = OpenAIJudgeSettings(
        "./nat_out.json",
        "http://localhost:9111/v1",
        "meno-tiny-0.1",
        api_key="None"
    )

    if settings:
        if isinstance(settings, InferenceSettings):
            inference(settings)
        elif isinstance(settings, SummarySettings):
            summarize_to_file(settings.file, settings.out_file)
        elif isinstance(settings, PlotSettings):
            plot_sums(
                settings.scan_dir,
                settings.out_path,
                settings.show,
                settings.html,
                settings.image,
            )
        elif isinstance(settings, MakeXLSettings):
            make_report(
                settings.scan_dir,
                settings.out_path,
            )
        else:
            judge(settings)


if __name__ == "__main__":
    main()
