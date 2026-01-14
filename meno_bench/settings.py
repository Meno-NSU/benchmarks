from argparse import ArgumentParser, _SubParsersAction, Action
from dotenv import load_dotenv
import os
from dataclasses import dataclass, fields, field
from pathlib import Path
from enum import StrEnum
from typing import TypedDict, get_args
import typer


app = typer.Typer()


class ArgumentMedata(TypedDict):
    help: str
    arg: str
    short_arg: str
    env_var: str


def add_to_typer(f):
    app.command(getattr(f, "_name"), help=getattr(f, "_help"))(f)
    return f


def add_function_to_typer(name: str, help: str | None = None):
    def adder_function_decorator(f):
        app.command(name, help=help)(f)

    return adder_function_decorator


@add_to_typer
@dataclass
class InferenceSettings:
    file: Path = field(
        metadata=ArgumentMedata(
            help="File with test cases",
            arg="--file",
            short_arg="-f",
            env_var="FILE",
        )
    )
    address: str = field(
        metadata=ArgumentMedata(
            help="Address of the model",
            arg="--address",
            short_arg="-a",
            env_var="ADDRESS",
        )
    )
    model: str = field(
        metadata=ArgumentMedata(
            help="Model name",
            arg="--model",
            short_arg="-m",
            env_var="MODEL",
        )
    )
    _name = "inference"
    _help = "Inference settings"

    def __post_init__(self):
        self.file = Path(self.file)


@add_to_typer
@dataclass
class SummarySettings:
    file: Path = field(
        metadata=ArgumentMedata(
            help="File with test cases",
            arg="--file",
            short_arg="-f",
            env_var="FILE",
        )
    )
    out_file: Path | None = field(
        default=None,
        metadata=ArgumentMedata(
            help="File to output. Defaults to '<file_input>_summary.json'",
            arg="--out-file",
            short_arg="-o",
            env_var="OUT_FILE",
        ),
    )
    _name = "summary"
    _help = "Summarize file"


@add_to_typer
@dataclass
class GoogleJudgeSettings:
    file: Path = field(
        metadata=ArgumentMedata(
            help="File with test cases",
            arg="--file",
            short_arg="-f",
            env_var="FILE",
        )
    )
    api_key: str = field(
        metadata=ArgumentMedata(
            help="API key",
            arg="--api-key",
            short_arg="-k",
            env_var="GEMINI_API_KEY",
        )
    )
    model: str = field(
        metadata=ArgumentMedata(
            help="Model name",
            arg="--model",
            short_arg="-m",
            env_var="MODEL",
        )
    )
    use_gemini_live: bool = field(
        default=False,
        metadata=ArgumentMedata(
            help="Use live API",
            arg="--live",
            short_arg="-l",
            env_var="USE_GEMINI_LIVE",
        ),
    )
    proxy: str | None = field(
        default=None,
        metadata=ArgumentMedata(
            help="Address of the model",
            arg="--address",
            short_arg="-a",
            env_var="ADDRESS",
        ),
    )
    _name = "google"
    _help = "Google Judge"

    def __post_init__(self):
        if self.proxy is not None:
            os.environ["wss_proxy"] = self.proxy
            os.environ["ws_proxy"] = self.proxy


@add_to_typer
@dataclass
class OpenAIJudgeSettings:
    file: Path = field(
        metadata=ArgumentMedata(
            help="File with test cases",
            arg="--file",
            short_arg="-f",
            env_var="FILE",
        )
    )
    address: str = field(
        metadata=ArgumentMedata(
            help="Address of the model",
            arg="--address",
            short_arg="-a",
            env_var="ADDRESS",
        )
    )
    model: str = field(
        metadata=ArgumentMedata(
            help="Model name",
            arg="--model",
            short_arg="-m",
            env_var="MODEL",
        )
    )
    api_key: str | None = field(
        default=None,
        metadata=ArgumentMedata(
            help="API key",
            arg="--api-key",
            short_arg="-k",
            env_var="OPENAI_API_KEY",
        ),
    )
    proxy: str | None = field(
        default=None,
        metadata=ArgumentMedata(
            help="Address of the model",
            arg="--address",
            short_arg="-a",
            env_var="ADDRESS",
        ),
    )
    _name = "openai"
    _help = "OpenAI Judge"


@add_to_typer
@dataclass
class GigaSettings:
    file: Path = field(
        metadata=ArgumentMedata(
            help="File with test cases",
            arg="--file",
            short_arg="-f",
            env_var="FILE",
        )
    )
    api_key: str = field(
        metadata=ArgumentMedata(
            help="API key",
            arg="--api-key",
            short_arg="-k",
            env_var="GEMINI_API_KEY",
        )
    )
    model: str = field(
        metadata=ArgumentMedata(
            help="Model name",
            arg="--model",
            short_arg="-m",
            env_var="MODEL",
        )
    )
    _name = "giga"
    _help = "GigaChat judge"


type AnySettings = (
    InferenceSettings
    | SummarySettings
    | GoogleJudgeSettings
    | OpenAIJudgeSettings
    | GigaSettings
)
type JudgeSettings = GoogleJudgeSettings | OpenAIJudgeSettings | GigaSettings


@add_function_to_typer(
    name="env", help="Load settings from environment variables file .env by default"
)
def get_settings_from_env_file(
    env_file: Path | None = None, file: Path | None = None
) -> AnySettings:
    load_dotenv(dotenv_path=env_file)
    assert "NAME" in os.environ, "NAME is not set in .env file"
    assert file is not None or "FILE" in os.environ, (
        "FILE is not set in .env file and not passed as arg"
    )
    for cls in get_args(AnySettings.__value__):
        if os.environ["NAME"].lower() == cls._name.lower():
            kwargs = {}
            if file is not None:
                kwargs["file"] = file
            for field in fields(cls):
                if field.name not in kwargs:
                    kwargs[field.name] = os.environ.get(field.metadata["env_var"])
            return cls(**kwargs)


def get_settings(env_file: Path | None = None, file: Path | None = None) -> AnySettings:
    if env_file is not None:
        return get_settings_from_env_file(env_file, file)
    return app(standalone_mode=False)
