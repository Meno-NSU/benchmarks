from argparse import ArgumentParser
from dotenv import load_dotenv
import os
from dataclasses import dataclass, field
from pathlib import Path
from enum import StrEnum


class Arguments(StrEnum):
    environment_file = "env_file"
    file = "file"
    inference = "inference"
    address = "address"
    model = "model"
    api_key = "api_key"
    use_gemini = "use_gemini"
    proxy = "proxy"

    def as_arg(self) -> str:
        return f"--{self.value}"

    def as_short_arg(self) -> str:
        return f"-{self.value[0]}"


class Environment(StrEnum):
    file = "FILE"
    inference = "INFERENCE"
    address = "ADDRESS"
    model = "MODEL"
    api_key = "API_KEY"
    use_gemini = "USE_GEMINI"
    proxy = "PROXY"


ARGS_TO_ENV = {
    Arguments.file: Environment.file,
    Arguments.inference: Environment.inference,
    Arguments.address: Environment.address,
    Arguments.model: Environment.model,
    Arguments.api_key: Environment.api_key,
    Arguments.use_gemini: Environment.use_gemini,
    Arguments.proxy: Environment.proxy,
}


@dataclass
class ProxySettings:
    proxy: str | None = None

    def __post_init__(self):
        if self.proxy:
            os.environ["wss_proxy"] = self.proxy
            os.environ["ws_proxy"] = self.proxy


@dataclass
class JudgeSettings(ProxySettings):
    file: Path
    address: str
    model: str
    api_key: str
    use_gemini: bool


@dataclass
class InferenceSettings(ProxySettings):
    file: Path
    address: str
    model: str = "menon-1"


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        Arguments.environment_file.as_arg(), "-e", type=Path, help="Environment file"
    )
    parser.add_argument(
        Arguments.inference.as_arg(),
        "-i",
        action="store_true",
        help="Inference mode. Please choose address of the backend",
    )
    parser.add_argument(
        Arguments.inference.as_arg(),
        "-i",
        action="store_true",
        help="Inference mode. Please choose address of the backend",
    )
    parser.add_argument(
        Arguments.address.as_arg(),
        "-a",
        type=str,
        default="localhost:8000",
        help="Address of the backend or model",
    )
    parser.add_argument(
        Arguments.model.as_arg(),
        "-m",
        type=str,
        default=None,
        help="Model name for OpenAI API or Gemini API",
    )
    parser.add_argument(
        Arguments.api_key.as_arg(),
        "-k",
        type=str,
        default=None,
        help="Api key for model.",
    )
    parser.add_argument(
        Arguments.use_gemini.as_arg(),
        "-g",
        action="store_true",
        help="Use gemini judge model. Api key required",
    )
    parser.add_argument(
        Arguments.proxy.as_arg(), "-p", type=str, help="Proxy server address."
    )
    return parser


def get_args():
    parser = create_parser()
    args = parser.parse_args()
    return args


def get_settings(env_file: Path | None = None) -> JudgeSettings | InferenceSettings:
    args = get_args()
    if args.environment_file:
        env_file = args.environment_file
    load_dotenv(dotenv_path=env_file)
    for arg, env in ARGS_TO_ENV.items():
        if getattr(args, arg) is not None:
            os.environ[env] = getattr(args, arg)
    if os.environ.get(ARGS_TO_ENV[Arguments.inference]):
        return InferenceSettings(
            **{k: os.environ.get(v) for k, v in ARGS_TO_ENV.items()}
        )
    else:
        return JudgeSettings(**{k: os.environ.get(v) for k, v in ARGS_TO_ENV.items()})


settings = get_settings()
