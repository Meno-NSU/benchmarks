from argparse import ArgumentParser
from dotenv import load_dotenv
import os
from dataclasses import dataclass, fields
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
    use_gemini_live = "use_gemini_live"
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
    use_gemini_live = "USE_GEMINI_LIVE"
    proxy = "PROXY"


ARGS_TO_ENV = {
    Arguments.file: Environment.file,
    Arguments.inference: Environment.inference,
    Arguments.address: Environment.address,
    Arguments.model: Environment.model,
    Arguments.api_key: Environment.api_key,
    Arguments.use_gemini: Environment.use_gemini,
    Arguments.use_gemini_live: Environment.use_gemini_live,
    Arguments.proxy: Environment.proxy,
}


@dataclass
class ProxySettings:
    def __post_init__(self):
        if getattr(self, "proxy") is not None:
            os.environ["wss_proxy"] = self.proxy
            os.environ["ws_proxy"] = self.proxy

    @classmethod
    def from_kwargs(cls, **kwargs):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in kwargs.items() if k in names})


@dataclass
class JudgeSettings(ProxySettings):
    file: Path
    address: str | None
    model: str
    api_key: str
    use_gemini: bool
    use_gemini_live: bool
    proxy: str | None = None

    def __post_init__(self):
        self.file = Path(self.file)
        self.use_gemini = bool(int(self.use_gemini))
        self.use_gemini_live = bool(int(self.use_gemini_live))


@dataclass
class InferenceSettings(ProxySettings):
    file: Path
    address: str
    model: str = "menon-1"
    proxy: str | None = None

    def __post_init__(self):
        self.file = Path(self.file)


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        Arguments.environment_file.as_arg(), "-e", type=Path, help="Environment file"
    )
    parser.add_argument(
        Arguments.file.as_arg(),
        "-f",
        help="File to judge or inference",
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
        Arguments.use_gemini_live.as_arg(),
        Arguments.use_gemini_live.as_short_arg(),
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
    if (env_file_from_args := getattr(args, Arguments.environment_file)) is not None:
        env_file = env_file_from_args
    load_dotenv(dotenv_path=env_file)
    for arg, env in ARGS_TO_ENV.items():
        val = getattr(args, arg)
        match val:
            case None:
                continue
            case True | False if os.environ.get(env) is None:
                os.environ[env] = str(int(val))
            case _ if os.environ.get(env) is None:
                os.environ[env] = str(val)
    is_inference = int(os.environ.get(ARGS_TO_ENV[Arguments.inference]))
    kwargs = {k: os.environ.get(v) for k, v in ARGS_TO_ENV.items()}
    if is_inference:
        return InferenceSettings.from_kwargs(**kwargs)
    else:
        return JudgeSettings.from_kwargs(**kwargs)
