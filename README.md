# This is a Library and a CLI tool to test rag system

# Installation

Just install as [uv](https://docs.astral.sh/uv/getting-started/installation/) tool:
```bash
uv tool install https://github.com/Meno-NSU/benchmarks.git
```

# Usage

This tool can work with command line arguments or corresponding environment variables.

Tool uses Typer for simple command line interface.

This tool provides 2 modes: Inference and Judge with 2 API.

## Inference

To inference you need a json file with format:
```json
[
  {
    "question": "...",
    "ground_truth": "...",
  },
  {
    "question": "...",
    "ground_truth": "...",
  },
  "..."
]
```

Example of env file for inference is `inference.example.env`:
```bash
FILE="correct_json.json"
ADDRESS="http://10.1.15.44:9006/v1/chat/completions"
MODEL="menon-1"
NAME=INFERENCE
```

or command line variant:

```bash
meno-bench inference correct_json.json http://10.1.15.44:9006/v1/chat/completions menon-1
```

The address is backend endpoint for inference. It must accept messages in format:
```json
{
    "model": "...",
    "messages": [
        {
            "role": "user",
            "content": "..."
        }
    ],
    "stream": false,
    "user": "test"
}
```

#### The output will be written to file `[INPUT FILE NAME]_out.json`

## Judge

There are two options to use judge: via google API or via OpenAI standard API interface.

### OpenAI API interface

this is standard interface for llms. Required config is `judge_openai_api.example.env`:
```bash
FILE="correct_json_out.json"
ADDRESS="http://192.168.3.213:9111/v1"
MODEL="meno-tiny-0.1"
API_KEY=""
NAME="OPENAI"
```

or command line variant:

```bash
meno-bench openai correct_json_out.json http://192.168.3.213:9111/v1 meno-tiny-0.1
```

It must accept file from inference stage.

### Google API

This is mode to judge via google gemini API. Required config is `judge_google_proxy.example.env`:

```bash
FILE="correct_json_out.json"
PROXY="socks5://127.0.0.1:9150"
MODEL="gemini-2.5-flash"
NAME=GOOGLE
USE_GEMINI=1
USE_GEMINI_LIVE=0
API_KEY=""
```

or command line variant:

```bash
meno-bench correct_json_out.json "API_KEY" "gemini-2.5-flash" --peoxy "socks5://127.0.0.1:9150" --no-use-gemini-live
```

If you want to force use live mode, set `USE_GEMINI_LIVE=1` or add `--use-gemini-live` argument. Since google swapping live models and often removes pure live text-to-text models, it is not recommended to use it. Also if model name contain `live` keyword it will be treated as live model.

#### The output will be written to file `[INPUT FILE NAME]_judged.json`

#### The summary will be written to file `[INPUT FILE NAME]_summary.json`
