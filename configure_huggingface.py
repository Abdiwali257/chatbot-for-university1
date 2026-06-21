"""Safely write Hugging Face inference settings to the local .env file."""

from __future__ import annotations

from getpass import getpass
from pathlib import Path

from huggingface_hub import InferenceClient

PROJECT_DIR = Path(__file__).resolve().parent
ENV_PATH = PROJECT_DIR / ".env"
DEFAULTS = {
    "HF_MODEL": "Qwen/Qwen2.5-7B-Instruct",
    "HF_PROVIDER": "auto",
    "HF_TIMEOUT_SECONDS": "90",
    "HF_MAX_TOKENS": "500",
}


def read_env() -> tuple[list[str], dict[str, str]]:
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.exists() else []
    values = {}
    for line in lines:
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return lines, values


def main() -> None:
    print("Create a new fine-grained Hugging Face token before continuing.")
    print("Enable the permission: Make calls to Inference Providers.")
    token = getpass("New Hugging Face token: ").strip()
    if not token.startswith("hf_"):
        raise SystemExit("That does not look like a Hugging Face token. Nothing was changed.")

    model = input(f"Model [{DEFAULTS['HF_MODEL']}]: ").strip() or DEFAULTS["HF_MODEL"]
    provider = input(f"Provider [{DEFAULTS['HF_PROVIDER']}]: ").strip() or DEFAULTS["HF_PROVIDER"]
    print("Validating token with Hugging Face Inference Providers...")
    try:
        response = InferenceClient(
            model=model,
            provider=provider,
            token=token,
            timeout=30,
        ).chat_completion(
            model=model,
            messages=[{"role": "user", "content": "Reply only OK"}],
            max_tokens=10,
            temperature=0.0,
        )
        if not response.choices[0].message.content:
            raise RuntimeError("Hugging Face returned an empty response.")
    except Exception as exc:
        raise SystemExit(
            "Token validation failed. Create a fine-grained token with the permission "
            "'Make calls to Inference Providers', then run this script again.\n"
            f"Provider error: {type(exc).__name__}"
        ) from exc

    lines, values = read_env()
    updates = {**DEFAULTS, "HF_TOKEN": token, "HF_MODEL": model, "HF_PROVIDER": provider}

    output = []
    updated = set()
    for line in lines:
        key = line.split("=", 1)[0].strip() if "=" in line else ""
        if key in updates:
            output.append(f"{key}={updates[key]}")
            updated.add(key)
        else:
            output.append(line)

    for key, value in updates.items():
        if key not in updated:
            output.append(f"{key}={value}")

    ENV_PATH.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")
    print(f"Hugging Face settings saved to {ENV_PATH.name}. Restart the Django server.")


if __name__ == "__main__":
    main()
