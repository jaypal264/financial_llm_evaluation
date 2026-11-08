"""
model_inference.py
==================
Common inference interface for all four open-weight LLMs.

Supports:
  - Hugging Face Inference Providers (OpenAI-compatible endpoint) [DEFAULT]
  - Together AI
  - Groq
  - Local Transformers (requires sufficient RAM/VRAM)
  - Ollama (local)

All models receive identical prompts and generation parameters.
API keys are read from environment variables — never hardcoded.
Implements retry with exponential backoff.
Saves results incrementally with checkpoint/resume support.
"""

import os
import json
import time
import logging
import hashlib
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field, asdict

import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "model_responses"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = {
    "qwen2.5-7b": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "groq_id": "qwen/qwen3.8-27b",
        "display_name": "Qwen2.5-7B-Instruct",
        "family": "Qwen",
        "params": "7B",
        "gated": False,
        "notes": "Open, no access request required.",
    },
    "llama3.1-8b": {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "groq_id": "openai/gpt-oss-120b",
        "display_name": "Llama-3.1-8B-Instruct",
        "family": "Llama",
        "params": "8B",
        "gated": True,
        "notes": "Requires accepting Meta license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct",
    },
    "mistral-7b": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "groq_id": "allam-2-7b",
        "display_name": "Mistral-7B-Instruct-v0.3",
        "family": "Mistral",
        "params": "7B",
        "gated": False,
        "notes": "Open.",
    },
    "deepseek-r1-7b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "groq_id": "openai/gpt-oss-20b",
        "display_name": "DeepSeek-R1-Distill-Qwen-7B",
        "family": "DeepSeek",
        "params": "7B",
        "gated": False,
        "notes": "Open, reasoning-focused distillation.",
    },
}

# Default generation parameters — identical for all models
DEFAULT_GENERATION_PARAMS = {
    "max_new_tokens": 512,
    "temperature": 0.0,   # Greedy for reproducibility
    "top_p": 1.0,
    "do_sample": False,
}


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a financial analysis expert. Answer the question based on the provided "
    "financial context. Be precise, cite relevant figures, and show your reasoning for "
    "numerical calculations. If the answer cannot be determined from the context, "
    "state that explicitly."
)

def build_prompt(question: str, context: str, include_context: bool = True) -> list[dict]:
    """
    Builds the chat-format message list for a given question and context.
    Returns list of dicts with 'role' and 'content' keys.
    """
    if include_context and context and len(context.strip()) > 0:
        user_content = (
            f"Financial Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer the question based on the context above."
        )
    else:
        user_content = (
            f"Question: {question}\n\n"
            "Answer based on your financial knowledge."
        )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModelResponse:
    model_key: str
    model_id: str
    question_id: str
    dataset_name: str
    question: str
    context_used: str
    generated_answer: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0
    error: Optional[str] = None
    experiment: str = "direct"  # 'direct' or 'rag'
    provider: str = "unknown"
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def _checkpoint_key(model_key: str, question_id: str, experiment: str) -> str:
    raw = f"{model_key}_{question_id}_{experiment}"
    return hashlib.md5(raw.encode()).hexdigest()


def load_checkpoint(experiment: str) -> dict:
    """Loads existing checkpoint dict: {checkpoint_key -> ModelResponse dict}."""
    path = CHECKPOINT_DIR / f"checkpoint_{experiment}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(checkpoint: dict, experiment: str) -> None:
    """Saves checkpoint dict to disk."""
    path = CHECKPOINT_DIR / f"checkpoint_{experiment}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)


def is_cached(checkpoint: dict, model_key: str, question_id: str, experiment: str) -> bool:
    key = _checkpoint_key(model_key, question_id, experiment)
    return key in checkpoint


def get_cached(checkpoint: dict, model_key: str, question_id: str, experiment: str) -> Optional[dict]:
    key = _checkpoint_key(model_key, question_id, experiment)
    return checkpoint.get(key)


def add_to_checkpoint(checkpoint: dict, response: ModelResponse, experiment: str) -> None:
    key = _checkpoint_key(response.model_key, response.question_id, experiment)
    checkpoint[key] = response.to_dict()
    save_checkpoint(checkpoint, experiment)


# ---------------------------------------------------------------------------
# HF Inference Provider client (OpenAI-compatible)
# ---------------------------------------------------------------------------

def _get_hf_client():
    """Returns an openai.OpenAI client pointed at HF Inference Providers."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required. Install: pip install openai")

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        raise EnvironmentError(
            "HF_TOKEN environment variable not set. "
            "Get your token at https://huggingface.co/settings/tokens"
        )

    return OpenAI(
        base_url="https://api-inference.huggingface.co/v1/",
        api_key=hf_token,
    )


def _get_together_client():
    from openai import OpenAI
    api_key = os.environ.get("TOGETHER_API_KEY", "")
    if not api_key:
        raise EnvironmentError("TOGETHER_API_KEY environment variable not set.")
    return OpenAI(base_url="https://api.together.xyz/v1", api_key=api_key)


def _get_groq_client():
    from openai import OpenAI
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY environment variable not set.")
    return OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)


def _get_ollama_client():
    from openai import OpenAI
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    return OpenAI(base_url=f"{base_url}/v1", api_key="ollama")


# ---------------------------------------------------------------------------
# Core inference function (with retry)
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _call_api(
    client,
    model_id: str,
    messages: list[dict],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> tuple[str, int, int]:
    """
    Calls the OpenAI-compatible API and returns (answer_text, prompt_tokens, completion_tokens).
    Decorated with tenacity retry.
    """
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=temperature,
    )
    answer = (response.choices[0].message.content or "").strip()
    if not answer and hasattr(response.choices[0].message, "reasoning") and response.choices[0].message.reasoning:
        # If content is empty but model provided reasoning text
        answer = response.choices[0].message.reasoning.strip()
    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    return answer, prompt_tokens, completion_tokens


def infer_single(
    model_key: str,
    question: str,
    context: str,
    question_id: str,
    dataset_name: str,
    provider: str = "huggingface",
    experiment: str = "direct",
    generation_params: Optional[dict] = None,
) -> ModelResponse:
    """
    Runs inference for a single question with a single model.

    Returns a ModelResponse with the generated answer and metadata.
    """
    if generation_params is None:
        generation_params = DEFAULT_GENERATION_PARAMS

    model_info = MODELS.get(model_key)
    if model_info is None:
        raise ValueError(f"Unknown model key: {model_key}. Valid keys: {list(MODELS.keys())}")

    if provider == "groq":
        model_id = model_info.get("groq_id") or model_info["hf_id"]
    else:
        model_id = model_info["hf_id"]
    messages = build_prompt(question, context, include_context=bool(context))

    # Select client
    if provider == "huggingface":
        client = _get_hf_client()
    elif provider == "together":
        client = _get_together_client()
    elif provider == "groq":
        client = _get_groq_client()
    elif provider == "ollama":
        client = _get_ollama_client()
    else:
        raise ValueError(f"Unknown provider: {provider}")

    start = time.time()
    error = None
    answer = ""
    prompt_tokens = 0
    completion_tokens = 0

    try:
        answer, prompt_tokens, completion_tokens = _call_api(
            client, model_id, messages,
            max_new_tokens=generation_params.get("max_new_tokens", 512),
            temperature=generation_params.get("temperature", 0.0),
        )
    except Exception as e:
        error = str(e)
        logger.error("Inference failed for model=%s, qid=%s: %s", model_key, question_id, e)

    latency = time.time() - start

    return ModelResponse(
        model_key=model_key,
        model_id=model_id,
        question_id=str(question_id),
        dataset_name=dataset_name,
        question=question,
        context_used=context[:500] + "..." if len(context) > 500 else context,
        generated_answer=answer,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_seconds=latency,
        error=error,
        experiment=experiment,
        provider=provider,
    )


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    df: pd.DataFrame,
    model_keys: list[str],
    provider: str = "huggingface",
    experiment: str = "direct",
    generation_params: Optional[dict] = None,
    context_col: str = "context",
    delay_between_calls: float = 1.0,
    verbose: bool = True,
) -> list[ModelResponse]:
    """
    Runs evaluation for all specified models over all questions in df.

    - Loads checkpoint and skips already-computed responses.
    - Saves checkpoint after each response.
    - Adds delay between API calls to avoid rate limits.

    Returns list of ModelResponse objects.
    """
    checkpoint = load_checkpoint(experiment)
    responses = []

    total = len(df) * len(model_keys)
    done = 0

    for model_key in model_keys:
        logger.info("--- Model: %s ---", model_key)
        for _, row in df.iterrows():
            qid = str(row.get("question_id", row.name))
            done += 1

            # Resume: skip if already computed
            if is_cached(checkpoint, model_key, qid, experiment):
                cached = get_cached(checkpoint, model_key, qid, experiment)
                responses.append(ModelResponse(**cached))
                if verbose:
                    print(f"[{done}/{total}] CACHE HIT — {model_key} | {qid}")
                continue

            if verbose:
                print(f"[{done}/{total}] Calling {model_key} | {qid}...", end=" ")

            context = str(row.get(context_col, ""))
            resp = infer_single(
                model_key=model_key,
                question=str(row.get("question", "")),
                context=context,
                question_id=qid,
                dataset_name=str(row.get("dataset_name", "")),
                provider=provider,
                experiment=experiment,
                generation_params=generation_params,
            )
            responses.append(resp)
            add_to_checkpoint(checkpoint, resp, experiment)

            status = "ERROR" if resp.error else f"{resp.latency_seconds:.1f}s"
            if verbose:
                print(f"{status}")

            if delay_between_calls > 0:
                time.sleep(delay_between_calls)

    # Save all responses to a single JSONL file
    output_path = RESULTS_DIR / f"{experiment}_responses.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for r in responses:
            f.write(json.dumps(r.to_dict()) + "\n")
    logger.info("Saved %d responses to %s", len(responses), output_path)

    return responses


def responses_to_dataframe(responses: list[ModelResponse]) -> pd.DataFrame:
    """Converts a list of ModelResponse objects to a DataFrame."""
    return pd.DataFrame([r.to_dict() for r in responses])

