"""Run evaluate_german_fluency's exact evaluation logic with local Llama 70B."""

import os
import sys
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import evaluate_german_fluency as base


DEFAULT_MODEL = os.path.expanduser("~/models/llama-70b")
DEFAULT_OUTPUT_DIR = "fluency_evaluation_results_llama70b"


class LocalLlamaBackend:
    def __init__(self, model_name):
        model_name = os.path.expanduser(model_name)
        print(f"Loading local Llama model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )

    def complete(self, messages, max_tokens, temperature, top_logprobs):
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.model.device)
        else:
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        generated = outputs.sequences[0][input_len:]
        content = self.tokenizer.decode(generated, skip_special_tokens=False)
        logprob_content = []
        for generated_id, scores in zip(generated, outputs.scores):
            log_probs = torch.log_softmax(scores[0], dim=-1)
            top_values, top_ids = torch.topk(log_probs, k=min(top_logprobs, log_probs.shape[-1]))
            chosen_id = int(generated_id)
            alternatives = [
                SimpleNamespace(
                    token=self.tokenizer.decode([int(token_id)]),
                    logprob=float(log_prob),
                )
                for token_id, log_prob in zip(top_ids, top_values)
            ]
            logprob_content.append(
                SimpleNamespace(
                    token=self.tokenizer.decode([chosen_id]),
                    logprob=float(log_probs[chosen_id]),
                    top_logprobs=alternatives,
                )
            )

        usage = SimpleNamespace(
            prompt_tokens=int(input_len),
            completion_tokens=int(generated.shape[-1]),
            total_tokens=int(input_len + generated.shape[-1]),
        )
        choice = SimpleNamespace(
            message=SimpleNamespace(content=content),
            logprobs=SimpleNamespace(content=logprob_content),
        )
        return SimpleNamespace(choices=[choice], usage=usage)


class LocalOpenAI:
    """Minimal local replacement for the OpenAI client used by the base script."""

    _backends = {}

    def __init__(self, api_key=None, base_url=None):
        self.chat = SimpleNamespace(completions=self)

    def create(self, model, messages, temperature, max_tokens, logprobs, top_logprobs):
        if model not in self._backends:
            self._backends[model] = LocalLlamaBackend(model)
        return self._backends[model].complete(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_logprobs=top_logprobs,
        )


def _has_option(name):
    return any(argument == name or argument.startswith(f"{name}=") for argument in sys.argv[1:])


if __name__ == "__main__":
    # Preserve every command-line option from evaluate_german_fluency.py, but
    # give this dedicated runner Llama-specific defaults.
    if not _has_option("--model"):
        sys.argv.extend(["--model", DEFAULT_MODEL])
    if not _has_option("--output-dir"):
        sys.argv.extend(["--output-dir", DEFAULT_OUTPUT_DIR])

    base.OpenAI = LocalOpenAI
    raise SystemExit(base.main())
