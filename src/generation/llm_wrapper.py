"""Phase 4 — unified wrapper for Llama-3, Mistral, and Phi-3."""
import gc
import os

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import get_bnb_config, GENERATION_CONFIG, MODEL_IDS

load_dotenv()


class LLMWrapper:
    """
    Unified interface for all three project LLMs.
    Identical .generate(system, user) call regardless of model.
    apply_chat_template handles per-model format differences automatically.

    Usage:
        llm = LLMWrapper("mistral")
        answer = llm.generate(system_prompt, user_prompt)
        llm.unload()   # free GPU memory before loading the next model
    """

    def __init__(self, model_key: str):
        if model_key not in MODEL_IDS:
            raise ValueError(
                f"Unknown model '{model_key}'. Valid: {list(MODEL_IDS)}"
            )

        self.model_key = model_key
        self.model_id  = MODEL_IDS[model_key]
        hf_token = os.getenv("HF_TOKEN")

        if hf_token is None:
            print(f"[{model_key}] WARNING: HF_TOKEN not set — gated models (Llama-3) will fail.")

        print(f"[{model_key}] loading tokenizer: {self.model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, token=hf_token
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[{model_key}] loading model in 4-bit NF4 ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=get_bnb_config(),
            device_map="auto",
            token=hf_token,
        )
        self.model.eval()

        mem_gb = torch.cuda.memory_allocated() / 1e9
        print(f"[{model_key}] loaded — GPU mem: {mem_gb:.2f} GB\n")

    def _build_messages(self, system_prompt: str, user_prompt: str) -> list[dict]:
        """
        Build the messages list, handling per-model quirks.
        Mistral-7B-Instruct-v0.2's chat template does not accept a 'system' role —
        we fold the system prompt into the first user turn.
        """
        if self.model_key == "mistral":
            return [{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}]
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

    @torch.no_grad()
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        **gen_overrides,
    ) -> str:
        """
        Generate an answer. Any kwargs override GENERATION_CONFIG defaults.
        """
        messages = self._build_messages(system_prompt, user_prompt)

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=3500,   # leave headroom for generation within 4k context
        ).to(self.model.device)

        config = {**GENERATION_CONFIG, **gen_overrides}
        config["pad_token_id"] = self.tokenizer.eos_token_id

        outputs = self.model.generate(**inputs, **config)

        # Decode only the newly generated tokens (not the prompt)
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def unload(self):
        """Free GPU memory so the next model can load. Critical on T4 / 16 GB GPUs."""
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[{self.model_key}] unloaded.")