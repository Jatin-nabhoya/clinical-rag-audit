"""Phase 4 — centralized config for generation."""

# Model registry — short keys map to full HF model IDs
MODEL_IDS = {
    "llama3":  "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "phi3":    "microsoft/Phi-3-mini-4k-instruct",
}


def get_bnb_config():
    """
    Lazy-initialize BitsAndBytesConfig so this module can be imported on
    non-CUDA machines (Mac / CPU-only).  Call this only when actually loading
    a model on a CUDA GPU.
    """
    import torch
    from transformers import BitsAndBytesConfig

    # 4-bit NF4 quantization — IDENTICAL across all three models
    # to keep the comparison fair (quantization is not a confound)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


# Keep BNB_CONFIG as a module-level alias for backwards compatibility —
# it is evaluated lazily only when called.
BNB_CONFIG = get_bnb_config

# Deterministic generation — required for reproducible hallucination auditing
GENERATION_CONFIG = {
    "do_sample":          False,  # greedy decoding → deterministic
    "temperature":        0.0,    # redundant with do_sample=False, explicit for clarity
    "max_new_tokens":     512,
    "repetition_penalty": 1.1,    # guards against loops in small quantized models
}

# RAG retrieval settings
DEFAULT_TOP_K = 5