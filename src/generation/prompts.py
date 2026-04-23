"""Phase 4 — RAG prompt templates for clinical question answering."""

SYSTEM_PROMPT = """You are a clinical information assistant. Your task is to answer medical questions using ONLY the information provided in the CONTEXT below.

Strict rules:
1. Base your answer EXCLUSIVELY on the provided context. Do not use outside medical knowledge.
2. If the context does not contain enough information to answer the question, respond with EXACTLY: "The provided context does not contain enough information to answer this question."
3. If the context only partially answers the question, answer the supported part and state what is missing.
4. Do not speculate, infer causes, or extrapolate beyond what is explicitly stated.
5. Do not invent statistics, dosages, study results, or clinical guidelines.
6. Be concise and factual. Quote or closely paraphrase the context when stating specific facts.
"""


def build_rag_prompt(question: str, context: str) -> tuple[str, str]:
    """
    Build the (system, user) prompt pair for a RAG query.
    Returns a tuple compatible with LLMWrapper.generate().
    """
    user_prompt = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:"
    )
    return SYSTEM_PROMPT, user_prompt


def build_no_rag_prompt(question: str) -> tuple[str, str]:
    """
    Ablation prompt — no retrieved context.
    Used in Phase 5 to measure how much RAG reduces hallucination
    compared to the model's parametric knowledge alone.
    """
    system = (
        "You are a clinical information assistant. Answer the following medical "
        "question using your own knowledge. Be concise and factual. If you are "
        "uncertain, say so."
    )
    return system, question