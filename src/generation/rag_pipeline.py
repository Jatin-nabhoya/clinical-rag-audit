"""Phase 4 — end-to-end RAG pipeline: retrieve → prompt → generate."""
from typing import Optional

from .config import DEFAULT_TOP_K
from .llm_wrapper import LLMWrapper
from .prompts import build_no_rag_prompt, build_rag_prompt


class RAGPipeline:
    """
    Orchestrates retrieval + generation for a single model.

    Usage:
        from src.retrieval.retriever import Retriever
        from src.generation import LLMWrapper, RAGPipeline

        retriever = Retriever("data/vector_store/medical")
        llm       = LLMWrapper("mistral")
        rag       = RAGPipeline(llm, retriever)

        result = rag.answer("What are the symptoms of Type 2 diabetes?")
        # result["answer"] → the LLM's response
        # result["retrieved_chunks"] → chunks fed as context
    """

    def __init__(self, llm: LLMWrapper, retriever, k: int = DEFAULT_TOP_K):
        self.llm       = llm
        self.retriever = retriever
        self.k         = k

    def answer(
        self,
        question: str,
        k: Optional[int] = None,
        use_rag: bool = True,
    ) -> dict:
        """
        Answer a question, optionally with RAG retrieval.

        Returns a dict with everything needed for RAGAS evaluation:
          model, question, use_rag, k, retrieved_chunks, context, answer
        """
        k = k or self.k

        if use_rag:
            chunks  = self.retriever.retrieve(question, k=k)
            context = self.retriever.format_context(chunks)
            system_prompt, user_prompt = build_rag_prompt(question, context)
        else:
            chunks  = []
            context = ""
            system_prompt, user_prompt = build_no_rag_prompt(question)

        answer = self.llm.generate(system_prompt, user_prompt)

        return {
            "model":             self.llm.model_key,
            "question":          question,
            "use_rag":           use_rag,
            "k":                 k if use_rag else 0,
            "retrieved_chunks":  chunks,
            "context":           context,
            "answer":            answer,
        }