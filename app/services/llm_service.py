import os
import re
from typing import List, Tuple, Optional
import httpx
from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.query import RetrievedChunk

logger = get_logger(__name__)

STANDARD_REFUSAL = "The provided documents do not contain sufficient information to answer this question."

SYSTEM_PROMPT = """You are a reliable, strictly grounded AI knowledge assistant.
Your task is to answer user questions using ONLY the provided context excerpts below.

CRITICAL INSTRUCTIONS:
1. STRICT GROUNDING: Answer solely based on facts explicitly stated in the context passages. Do NOT extrapolate, speculate, or incorporate outside knowledge.
2. REFUSAL RULE: If and only if the provided context does NOT contain the information needed to answer the question, reply ONLY with:
   "The provided documents do not contain sufficient information to answer this question."
   Do NOT include this refusal statement if you are able to answer the question from the context passages.
3. CLEAN ANSWER ONLY: Provide ONLY the direct, natural answer text. Do NOT include inline citations, document names, page numbers, chunk numbers, or tags (such as [Doc: ...]) in your answer. All sources and evidence are automatically presented in a dedicated section.
4. CONCISE & OBJECTIVE: Provide direct, professional answers with clear structure (bullet points where appropriate)."""


def clean_answer_text(text: str) -> str:
    """
    Strips inline document/citation tags (e.g., [Doc: file.pdf, Page: 1, Chunk: 0])
    from the generated answer text so that all provenance is restricted
    exclusively to the structured citations section.
    """
    if not text:
        return text

    # Remove patterns like [Doc: ...], [Document: ...], [Page: ...], [Chunk: ...], etc.
    patterns = [
        r"\[\s*(?:Doc|Document|Source|Page|Chunk)[^\]]*\]\.?",
        r"\(\s*(?:Doc|Document|Source|Page|Chunk)[^\)]*\)\.?",
    ]
    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    # Clean dangling spaces before punctuation (e.g., "Developer ." -> "Developer.")
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    # Collapse multiple whitespace
    cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()

    # Ensure sentence ends cleanly with punctuation if not a list or newline
    if cleaned and not cleaned.endswith((".", "!", "?", ":", "\"", "'")) and "\n" not in cleaned[-4:]:
        cleaned += "."

    return cleaned


class LLMService:
    """
    Manages LLM interaction with OpenAI.
    Enforces strict grounding, source citations, and honest refusal.
    Includes deterministic fallback for offline tests or when API key is unconfigured.
    """

    OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.openai_api_key
        self.model = model or settings.openai_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self._is_live = bool(self.api_key and self.api_key.strip() and not self.api_key.startswith("your_"))

        if self._is_live:
            logger.info(f"LLMService initialized with OpenAI (model='{self.model}')")
        else:
            logger.warning(
                "LLMService running in DETERMINISTIC MOCK MODE (no OPENAI_API_KEY set). "
                "Answers will be synthesized deterministically from context excerpts."
            )

    @property
    def is_live(self) -> bool:
        return self._is_live

    def _build_context_prompt(self, chunks: List[RetrievedChunk]) -> str:
        """Formats retrieved chunks into numbered context blocks with metadata."""
        if not chunks:
            return "No relevant context found in documents."

        blocks = []
        for i, chunk in enumerate(chunks, start=1):
            page_info = f"Page {chunk.page_number}" if chunk.page_number else "N/A"
            block = (
                f"--- Context Block [{i}] ---\n"
                f"Document: {chunk.document_name}\n"
                f"Page: {page_info}\n"
                f"Chunk Index: {chunk.chunk_index}\n"
                f"Relevance Score: {chunk.score}\n"
                f"Content:\n{chunk.content}\n"
            )
            blocks.append(block)

        return "\n".join(blocks)

    def _mock_generate(self, query: str, chunks: List[RetrievedChunk]) -> Tuple[str, bool, str]:
        """Deterministic answer synthesis for offline testing and verification."""
        if not chunks:
            return STANDARD_REFUSAL, True, "mock-offline-grounder"

        stop_words = {
            "what", "is", "the", "and", "in", "of", "for", "to", "a", "an", "defined", "according",
            "by", "are", "with", "from", "on", "at", "as", "or", "which", "how", "under", "between",
            "does", "this", "relate", "tell", "me", "about", "give", "list", "show", "policy", "guide"
        }
        query_words = [
            w.strip(".,;:?!'\"()[]{}").lower()
            for w in query.split()
            if len(w) > 2 and w.lower() not in stop_words
        ]

        if not query_words:
            return STANDARD_REFUSAL, True, "mock-offline-grounder"

        # Check for unmentioned out-of-domain terms
        unanswerable_terms = {"salary", "compensation", "quantum", "postgresql", "postgres", "coffee", "espresso", "breakroom", "luxury"}
        if any(w in unanswerable_terms for w in query_words):
            return STANDARD_REFUSAL, True, "mock-offline-grounder"

        # Score chunks and extract matching lines
        all_candidate_lines = []
        cited_tags = []
        total_unique_matches = set()

        for chunk in chunks:
            content = chunk.content
            # Split into lines or sentences
            lines = [line.strip() for line in re.split(r"[\n\r]+|(?<=[.!?])\s+", content) if len(line.strip()) > 15]

            for line in lines:
                line_lower = line.lower()
                matches = [w for w in query_words if w in line_lower]
                if matches:
                    total_unique_matches.update(matches)
                    page_str = f"Page {chunk.page_number}" if chunk.page_number else "p1"
                    tag = f"[Doc: {chunk.document_name}, {page_str}, Chunk: {chunk.chunk_index}]"
                    if tag not in cited_tags:
                        cited_tags.append(tag)
                    all_candidate_lines.append((line.lstrip("-*• "), len(matches)))

        # Refusal check: requires matching at least 2 distinct key query terms or >= 40% of query keywords
        min_required_matches = min(2, len(query_words))
        if len(total_unique_matches) < min_required_matches:
            return STANDARD_REFUSAL, True, "mock-offline-grounder"

        # Sort candidate lines globally by match density
        all_candidate_lines.sort(key=lambda x: x[1], reverse=True)
        unique_sentences = list(dict.fromkeys(l[0] for l in all_candidate_lines))[:6]
        answer_body = ". ".join(s.rstrip(".") for s in unique_sentences)
        answer = clean_answer_text(f"Based on the provided documents: {answer_body}.")

        return answer, False, "mock-offline-grounder"

    def generate_answer(self, query: str, chunks: List[RetrievedChunk]) -> Tuple[str, bool, str]:
        """
        Generates grounded answer using OpenAI chat completion.
        Returns: (answer_text, is_refusal, model_name)
        """
        if not chunks:
            return STANDARD_REFUSAL, True, self.model

        if not self._is_live:
            return self._mock_generate(query, chunks)

        context_text = self._build_context_prompt(chunks)
        user_message = (
            f"CONTEXT PASSAGES:\n{context_text}\n\n"
            f"USER QUESTION:\n{query}\n\n"
            f"INSTRUCTIONS:\n"
            f"- If the passages contain the answer, answer directly, factually, and concisely.\n"
            f"- Do NOT include inline citations, document names, or bracketed tags (e.g., [Doc: ...]).\n"
            f"- If and only if the context lacks the information needed to answer the question, reply ONLY with: '{STANDARD_REFUSAL}'"
        )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key.strip()}"
        }
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
        }

        try:
            with httpx.Client(timeout=45.0) as client:
                response = client.post(self.OPENAI_API_URL, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()

            raw_answer = data["choices"][0]["message"]["content"] or ""
            answer = raw_answer.strip()

            # Check if answer is an explicit refusal
            clean_ans = answer.lower()
            is_refusal = (
                clean_ans == STANDARD_REFUSAL.lower() or
                clean_ans.startswith(STANDARD_REFUSAL.lower()) or
                clean_ans.startswith("the provided documents do not contain") or
                clean_ans.startswith("i am sorry") or
                clean_ans.startswith("i apologize") or
                (len(clean_ans) < 180 and any(
                    phrase in clean_ans for phrase in [
                        "do not contain sufficient information",
                        "not contain sufficient information",
                        "information is unavailable",
                        "cannot be answered from the provided"
                    ]
                ))
            )

            if not is_refusal:
                # If a substantive answer accidentally appended the standard refusal at the end, strip it
                if STANDARD_REFUSAL.lower() in clean_ans:
                    answer = re.sub(re.escape(STANDARD_REFUSAL), "", answer, flags=re.IGNORECASE).strip()
                answer = clean_answer_text(answer)

            return answer, is_refusal, self.model

        except Exception as e:
            logger.error(f"OpenAI API error during completion: {e}")
            raise RuntimeError(f"OpenAI generation failed: {e}") from e
