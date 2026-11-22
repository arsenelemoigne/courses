"""
A drop-in replacement for LiteLLMModel that prints the model's hidden
reasoning to the terminal on every call.

Why this file exists: smolagents 1.13 predates reasoning models. When it
builds its own ChatMessage it keeps only role / content / tool_calls and
discards everything else - including the `reasoning` field Ollama returns.
It does, however, attach the untouched server reply as `.raw`, so the
reasoning is recoverable. We read around the abstraction rather than
editing the library.

Usage:
    from reasoning_model import ReasoningLiteLLMModel
    model = ReasoningLiteLLMModel(model_id="ollama_chat/qwen3:1.7b", ...)
"""
from smolagents import LiteLLMModel

AMBER, DIM, RESET = "\033[33m", "\033[2m", "\033[0m"

# Providers disagree on the name: HuggingFace says reasoning_content,
# Ollama says reasoning, some say thinking.
_FIELDS = ("reasoning_content", "reasoning", "thinking")


def extract_reasoning(chat_message):
    """Dig the hidden reasoning out of a smolagents ChatMessage, or None."""
    for f in _FIELDS:
        v = getattr(chat_message, f, None)
        if v:
            return v
    raw = getattr(chat_message, "raw", None)
    try:
        inner = raw.choices[0].message
        for f in _FIELDS:
            v = getattr(inner, f, None)
            if v:
                return v
        extra = getattr(inner, "model_extra", None) or {}
        for f in _FIELDS:
            if extra.get(f):
                return extra[f]
    except Exception:
        pass
    return None


class ReasoningLiteLLMModel(LiteLLMModel):
    """LiteLLMModel that shows you what you are actually paying for.

    Set .quiet = True when the caller prints the reasoning itself, so the two
    do not interleave (this is what made ledger.py print out of order).
    """
    quiet = False

    def __call__(self, *args, **kwargs):
        msg = super().__call__(*args, **kwargs)
        if self.quiet:
            return msg

        reasoning = extract_reasoning(msg)
        usage = getattr(getattr(msg, "raw", None), "usage", None)
        billed = getattr(usage, "completion_tokens", None)

        print()
        if reasoning:
            head = f"REASONING — {len(reasoning):,} chars"
            if billed:
                head += f"   ({billed:,} completion tokens billed for this reply)"
            print(f"{AMBER}┌─ {head} {'─' * max(0, 74 - len(head))}{RESET}")
            for line in str(reasoning).strip().splitlines():
                print(f"{AMBER}│{RESET} {DIM}{line}{RESET}")
            print(f"{AMBER}└{'─' * 76}{RESET}")
        else:
            print(f"{DIM}(no reasoning channel — this model thinks only in what it writes){RESET}")

        return msg
