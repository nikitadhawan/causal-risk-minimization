"""Prompt format for HuggingFace causal LM models.

Each dataset can define a PromptFormat subclass to express task semantics
in natural language, helping the LLM leverage its pre-trained priors.
"""


class PromptFormat:
    """Generic dataset-agnostic prompt format (fallback)."""

    pos_token: str = " Yes"
    neg_token: str = " No"

    def x_to_text(self, x_val: int) -> str:
        """Convert integer X value to its natural-language representation."""
        raise NotImplementedError

    def outcome_seq(self, x_text: str, t_text: str, y: int = None) -> str:
        """Sequence for outcome model training/inference.

        With y=None returns the prompt prefix (inference);
        with y provided appends the outcome token (training).
        """
        s = f"{x_text}\n{t_text}\nOutcome:"
        if y is not None:
            s += self.pos_token if int(y) == 1 else self.neg_token
        return s

    def propensity_prefix(self, x_text: str) -> str:
        """The X-only prefix for a propensity sequence.

        The token length of this string is used to mask X tokens in the
        training labels so the LM only learns p(T | X).
        Must satisfy: propensity_seq(x, t) == propensity_prefix(x) + t_text.
        """
        return f"{x_text}\n"

    def propensity_seq(self, x_text: str, t_text: str) -> str:
        """Full X+T sequence for propensity training and log-prob scoring."""
        return self.propensity_prefix(x_text) + t_text

    def apo_seq(self, t_text: str, y: int = None) -> str:
        """Sequence for APO model training/inference."""
        s = f"{t_text}\nOutcome:"
        if y is not None:
            s += self.pos_token if int(y) == 1 else self.neg_token
        return s
