"""Pull the model's JSON object out of its raw text. Dependency-free on purpose:
the serving container imports this without scikit-learn or torch."""


def extract_json(text: str) -> str:
    """First balanced {...} block — models love fences and preambles; being
    unable to find ANY object still counts as a parse failure downstream."""
    start = text.find("{")
    if start < 0:
        return text.strip()
    depth = 0
    for i, ch in enumerate(text[start:], start):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            return text[start : i + 1]
    return text[start:].strip()
