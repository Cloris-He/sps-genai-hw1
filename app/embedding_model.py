import spacy


class EmbeddingModel:
    """Generate word embeddings with spaCy's en_core_web_lg model."""

    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_lg")

    def get_embedding(self, word: str) -> dict:
        cleaned_word = word.strip()

        if not cleaned_word:
            raise ValueError("The query word cannot be empty.")

        doc = self.nlp(cleaned_word)

        if len(doc) != 1:
            raise ValueError("Please provide exactly one word.")

        token = doc[0]

        return {
            "word": cleaned_word,
            "dimension": len(token.vector),
            "has_vector": token.has_vector,
            "embedding": token.vector.tolist(),
        }