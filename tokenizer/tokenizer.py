import re
from typing import Dict, List


class Tokenizer:
    def __init__(self):

        self.sentence_end = r"[.!?]+"
        self.abbreviations = r"(?<=[а-яА-ЯіІїЇєЄ])\."

    def _split_into_sentences(self, text: str) -> List[str]:

        text = re.sub(r"\n+", " ", text)

        text = re.sub(self.sentence_end + r"\s+", "\n", text)

        sentences = [s.strip() for s in text.split("\n") if s.strip()]

        return sentences

    def _split_into_words(self, text: str) -> List[str]:

        words = re.findall(r"\b\w+\b", text)
        return words

    def tokenize_text(
        self, text: str, max_chunk_size: int = 1200, overlap: int = 200
    ) -> List[Dict]:

        sentences = self._split_into_sentences(text)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            words = self._split_into_words(sentence)
            sentence_length = len(words)

            if current_length + sentence_length > max_chunk_size and current_chunk:

                chunk_text = " ".join(current_chunk)
                chunks.append({"text": chunk_text, "length": current_length})

                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    s_words = self._split_into_words(s)
                    if overlap_length + len(s_words) > overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_length += len(s_words)

                current_chunk = overlap_sentences
                current_length = overlap_length

            current_chunk.append(sentence)
            current_length += sentence_length

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({"text": chunk_text, "length": current_length})

        return chunks

    def extract_metadata(self, chunk: str) -> Dict:

        article_match = re.search(
            r"Стаття\s+(\d+(?:\.\d+)?)\s*\.?\s*([^0-9\n]+)", chunk
        )
        section_match = re.search(r"Розділ\s+[IVХ]+\.*\s*[^\.]+", chunk)
        head_match = re.search(r"Глава\s+\d+[\.\-]?\d*\.*\s*[^\.]+", chunk)

        metadata = {
            "article": article_match.group(0) if article_match else None,
            "section": section_match.group(0) if section_match else None,
            "head": head_match.group(0) if head_match else None,
        }

        point_match = re.search(r"\b(\d+\.\d+)\b(?!\.\d+)", chunk)
        if point_match:
            metadata["point"] = point_match.group(1)

        subpoint_match = re.search(r"\b(\d+\.\d+\.\d+)\b", chunk)
        if subpoint_match:
            metadata["subpoint"] = subpoint_match.group(1)

        return metadata
