import logging
import os
import re
from typing import Dict, List, Optional, Union

import pandas as pd
import PyPDF2

from tokenizer.tokenizer import Tokenizer


class Dataset:
    PATTERNS = {
        "article": r"Стаття (\d+)",
        "point": r"(\d+)\.(\d+)(?:\.(\d+))?(?:\.(\d+))?",
        "subpoint": r"[а-я]\)",
    }

    def __init__(self, pdf_paths: Union[str, List[str]]):

        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
        )
        self.logger = logging.getLogger(__name__)

        self.tokenizer = Tokenizer()

        self.pdf_paths = self._normalize_pdf_paths(pdf_paths)

        if not self.pdf_paths:
            raise ValueError("No PDF files found")

    def _normalize_pdf_paths(self, pdf_paths: Union[str, List[str]]) -> List[str]:

        if isinstance(pdf_paths, str):

            if os.path.isdir(pdf_paths):
                paths = [
                    os.path.join(pdf_paths, f)
                    for f in os.listdir(pdf_paths)
                    if f.endswith(".pdf")
                ]
            else:
                paths = [pdf_paths]
        else:
            paths = pdf_paths

        # Filter existing paths
        return [path for path in paths if os.path.exists(path)]

    def _clean_text(self, text: str) -> str:

        cleaned = re.sub(r"\.{3,}", " ", text)
        cleaned = re.sub(r"\s+", " ", text)
        cleaned = cleaned.strip()
        return cleaned

    def _extract_structure_info(self, text: str, page_num: int) -> Dict:

        page_pattern = (
            r'Газета\s+"Все\s+про\s+бухгалтерський\s+облік"\s+(\d+)\s+gazeta\.vobu\.ua'
        )
        page_match = re.search(page_pattern, text)
        page_num = int(page_match.group(1)) if page_match else None

        articles = [
            f"Стаття {match}" for match in re.findall(self.PATTERNS["article"], text)
        ]

        points = []
        for match in re.finditer(self.PATTERNS["point"], text):
            groups = match.groups()

            point_number = ".".join(str(g) for g in groups if g is not None)
            points.append(point_number)

        return {
            "articles": sorted(set(articles)),  # Унікальні статті
            "points": sorted(set(points)),  # Унікальні пункти
            "page": page_num,
        }

    def _extract_text_from_pdf(self, pdf_path: str) -> Dict:

        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                full_text = ""
                page_details = []

                for page_num, page in enumerate(reader.pages, 1):
                    page_text = page.extract_text()
                    cleaned_text = self._clean_text(page_text)
                    full_text += cleaned_text + "\n\n"

                    structure_info = self._extract_structure_info(
                        cleaned_text, page_num
                    )

                    page_details.append(
                        {
                            "page_number": page_num,
                            "text_preview": cleaned_text[:200],
                            "structure": structure_info,
                        }
                    )

                return {
                    "path": pdf_path,
                    "filename": os.path.basename(pdf_path),
                    "text": full_text,
                    "total_pages": len(reader.pages),
                    "page_details": page_details,
                }
        except Exception as e:
            self.logger.error(f"Error processing {pdf_path}: {e}")
            return None

    def prepare_dataset(self, chunk_size: int = 512, overlap: int = 100) -> List[Dict]:

        documents = [
            doc
            for doc in [self._extract_text_from_pdf(path) for path in self.pdf_paths]
            if doc is not None
        ]

        dataset = []

        for doc in documents:
            chunks = self._tokenize_text(
                doc["text"], doc, chunk_size=chunk_size, overlap=overlap
            )
            dataset.extend(chunks)

        return dataset

    def _tokenize_text(
        self, text: str, document_info: Dict, chunk_size: int = 512, overlap: int = 50
    ) -> List[Dict]:

        chunks = self.tokenizer.tokenize_text(
            text, max_chunk_size=chunk_size, overlap=overlap
        )

        processed_chunks = []
        for chunk in chunks:
            structure_info = self._extract_structure_info(
                chunk["text"],
                self._find_page_number(chunk["text"], document_info["page_details"]),
            )

            chunk_data = {
                "text": chunk["text"],
                "source_file": document_info["filename"],
                "length": chunk["length"],
                "structure": structure_info,
                "document_metadata": {
                    "total_pages": document_info["total_pages"],
                    "document_path": document_info["path"],
                },
            }
            processed_chunks.append(chunk_data)

        return processed_chunks

    def _find_page_number(self, chunk_text: str, page_details: List[Dict]) -> int:

        for page in page_details:
            if chunk_text[:100] in page["text_preview"]:
                return page["page_number"]
        return 1

    def save_dataset(
        self,
        dataset: List[Dict],
        output_formats: Union[str, List[str]] = "json",
        base_path: Optional[str] = None,
        filename: str = "dataset",
    ):

        if isinstance(output_formats, str):
            output_formats = [output_formats]

        if base_path is None:
            base_path = os.getcwd()

        os.makedirs(base_path, exist_ok=True)

        df = pd.DataFrame(dataset)

        for format in output_formats:
            filename = f"{filename}.{format}"
            filepath = os.path.join(base_path, filename)

            if format == "json":
                df.to_json(filepath, orient="records", force_ascii=False, indent=2)
            elif format == "csv":
                df.to_csv(filepath, index=False, encoding="utf-8")
            elif format == "parquet":
                df.to_parquet(filepath, index=False, compression="snappy")
            else:
                self.logger.warning(f"Unsupported format: {format}")

            self.logger.info(f"Saved {format.upper()} dataset: {filepath}")

    @classmethod
    def load_dataset(cls, dataset_path: str, file_type: str = "json") -> List[Dict]:

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")

        if file_type == "json":
            return pd.read_json(dataset_path, orient="records").to_dict("records")
        elif file_type == "csv":
            return pd.read_csv(dataset_path).to_dict("records")
        elif file_type == "parquet":
            return pd.read_parquet(dataset_path).to_dict("records")
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
