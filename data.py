from data.dataset import Dataset
from embeddings.embeddings_faiss import EmbeddingsManager

pdf_path = "/app/code/tax_code.pdf"

processor = Dataset(pdf_path)

dataset = processor.prepare_dataset(chunk_size=1000, overlap=200)

processor.save_dataset(
    dataset,
    output_formats=["json"],
    base_path="/app/dataset",
    filename="tax_code1000",
)


embeddings_manager = EmbeddingsManager(persist_directory="/app/db")

vectorstore = embeddings_manager.create_vectorstore(
    dataset_path="/app/dataset/tax_code1000.json",
    file_type="json",
)
