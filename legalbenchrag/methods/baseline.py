import os
import sqlite3
import struct
from typing import Literal, cast

import sqlite_vec  # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from tqdm import tqdm

from legalbenchrag.benchmark_types import (
    Document,
    QueryResponse,
    RetrievalMethod,
    RetrievedSnippet,
)
from legalbenchrag.utils.ai import (
    AIEmbeddingModel,
    AIEmbeddingType,
    AIRerankModel,
    ai_embedding,
    ai_rerank,
)


def serialize_f32(vector: list[float]) -> bytes:
    """serializes a list of floats into a compact "raw bytes" format"""
    return struct.pack(f"{len(vector)}f", *vector)


SHOW_LOADING_BAR = True


class ChunkingStrategy(BaseModel):
    strategy_name: Literal["naive", "rcts"]
    chunk_size: int


class RetrievalStrategy(BaseModel):
    chunking_strategy: ChunkingStrategy
    embedding_model: AIEmbeddingModel
    embedding_topk: int
    rerank_model: AIRerankModel | None
    rerank_topk: int
    token_limit: int | None


class EmbeddingInfo(BaseModel):
    document_id: str
    span: tuple[int, int]


class BaselineRetrievalMethod(RetrievalMethod):
    retrieval_strategy: RetrievalStrategy
    documents: dict[str, Document]
    embedding_infos: list[EmbeddingInfo] | None
    sqlite_db: sqlite3.Connection | None
    sqlite_db_file_path: str | None

    def __init__(self, retrieval_strategy: RetrievalStrategy):
        self.retrieval_strategy = retrieval_strategy
        self.documents = {}
        self.embedding_infos = None
        self.sqlite_db = None
        self.sqlite_db_file_path = None

    async def cleanup(self) -> None:
        if self.sqlite_db is not None:
            self.sqlite_db.close()
            self.sqlite_db = None
        if self.sqlite_db_file_path is not None and os.path.exists(
            self.sqlite_db_file_path
        ):
            os.remove(self.sqlite_db_file_path)
            self.sqlite_db_file_path = None

    async def ingest_document(self, document: Document) -> None:
        self.documents[document.file_path] = document

    async def sync_all_documents(self) -> None:
        class Chunk(BaseModel):
            document_id: str
            span: tuple[int, int]
            content: str

        # Calculate chunks
        chunks: list[Chunk] = []
        for document_id, document in self.documents.items():
            # Get chunks
            chunk_size = self.retrieval_strategy.chunking_strategy.chunk_size
            match self.retrieval_strategy.chunking_strategy.strategy_name:
                case "naive":
                    text_splits: list[str] = []
                    for i in range(0, len(document.content), chunk_size):
                        text_splits.append(document.content[i : i + chunk_size])
                case "rcts":
                    synthetic_data_splitter = RecursiveCharacterTextSplitter(
                        separators=[
                            "\n\n",
                            "\n",
                            "!",
                            "?",
                            ".",
                            ":",
                            ";",
                            ",",
                            " ",
                            "",
                        ],
                        chunk_size=chunk_size,
                        chunk_overlap=0,
                        length_function=len,
                        is_separator_regex=False,
                        strip_whitespace=False,
                    )
                    text_splits = synthetic_data_splitter.split_text(document.content)
            assert sum(len(text_split) for text_split in text_splits) == len(
                document.content
            )
            assert "".join(text_splits) == document.content

            # Get spans from chunks
            prev_span: tuple[int, int] | None = None
            for text_split in text_splits:
                prev_index = prev_span[1] if prev_span is not None else 0
                span = (prev_index, prev_index + len(text_split))
                chunks.append(
                    Chunk(
                        document_id=document_id,
                        span=span,
                        content=text_split,
                    )
                )
                prev_span = span

        # Calculate embeddings
        progress_bar: tqdm | None = None
        if SHOW_LOADING_BAR:
            progress_bar = tqdm(
                total=len(chunks), desc="Processing Embeddings", ncols=100
            )

        EMBEDDING_BATCH_SIZE = 2048
        self.embedding_infos = []
        for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            chunk_batch = chunks[i : i + EMBEDDING_BATCH_SIZE]
            assert len(chunk_batch) > 0
            embeddings = await ai_embedding(
                self.retrieval_strategy.embedding_model,
                [chunk.content for chunk in chunk_batch],
                AIEmbeddingType.DOCUMENT,
                callback=lambda: (progress_bar.update(1), None)[1]
                if progress_bar
                else None,
            )
            assert len(chunk_batch) == len(embeddings)
            # Save the Info
            if self.sqlite_db is None:
                # random_id = str(uuid4())
                self.sqlite_db_file_path = "./data/cache/baseline.db"
                if os.path.exists(self.sqlite_db_file_path):
                    os.remove(self.sqlite_db_file_path)
                self.sqlite_db = sqlite3.connect(self.sqlite_db_file_path)
                self.sqlite_db.enable_load_extension(True)
                sqlite_vec.load(self.sqlite_db)
                self.sqlite_db.enable_load_extension(False)
                # Set RAM Usage and create vector table
                self.sqlite_db.execute(f"PRAGMA mmap_size = {3*1024*1024*1024}")
                self.sqlite_db.execute(
                    f"CREATE VIRTUAL TABLE vec_items USING vec0(embedding float[{len(embeddings[0])}])"
                )

            with self.sqlite_db as db:
                insert_data = [
                    (len(self.embedding_infos) + i, serialize_f32(embedding))
                    for i, embedding in enumerate(embeddings)
                ]
                db.executemany(
                    "INSERT INTO vec_items(rowid, embedding) VALUES (?, ?)",
                    insert_data,
                )
                for chunk, embedding in zip(chunk_batch, embeddings):
                    self.embedding_infos.append(
                        EmbeddingInfo(
                            document_id=chunk.document_id,
                            span=chunk.span,
                            embedding=embedding,
                        )
                    )
        if progress_bar:
            progress_bar.close()

    async def query(self, query: str) -> QueryResponse:
        if self.sqlite_db is None or self.embedding_infos is None:
            raise ValueError("Sync documents before querying!")
        # Get TopK Embedding results
        query_embedding = (
            await ai_embedding(
                self.retrieval_strategy.embedding_model, [query], AIEmbeddingType.QUERY
            )
        )[0]
        rows = self.sqlite_db.execute(
            """
            SELECT
                rowid,
                distance
            FROM vec_items
            WHERE embedding MATCH ?
            ORDER BY distance ASC
            LIMIT ?
            """,
            [serialize_f32(query_embedding), self.retrieval_strategy.embedding_topk],
        ).fetchall()
        indices = [cast(int, row[0]) for row in rows]
        retrieved_embedding_infos = [self.embedding_infos[i] for i in indices]

        # Rerank
        if self.retrieval_strategy.rerank_model is not None:
            reranked_indices = await ai_rerank(
                self.retrieval_strategy.rerank_model,
                query,
                texts=[
                    self.get_embedding_info_text(embedding_info)
                    for embedding_info in retrieved_embedding_infos
                ],
            )
            retrieved_embedding_infos = [
                retrieved_embedding_infos[i]
                for i in reranked_indices[: self.retrieval_strategy.rerank_topk]
            ]

        # Get the top retrieval snippets, up until the token limit
        remaining_tokens = self.retrieval_strategy.token_limit
        retrieved_snippets: list[RetrievedSnippet] = []
        for i, embedding_info in enumerate(retrieved_embedding_infos):
            if remaining_tokens is not None and remaining_tokens <= 0:
                break
            span = embedding_info.span
            if remaining_tokens is not None:
                span = (span[0], min(span[1], span[0] + remaining_tokens))
            retrieved_snippets.append(
                RetrievedSnippet(
                    file_path=embedding_info.document_id,
                    span=span,
                    score=1.0 / (i + 1),
                )
            )
            if remaining_tokens is not None:
                remaining_tokens -= span[1] - span[0]
        return QueryResponse(retrieved_snippets=retrieved_snippets)

    def get_embedding_info_text(self, embedding_info: EmbeddingInfo) -> str:
        return self.documents[embedding_info.document_id].content[
            embedding_info.span[0] : embedding_info.span[1]
        ]
