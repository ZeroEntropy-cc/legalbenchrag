import asyncio
from typing import cast

import faiss  # type: ignore
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

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


class RetrievalStrategy(BaseModel):
    name: str
    embedding_model: AIEmbeddingModel
    embedding_topk: int
    rerank_model: AIRerankModel | None
    rerank_topk: int
    token_limit: int


class EmbeddingInfo(BaseModel):
    document_id: str
    span: tuple[int, int]
    embedding: list[float]


class BaselineRetrievalMethod(RetrievalMethod):
    retrieval_strategy: RetrievalStrategy
    documents: dict[str, Document]
    embedding_infos: list[EmbeddingInfo] | None
    faiss_index: faiss.IndexFlatL2 | None

    def __init__(self, retrieval_strategy: RetrievalStrategy):
        self.retrieval_strategy = retrieval_strategy
        self.documents = {}
        self.embedding_infos = None
        self.faiss_index = None

    async def ingest_document(self, document: Document) -> None:
        self.documents[document.file_path] = document

    async def sync_all_documents(self) -> None:
        embedding_infos: list[EmbeddingInfo] = []
        for document_id, document in self.documents.items():
            # Get chunks
            synthetic_data_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", "!", "?", ".", ":", ";", ",", " ", ""],
                chunk_size=500,
                chunk_overlap=0,
                length_function=len,
                is_separator_regex=False,
                strip_whitespace=False,
            )
            chunks = synthetic_data_splitter.split_text(document.content)
            assert sum(len(chunk) for chunk in chunks) == len(document.content)

            # Get spans from chunks
            spans: list[tuple[int, int]] = []
            for chunk in chunks:
                prev_index = spans[-1][1] if len(spans) > 0 else 0
                spans.append((prev_index, prev_index + len(chunk)))

            # Calculate embeddings
            embeddings = await asyncio.gather(
                *[
                    ai_embedding(
                        self.retrieval_strategy.embedding_model,
                        chunk,
                        AIEmbeddingType.DOCUMENT,
                    )
                    for chunk in chunks
                ]
            )

            # Store the embedding infos
            assert len(spans) == len(embeddings)
            for span, embedding in zip(spans, embeddings):
                embedding_infos.append(
                    EmbeddingInfo(
                        document_id=document_id,
                        span=span,
                        embedding=embedding,
                    )
                )

        # Create a FAISS index over all embedding infos
        all_embeddings = [
            embedding_info.embedding for embedding_info in embedding_infos
        ]
        dimension = len(all_embeddings[0])
        self.embedding_infos = embedding_infos
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(np.array(all_embeddings).astype("float32"))

    async def query(self, query: str) -> QueryResponse:
        if self.faiss_index is None or self.embedding_infos is None:
            raise ValueError("Sync documents before querying!")
        # Get TopK Embedding results
        query_embedding = await ai_embedding(
            self.retrieval_strategy.embedding_model, query, AIEmbeddingType.QUERY
        )
        _, faiss_indices = self.faiss_index.search(
            np.array([query_embedding]).astype("float32"),
            self.retrieval_strategy.embedding_topk,
        )
        faiss_indices = cast(list[list[int]], [a.tolist() for a in faiss_indices])
        indices = [index for index in faiss_indices[0] if index >= 0]
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
            if remaining_tokens <= 0:
                break
            span = embedding_info.span
            span = (span[0], min(span[1], span[0] + remaining_tokens))
            retrieved_snippets.append(
                RetrievedSnippet(
                    file_path=embedding_info.document_id,
                    span=span,
                    score=1.0 / (i + 1),
                )
            )
            remaining_tokens -= span[1] - span[0]
        return QueryResponse(retrieved_snippets=retrieved_snippets)

    def get_embedding_info_text(self, embedding_info: EmbeddingInfo) -> str:
        return self.documents[embedding_info.document_id].content[
            embedding_info.span[0] : embedding_info.span[1]
        ]
