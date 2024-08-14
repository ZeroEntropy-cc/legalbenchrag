import asyncio
import hashlib
import logging
import os
from collections.abc import Callable, Coroutine
from enum import Enum
from typing import Any, Literal, cast

import anthropic
import cohere
import diskcache as dc  # type: ignore
import httpx
import openai
import tiktoken
import voyageai  # type: ignore
import voyageai.error  # type: ignore
from anthropic import NOT_GIVEN, Anthropic, AsyncAnthropic, NotGiven
from anthropic.types import MessageParam
from openai import AsyncOpenAI, RateLimitError
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, computed_field

from legalbenchrag.utils.credentials import credentials

logger = logging.getLogger("uvicorn")

# AI Types


class AIModel(BaseModel):
    company: Literal["openai", "anthropic"]
    model: str

    @computed_field  # type: ignore[misc]
    @property
    def ratelimit_tpm(self) -> float:
        match self.company:
            case "openai":
                # Tier 5
                match self.model:
                    case "gpt-4o-mini":
                        return 150000000
                    case "gpt-4o":
                        return 30000000
                    case m if m.startswith("gpt-4-turbo"):
                        return 2000000
                    case _:
                        return 1000000
            case "anthropic":
                # Tier 4
                return 400000


class AIMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class AIEmbeddingModel(BaseModel):
    company: Literal["openai", "cohere", "voyageai"]
    model: str

    @computed_field  # type: ignore[misc]
    @property
    def ratelimit_tpm(self) -> float:
        match self.company:
            case "openai":
                return 1000000
            case "cohere":
                # 96 texts per embed
                return 10000 * 96
            case "voyageai":
                # It says 300RPM but I can only get 30 out of it
                return 1000000

    @computed_field  # type: ignore[misc]
    @property
    def ratelimit_rpm(self) -> float:
        match self.company:
            case "openai":
                return 5000
            case "cohere":
                return 10000
            case "voyageai":
                # It says 300RPM but I can only get 30 out of it
                return 30

    @computed_field  # type: ignore[misc]
    @property
    def max_batch_len(self) -> int:
        match self.company:
            case "openai":
                return 2048
            case "cohere":
                return 96
            case "voyageai":
                return 128


class AIEmbeddingType(Enum):
    DOCUMENT = 1
    QUERY = 2


class AIRerankModel(BaseModel):
    company: Literal["cohere", "voyageai"]
    model: str

    @computed_field  # type: ignore[misc]
    @property
    def ratelimit_rpm(self) -> float:
        match self.company:
            case "cohere":
                return 10000
            case "voyageai":
                # It says 100RPM but I can only get 60 out of it
                return 60


# Cache
os.makedirs("./data/cache", exist_ok=True)
cache = dc.Cache("./data/cache/ai_cache.db")

RATE_LIMIT_RATIO = 0.95


class AIConnection:
    openai_client: AsyncOpenAI
    voyageai_client: voyageai.AsyncClient
    cohere_client: cohere.AsyncClient
    anthropic_client: AsyncAnthropic
    sync_anthropic_client: Anthropic
    # Share one global Semaphore across all threads
    cohere_ratelimit_semaphore = asyncio.Semaphore(1)
    voyageai_ratelimit_semaphore = asyncio.Semaphore(1)
    openai_ratelimit_semaphore = asyncio.Semaphore(1)
    anthropic_ratelimit_semaphore = asyncio.Semaphore(1)

    def __init__(self) -> None:
        self.openai_client = AsyncOpenAI(
            api_key=credentials.ai.openai_api_key.get_secret_value()
        )
        self.anthropic_client = AsyncAnthropic(
            api_key=credentials.ai.anthropic_api_key.get_secret_value()
        )
        self.sync_anthropic_client = Anthropic(
            api_key=credentials.ai.anthropic_api_key.get_secret_value()
        )
        self.voyageai_client = voyageai.AsyncClient(
            api_key=credentials.ai.voyageai_api_key.get_secret_value()
        )
        self.cohere_client = cohere.AsyncClient(
            api_key=credentials.ai.cohere_api_key.get_secret_value()
        )


# NOTE: API Clients cannot be called from multiple event loops,
# So every asyncio event loop needs its own API connection
ai_connections: dict[asyncio.AbstractEventLoop, AIConnection] = {}


def get_ai_connection() -> AIConnection:
    event_loop = asyncio.get_event_loop()
    if event_loop not in ai_connections:
        ai_connections[event_loop] = AIConnection()
    return ai_connections[event_loop]


class AIError(Exception):
    """A class for AI Task Errors"""


class AIValueError(AIError, ValueError):
    """A class for AI Value Errors"""


class AITimeoutError(AIError, TimeoutError):
    """A class for AI Task Timeout Errors"""


def ai_num_tokens(model: AIModel | AIEmbeddingModel | AIRerankModel, s: str) -> int:
    if isinstance(model, AIModel):
        if model.company == "anthropic":
            # Doesn't actually connect to the network
            return get_ai_connection().sync_anthropic_client.count_tokens(s)
        elif model.company == "openai":
            encoding = tiktoken.encoding_for_model(model.model)
            num_tokens = len(encoding.encode(s))
            return num_tokens
    if isinstance(model, AIEmbeddingModel):
        if model.company == "openai":
            encoding = tiktoken.encoding_for_model(model.model)
            num_tokens = len(encoding.encode(s))
            return num_tokens
        elif model.company == "voyageai":
            return get_ai_connection().voyageai_client.count_tokens([s], model.model)
    # Otherwise, estimate
    logger.warning("Estimating Tokens!")
    return int(len(s) / 4)


def get_call_cache_key(
    model: AIModel,
    messages: list[AIMessage],
) -> str:
    # Hash the array of texts
    md5_hasher = hashlib.md5()
    md5_hasher.update(model.model_dump_json().encode())
    for message in messages:
        md5_hasher.update(md5_hasher.hexdigest().encode())
        md5_hasher.update(message.model_dump_json().encode())
    key = md5_hasher.hexdigest()

    return key


async def ai_call(
    model: AIModel,
    messages: list[AIMessage],
    *,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    # When using anthropic, the first message must be from the user.
    # If the first message is not a User, this message will be prepended to the messages.
    anthropic_initial_message: str | None = "<START>",
    # If two messages of the same role are given to anthropic, they must be concatenated.
    # This is the delimiter between concatenated.
    anthropic_combine_delimiter: str = "\n",
    # Throw an AITimeoutError after this many retries fail
    num_ratelimit_retries: int = 10,
    # Backoff function (Receives index of attempt)
    backoff_algo: Callable[[int], float] = lambda i: min(2**i, 5),
) -> str:
    cache_key = get_call_cache_key(model, messages)
    cached_call = cache.get(cache_key)

    if cached_call is not None:
        return cached_call

    num_tokens_input: int = sum(
        [ai_num_tokens(model, message.content) for message in messages]
    )

    return_value: str | None = None
    match model.company:
        case "openai":
            for i in range(num_ratelimit_retries):
                try:
                    # Guard with ratelimit
                    async with get_ai_connection().openai_ratelimit_semaphore:
                        tpm = model.ratelimit_tpm * RATE_LIMIT_RATIO
                        expected_wait = num_tokens_input / (tpm / 60)
                        await asyncio.sleep(expected_wait)

                    def ai_message_to_openai_message_param(
                        message: AIMessage,
                    ) -> ChatCompletionMessageParam:
                        if message.role == "system":  # noqa: SIM114
                            return {"role": message.role, "content": message.content}
                        elif message.role == "user":  # noqa: SIM114
                            return {"role": message.role, "content": message.content}
                        elif message.role == "assistant":
                            return {"role": message.role, "content": message.content}

                    if i > 0:
                        logger.debug("Trying again after RateLimitError...")
                    response = (
                        await get_ai_connection().openai_client.chat.completions.create(
                            model=model.model,
                            messages=[
                                ai_message_to_openai_message_param(message)
                                for message in messages
                            ],
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                    )
                    assert response.choices[0].message.content is not None
                    return_value = response.choices[0].message.content
                    break
                except RateLimitError:
                    logger.warning("OpenAI RateLimitError")
                    async with get_ai_connection().openai_ratelimit_semaphore:
                        await asyncio.sleep(backoff_algo(i))
            if return_value is None:
                raise AITimeoutError("Cannot overcome OpenAI RateLimitError")

        case "anthropic":
            for i in range(num_ratelimit_retries):
                try:
                    # Guard with ratelimit
                    async with get_ai_connection().anthropic_ratelimit_semaphore:
                        tpm = model.ratelimit_tpm * RATE_LIMIT_RATIO
                        expected_wait = num_tokens_input / (tpm / 60)
                        await asyncio.sleep(expected_wait)

                    def ai_message_to_anthropic_message_param(
                        message: AIMessage,
                    ) -> MessageParam:
                        if message.role == "user" or message.role == "assistant":
                            return {"role": message.role, "content": message.content}
                        elif message.role == "system":
                            raise AIValueError(
                                "system not allowed in anthropic message param"
                            )

                    if i > 0:
                        logger.debug("Trying again after RateLimitError...")

                    # Extract system message if it exists
                    system: str | NotGiven = NOT_GIVEN
                    if len(messages) > 0 and messages[0].role == "system":
                        system = messages[0].content
                        messages = messages[1:]
                    # Insert initial message if necessary
                    if (
                        anthropic_initial_message is not None
                        and len(messages) > 0
                        and messages[0].role != "user"
                    ):
                        messages = [
                            AIMessage(role="user", content=anthropic_initial_message)
                        ] + messages
                    # Combined messages (By combining consecutive messages of the same role)
                    combined_messages: list[AIMessage] = []
                    for message in messages:
                        if (
                            len(combined_messages) == 0
                            or combined_messages[-1].role != message.role
                        ):
                            combined_messages.append(message)
                        else:
                            # Copy before edit
                            combined_messages[-1] = combined_messages[-1].model_copy(
                                deep=True
                            )
                            # Merge consecutive messages with the same role
                            combined_messages[-1].content += (
                                anthropic_combine_delimiter + message.content
                            )
                    # Get the response
                    response_message = (
                        await get_ai_connection().anthropic_client.messages.create(
                            model=model.model,
                            system=system,
                            messages=[
                                ai_message_to_anthropic_message_param(message)
                                for message in combined_messages
                            ],
                            temperature=0.0,
                            max_tokens=max_tokens,
                        )
                    )
                    assert isinstance(
                        response_message.content[0], anthropic.types.TextBlock
                    )
                    assert isinstance(response_message.content[0].text, str)
                    return_value = response_message.content[0].text
                    break
                except anthropic.RateLimitError as e:
                    logger.warning(f"Anthropic Error: {repr(e)}")
                    async with get_ai_connection().anthropic_ratelimit_semaphore:
                        await asyncio.sleep(backoff_algo(i))
            if return_value is None:
                raise AITimeoutError("Cannot overcome Anthropic RateLimitError")

    cache.set(cache_key, return_value)
    return return_value


def get_embeddings_cache_key(
    model: AIEmbeddingModel, text: str, embedding_type: AIEmbeddingType
) -> str:
    key = f"{model.company}||||{model.model}||||{embedding_type.name}||||{hashlib.md5(text.encode()).hexdigest()}"
    return key


async def ai_embedding(
    model: AIEmbeddingModel,
    texts: list[str],
    embedding_type: AIEmbeddingType,
    *,
    # Throw an AITimeoutError after this many retries fail
    num_ratelimit_retries: int = 10,
    # Backoff function (Receives index of attempt)
    backoff_algo: Callable[[int], float] = lambda i: min(2**i, 5),
    # Callback (For tracking progress)
    callback: Callable[[], None] = lambda: None,
) -> list[list[float]]:
    # Extract cache miss indices
    text_embeddings: list[list[float] | None] = [None] * len(texts)
    for i, text in enumerate(texts):
        cache_key = get_embeddings_cache_key(model, text, embedding_type)
        text_embeddings[i] = cache.get(cache_key)
        if text_embeddings[i] is not None:
            callback()
    if not any(embedding is None for embedding in text_embeddings):
        return cast(list[list[float]], text_embeddings)
    required_text_embeddings_indices = [
        i for i in range(len(text_embeddings)) if text_embeddings[i] is None
    ]

    # Recursively Batch if necessary
    if len(required_text_embeddings_indices) > model.max_batch_len:
        # Calculate embeddings in batches
        tasks: list[Coroutine[Any, Any, list[list[float]]]] = []
        for i in range(0, len(required_text_embeddings_indices), model.max_batch_len):
            batch_indices = required_text_embeddings_indices[
                i : i + model.max_batch_len
            ]
            tasks.append(
                ai_embedding(
                    model,
                    [texts[i] for i in batch_indices],
                    embedding_type,
                    num_ratelimit_retries=num_ratelimit_retries,
                    backoff_algo=backoff_algo,
                    callback=callback,
                )
            )
        preflattened_results = await asyncio.gather(*tasks)
        results: list[list[float]] = []
        for embeddings_list in preflattened_results:
            results.extend(embeddings_list)
        # Merge with cache hits
        assert len(required_text_embeddings_indices) == len(results)
        for i, embedding in zip(required_text_embeddings_indices, results):
            text_embeddings[i] = embedding
        assert all(embedding is not None for embedding in text_embeddings)
        return cast(list[list[float]], text_embeddings)

    num_tokens_input: int = sum(
        [
            ai_num_tokens(model, texts[index])
            for index in required_text_embeddings_indices
        ]
    )

    input_texts = [texts[i] for i in required_text_embeddings_indices]
    text_embeddings_response: list[list[float]] | None = None
    match model.company:
        case "openai":
            for i in range(num_ratelimit_retries):
                try:
                    async with get_ai_connection().openai_ratelimit_semaphore:
                        rpm = model.ratelimit_rpm * RATE_LIMIT_RATIO
                        tpm = model.ratelimit_tpm * RATE_LIMIT_RATIO
                        expected_wait = max(60.0 / rpm, num_tokens_input / (tpm / 60))
                        await asyncio.sleep(expected_wait)
                    response = (
                        await get_ai_connection().openai_client.embeddings.create(
                            input=input_texts,
                            model=model.model,
                        )
                    )
                    text_embeddings_response = [
                        embedding.embedding for embedding in response.data
                    ]
                    break
                except (
                    openai.RateLimitError,
                    openai.APIConnectionError,
                    openai.APITimeoutError,
                ):
                    logger.warning("OpenAI RateLimitError")
                    async with get_ai_connection().openai_ratelimit_semaphore:
                        await asyncio.sleep(backoff_algo(i))
            if text_embeddings_response is None:
                raise AITimeoutError("Cannot overcome OpenAI RateLimitError")
        case "cohere":
            for i in range(num_ratelimit_retries):
                try:
                    async with get_ai_connection().cohere_ratelimit_semaphore:
                        rpm = model.ratelimit_rpm * RATE_LIMIT_RATIO
                        tpm = model.ratelimit_tpm * RATE_LIMIT_RATIO
                        expected_wait = max(60.0 / rpm, num_tokens_input / (tpm / 60))
                        await asyncio.sleep(expected_wait)
                    result = await get_ai_connection().cohere_client.embed(
                        texts=input_texts,
                        model=model.model,
                        input_type=(
                            "search_document"
                            if embedding_type == AIEmbeddingType.DOCUMENT
                            else "search_query"
                        ),
                    )
                    assert isinstance(result.embeddings, list)
                    text_embeddings_response = result.embeddings
                    break
                except voyageai.error.RateLimitError:
                    logger.warning("Cohere RateLimitError")
                    async with get_ai_connection().cohere_ratelimit_semaphore:
                        await asyncio.sleep(backoff_algo(i))
            if text_embeddings_response is None:
                raise AITimeoutError("Cannot overcome Cohere RateLimitError")
        case "voyageai":
            for i in range(num_ratelimit_retries):
                try:
                    async with get_ai_connection().voyageai_ratelimit_semaphore:
                        rpm = model.ratelimit_rpm * RATE_LIMIT_RATIO
                        tpm = model.ratelimit_tpm * RATE_LIMIT_RATIO
                        expected_wait = max(60.0 / rpm, num_tokens_input / (tpm / 60))
                        await asyncio.sleep(expected_wait)
                    result = await get_ai_connection().voyageai_client.embed(
                        input_texts,
                        model=model.model,
                        input_type=(
                            "document"
                            if embedding_type == AIEmbeddingType.DOCUMENT
                            else "query"
                        ),
                    )
                    assert isinstance(result.embeddings, list)
                    text_embeddings_response = result.embeddings
                    break
                except voyageai.error.RateLimitError:
                    logger.warning("VoyageAI RateLimitError")
                    async with get_ai_connection().voyageai_ratelimit_semaphore:
                        await asyncio.sleep(backoff_algo(i))
            if text_embeddings_response is None:
                raise AITimeoutError("Cannot overcome VoyageAI RateLimitError")

    assert len(text_embeddings_response) == len(required_text_embeddings_indices)
    for index, embedding in zip(
        required_text_embeddings_indices, text_embeddings_response
    ):
        cache_key = get_embeddings_cache_key(model, texts[index], embedding_type)
        cache.set(cache_key, embedding)
        text_embeddings[index] = embedding
        callback()
    assert all(embedding is not None for embedding in text_embeddings)
    return cast(list[list[float]], text_embeddings)


def get_rerank_cache_key(
    model: AIRerankModel, query: str, texts: list[str], top_k: int | None
) -> str:
    # Hash the array of texts
    md5_hasher = hashlib.md5()
    md5_hasher.update(query.encode())
    for text in texts:
        md5_hasher.update(md5_hasher.hexdigest().encode())
        md5_hasher.update(text.encode())
    texts_hash = md5_hasher.hexdigest()

    key = f"{model.company}||||{model.model}||||{top_k}||||{texts_hash}"
    return key


# Gets the list of indices that reranks the original texts
async def ai_rerank(
    model: AIRerankModel,
    query: str,
    texts: list[str],
    *,
    top_k: int | None = None,
    # Throw an AITimeoutError after this many retries fail
    num_ratelimit_retries: int = 10,
    # Backoff function (Receives index of attempt)
    backoff_algo: Callable[[int], float] = lambda i: min(2**i, 5),
) -> list[int]:
    cache_key = get_rerank_cache_key(model, query, texts, top_k)
    cached_reranking = cache.get(cache_key)

    if cached_reranking is not None:
        return cached_reranking

    indices: list[int] | None = None
    match model.company:
        case "cohere":
            for i in range(num_ratelimit_retries):
                try:
                    async with get_ai_connection().cohere_ratelimit_semaphore:
                        rpm = model.ratelimit_rpm * RATE_LIMIT_RATIO
                        await asyncio.sleep(60.0 / rpm)
                    response = await get_ai_connection().cohere_client.rerank(
                        model=model.model,
                        query=query,
                        documents=texts,
                        top_n=top_k,
                    )
                    indices = [result.index for result in response.results]
                    break
                except (
                    cohere.errors.TooManyRequestsError,
                    httpx.ConnectError,
                    httpx.RemoteProtocolError,
                ):
                    logger.warning("Cohere RateLimitError")
                    async with get_ai_connection().cohere_ratelimit_semaphore:
                        await asyncio.sleep(backoff_algo(i))
            if indices is None:
                raise AITimeoutError("Cannot overcome Cohere RateLimitError")
        case "voyageai":
            for i in range(num_ratelimit_retries):
                try:
                    async with get_ai_connection().voyageai_ratelimit_semaphore:
                        rpm = model.ratelimit_rpm * RATE_LIMIT_RATIO
                        await asyncio.sleep(60.0 / rpm)
                    voyageai_response = (
                        await get_ai_connection().voyageai_client.rerank(
                            query=query,
                            documents=texts,
                            model=model.model,
                            top_k=top_k,
                        )
                    )
                    indices = [
                        int(result.index) for result in voyageai_response.results
                    ]
                    break
                except voyageai.error.RateLimitError:
                    logger.warning("VoyageAI RateLimitError")
                    async with get_ai_connection().voyageai_ratelimit_semaphore:
                        await asyncio.sleep(backoff_algo(i))
            if indices is None:
                raise AITimeoutError("Cannot overcome VoyageAI RateLimitError")
    cache.set(cache_key, indices)
    return indices
