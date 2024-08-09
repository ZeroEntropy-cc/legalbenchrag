import tomllib
from pydantic import BaseModel, SecretStr


class AICredentials(BaseModel):
    openai_api_key: SecretStr
    anthropic_api_key: SecretStr
    cohere_api_key: SecretStr
    voyageai_api_key: SecretStr


class Credentials(BaseModel):
    ai: AICredentials


with open("./credentials/credentials.toml", "rb") as credentials_file:
    credentials = Credentials.model_validate(tomllib.load(credentials_file))
