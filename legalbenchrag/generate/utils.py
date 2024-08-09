from pydantic import BaseModel

from legalbenchrag.utils.ai import AIMessage, AIModel, ai_call

WRITE_TITLES = False

TITLE_SYSTEM_PROMPT = (
    """
# Instructions

The User will provide to you a very long contract. Your job is to create a reasonable title of it.
Keep the title succinct and clear.
You MUST mention the relevant companies involved in the contract, and what the contract purpose and topic is.
{EXTRA_INSTRUCTIONS}
# Format

Your output format should be JSON, matching the following JSON schema.

{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "thoughts": {
      "type": "array",
      "description": "An array of thoughts or considerations related to the parties involved in the contract and what the contract's purpose is.",
      "items": {
        "type": "string"
      }
    },
    "title": {
      "type": ["string"],
      "description": "A reasonable and succinct title for the contract, including all parties and companies involved, and what the contract's purpose is."
    }
  },
  "required": ["thoughts", "title"],
  "additionalProperties": false
}


Here's an example,
{
  "thoughts": [
    "I read the Licensing Agreement and the parties involved appear to be Company A and Company B."
  ],
  "title": "Licensing Agreement between Company A and Company B",
}

Do NOT output ```json OR ```.
""".strip()
    + "\n"
)

TITLE_USER_PROMPT = (
    """
FILENAME: {FILENAME}
CONTENT:
{CONTENT}
""".strip()
    + "\n"
)


class TitleResponse(BaseModel):
    thoughts: list[str]
    title: str


async def create_title(
    filename: str, text: str, *, extra_instructions: str = ""
) -> str:
    if len(text) > 30000:
        text = text[:10000] + text[-10000:]
    response = await ai_call(
        model=AIModel(company="openai", model="gpt-4o-mini"),
        messages=[
            AIMessage(
                role="system",
                content=TITLE_SYSTEM_PROMPT.replace(
                    "{EXTRA_INSTRUCTIONS}",
                    extra_instructions,
                ),
            ),
            AIMessage(
                role="user",
                content=TITLE_USER_PROMPT.format(
                    FILENAME=filename,
                    CONTENT=text,
                ),
            ),
        ],
    )
    response = response.strip()
    if response.endswith(",\n}"):
        response = response[:-3] + "\n}"
    title_response = TitleResponse.model_validate_json(response)
    return title_response.title
