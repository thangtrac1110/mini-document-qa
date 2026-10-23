"""Generate an answer from a question and labeled sources, and parse it into a typed model."""

from pydantic import BaseModel, ConfigDict, ValidationError

from app.errors import MalformedLLMResponseError
from app.generation.llm_client import LLMClient
from app.generation.prompts import ANSWER_SCHEMA, SYSTEM_PROMPT, ContextSource, build_user_message


class GeneratedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    supporting_source_ids: list[str]
    insufficient_information: bool


class AnswerGenerator:
    def __init__(self, llm: LLMClient):
        self._llm = llm

    def generate(self, question: str, sources: list[ContextSource]) -> GeneratedAnswer:
        raw = self._llm.complete_json(
            system=SYSTEM_PROMPT,
            user=build_user_message(question, sources),
            schema=ANSWER_SCHEMA,
        )
        try:
            return GeneratedAnswer.model_validate(raw)
        except ValidationError as exc:
            raise MalformedLLMResponseError("The model output does not match the answer schema") from exc
