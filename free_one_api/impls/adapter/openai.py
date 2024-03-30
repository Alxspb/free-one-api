import typing
import traceback
import uuid
import random

import openai
from openai.types.chat.chat_completion_chunk import Choice

from free_one_api.entities import request, response

from ...models import adapter
from ...models.adapter import llm
from ...entities import request, response, exceptions
from ...models.channel import evaluation


@adapter.llm_adapter
class OpenAI(llm.LLMLibAdapter):

    @classmethod
    def name(cls) -> str:
        return "openai"

    @classmethod
    def description(self) -> str:
        return "Use official openai api (with optional custom base url)"

    def supported_models(self) -> list[str]:
        return self.config.get("supported_models", [
            "gpt-3.5-turbo",
            "gpt-4"
        ])

    def function_call_supported(self) -> bool:
        return True

    def stream_mode_supported(self) -> bool:
        return True

    def multi_round_supported(self) -> bool:
        return True

    @classmethod
    def config_comment(cls) -> str:
        return \
            """You shoule provide api_key or/and base_url:
            {
                "base_url": "https://chat.openai.com",
                "api_key": "sk-12345"
            }
            """

    @classmethod
    def supported_path(cls) -> str:
        return "/v1/chat/completions"

    _chatbot: openai.OpenAI = None

    @property
    def chatbot(self) -> openai.OpenAI:
        if self._chatbot is None:
            self._chatbot = openai.OpenAI(
                base_url=self.config.get("base_url"),
                api_key=self.config.get("api_key"))
        return self._chatbot

    def __init__(self, config: dict, eval: evaluation.AbsChannelEvaluation):
        self.config = config
        self.eval = eval

    async def test(self) -> (bool, str):
        try:
            self.chatbot.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": "hi"}
                ]
            )
            return True, ""
        except Exception as e:
            traceback.print_exc()
            return False, str(e)

    async def query(self, req: request.Request) -> typing.AsyncGenerator[response.Response, None]:
        random_int = random.randint(0, 1000000000).__str__()
        for resp in self.chatbot.chat.completions.create(
            model=req.model,
            messages=req.messages,
            functions=req.functions,
            stream=True,

        ):
            choice: Choice = resp.choices[0]
            yield response.Response(
                id=resp.id,
                finish_reason=response.FinishReason.NULL,
                normal_message=choice.delta.content,
                function_call=choice.delta.function_call
            )

        yield response.Response(
            id=random_int,
            finish_reason=response.FinishReason.STOP,
            normal_message="",
            function_call=None
        )
