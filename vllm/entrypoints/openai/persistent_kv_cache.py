from typing import (Optional, List)
from urllib.request import Request
import uuid
from pydantic import BaseModel
from vllm.config import ModelConfig
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_engine import LoRAModulePath
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam


class IndexContextRequest(BaseModel):
    content: str
    model: str = "mistralai/Mistral-7B-Instruct-v0.3"

class PersistentKvCache():
    def __init__(self, engine: AsyncLLMEngine, model_config: ModelConfig,
                 served_model_names: List[str],
                 lora_modules: Optional[List[LoRAModulePath]]) -> None:
        self.openai_serving_chat = OpenAIServingChat(engine, model_config, served_model_names, lora_modules)
        pass

    async def populate(self, request: IndexContextRequest): 
        index_id = self._generate_index_id() #hash? 
        messages = [ChatCompletionUserMessageParam(role="user", content=request.content)]
        output = await self.openai_serving_chat.create_chat_completion(ChatCompletionRequest(
            messages = messages, 
            index_id = index_id, 
            model = request.model,
            max_tokens=1)
        )
        # print("result is:" + output.usage.completion_tokens)
        return index_id

    def _generate_index_id(self):
        return str(uuid.uuid4())
        

