from typing import (Optional, List)
from urllib.request import Request
import uuid
from pydantic import BaseModel
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.serving_engine import PromptAdapterPath
from vllm.config import ModelConfig
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, CompletionRequest
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_engine import LoRAModulePath
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam


class IndexContextRequest(BaseModel):
    content: str
    model: str = "mistralai/Mistral-7B-Instruct-v0.3"

class OptimizedCompletionRequest(ChatCompletionRequest):
    index_id: str

class PersistentKvCache():
    def __init__(
        self,
        engine: AsyncLLMEngine,
        model_config: ModelConfig,
        served_model_names: List[str],
        response_role: str,
        *,
        lora_modules: Optional[List[LoRAModulePath]],
        prompt_adapters: Optional[List[PromptAdapterPath]],
        request_logger: Optional[RequestLogger],
        chat_template: Optional[str],
        return_tokens_as_token_ids: bool = False,
    ):
      self.openai_serving_chat = OpenAIServingChat(
        engine,
        model_config,
        served_model_names,
        'user',
        lora_modules=lora_modules,
        prompt_adapters=prompt_adapters,
        request_logger=request_logger,
        chat_template=chat_template,
        return_tokens_as_token_ids=return_tokens_as_token_ids,
    )
      pass

    async def populate(self, request: IndexContextRequest): 
        index_id = self._generate_index_id() #hash? 
        messages = [ChatCompletionUserMessageParam(role="user", content=request.content)]
        output = await self.openai_serving_chat.create_chat_completion(ChatCompletionRequest(
            messages = messages, 
            index_id = index_id, 
            should_index = True,
            model = request.model,
            max_tokens=1)
        )
        # print("result is:" + output.usage.completion_tokens)
        return index_id
    
    async def create_chat_opt_completion(self, request: OptimizedCompletionRequest):
        return await self.openai_serving_chat.create_chat_completion(request)


    def _generate_index_id(self):
        return str(uuid.uuid4())
        

