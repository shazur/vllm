
from typing import Any, List, Optional
from typing_extensions import NotRequired

from vllm.inputs import LLMInputs


class MeowLLMInputs(LLMInputs):
    """
    An extended version of LLMInputs with additional parameters.
    """
    indexed_prompt: Optional[str]
    """The indexed prompt text, if available."""

    indexed_prompt_ids: Optional[List[int]]
    """The token IDs of the indexed prompt."""

    new_prompt: Optional[str]
    """The new prompt text, if available."""

    new_prompt_token_ids: List[int]
    """The token IDs of the new prompt."""

    indexed_kv_cache: List[Any]
    """The indexed key-value cache."""

    num_of_computed_token: List[int]
    """The number of computed tokens."""