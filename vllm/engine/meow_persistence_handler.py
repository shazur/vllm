import torch
from datetime import datetime
from typing import (Any, List, TypedDict, Dict)
import asyncio
from concurrent.futures import ThreadPoolExecutor
from vllm.utils import singleton

from vllm.meow_stats import MeowStats
from vllm.logger import init_logger

logger = init_logger(__name__)

meow_stats = MeowStats()

async def async_torch_load(filepath: str, map_location: str = 'cpu'):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        loaded_from_disk_cache = await loop.run_in_executor(
            pool,
            lambda: torch.load(filepath, map_location=map_location)
        )
    return loaded_from_disk_cache

class MeowPersistedKVCaches(TypedDict):
    kv_cache: List[Any]

class MeowPersistedMetadata(TypedDict):
    computed_token_ids: List[int]
    eos_token: int
    index_id: str

@singleton
class MeowPersistenceHandler:
    meow_cache_dict: Dict[str, MeowPersistedKVCaches] = {}
    def __init__(self):
       pass 
      
    def persistCache(self, index_id, kv_caches, blocks, computed_token_ids, eos_token: int):
        persistedKVCache = MeowPersistedKVCaches({
            "kv_cache": self._select_blocks(kv_caches, blocks)
            })
        persistedMetadata = MeowPersistedMetadata({
            "computed_token_ids": computed_token_ids,
            "index_id": index_id,
            "eos_token": eos_token
        })
        self.meow_cache_dict[index_id] = persistedKVCache
        torch.save(persistedKVCache, index_id + ".data.pt") 
        torch.save(persistedMetadata, index_id + ".metadata.pt")

    def getPersistedKvCaches(self, index_id): #make sure to load it form disk first(using async)
        return self.meow_cache_dict.get(index_id)
    def getPersistedMetadata(self):
        return self.persistedMetadata

    def _select_blocks(self, tensor_list, indices):
        # Convert indices to a tensor if it's not already
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices)
        
        # Ensure indices is 1D
        indices = indices.squeeze()
        
        # Get the value of x from the indices
        x = indices.size(0)
        
        results = []
        for tensor in tensor_list:
            # Select the indices from the second dimension of the tensor
            result = tensor[:, indices, :, :, :]
            
            # *** todo meow todo: use model config instead of guessing number of heads, dimension, block size....
            assert result.shape == (2, x, 16, 8, 128), f"Unexpected shape: {result.shape}" 
            
            results.append(result)
    
        return results
    
    async def load_cache(self, index_id):
        start_time = datetime.now()
        if not self.meow_cache_dict.get(index_id):
          filepath = f"{index_id}.data.pt" 
          # Load the dictionary from disk
          #loaded_dict = torch.load(filepath, mmap=True, map_location='cpu') #nmap - doesnt load until in use
          cache = await async_torch_load(filepath, map_location='cpu')
          
          self.meow_cache_dict[index_id] = cache
          
          duration = (datetime.now() - start_time).total_seconds()
          #todo: add stats per MB \ per loaded block
          meow_stats.add_operation_duration("load_cache_from_disk", duration) 

          logger.info(f"loading cache from disk took: {duration} seconds. exact_time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        else:
            logger.info(f"Skipped loading from disk")
       

    async def load_metadata_from_disk(self, index_id):
        # Load the dictionary from disk
        filepath = f"{index_id}.metadata.pt"
        return await async_torch_load(filepath, map_location='cpu')
        