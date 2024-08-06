import torch
from datetime import datetime
from typing import (Any, List, TypedDict)

from vllm.meow_stats import MeowStats

meow_stats = MeowStats()

class MeowPersistedKVCaches(TypedDict):
    kv_cache: List[Any]

class MeowPersistedMetadata(TypedDict):
    computed_token_ids: List[int]
    eos_token: int
    index_id: str

class MeowPersistenceHandler:
    def __init__(self, index_id, kv_caches, blocks, computed_token_ids, eos_token: int):
        
        self.persistedKVCache = MeowPersistedKVCaches({
            "kv_cache": self._select_blocks(kv_caches, blocks)
            })
        self.persistedMetadata = MeowPersistedMetadata({
            "computed_token_ids": computed_token_ids,
            "index_id": index_id,
            "eos_token": eos_token
        })
    
    def writeToDisk(self, index_id):
        #write kv_caches and metadata to disk -
        # meow TODO meow: should be async, and check that the dictionaries are not empty and such
        torch.save(self.persistedKVCache, index_id + ".data.pt") 
        torch.save(self.persistedMetadata, index_id + ".metadata.pt") 

    def getPersistedKvCaches(self):
        return self.persistedKVCache
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
    
    @classmethod
    def load_cache_from_disk(cls, filepath):
        start_time = datetime.now() 
        # Load the dictionary from disk
        loaded_dict = torch.load(filepath, mmap=True, map_location='cpu') #todo meow- go back to gpu, this is slow
        
        # Create an instance of the class
        instance = cls.__new__(cls)
        
        # Directly set the dict attribute
        instance.persistedKVCache = loaded_dict
        
        duration = (datetime.now() - start_time).total_seconds()

        #todo: add stats per MB \ per loaded block
        meow_stats.add_operation_duration("load_cache_from_disk", duration) 

        print(f"loading cache from disk took: {duration} seconds. exact_time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")

        return instance
    @classmethod
    def load_metadata_from_disk(cls, filepath):
        # Load the dictionary from disk
        loaded_dict = torch.load(filepath, mmap=True, map_location='cpu')
        
        # Create an instance of the class
        instance = cls.__new__(cls)
        
        # Directly set the dict attribute
        instance.persistedMetadata = loaded_dict

        return instance