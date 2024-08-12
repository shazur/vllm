import threading
import time
from vllm.logger import init_logger
from vllm.utils import singleton

logger = init_logger(__name__)

@singleton
class MeowStats:
    # _instance = None

    # def __new__(cls):
    #     if cls._instance is None:
    #         cls._instance = super(MeowStats, cls).__new__(cls)
    #         cls._instance._init_stats()
    #     return cls._instance

    def __init__(self):
        self.stats = {
            'index': {
                'prompt_phase': {'total_time': 0, 'total_tokens': 0, 'total_computed_tokens': 0, 'total_requests': 0},
                'inference_phase': {'total_time': 0, 'total_tokens': 0}
            },
            'regular': {
                'prompt_phase': {'total_time': 0, 'total_tokens': 0, 'total_requests': 0},
                'inference_phase': {'total_time': 0, 'total_tokens': 0}
            },
            'operations': {}
        }

    def add_timing(self, duration: float, number_of_input_tokens: int, is_opt_from_index_request: bool, is_prompt_phase: bool, number_of_computed_tokens: int = 0):
        request_type = 'index' if is_opt_from_index_request else 'regular'
        phase = 'prompt_phase' if is_prompt_phase else 'inference_phase'
        
        self.stats[request_type][phase]['total_time'] += duration
        self.stats[request_type][phase]['total_tokens'] += number_of_input_tokens
            
        if is_prompt_phase: # because this only happens once
            if is_opt_from_index_request:
              self.stats[request_type][phase]['total_computed_tokens'] += number_of_computed_tokens
            self.stats[request_type][phase]['total_requests'] += 1

    def add_operation_duration(self, operation_name: str, duration: float):
        if operation_name not in self.stats['operations']:
            self.stats['operations'][operation_name] = {
                'durations': [],
                'min_duration': float('inf'),
                'max_duration': float('-inf'),
                'total_duration': 0,
                'count': 0
            }
        self.stats['operations'][operation_name]['durations'].append(duration)
        self.stats['operations'][operation_name]['min_duration'] = min(self.stats['operations'][operation_name]['min_duration'], duration)
        self.stats['operations'][operation_name]['max_duration'] = max(self.stats['operations'][operation_name]['max_duration'], duration)
        self.stats['operations'][operation_name]['total_duration'] += duration
        self.stats['operations'][operation_name]['count'] += 1

    def get_operation_stats(self, operation_name: str):
        if operation_name in self.stats['operations']:
            op_stats = self.stats['operations'][operation_name]
            average_duration = op_stats['total_duration'] / op_stats['count'] if op_stats['count'] > 0 else 0
            return {
                'min_duration': op_stats['min_duration'],
                'max_duration': op_stats['max_duration'],
                'average_duration': round(average_duration, 3)
            }
        return {}

    def get_stats(self, phase: str, is_opt_from_index_request: bool):
        request_type = 'index' if is_opt_from_index_request else 'regular'
        total_time = self.stats[request_type][phase]['total_time']
        total_tokens = self.stats[request_type][phase]['total_tokens']
        total_computed_tokens = self.stats[request_type][phase].get('total_computed_tokens', 0)
        total_requests = self.stats[request_type][phase].get('total_requests', 0)
        time_per_token = total_time / total_tokens if total_tokens else 0
        tokens_per_second = total_tokens / total_time if total_time else 0
        average_prompt_length = total_tokens / total_requests if total_requests else 0
        combined_tokens_per_second = (total_tokens + total_computed_tokens) / total_time if total_time else 0
        return {
            'total_time': total_time,
            'total_tokens': total_tokens,
            'total_computed_tokens': total_computed_tokens,
            'total_requests': total_requests,
            'time_per_token': round(time_per_token, 7),
            'tokens_per_second': round(tokens_per_second, 3),
            'average_prompt_length': round(average_prompt_length, 2),
            'combined_tokens_per_second': round(combined_tokens_per_second, 3)
        }

    def get_index_prompt_stats(self):
        return self.get_stats('prompt_phase', True)

    def get_regular_prompt_stats(self):
        return self.get_stats('prompt_phase', False)

    def get_index_inference_stats(self):
        return self.get_stats('inference_phase', True)

    def get_regular_inference_stats(self):
        return self.get_stats('inference_phase', False)

    def log_stats(self):
        index_prompt_stats = self.get_index_prompt_stats()
        regular_prompt_stats = self.get_regular_prompt_stats()
        index_inference_stats = self.get_index_inference_stats()
        regular_inference_stats = self.get_regular_inference_stats()

        log_message = ""

        if any(stat['total_time'] > 0 or stat['total_tokens'] > 0 for stat in [index_prompt_stats, regular_prompt_stats, index_inference_stats, regular_inference_stats]):
            log_message += (
                "------------------------------------------------------------------------\n"
                "Opt indexed request:\n"
                f"  Prompt: secs/token={index_prompt_stats['time_per_token']}, tokens/second={index_prompt_stats['tokens_per_second']}, avg_prompt_length={index_prompt_stats['average_prompt_length']}, combined_tokens/second={index_prompt_stats['combined_tokens_per_second']}\n"
                f"  Inference: secs/token={index_inference_stats['time_per_token']}, tokens/second={index_inference_stats['tokens_per_second']}\n"
                "Regular request:\n"
                f"  Prompt: secs/token={regular_prompt_stats['time_per_token']}, tokens/second={regular_prompt_stats['tokens_per_second']}, avg_prompt_length={regular_prompt_stats['average_prompt_length']}\n"
                f"  Inference: secs/token={regular_inference_stats['time_per_token']}, tokens/second={regular_inference_stats['tokens_per_second']}\n"
                "------------------------------------------------------------------------\n"
            )

        for operation_name, operation_stats in self.stats['operations'].items():
            if operation_stats['count'] > 0:
                duration_stats = self.get_operation_stats(operation_name)
                log_message += (
                    f"{operation_name}:\n"
                    f"  min_duration={duration_stats['min_duration']}, max_duration={duration_stats['max_duration']}, average_duration={duration_stats['average_duration']}\n"
                )

        if log_message:
            log_message += "------------------------------------------------------------------------\n"
            #logger.info(log_message.strip()) todo meow - uncomment 


