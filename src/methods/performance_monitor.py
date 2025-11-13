from loguru import logger
import time
import torch
from functools import wraps

def log_execution_time(use_cuda_timing=True):
    """
    Universal decorator to log execution time for both CUDA and non-CUDA operations.
    
    Args:
        use_cuda_timing (bool): If True and CUDA is available, uses CUDA events for timing.
                               If False or CUDA unavailable, uses CPU timing.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Determine timing method
            cuda_available = torch.cuda.is_available() and use_cuda_timing
            device = "CUDA" if cuda_available else "CPU"
            
            if cuda_available:
                # CUDA timing with events
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                result = func(*args, **kwargs)
                end_event.record()
                
                # Wait for GPU to finish and get timing
                torch.cuda.synchronize()
                elapsed_time = start_event.elapsed_time(end_event)
                time_unit = "ms"
                
            else:
                # CPU timing
                start_time = time.perf_counter()  # More precise than time.time()
                result = func(*args, **kwargs)
                elapsed_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
                time_unit = "ms"
            
            # Log the timing information
            logger.info(f"Execution time for {func.__name__} on {device}: {elapsed_time:.2f} {time_unit}")
            
            return result
        return wrapper
    return decorator