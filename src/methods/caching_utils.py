"""Simple wrapper for caching intermediate results."""

import hashlib
import os
import pickle
from functools import wraps
import warnings
from joblib import dump, load

import pandas as pd
from loguru import logger


def get_dataframe_hash(X: pd.DataFrame) -> str:
    """Compute a hash for the DataFrame to check if cached hiddens match.

    Parameters
    ----------
    X : pd.DataFrame
        Input DataFrame to compute the hash.

    Returns
    -------
    str : Hash value representing the contents of the DataFrame.

    """
    X_bytes = X.to_string().encode()
    return hashlib.md5(X_bytes).hexdigest()


def cache_result(cache_dir: str, message: str = "Processing"):
    """Cache the result of a function call.

    Note that you need to pass argument "hash" or "cache_name" as keyword argument whenever you call wrapped function.

    Args:
    ----
        default_cache_dir: The directory where the cache will be stored.
        If cache_dir keyword argument is passed in wrapped function call,
        then it will supersede value specified in decorator
        message: A message to log when processing.

    Returns:
    -------
        The wrapped function with caching enabled.

    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            hash = kwargs.pop("hash", None)
            cache_name = kwargs.pop("cache_name", None)
            cache_dir_from_func_call = kwargs.pop("cache_dir", None)
            used_cache_dir = (
                cache_dir_from_func_call
                if cache_dir_from_func_call is not None
                else cache_dir
            )
            if (hash is None) and (cache_name is None):
                raise ValueError(
                    "You need to pass hash as keyword argument or pass cachefile name"
                )

            if cache_name is None:
                cache_file = os.path.join(used_cache_dir, f"{func.__name__}_{hash}.pkl")
            else:
                cache_file = os.path.join(used_cache_dir, cache_name)

            os.makedirs(used_cache_dir, exist_ok=True)

            if os.path.exists(cache_file):
                logger.info(f"Loading cached result from {cache_file}.")
                # breakpoint()
                with open(cache_file, "rb") as f:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", FutureWarning)
                        data = load(f)
                logger.info(f"Successfully loaded cached result from {cache_file}")
                return data
            else:
                logger.info(
                    f"{message}: No cache found, for {cache_file} running {func.__name__}."
                )
                result = func(*args, **kwargs)
                logger.info(f"Saving result to {cache_file}")
                with open(cache_file, "wb") as f:
                    dump(result, f)
                logger.info(f"{message}: Saved result to {cache_file}.")

                return result

        return wrapper

    return decorator
