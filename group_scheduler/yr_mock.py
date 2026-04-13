# mock_yr.py
"""
Mock yr module for testing GroupScheduler without yr framework.
Supports async pattern with Future for realistic GS behavior.
"""
from types import SimpleNamespace
from typing import Any, List
from concurrent.futures import Future


class MockKVStore:
    """Mock key-value store"""
    def __init__(self):
        self._store = {}

    def kv_set(self, key: Any, value: Any):
        """Mock kv_set - does nothing"""
        pass

    def kv_get(self, key: Any):
        """Mock kv_get"""
        return self._store.get(key)

    def __contains__(self, key: Any):
        return key in self._store


# Global mock instances
_kv_store = MockKVStore()


class MockFuture:
    """Mock future for async operations"""
    def __init__(self):
        self._future = Future()

    def set_result(self, result):
        """Set result of future"""
        self._future.set_result(result)

    def get(self, timeout=None):
        """
        Get result, blocking if not ready

        Args:
            timeout: Optional timeout in seconds

        Returns:
            The result
        """
        return self._future.result(timeout=timeout if timeout else None)


class MockInstance:
    """Mock @yr.instance decorator"""
    def __init__(self, cls):
        self._cls = cls

    def __call__(self, *args, **kwargs):
        """Return an instance of class directly, bypassing decorator"""
        return self._cls(*args, **kwargs)

    def invoke(self, *args, **kwargs):
        """
        Mock invoke method - returns MockFuture for async behavior

        In real yr framework, this returns an object_ref (future),
        not result directly. We mimic this behavior.
        """
        future = MockFuture()
        # Immediately set result (since we're mocking, no real async)
        result = self._cls(*args, **kwargs)
        future.set_result(result)
        return future


def mock_get(object_ref_or_refs, timeout=None):
    """
    Mock yr.get() - wait for futures and return results

    Args:
        object_ref_or_refs: List of MockFuture objects or single MockFuture
        timeout: Optional timeout in seconds

    Returns:
        List of results or single result
    """
    if isinstance(object_ref_or_refs, list):
        return [ref.get(timeout=timeout) for ref in object_ref_or_refs]
    else:
        return object_ref_or_refs.get(timeout=timeout)


# Mock yr module
mock_yr = SimpleNamespace(
    instance=MockInstance,
    kv_set=_kv_store.kv_set,
    kv_get=_kv_store.kv_get,
    kv_del=lambda key: None,
    init=lambda: None,
    finalize=lambda: None,
    resources=lambda: [],
    get=mock_get,
)
