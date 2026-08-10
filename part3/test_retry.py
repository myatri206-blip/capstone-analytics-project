"""
Standalone proof that call_llm()'s retry logic works: simulates a fake
API that fails with a network error twice, then succeeds on the 3rd
attempt, and confirms call_llm() still returns the correct result
instead of crashing or giving up early.

Run: python test_retry.py
"""
import unittest.mock as mock
import llm_client

call_count = {"n": 0}


class FakeResponse:
    def __init__(self, status_code, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data
        self.text = text

    def json(self):
        return self._json


def fake_post(*args, **kwargs):
    call_count["n"] += 1
    if call_count["n"] < 3:
        raise llm_client.requests.exceptions.ConnectionError("simulated network drop")
    return FakeResponse(
        200,
        {"candidates": [{"content": {"parts": [
            {"text": '{"label":"positive","confidence":"high","reason":"test"}'}
        ]}}]},
    )


if __name__ == "__main__":
    llm_client.API_KEY = "fake-key-for-test"
    with mock.patch("llm_client.requests.post", side_effect=fake_post):
        with mock.patch("llm_client.time.sleep", return_value=None):
            result = llm_client.call_llm("test prompt", temperature=0.2, max_tokens=100)

    print("Result after simulated failures:", result)
    print("Total attempts made:", call_count["n"])
    assert call_count["n"] == 3, "Expected exactly 3 attempts (2 failures + 1 success)"
    assert "positive" in result, "Expected the successful response to come through"
    print("\nPASS: retry logic correctly recovered after 2 simulated failures.")
