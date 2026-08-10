"""
Standalone tests proving both tools work correctly, including their
error/safety paths (no network calls needed for these — get_weather's
HTTP layer is mocked so this runs instantly and doesn't depend on your
API key or internet connection).

Run: python test_tools.py
"""
import unittest.mock as mock
import requests
from tools import get_order_policy, get_weather


class FakeResp:
    def __init__(self, status_code, data):
        self.status_code = status_code
        self._data = data

    def json(self):
        return self._data


def test_order_policy_known_topic():
    result = get_order_policy.invoke({"topic": "returns"})
    assert "30 days" in result
    print("PASS: get_order_policy returns correct info for a known topic")


def test_order_policy_unknown_topic():
    result = get_order_policy.invoke({"topic": "nonexistent topic xyz"})
    assert "Error" in result
    print("PASS: get_order_policy returns an error string (not a crash) for an unknown topic")


def test_weather_success():
    def fake_get(url, params=None, timeout=None):
        if "geocoding" in url:
            return FakeResp(200, {"results": [{"latitude": 48.85, "longitude": 2.35,
                                                "name": "Paris", "country": "France"}]})
        return FakeResp(200, {"current": {"temperature_2m": 22.5, "weather_code": 1}})

    with mock.patch("tools.requests.get", side_effect=fake_get):
        result = get_weather.invoke({"city": "Paris"})
    assert "Paris" in result and "22.5" in result
    print("PASS: get_weather correctly parses a successful response")


def test_weather_city_not_found():
    def fake_get(url, params=None, timeout=None):
        return FakeResp(200, {"results": None})

    with mock.patch("tools.requests.get", side_effect=fake_get):
        result = get_weather.invoke({"city": "Nonexistentplacexyz"})
    assert "Error" in result
    print("PASS: get_weather returns an error string for an unfindable city")


def test_weather_network_failure_does_not_crash():
    def fake_get(url, params=None, timeout=None):
        raise requests.exceptions.ConnectionError("simulated network drop")

    with mock.patch("tools.requests.get", side_effect=fake_get):
        result = get_weather.invoke({"city": "London"})
    assert "Error" in result
    print("PASS: get_weather returns an error string instead of crashing on a network failure")


if __name__ == "__main__":
    test_order_policy_known_topic()
    test_order_policy_unknown_topic()
    test_weather_success()
    test_weather_city_not_found()
    test_weather_network_failure_does_not_crash()
    print("\nALL TOOL TESTS PASSED.")
