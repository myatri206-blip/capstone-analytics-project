"""
Task: Define at least two tools, at least one calling a real, live
external API (not a hardcoded dictionary).

Both tools follow the four taught good-tool properties:
- Clear name: get_weather, get_order_policy — the name alone tells you
  what each does.
- Honest/accurate description: the docstring (which the LLM reads to
  decide when to call the tool) describes exactly what it does, no more.
- Atomic: each does exactly one job — one looks up weather, the other
  looks up a policy. Neither is a "do several things" grab-bag tool.
- Safe: neither tool ever raises an exception up to the agent. Every
  failure path (bad city name, network error, unknown policy topic)
  returns a descriptive string as DATA, so the agent can react to it
  in conversation instead of the whole run crashing.
"""
import requests
from langchain_core.tools import tool

WEATHER_ATTRIBUTION = "Weather data by Open-Meteo.com (CC BY 4.0)"

# Open-Meteo's WMO weather codes, abbreviated to the common ones
_WEATHER_CODE_MAP = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    71: "slight snow", 73: "moderate snow", 75: "heavy snow",
    80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    95: "thunderstorm",
}


@tool
def get_weather(city: str) -> str:
    """Looks up the CURRENT real-time weather for a named city using the
    live Open-Meteo API (no API key needed). Use this when the user asks
    about current weather, temperature, or conditions in a specific city,
    e.g. for questions about whether weather might delay a delivery.
    Input should be just the city name, e.g. "Paris" or "Berlin"."""
    try:
        geo_resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=10,
        )
        if geo_resp.status_code != 200:
            return f"Error: geocoding lookup failed with status {geo_resp.status_code} for city '{city}'."

        geo_data = geo_resp.json()
        results = geo_data.get("results")
        if not results:
            return f"Error: could not find a location matching '{city}'. Please check the spelling."

        lat = results[0]["latitude"]
        lon = results[0]["longitude"]
        resolved_name = results[0].get("name", city)
        country = results[0].get("country", "")

        forecast_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,weather_code",
            },
            timeout=10,
        )
        if forecast_resp.status_code != 200:
            return f"Error: forecast lookup failed with status {forecast_resp.status_code} for '{resolved_name}'."

        current = forecast_resp.json().get("current", {})
        temp = current.get("temperature_2m")
        code = current.get("weather_code")
        condition = _WEATHER_CODE_MAP.get(code, f"weather code {code}")

        return (
            f"Current weather in {resolved_name}, {country}: {temp}°C, {condition}. "
            f"({WEATHER_ATTRIBUTION})"
        )

    except requests.exceptions.RequestException as e:
        return f"Error: network problem while fetching weather for '{city}': {e}"


# Local/mock data — a small customer-support policy knowledge base.
# Deliberately NOT calling an external service; this is the tool the
# brief allows to "read from local/mock data".
_POLICY_KB = {
    "returns": "Items can be returned within 30 days of delivery, unworn "
               "and with tags attached, for a full refund to the original "
               "payment method.",
    "shipping": "Standard shipping takes 5-7 business days. Expedited "
                "shipping (2-3 business days) is available at checkout "
                "for an additional fee.",
    "refunds": "Refunds are issued within 5-10 business days after we "
               "receive the returned item, back to the original payment "
               "method.",
    "exchanges": "Size or color exchanges are free within 30 days of "
                 "delivery, subject to stock availability.",
    "delivery delay": "If a delivery is delayed beyond the estimated "
                       "window, customers can contact support for a "
                       "shipping credit or expedited replacement.",
}


@tool
def get_order_policy(topic: str) -> str:
    """Looks up this store's official policy on a given order-related
    topic from our internal knowledge base. Valid topics include:
    "returns", "shipping", "refunds", "exchanges", "delivery delay".
    Use this when the user asks about store policy rather than a
    specific order's status. Input should be a short topic phrase."""
    topic_lower = topic.lower().strip()
    for key, policy_text in _POLICY_KB.items():
        if key in topic_lower or topic_lower in key:
            return f"Policy on {key}: {policy_text}"
    available = ", ".join(_POLICY_KB.keys())
    return (
        f"Error: no policy found matching '{topic}'. "
        f"Available policy topics are: {available}."
    )
