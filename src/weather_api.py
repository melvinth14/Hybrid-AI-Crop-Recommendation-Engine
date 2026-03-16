"""
Weather API module — fetches live weather for Tamil Nadu districts
using OpenWeatherMap Current Weather API (free tier).
"""

import requests

# Tamil Nadu districts with lat/lon
TN_DISTRICTS = {
    "Ariyalur":        (11.1372, 79.0778),
    "Chengalpattu":    (12.6831, 79.9770),
    "Chennai":         (13.0825, 80.2750),
    "Coimbatore":      (11.0168, 76.9633),
    "Cuddalore":       (11.7500, 79.7500),
    "Dharmapuri":      (12.1065, 78.1361),
    "Dindigul":        (10.3500, 77.9500),
    "Erode":           (11.3410, 77.7171),
    "Kallakurichi":    (11.7382, 78.9639),
    "Kanchipuram":     (12.8342, 79.7036),
    "Kanyakumari":     (8.0883,  77.5385),
    "Karur":           (10.9600, 78.0750),
    "Krishnagiri":     (12.5516, 78.1976),
    "Madurai":         (9.9391,  78.1217),
    "Mayiladuthurai":  (11.1018, 79.6522),
    "Nagapattinam":    (10.7906, 79.8428),
    "Namakkal":        (11.2296, 78.1712),
    "Nilgiris":        (11.4000, 76.7000),
    "Perambalur":      (11.2300, 78.8800),
    "Pudukkottai":     (10.3833, 78.8001),
    "Ramanathapuram":  (9.3639,  78.8395),
    "Ranipet":         (12.9321, 79.3335),
    "Salem":           (11.6500, 78.1500),
    "Sivaganga":       (9.8470,  78.4836),
    "Tenkasi":         (8.9590,  77.3129),
    "Thanjavur":       (10.7833, 79.1383),
    "Theni":           (10.0104, 77.4768),
    "Thoothukudi":     (8.8056,  78.1450),
    "Tiruchirappalli": (10.7830, 78.6830),
    "Tirunelveli":     (8.6500,  77.3830),
    "Tirupattur":      (12.4927, 78.5681),
    "Tiruppur":        (11.1085, 77.3411),
    "Tiruvallur":      (13.1444, 79.8940),
    "Tiruvannamalai":  (12.2286, 79.0665),
    "Tiruvarur":       (10.7730, 79.6370),
    "Vellore":         (12.9349, 79.1469),
    "Viluppuram":      (11.9401, 79.4861),
    "Virudhunagar":    (9.5872,  77.9514),
}

# Seasonal average rainfall (mm) estimates for Tamil Nadu by month
MONTHLY_RAINFALL = {
    1: 30,  2: 15,  3: 10,  4: 25,   # Winter / Pre-summer
    5: 55,  6: 40,  7: 60,  8: 90,   # Summer / Kharif start
    9: 110, 10: 180, 11: 220, 12: 100 # NE Monsoon peak
}


def get_weather(district: str, api_key: str) -> dict:
    """
    Fetch current weather for a Tamil Nadu district.
    Returns dict with temperature, humidity, rainfall (estimated).
    Falls back to seasonal averages if API call fails.
    """
    if not api_key or api_key in ("YOUR_API_KEY_HERE", ""):
        return _fallback_weather(district)

    if district not in TN_DISTRICTS:
        return _fallback_weather(district)

    lat, lon = TN_DISTRICTS[district]
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    )

    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        temp     = data['main']['temp']
        humidity = data['main']['humidity']
        # Rainfall last 1h if available (convert mm/h → seasonal estimate)
        rain_1h  = data.get('rain', {}).get('1h', 0)

        import datetime
        month = datetime.datetime.now().month
        rainfall = MONTHLY_RAINFALL.get(month, 80)

        return {
            'temperature': round(temp, 1),
            'humidity':    round(humidity, 1),
            'rainfall':    rainfall,
            'source':      'live',
            'district':    district,
        }
    except Exception:
        return _fallback_weather(district)


def _fallback_weather(district: str) -> dict:
    """Return seasonal average weather for a district."""
    import datetime
    month = datetime.datetime.now().month
    # Typical TN averages by month
    month_temp = {
        1: 26, 2: 28, 3: 31, 4: 34, 5: 36, 6: 34,
        7: 32, 8: 31, 9: 30, 10: 28, 11: 26, 12: 25
    }
    month_humidity = {
        1: 72, 2: 68, 3: 65, 4: 68, 5: 65, 6: 68,
        7: 74, 8: 78, 9: 80, 10: 82, 11: 80, 12: 75
    }
    return {
        'temperature': month_temp.get(month, 30),
        'humidity':    month_humidity.get(month, 72),
        'rainfall':    MONTHLY_RAINFALL.get(month, 80),
        'source':      'seasonal_estimate',
        'district':    district,
    }
