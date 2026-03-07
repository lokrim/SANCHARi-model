import requests
import json
import time

APIS = [
    {
        "name": "Local GeoTIFF API",
        "url": "http://localhost:8000/predict",
        "payload": {
            "latitude": 30.224949915094008,
            "longitude": -97.78460932372762
        }
    },
    {
        "name": "GEE API",
        "url": "http://localhost:8001/predict",
        "payload": {
            "latitude": 34.09452,
            "longitude": -118.27286
        }
    }
]

TIMEOUT = 200


def test_api(api):
    name = api["name"]
    url = api["url"]
    payload = api["payload"]

    print("\n--------------------------------")
    print("Testing:", name)
    print("Endpoint:", url)
    print("Payload:", payload)
    print("--------------------------------")

    try:
        start = time.time()

        response = requests.post(
            url,
            json=payload,
            timeout=TIMEOUT
        )

        elapsed = time.time() - start

        print("Status Code:", response.status_code)
        print("Response Time: {:.2f}s".format(elapsed))

        if response.status_code != 200:
            print("Response:")
            print(response.text)
            return

        try:
            data = response.json()
        except ValueError:
            print("Response is not valid JSON")
            print(response.text)
            return

        print("GeoJSON Response:")
        print(json.dumps(data, indent=2))

    except requests.exceptions.ConnectionError:
        print("Connection failed. API may not be running.")

    except requests.exceptions.Timeout:
        print("Request timed out.")

    except Exception as e:
        print("Unexpected error:", str(e))


if __name__ == "__main__":

    print("SANCHARi API Test Runner")

    for api in APIS:
        test_api(api)

    print("\nTest complete.")