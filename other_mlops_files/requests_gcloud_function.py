import requests

# URL of the API endpoint
url = "https://my-first-function-gs2vkyucma-ew.a.run.app"

params = {
    "name": "Bjarke"

}
# Sending a GET request to the API endpoint
response = requests.get(url, params=params)

# Checking if the request was successful
if response.status_code == 200:
    print("Request successful.")
    print("Response:", response.text)
else:
    print(f"Request failed with status code: {response.status_code}")
      