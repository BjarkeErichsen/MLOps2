
import numpy as np
import requests
import json

# URL of the API endpoint
url = "https://europe-west1-woven-plane-410710.cloudfunctions.net/function1"

#we access the content of the json file via its key. Look at the cloud funciton script to see how input data is used.
data = {
    "input_data": "1,1,1,1"
}
#the only reason we use a list instead of another datastructure is how the script is made, ie. you can use another datastructure if the cloud function is different. 

# Sending a GET request to the API endpoint
headers = {
    'Content-Type': 'application/json'
}

# Sending a POST request to the API endpoint with JSON data
response = requests.post(url, data=json.dumps(data), headers=headers)


print(response.content)




  