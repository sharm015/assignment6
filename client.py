import requests
import json

url = "http://localhost:5000/predict"

r = requests.get(url)
print(r.text)

payload = {'data': 'paper5.png'}
headers = {
        'Content-Type': 'application/json'
    }
response = requests.post(url, headers=headers, data=json.dumps(payload))
print(response.text, response.status_code)