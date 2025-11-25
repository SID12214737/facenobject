import requests

print(requests.post("http://localhost:8088/register-name", json={"unknown_id": "1", "name": "mir"}).json())