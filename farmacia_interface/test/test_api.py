# test_api.py
import requests
import json

# Test health endpoint
response = requests.get("http://localhost:8000/health")
print(f"Health check: {response.json()}")

# Test detection endpoint
with open("dataset/yolo_format/images/test/codigos10.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(
        "http://localhost:8000/detect-codes",
        files=files,
        params={"validate_cima": True}
    )
    
if response.status_code == 200:
    result = response.json()
    print(f"✅ API funcionando: {result['detection_summary']['total_codes_detected']} códigos")
else:
    print(f"❌ Error API: {response.status_code}")
