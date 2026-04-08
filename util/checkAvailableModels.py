import requests
import time
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY1")

# Load your models JSON (paste or load from file)
url = "https://generativelanguage.googleapis.com/v1beta/models"
response = requests.post(url, timeout=10)
data = response.json()

models = [m["name"] for m in data["models"]]

URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/{model}:generateContent?key={api_key}"

payload = {
    "contents": [
        {
            "parts": [
                {"text": "Say OK"}
            ]
        }
    ]
}

results = []

for model in models:
    url = URL_TEMPLATE.format(model=model, api_key=API_KEY)

    try:
        response = requests.post(url, json=payload, timeout=10)

        if response.status_code == 200:
            results.append((model, "✅ WORKING"))
            print(f"[✔] {model}")

        else:
            error_text = response.text.lower()

            if "quota" in error_text or "billing" in error_text:
                status = "❌ QUOTA/BILLING"
            elif "permission" in error_text or "not found" in error_text:
                status = "❌ NOT ALLOWED"
            else:
                status = f"⚠️ ERROR {response.status_code}"

            results.append((model, status))
            print(f"[✘] {model} -> {status}")

    except Exception as e:
        results.append((model, "⚠️ EXCEPTION"))
        print(f"[!] {model} -> Exception: {str(e)}")

    time.sleep(1)  # avoid rate limiting

# Save results
with open("model_results.txt", "w") as f:
    for model, status in results:
        f.write(f"{model} : {status}\n")

print("\nDone! Results saved in model_results.txt")