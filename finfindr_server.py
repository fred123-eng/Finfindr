#!/usr/bin/env python3
"""
FinFindr – Fishial.ai Identification App
Upload photos → get real species identification via Fishial API.
Ready for local use or deployment on Render.
"""

from flask import Flask, request, jsonify, render_template_string
import requests
import hashlib
import base64
import os

app = Flask(__name__)

# ============================================================
# FISHIAL CREDENTIALS (set as environment variables for production)
# ============================================================
CLIENT_ID = os.environ.get("FISHIAL_CLIENT_ID", "34920f6556702d993e17d75d")
CLIENT_SECRET = os.environ.get("FISHIAL_CLIENT_SECRET", "163052c6e9dc46123cf64316d67fd05e")
# ============================================================

AUTH_URL = "https://api-users.fishial.ai/v1/auth/token"
UPLOAD_URL = "https://api.fishial.ai/v1/recognition/upload"
RECOGNITION_URL = "https://api.fishial.ai/v1/recognition/image"

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FinFindr</title>
<style>
  :root { --deep: #0a2540; --teal: #0d9488; --aqua: #22d3ee; --white: #f8fafc; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, -apple-system, sans-serif; background: linear-gradient(160deg, var(--deep) 0%, #134e4a 100%); color: var(--white); min-height: 100vh; padding: 1.5rem; }
  header { text-align: center; margin-bottom: 1.5rem; }
  header h1 { font-size: 2.6rem; background: linear-gradient(90deg, var(--aqua), #67e8f9); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  header p { opacity: 0.8; margin-top: 0.3rem; }
  .upload-zone { width: 100%; max-width: 560px; margin: 0 auto 1.5rem; border: 2px dashed rgba(255,255,255,0.3); border-radius: 20px; padding: 2.5rem 2rem; text-align: center; background: rgba(255,255,255,0.05); cursor: pointer; }
  .upload-zone:hover, .upload-zone.dragover { border-color: var(--aqua); background: rgba(34,211,238,0.08); }
  #fileInput { display: block; margin: 1rem auto; }
  #results { max-width: 800px; margin: 0 auto; }
  .result-card { background: rgba(0,0,0,0.35); border-radius: 16px; margin-bottom: 1.25rem; border: 1px solid rgba(255,255,255,0.1); overflow: hidden; }
  .result-card img { width: 100%; max-height: 280px; object-fit: cover; }
  .result-body { padding: 1rem; }
  .result-body h3 { margin-bottom: 0.3rem; font-size: 1.35rem; }
  .sci { opacity: 0.7; font-style: italic; margin-bottom: 0.4rem; }
  .confidence { display: inline-block; background: var(--teal); padding: 0.25rem 0.7rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600; margin: 0.4rem 0; }
  .error { color: #fb7185; }
  .identifying { opacity: 0.6; font-style: italic; }
  .actions { text-align: center; margin-top: 1.5rem; }
  button { background: var(--teal); color: white; border: none; padding: 0.7rem 1.5rem; border-radius: 12px; cursor: pointer; font-weight: 600; }
  footer { text-align: center; margin-top: 2.5rem; opacity: 0.5; font-size: 0.8rem; }
</style>
</head>
<body>
<header>
  <h1>FinFindr</h1>
  <p>Upload a photo · Get the species</p>
</header>

<div class="upload-zone" id="dropZone">
  <h2>Choose a photo</h2>
  <p>Use the button below</p>
  <input type="file" id="fileInput" accept="image/*" style="display:block; margin:1rem auto; font-size:1.1rem;">
</div>

<div id="results"></div>

<div class="actions" id="actions" style="display:none;">
  <button onclick="reset()">Identify another photo</button>
</div>

<footer>FinFindr</footer>

<script>
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const results = document.getElementById("results");
const actions = document.getElementById("actions");

// Direct file input - no click proxy needed
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

async function handleFile(file) {
  if (!file.type.startsWith("image/")) return alert("Please upload an image.");

  dropZone.style.display = "none";
  actions.style.display = "block";
  results.innerHTML = "";

  const card = document.createElement("div");
  card.className = "result-card";
  card.innerHTML = `
    <img class="upload-preview" alt="Uploaded photo">
    <div class="result-body">
      <h3 class="identifying">Identifying…</h3>
      <div class="sci"></div>
      <div class="confidence"></div>
      <div class="error"></div>
    </div>
  `;
  results.appendChild(card);

  const reader = new FileReader();
  reader.onload = (e) => card.querySelector(".upload-preview").src = e.target.result;
  reader.readAsDataURL(file);

  const formData = new FormData();
  formData.append("image", file);

  try {
    const resp = await fetch("/identify", { method: "POST", body: formData });
    if (!resp.ok) {
      const errText = await resp.text();
      throw new Error("Server error " + resp.status + ": " + errText.slice(0, 200));
    }
    const data = await resp.json();

    if (data.error) {
      card.querySelector("h3").textContent = "Identification failed";
      card.querySelector("h3").classList.remove("identifying");
      card.querySelector(".error").textContent = data.error;
      return;
    }

    const text = data.result_text || "No response from model";
    card.querySelector("h3").textContent = "Result";
    card.querySelector("h3").classList.remove("identifying");
    card.querySelector(".sci").textContent = text;
  } catch (err) {
    card.querySelector("h3").textContent = "Error";
    card.querySelector("h3").classList.remove("identifying");
    card.querySelector(".error").textContent = err.message || "Could not reach the server.";
    const retryBtn = document.createElement("button");
    retryBtn.textContent = "Retry";
    retryBtn.style.marginTop = "0.75rem";
    retryBtn.onclick = () => handleFile(file);
    card.querySelector(".result-body").appendChild(retryBtn);
  }
}

function reset() {
  results.innerHTML = "";
  actions.style.display = "none";
  dropZone.style.display = "block";
  fileInput.value = "";
}
</script>
</body>
</html>
"""

def get_access_token():
    # no longer needed
    pass

def get_checksum(data: bytes) -> str:
    # no longer needed
    pass

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/identify", methods=["POST"])
def identify():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image_bytes = file.read()
    content_type = file.content_type or "image/jpeg"

    xai_key = os.environ.get("XAI_API_KEY")
    if not xai_key:
        return jsonify({"error": "XAI_API_KEY not set on the server"}), 500

    # Encode image as base64 data URL
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{content_type};base64,{b64}"

    prompt = (
        "This underwater photo was taken in the Sea of Cortez near La Paz, Mexico. "
        "Identify the main fish or marine animal. "
        "Reply in this exact format:\n"
        "Common name: <name>\n"
        "Scientific name: <name>\n"
        "Group: <group or family>\n"
        "Confidence: <low/medium/high>\n"
        "Notes: <one short sentence>\n"
        "If you are unsure, say so clearly."
    )

    payload = {
        "model": "grok-4",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": data_url, "detail": "high"},
                    {"type": "input_text", "text": prompt}
                ]
            }
        ]
    }

    try:
        resp = requests.post(
            "https://api.x.ai/v1/responses",
            headers={
                "Authorization": f"Bearer {xai_key}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=60
        )
        resp.raise_for_status()
        data = resp.json()

        # Extract the text response
        text = ""
        if "output" in data and data["output"]:
            for item in data["output"]:
                if item.get("type") == "message":
                    for c in item.get("content", []):
                        if c.get("type") == "output_text":
                            text += c.get("text", "")
        if not text and "choices" in data:
            text = data["choices"][0].get("message", {}).get("content", "")

        return jsonify({"result_text": text or "No response from model"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\nFinFindr is running on port {port}")
    print(f"Local:  http://localhost:{port}")
    print("Press Ctrl+C to stop.\n")
    app.run(host="0.0.0.0", port=port, debug=False)
