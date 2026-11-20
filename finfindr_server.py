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
  #fileInput { display: none; }
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
  <p>Upload a photo · Get the species from Fishial.ai</p>
</header>

<div class="upload-zone" id="dropZone">
  <h2>Drop a photo here</h2>
  <p>or click to choose an image</p>
  <input type="file" id="fileInput" accept="image/*">
</div>

<div id="results"></div>

<div class="actions" id="actions" style="display:none;">
  <button onclick="reset()">Identify another photo</button>
</div>

<footer>FinFindr · Powered by Fishial.ai</footer>

<script>
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const results = document.getElementById("results");
const actions = document.getElementById("actions");

dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("dragover"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
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
      <h3 class="identifying">Identifying with Fishial…</h3>
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
    const data = await resp.json();

    if (data.error) {
      card.querySelector("h3").textContent = "Identification failed";
      card.querySelector("h3").classList.remove("identifying");
      card.querySelector(".error").textContent = data.error;
      return;
    }

    const fish = data.results && data.results[0];
    if (!fish || !fish.species || !fish.species.length) {
      card.querySelector("h3").textContent = "No fish detected";
      card.querySelector("h3").classList.remove("identifying");
      return;
    }

    const top = fish.species[0];
    card.querySelector("h3").textContent = top.common_name || top.scientific_name || "Unknown";
    card.querySelector("h3").classList.remove("identifying");
    card.querySelector(".sci").textContent = top.scientific_name || "";
    if (top.accuracy) {
      card.querySelector(".confidence").textContent = Math.round(top.accuracy * 100) + "% confidence";
    }
  } catch (err) {
    card.querySelector("h3").textContent = "Error";
    card.querySelector("h3").classList.remove("identifying");
    card.querySelector(".error").textContent = "Could not reach the server.";
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
    resp = requests.post(AUTH_URL, json={
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET
    })
    resp.raise_for_status()
    return resp.json()["access_token"]

def get_checksum(data: bytes) -> str:
    md5 = hashlib.md5(data).digest()
    return base64.b64encode(md5).decode("utf-8")

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/identify", methods=["POST"])
def identify():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image_bytes = file.read()
    filename = file.filename or "photo.jpg"
    content_type = file.content_type or "image/jpeg"

    try:
        token = get_access_token()
        checksum = get_checksum(image_bytes)

        # 1. Request upload URL
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        upload_payload = {
            "blob": {
                "filename": filename,
                "content_type": content_type,
                "byte_size": len(image_bytes),
                "checksum": checksum
            }
        }
        resp = requests.post(UPLOAD_URL, json=upload_payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        signed_id = data.get("signed-id")
        direct = data.get("direct-upload", {})
        upload_url = direct.get("url")
        content_disposition = direct.get("headers", {}).get("Content-Disposition", "")

        if not signed_id or not upload_url:
            return jsonify({"error": "Failed to get upload URL from Fishial"}), 500

        # 2. Upload the image
        put_headers = {
            "Content-Disposition": content_disposition,
            "Content-Md5": checksum,
            "Content-Type": ""
        }
        put_resp = requests.put(upload_url, data=image_bytes, headers=put_headers)
        put_resp.raise_for_status()

        # 3. Request recognition
        rec_url = f"{RECOGNITION_URL}?q={signed_id}"
        rec_resp = requests.get(rec_url, headers={"Authorization": f"Bearer {token}"})
        rec_resp.raise_for_status()
        result = rec_resp.json()

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\nFinFindr is running on port {port}")
    print(f"Local:  http://localhost:{port}")
    print("Press Ctrl+C to stop.\n")
    app.run(host="0.0.0.0", port=port, debug=False)
