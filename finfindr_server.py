#!/usr/bin/env python3
"""
FinFindr – Marine species identification
"""

from flask import Flask, request, jsonify, render_template_string
import requests
import base64
import os

app = Flask(__name__)

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
  .upload-zone { width: 100%; max-width: 560px; margin: 0 auto 1.5rem; border: 2px dashed rgba(255,255,255,0.3); border-radius: 20px; padding: 2rem; text-align: center; background: rgba(255,255,255,0.05); }
  #results { max-width: 800px; margin: 0 auto; }
  .result-card { background: rgba(0,0,0,0.35); border-radius: 16px; margin-bottom: 1.25rem; border: 1px solid rgba(255,255,255,0.1); overflow: hidden; }
  .result-card img { width: 100%; max-height: 280px; object-fit: cover; }
  .result-body { padding: 1rem; }
  .result-body h3 { margin-bottom: 0.3rem; font-size: 1.35rem; }
  .sci { opacity: 0.7; font-style: italic; margin-bottom: 0.4rem; }
  .confidence { display: inline-block; background: var(--teal); padding: 0.25rem 0.7rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600; margin: 0.4rem 0; }
  .error { color: #fb7185; }
  button { background: var(--teal); color: white; border: none; padding: 0.75rem 1.5rem; border-radius: 12px; cursor: pointer; font-weight: 600; font-size: 1rem; }
  footer { text-align: center; margin-top: 2.5rem; opacity: 0.5; font-size: 0.8rem; }
</style>
</head>
<body>
<header>
  <h1>FinFindr</h1>
  <p>Upload a photo · Get the species</p>
</header>

<div class="upload-zone" id="uploadZone">
  <h2>Choose a photo</h2>
  <input type="file" id="fileInput" accept="image/*" style="margin:1rem 0; font-size:1rem;">
  <br>
  <button type="button" id="identifyBtn" disabled>Identify Photo</button>
</div>

<div id="results"></div>

<footer>FinFindr</footer>

<script>
const fileInput = document.getElementById("fileInput");
const identifyBtn = document.getElementById("identifyBtn");
const results = document.getElementById("results");
const uploadZone = document.getElementById("uploadZone");

fileInput.addEventListener("change", function() {
  if (this.files && this.files[0]) {
    identifyBtn.disabled = false;
    identifyBtn.textContent = "Identify: " + this.files[0].name;
  } else {
    identifyBtn.disabled = true;
    identifyBtn.textContent = "Identify Photo";
  }
});

identifyBtn.addEventListener("click", function() {
  if (!fileInput.files || !fileInput.files[0]) return;
  handleFile(fileInput.files[0]);
});

async function handleFile(file) {
  uploadZone.style.display = "none";
  results.innerHTML = "";

  const card = document.createElement("div");
  card.className = "result-card";
  card.innerHTML = `
    <img class="preview" alt="Photo" style="width:100%; max-height:280px; object-fit:cover;">
    <div class="result-body">
      <h3>Identifying…</h3>
      <div class="sci"></div>
      <div class="confidence"></div>
      <div class="error"></div>
    </div>
  `;
  results.appendChild(card);

  const reader = new FileReader();
  reader.onload = (e) => card.querySelector(".preview").src = e.target.result;
  reader.readAsDataURL(file);

  const formData = new FormData();
  formData.append("image", file);

  try {
    const resp = await fetch("/identify", { method: "POST", body: formData });
    const data = await resp.json();

    if (data.error) {
      card.querySelector("h3").textContent = "Error";
      card.querySelector(".error").textContent = data.error;
      return;
    }

    const text = data.result_text || "No response";
    const lines = text.split("\\n").map(l => l.trim()).filter(Boolean);
    let common = "", scientific = "", group = "", altGroup = "", confidence = "", notes = "";
    lines.forEach(line => {
      const lower = line.toLowerCase();
      if (lower.startsWith("common name:")) common = line.split(":").slice(1).join(":").trim();
      else if (lower.startsWith("scientific name:")) scientific = line.split(":").slice(1).join(":").trim();
      else if (lower.startsWith("group:")) group = line.split(":").slice(1).join(":").trim();
      else if (lower.startsWith("alternative group considered:")) altGroup = line.split(":").slice(1).join(":").trim();
      else if (lower.startsWith("confidence:")) confidence = line.split(":").slice(1).join(":").trim();
      else if (lower.startsWith("notes:")) notes = line.split(":").slice(1).join(":").trim();
    });

    card.querySelector("h3").textContent = common || "Result";
    card.querySelector(".sci").textContent = scientific;
    if (group) {
      const g = document.createElement("div");
      g.style.margin = "0.3rem 0";
      g.textContent = "Group: " + group;
      card.querySelector(".result-body").appendChild(g);
    }
    if (altGroup && altGroup.toLowerCase() !== "none") {
      const a = document.createElement("div");
      a.style.margin = "0.2rem 0";
      a.style.opacity = "0.75";
      a.style.fontSize = "0.9rem";
      a.textContent = "Also considered: " + altGroup;
      card.querySelector(".result-body").appendChild(a);
    }
    if (confidence) card.querySelector(".confidence").textContent = confidence + " confidence";
    if (notes) {
      const n = document.createElement("div");
      n.style.marginTop = "0.5rem";
      n.textContent = notes;
      card.querySelector(".result-body").appendChild(n);
    }
    if (!common && !scientific) card.querySelector(".sci").textContent = text;

  } catch (err) {
    card.querySelector("h3").textContent = "Error";
    card.querySelector(".error").textContent = err.message || "Something went wrong";
  }
}
</script>
</body>
</html>
"""

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
        return jsonify({"error": "XAI_API_KEY not set"}), 500

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{content_type};base64,{b64}"

    prompt = (
        "You are a marine biologist specializing in the Sea of Cortez. "
        "This photo was taken near La Paz, Mexico.\\n\\n"
        "Examine the animal carefully. First note the overall body plan "
        "(arms/rays, fins, rounded spiny body, elongated form, hard shell, soft body, etc.).\\n\\n"
        "Then consider the two most likely groups this could belong to. "
        "Do not lock in the first group — if features could fit either, weigh both.\\n"
        "Example: an elongated animal could be an eel or a sea snake; "
        "a spiny animal could be a starfish or a sea urchin. "
        "Choose the species that best matches the visible features, even if it is not in the first group you thought of.\\n\\n"
        "Key rules:\\n"
        "- Arms or rays from a central disc → starfish (not sea urchin).\\n"
        "- Clearly rounded/globular body with long spines and no arms → sea urchin.\\n"
        "- Prefer Sea of Cortez species when ambiguous.\\n\\n"
        "Reply in this exact format:\\n"
        "Common name: <name>\\n"
        "Scientific name: <name>\\n"
        "Group: <group>\\n"
        "Alternative group considered: <other possible group or none>\\n"
        "Confidence: <high / medium / low>\\n"
        "Notes: <one cool fact about the species>\\n"
        "If still unsure, list the closest alternatives."
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
            headers={"Authorization": f"Bearer {xai_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=60
        )
        resp.raise_for_status()
        data = resp.json()

        text = ""
        if "output" in data and data["output"]:
            for item in data["output"]:
                if item.get("type") == "message":
                    for c in item.get("content", []):
                        if c.get("type") == "output_text":
                            text += c.get("text", "")
        if not text and "choices" in data:
            text = data["choices"][0].get("message", {}).get("content", "")

        return jsonify({"result_text": text or "No response"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
