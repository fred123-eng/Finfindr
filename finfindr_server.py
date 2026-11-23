#!/usr/bin/env python3
"""
FinFindr – Marine species identification
"""

import base64
import fcntl
import logging
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import requests
from flask import Flask, jsonify, render_template_string, request
from requests.adapters import HTTPAdapter
from PIL import Image, ImageOps
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
# Phone photos are often 8–15MB; reject anything larger before it ties up the worker.
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
# Cap decoded pixels so a huge upload cannot exhaust a small instance.
Image.MAX_IMAGE_PIXELS = 50_000_000

MIN_DIMENSION = 32
DEFAULT_MAX_IMAGE_PX = 1536
JPEG_QUALITY = 85
# grok-4.7 is the current flagship and accepts image input (text + image → text).
# https://docs.x.ai/docs/models
DEFAULT_XAI_MODEL = "grok-4.7"

# One connection pool for xAI and iNaturalist so repeat calls reuse TLS.
http = requests.Session()
_http_adapter = HTTPAdapter(pool_connections=8, pool_maxsize=16)
http.mount("https://", _http_adapter)
http.mount("http://", _http_adapter)

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
  body { font-family: system-ui, -apple-system, sans-serif; background: linear-gradient(160deg, var(--deep) 0%, #134e4a 100%); color: var(--white); min-height: 100vh; padding: 1.5rem; overflow-x: hidden; }
  header { text-align: center; margin-bottom: 1.5rem; }
  header h1 { font-size: 2.6rem; background: linear-gradient(90deg, var(--aqua), #67e8f9); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  header p { opacity: 0.8; margin-top: 0.3rem; }
  .upload-zone { width: 100%; max-width: 560px; margin: 0 auto 1.5rem; border: 2px dashed rgba(255,255,255,0.3); border-radius: 20px; padding: 2rem; text-align: center; background: rgba(255,255,255,0.05); transition: border-color 0.15s ease, background 0.15s ease; }
  .upload-zone.dragover { border-color: var(--aqua); background: rgba(34, 211, 238, 0.16); }
  .hint { opacity: 0.8; margin-top: 0.35rem; }
  .upload-zone.dragover .hint { color: var(--aqua); font-weight: 600; }
  .region-panel { width: 100%; max-width: 560px; margin: 0 auto 1rem; text-align: left; }
  .region-panel label { display: block; font-weight: 700; margin: 0.55rem 0 0.35rem; }
  #regionSelect, #speciesListSelect, #modelInput { width: 100%; max-width: 100%; font-size: 16px; line-height: 1.3; min-height: 44px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.35); background: rgba(0,0,0,0.28); color: var(--white); padding: 0.45rem 0.7rem; color-scheme: dark; }
  .location-status { min-height: 1.25rem; margin-top: 0.45rem; font-size: 0.92rem; opacity: 0.9; overflow-wrap: anywhere; }
  .location-status:empty { display: none; }
  .test-options { margin-top: 0.75rem; font-size: 0.85rem; opacity: 0.8; }
  .test-options[hidden] { display: none; }
  .test-options summary { cursor: pointer; min-height: 44px; display: flex; align-items: center; }
  .region-line { margin: 0.15rem 0 0.45rem; color: var(--aqua); font-weight: 600; overflow-wrap: anywhere; }
  #fileInput { display: block; width: 100%; max-width: 100%; margin: 1rem auto; font-size: 1rem; color: var(--white); }
  #results { max-width: 800px; margin: 0 auto; }
  .result-card { background: rgba(0,0,0,0.35); border-radius: 16px; margin-bottom: 1.25rem; border: 1px solid rgba(255,255,255,0.1); overflow: hidden; }
  .result-card img { width: 100%; max-height: 280px; object-fit: cover; display: block; background: rgba(0,0,0,0.25); }
  .result-body { padding: 1rem; }
  .result-body h3 { margin-bottom: 0.3rem; font-size: 1.35rem; overflow-wrap: anywhere; }
  .scientific { margin: 0.45rem 0 0.7rem; padding: 0.55rem 0.75rem; border-radius: 12px; background: rgba(34, 211, 238, 0.12); border: 1px solid rgba(34, 211, 238, 0.45); }
  .scientific[hidden] { display: none; }
  .sci-label { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: var(--aqua); margin-bottom: 0.15rem; }
  .sci-label[hidden] { display: none; }
  .sci { font-style: italic; font-weight: 700; font-size: 1.2rem; line-height: 1.35; color: var(--white); opacity: 1; overflow-wrap: anywhere; }
  .confidence { display: inline-block; background: var(--teal); padding: 0.25rem 0.7rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600; margin: 0.4rem 0; }
  .confidence:empty { display: none; }
  .error { color: #fb7185; overflow-wrap: anywhere; }
  .client-error { margin-top: 0.85rem; }
  .spinner { width: 44px; height: 44px; margin: 0.15rem auto 0.85rem; border-radius: 50%; border: 4px solid rgba(255,255,255,0.22); border-top-color: var(--aqua); animation: spin 0.75s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  @media (prefers-reduced-motion: reduce) {
    .spinner { animation: none; border-color: var(--aqua); }
  }
  .is-loading h3 { color: var(--aqua); }
  .result-actions { padding: 0 1rem 1rem; }
  button { background: var(--teal); color: white; border: 2px solid transparent; padding: 0.8rem 1.25rem; border-radius: 12px; cursor: pointer; font-weight: 600; font-size: 1rem; min-height: 44px; max-width: 100%; white-space: normal; overflow-wrap: anywhere; -webkit-tap-highlight-color: transparent; }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  button:focus-visible, #fileInput:focus-visible { outline: 2px solid var(--aqua); outline-offset: 2px; }
  .retry-btn { width: 100%; }
  footer { text-align: center; margin-top: 2.5rem; opacity: 0.5; font-size: 0.8rem; }
  @media (max-width: 640px) {
    body { padding: 1rem 0.75rem; }
    header h1 { font-size: 2rem; }
    .upload-zone { padding: 1.25rem 0.9rem; }
    #identifyBtn, .retry-btn { width: 100%; }
    .result-card img { max-height: 220px; }
    .sci { font-size: 1.05rem; }
    .region-panel { text-align: left; }
  }
</style>
</head>
<body>
<header>
  <h1>FinFindr</h1>
  <p>Upload a photo · Get the species</p>
</header>

<div class="region-panel">
  <label for="regionSelect">Region</label>
  <select id="regionSelect">
    <option value="sea-of-cortez">Sea of Cortez / La Paz</option>
    <option value="caribbean">Caribbean</option>
    <option value="hawaii">Hawaii</option>
    <option value="red-sea">Red Sea</option>
    <option value="great-barrier-reef">Great Barrier Reef</option>
    <option value="mediterranean">Mediterranean</option>
    <option value="other">Other</option>
    <option value="my-location">Use my location</option>
  </select>
  <p id="locationStatus" class="location-status" role="status"></p>
  <details id="speciesListOptions" class="test-options" hidden>
    <summary>Testing: species list size</summary>
    <label for="speciesListSelect">List sent with the photo</label>
    <select id="speciesListSelect">
      <option value="">Server default</option>
      <option value="short">Short — top 250</option>
      <option value="full">Full — all recorded</option>
    </select>
    <label for="modelInput">Model</label>
    <input id="modelInput" type="text" placeholder="Server default" autocomplete="off" spellcheck="false">
  </details>
</div>

<div class="upload-zone" id="uploadZone">
  <h2>Choose a photo</h2>
  <p class="hint" id="dropHint">Drop a photo here, or select one</p>
  <input type="file" id="fileInput" accept="image/*" aria-label="Photo to identify">
  <button type="button" id="identifyBtn" disabled>Identify Photo</button>
  <p id="clientError" class="error client-error" role="alert" hidden></p>
</div>

<div id="results" aria-live="polite"></div>

<footer>FinFindr</footer>

<script>
const fileInput = document.getElementById("fileInput");
const identifyBtn = document.getElementById("identifyBtn");
const results = document.getElementById("results");
const uploadZone = document.getElementById("uploadZone");
const clientError = document.getElementById("clientError");
const dropHint = document.getElementById("dropHint");
const regionSelect = document.getElementById("regionSelect");
const speciesListSelect = document.getElementById("speciesListSelect");
const modelInput = document.getElementById("modelInput");
const locationStatus = document.getElementById("locationStatus");
const MIN_DIMENSION = 32;
const MAX_BYTES = 20 * 1024 * 1024;
const REGION_KEY = "finfindr.region";
const LOCATION_KEY = "finfindr.location";
const LIST_KEY = "finfindr.speciesList";
let busy = false;
let validateToken = 0;
let dragDepth = 0;
let browserLat = null;
let browserLng = null;
let debugMode = false;

function resultLines(text) {
  const newline = String.fromCharCode(10);
  const cr = String.fromCharCode(13);
  const slashN = String.fromCharCode(92) + "n";
  let normalized = String(text || "").split(cr + newline).join(newline).split(cr).join(newline);
  if (normalized.indexOf(newline) === -1 && normalized.indexOf(slashN) !== -1) {
    normalized = normalized.split(slashN).join(newline);
  }
  return normalized.split(newline).map(function (line) { return line.trim(); }).filter(Boolean);
}

function setLocationStatus(message) {
  locationStatus.textContent = message || "";
}

function readSavedLocation() {
  try {
    const saved = JSON.parse(localStorage.getItem(LOCATION_KEY) || "null");
    if (!saved || typeof saved.lat !== "number" || typeof saved.lng !== "number") return null;
    return saved;
  } catch (err) {
    return null;
  }
}

function requestLocation() {
  const saved = readSavedLocation();
  if (saved) {
    browserLat = saved.lat;
    browserLng = saved.lng;
    setLocationStatus("Using your location (" + saved.lat.toFixed(2) + ", " + saved.lng.toFixed(2) + ").");
  }
  if (!navigator.geolocation) {
    setLocationStatus("Location is not available in this browser. Pick a region, or photos will be identified as Sea of Cortez / La Paz. Photo GPS is still used when the picture has it.");
    return;
  }
  if (browserLat == null) setLocationStatus("Finding your location…");
  navigator.geolocation.getCurrentPosition(function (pos) {
    browserLat = pos.coords.latitude;
    browserLng = pos.coords.longitude;
    try {
      localStorage.setItem(LOCATION_KEY, JSON.stringify({ lat: browserLat, lng: browserLng, ts: Date.now() }));
    } catch (err) {}
    setLocationStatus("Using your location (" + browserLat.toFixed(2) + ", " + browserLng.toFixed(2) + ").");
  }, function () {
    if (browserLat == null) {
      setLocationStatus("Could not read your location. Pick a region, or photos will be identified as Sea of Cortez / La Paz.");
    }
  }, { enableHighAccuracy: false, timeout: 8000, maximumAge: 300000 });
}

function restoreRegionChoice() {
  let stored = null;
  try { stored = localStorage.getItem(REGION_KEY); } catch (err) { stored = null; }
  if (stored && Array.prototype.some.call(regionSelect.options, function (option) { return option.value === stored; })) {
    regionSelect.value = stored;
  }
  const params = new URLSearchParams(window.location.search);
  debugMode = params.get("debug") === "1";
  const speciesListOptions = document.getElementById("speciesListOptions");
  if (debugMode && speciesListOptions) speciesListOptions.hidden = false;
  let listMode = null;
  if (debugMode) {
    try { listMode = sessionStorage.getItem(LIST_KEY); } catch (err) { listMode = null; }
  }
  const fromUrl = params.get("species_list");
  if (fromUrl === "short" || fromUrl === "full") listMode = fromUrl;
  if (listMode === "short" || listMode === "full") {
    speciesListSelect.value = listMode;
    if (debugMode && speciesListOptions) speciesListOptions.open = true;
  }
  if (regionSelect.value === "my-location") requestLocation();
}

regionSelect.addEventListener("change", function () {
  try { localStorage.setItem(REGION_KEY, regionSelect.value); } catch (err) {}
  if (regionSelect.value === "my-location") {
    requestLocation();
  } else {
    setLocationStatus("");
  }
});

speciesListSelect.addEventListener("change", function () {
  try {
    if (speciesListSelect.value === "short" || speciesListSelect.value === "full") {
      sessionStorage.setItem(LIST_KEY, speciesListSelect.value);
    } else {
      sessionStorage.removeItem(LIST_KEY);
    }
  } catch (err) {}
});

restoreRegionChoice();

function showRegion(card, label) {
  if (!label) return;
  let line = card.querySelector(".region-line");
  if (!line) {
    line = document.createElement("div");
    line.className = "region-line";
    const title = card.querySelector("h3");
    if (title) title.insertAdjacentElement("afterend", line);
    else card.querySelector(".result-body").prepend(line);
  }
  line.textContent = "Region: " + label;
}

function showClientError(message) {
  clientError.textContent = message;
  clientError.hidden = false;
}

function clearClientError() {
  clientError.textContent = "";
  clientError.hidden = true;
}

function friendlyMessage(message) {
  const fallback = "Something went wrong identifying this photo. Please try again.";
  if (typeof message !== "string") return fallback;
  const trimmed = message.trim();
  if (!trimmed || trimmed.length > 240) return fallback;
  if (/traceback|exception|https?:|client error|<!doctype|<html|errno|stack/i.test(trimmed)) return fallback;
  return trimmed;
}

function validatePhoto(file) {
  return new Promise((resolve) => {
    if (!file) {
      resolve("Please upload a valid photo");
      return;
    }
    if (file.type && !file.type.startsWith("image/")) {
      resolve("Please upload a valid photo");
      return;
    }
    if (!file.size) {
      resolve("Please upload a valid photo");
      return;
    }
    if (file.size > MAX_BYTES) {
      resolve("That photo is too large. Please choose a smaller one.");
      return;
    }
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      if (img.naturalWidth < MIN_DIMENSION || img.naturalHeight < MIN_DIMENSION) {
        resolve("Please upload a valid photo");
      } else {
        resolve("");
      }
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      resolve("Please upload a valid photo");
    };
    img.src = url;
  });
}

function resetUI() {
  busy = false;
  dragDepth = 0;
  results.innerHTML = "";
  fileInput.value = "";
  identifyBtn.disabled = true;
  identifyBtn.textContent = "Identify Photo";
  uploadZone.style.display = "";
  uploadZone.classList.remove("dragover");
  dropHint.textContent = "Drop a photo here, or select one";
  clearClientError();
  fileInput.focus();
}

function addRetry(card) {
  if (card.querySelector(".retry-btn")) return;
  const wrap = document.createElement("div");
  wrap.className = "result-actions";
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "retry-btn";
  btn.textContent = "Try another photo";
  btn.addEventListener("click", resetUI);
  wrap.appendChild(btn);
  card.appendChild(wrap);
}

function showError(card, message) {
  card.classList.remove("is-loading");
  card.classList.add("is-error");
  card.removeAttribute("aria-busy");
  const preview = card.querySelector(".preview");
  if (preview) preview.remove();
  const confidence = card.querySelector(".confidence");
  if (confidence) confidence.remove();
  const spinner = card.querySelector(".spinner");
  if (spinner) spinner.remove();
  const sci = card.querySelector(".sci");
  if (sci) sci.textContent = "";
  const sciWrap = card.querySelector(".scientific");
  if (sciWrap) sciWrap.hidden = true;
  const sciLabel = card.querySelector(".sci-label");
  if (sciLabel) sciLabel.hidden = true;
  card.querySelector("h3").textContent = "Couldn't identify";
  card.querySelector(".error").textContent = message;
  addRetry(card);
}

async function readJsonResponse(resp) {
  const fallback = "Something went wrong identifying this photo. Please try again.";
  let data = null;
  try {
    const raw = await resp.text();
    data = raw ? JSON.parse(raw) : null;
  } catch (err) {
    return { ok: false, message: fallback };
  }
  if (!resp.ok || !data || data.error) {
    return { ok: false, message: friendlyMessage(data && data.error), region_label: data && data.region_label };
  }
  return { ok: true, data: data, region_label: data.region_label };
}

fileInput.addEventListener("change", async function() {
  const token = ++validateToken;
  clearClientError();
  if (!this.files || !this.files[0]) {
    identifyBtn.disabled = true;
    identifyBtn.textContent = "Identify Photo";
    return;
  }
  const file = this.files[0];
  const problem = await validatePhoto(file);
  if (token !== validateToken) return;
  if (problem) {
    identifyBtn.disabled = true;
    identifyBtn.textContent = "Identify Photo";
    showClientError(problem);
    return;
  }
  identifyBtn.disabled = false;
  identifyBtn.textContent = "Identify: " + file.name;
});

identifyBtn.addEventListener("click", function() {
  if (!fileInput.files || !fileInput.files[0]) return;
  handleFile(fileInput.files[0]);
});

["dragenter", "dragover"].forEach((eventName) => {
  uploadZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    if (eventName === "dragenter") dragDepth += 1;
    uploadZone.classList.add("dragover");
    if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
  });
});

uploadZone.addEventListener("dragleave", (event) => {
  event.preventDefault();
  dragDepth -= 1;
  if (dragDepth <= 0) {
    dragDepth = 0;
    uploadZone.classList.remove("dragover");
  }
});

uploadZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dragDepth = 0;
  uploadZone.classList.remove("dragover");
  const files = event.dataTransfer && event.dataTransfer.files;
  if (!files || !files[0]) return;
  handleFile(files[0]);
});

document.addEventListener("dragover", (event) => event.preventDefault());
document.addEventListener("drop", (event) => event.preventDefault());

async function handleFile(file) {
  if (busy) return;
  busy = true;
  clearClientError();
  try {
    const problem = await validatePhoto(file);
    if (problem) {
      showClientError(problem);
      identifyBtn.disabled = true;
      identifyBtn.textContent = "Identify Photo";
      return;
    }

    uploadZone.style.display = "none";
    results.innerHTML = "";

    const card = document.createElement("div");
    card.className = "result-card is-loading";
    card.setAttribute("aria-busy", "true");
    card.innerHTML = `
      <img class="preview" alt="Photo being identified" style="width:100%; max-height:280px; object-fit:cover;">
      <div class="result-body">
        <div class="spinner" role="status" aria-label="Identifying photo"></div>
        <h3>Identifying…</h3>
        <div class="scientific" hidden>
          <div class="sci-label" hidden>Scientific name</div>
          <div class="sci" translate="no"></div>
        </div>
        <div class="confidence"></div>
        <div class="error"></div>
      </div>
    `;
    results.appendChild(card);

    const reader = new FileReader();
    reader.onload = (e) => {
      const preview = card.querySelector(".preview");
      if (preview) preview.src = e.target.result;
    };
    reader.readAsDataURL(file);

    const formData = new FormData();
    formData.append("image", file);
    formData.append("region", regionSelect.value);
    if (regionSelect.value === "my-location" && browserLat != null && browserLng != null) {
      formData.append("lat", String(browserLat));
      formData.append("lng", String(browserLng));
    }
    if (speciesListSelect.value === "short" || speciesListSelect.value === "full") {
      formData.append("species_list", speciesListSelect.value);
    }
    if (debugMode) {
      formData.append("debug", "1");
      const modelName = modelInput.value.trim();
      if (modelName) formData.append("model", modelName);
    }

    const resp = await fetch("/identify", { method: "POST", body: formData });
    const parsed = await readJsonResponse(resp);
    if (parsed.region_label) showRegion(card, parsed.region_label);
    if (!parsed.ok) {
      showError(card, parsed.message);
      return;
    }
    const data = parsed.data;
    card.classList.remove("is-loading");
    card.removeAttribute("aria-busy");
    const spinner = card.querySelector(".spinner");
    if (spinner) spinner.remove();

    const text = data.result_text || "No response";
    const lines = resultLines(text);
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

    const sciWrap = card.querySelector(".scientific");
    const sciLabel = card.querySelector(".sci-label");
    const sciEl = card.querySelector(".sci");
    if (sciLabel) sciLabel.hidden = !scientific;
    if (sciWrap) sciWrap.hidden = !(sciEl && sciEl.textContent.trim());

    const preview = card.querySelector(".preview");
    if (preview && common) preview.alt = common;
    addRetry(card);
  } catch (err) {
    const card = results.querySelector(".result-card");
    if (card) {
      showError(card, "Something went wrong identifying this photo. Please try again.");
    } else {
      uploadZone.style.display = "";
      showClientError("Something went wrong identifying this photo. Please try again.");
    }
  } finally {
    busy = false;
  }
}
</script>
</body>
</html>
"""


def _to_rgb(img):
    if img.mode == "RGB":
        return img
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    return img.convert("RGB")


def max_image_px():
    """Longest edge sent to the model. MAX_IMAGE_PX overrides the 1536 default."""
    raw = (os.environ.get("MAX_IMAGE_PX") or "").strip()
    if not raw:
        return DEFAULT_MAX_IMAGE_PX
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_IMAGE_PX
    if value < 64 or value > 4096:
        return DEFAULT_MAX_IMAGE_PX
    return value


def prepare_image(image_bytes):
    """Return (JPEG bytes, width, height), or None when the upload is not a photo."""
    if not image_bytes:
        app.logger.warning("Rejected empty upload")
        return None

    img = None
    try:
        img = Image.open(BytesIO(image_bytes))
        transposed = ImageOps.exif_transpose(img)
        if transposed is not None and transposed is not img:
            img.close()
            img = transposed
        img.load()

        width, height = img.size
        if width < MIN_DIMENSION or height < MIN_DIMENSION:
            app.logger.warning("Rejected tiny image %sx%s", width, height)
            return None

        rgb = _to_rgb(img)
        if rgb is not img:
            img.close()
            img = rgb

        limit = max_image_px()
        longest = max(img.size)
        if longest > limit:
            scale = limit / float(longest)
            new_size = (
                max(1, min(limit, int(round(img.size[0] * scale)))),
                max(1, min(limit, int(round(img.size[1] * scale)))),
            )
            resized = img.resize(new_size, Image.Resampling.LANCZOS)
            img.close()
            img = resized

        out = BytesIO()
        img.save(out, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        width, height = img.size
        return out.getvalue(), width, height
    except Exception as exc:
        app.logger.warning("Rejected unreadable image: %s", exc)
        return None
    finally:
        if img is not None:
            try:
                img.close()
            except Exception:
                pass



# Connect + read caps. A slow vision call already uses up to 60s, and the
# gunicorn worker timeout is 120s, so this lookup cannot be what kills the request.
INAT_TIMEOUT = (1.0, 2.0)
INAT_CACHE_TTL = 24 * 60 * 60
INAT_ERROR_TTL = 60
_INAT_URL = "https://api.inaturalist.org/v1/taxa"
_SPECIES_RANKS = {"species", "subspecies", "variety", "form"}
_cache_lock = threading.Lock()
_inat_cache = {}

# Sea of Cortez names keyed by exact scientific name. Values match iNaturalist's
# preferred English name, except Gymnothorax castaneus: that is the Panamic green
# moray, not the green moray (G. funebris). Phataria unifascialis is the Blue Sea
# Star (iNaturalist and Reef Life Survey); Tan sea star is the other regional name.
CURATED_COMMON_NAMES = {
    'phataria unifascialis': 'Blue Sea Star',
    'pharia pyramidata': 'Pyramid Sea Star',
    'heliaster kubiniji': 'Gulf Sun Star',
    'nidorellia armata': 'Chocolate Chip Sea Star',
    'pentaceraster cumingi': 'Panamic Cushion Star',
    'oreaster occidentalis': 'Panamic Cushion Star',
    'linckia columbiae': 'Tamarisk Sea Star',
    'mithrodia bradleyi': "Bradley's Sea Star",
    'diadema mexicanum': 'Needle Sea Urchin',
    'eucidaris thouarsii': 'slate pencil urchin',
    'toxopneustes roseus': 'Rose Flower Urchin',
    'tripneustes depressus': 'Sea Egg',
    'centrostephanus coronatus': 'Crowned sea urchin',
    'echinometra vanbrunti': 'Central American Rock-boring Urchin',
    'astropyga pulvinata': 'Pin Cushion Sea Urchin',
    'isostichopus fuscus': 'Brown Sea Cucumber',
    'euapta godeffroyi': "Lion's Paw Sea Cucumber",
    'holothuria inornata': 'Sandy Sea Cucumber',
    'rhincodon typus': 'Whale Shark',
    'mobula munkiana': "Munk's Pygmy Devil Ray",
    'mobula birostris': 'Oceanic Manta Ray',
    'mobula thurstoni': 'Bentfin Devil Ray',
    'mobula tarapacana': 'Sicklefin Devil Ray',
    'sphyrna lewini': 'Scalloped Hammerhead',
    'sphyrna corona': 'Scalloped Bonnethead',
    'triaenodon obesus': 'Whitetip Reef Shark',
    'carcharhinus limbatus': 'Common Blacktip Shark',
    'carcharhinus falciformis': 'Silky Shark',
    'galeocerdo cuvier': 'Tiger Shark',
    'ginglymostoma unami': 'Pacific Nurse Shark',
    'heterodontus francisci': 'Horn Shark',
    'mustelus lunulatus': 'Sicklefin Smooth-hound',
    'diodon holocanthus': 'Longspined Porcupinefish',
    'diodon hystrix': 'Spotted Porcupinefish',
    'arothron meleagris': 'Guineafowl Puffer',
    'arothron hispidus': 'White-spotted Puffer',
    'canthigaster punctatissima': 'Spotted Sharpnose Puffer',
    'sufflamen verres': 'Orangeside Triggerfish',
    'balistes polylepis': 'Finescale Triggerfish',
    'pseudobalistes naufragium': 'Stone Triggerfish',
    'cantherhines dumerilii': 'Barred Filefish',
    'aluterus scriptus': 'Scrawled Filefish',
    'melichthys niger': 'Black Triggerfish',
    'holacanthus passer': 'King Angelfish',
    'holacanthus clarionensis': 'Clarión Angelfish',
    'pomacanthus zonipectus': 'Cortez Angelfish',
    'abudefduf troschelii': 'Panamic Sergeant Major',
    'thalassoma lucasanum': 'Cortez Rainbow Wrasse',
    'bodianus diplotaenia': 'Mexican Hogfish',
    'scarus ghobban': 'Blue-barred Parrotfish',
    'scarus perrico': 'Bumphead Parrotfish',
    'scarus rubroviolaceus': 'Redlip Parrotfish',
    'nicholsina denticulata': 'Loosetooth Parrotfish',
    'johnrandallia nigrirostris': 'Black-nosed Butterflyfish',
    'chaetodon humeralis': 'Three-banded Butterflyfish',
    'prionurus punctatus': 'Razor Surgeonfish',
    'prionurus laticlavius': 'Razor Surgeonfish',
    'acanthurus xanthopterus': 'Yellowfin Surgeonfish',
    'microspathodon dorsalis': 'Giant Damselfish',
    'microspathodon bairdii': 'Bumphead Damselfish',
    'chromis atrilobata': 'Scissortail Chromis',
    'azurina atrilobata': 'Scissortail Chromis',
    'stegastes rectifraenum': 'Cortez Damselfish',
    'stegastes flavilatus': 'Beaubrummel',
    'fistularia commersonii': 'Blue-spotted Cornetfish',
    'aulostomus chinensis': 'Pacific Trumpetfish',
    'zanclus cornutus': 'Moorish Idol',
    'hippocampus ingens': 'Giant Seahorse',
    'gymnothorax castaneus': 'Panamic green moray',
    'gymnothorax dovii': 'Fine-spotted Moray',
    'gymnothorax funebris': 'Green Moray',
    'muraena lentiginosa': 'Jewel Moray',
    'gymnomuraena zebra': 'Zebra Moray',
    'echidna nocturna': 'Palenose Moray',
    'ostracion meleagris': 'Spotted Boxfish',
    'lutjanus argentiventris': 'Yellow Snapper',
    'lutjanus novemfasciatus': 'Pacific Cubera Snapper',
    'lutjanus viridis': 'Blue-and-gold Snapper',
    'lutjanus peru': 'South Pacific Red Snapper',
    'mycteroperca rosacea': 'Leopard Grouper',
    'mycteroperca jordani': 'Gulf Grouper',
    'epinephelus labriformis': 'Starry Grouper',
    'epinephelus analogus': 'Spotted Grouper',
    'paranthias colonus': 'Pacific Creolefish',
    'cephalopholis colonus': 'Pacific Creolefish',
    'alphestes immaculatus': 'Pacific Mutton Hamlet',
    'dermatolepis dermatolepis': 'Leather Bass',
    'seriola lalandi': 'Yellowtail Kingfish',
    'caranx sexfasciatus': 'Bigeye Trevally',
    'caranx caballus': 'Green Jack',
    'trachinotus rhodopus': 'Gafftopsail Pompano',
    'mugil curema': 'White Mullet',
    'kyphosus elegans': 'Cortez Sea Chub',
    'kyphosus vaigiensis': 'Brassy Drummer',
    'mulloidichthys dentatus': 'Mexican Goatfish',
    'haemulon sexfasciatum': 'Graybar Grunt',
    'haemulon maculicauda': 'Spottail Grunt',
    'haemulon flaviguttatum': 'Cortez Grunt',
    'anisotremus interruptus': 'Burrito Grunt',
    'anisotremus taeniatus': 'Panamic Porkfish',
    'hoplopagrus guentherii': 'Barred Pargo',
    'sphoeroides annulatus': 'Bullseye Puffer',
    'scorpaena mystes': 'Stone Scorpionfish',
    'cirrhitichthys oxycephalus': 'Pixy Hawkfish',
    'cirrhitus rivulatus': 'Giant Hawkfish',
    'plagiotremus azaleus': 'Sabertooth Blenny',
    'ophioblennius steindachneri': 'Panamic Fanged Blenny',
    'malacoctenus hubbsi': 'Redside Blenny',
    'rypticus bicolor': 'Mottled Soapfish',
    'chelonia mydas': 'Green Sea Turtle',
    'eretmochelys imbricata': 'Hawksbill Sea Turtle',
    'lepidochelys olivacea': 'Olive Ridley Sea Turtle',
    'caretta caretta': 'Loggerhead Sea Turtle',
    'dermochelys coriacea': 'Leatherback Sea Turtle',
    'zalophus californianus': 'California Sea Lion',
    'tursiops truncatus': 'Common Bottlenose Dolphin',
    'stenella longirostris': 'Spinner Dolphin',
    'octopus hubbsorum': "Hubb's Octopus",
    'octopus bimaculatus': "Verrill's Two-spot Octopus",
    'aplysia californica': 'California Seahare',
    'grapsus grapsus': 'Sally Lightfoot Crab',
    'callinectes bellicosus': 'Warrior Swimming Crab',
    'stenorhynchus debilis': 'Panamic Arrow Crab',
    'panulirus inflatus': 'Blue spiny lobster',
    'panulirus gracilis': 'Green Spiny Lobster',
    'panulirus interruptus': 'California Spiny Lobster',
    'hydrophis platurus': 'Yellow-bellied Sea Snake',
    'pelamis platura': 'Yellow-bellied Sea Snake',
    'titanostrombus galeatus': 'Eastern Pacific Giant Conch',
    'hypanus dipterurus': 'Diamond Stingray',
    'hypanus longus': 'Longtail Stingray',
    'dasyatis longa': 'Longtail Stingray',
    'urobatis halleri': "Haller's Round Ray",
    'aetobatus laticeps': 'Pacific Eagle Ray',
    'rhinoptera steindachneri': 'Golden Cownose Ray',
    'narcine entemedor': 'Giant Electric Ray',
    'rhinobatos productus': 'Shovelnose Guitarfish',
    'pseudobatos productus': 'Shovelnose Guitarfish',
    'zapteryx exasperata': 'Banded Guitarfish',
    'acanthaster planci': 'Long-spined Crown-of-Thorns',
}

_UNCERTAIN_NAME = re.compile(r"\b(?:cf\.?|aff\.?|spp?\.?|species|indet)\b|[?]", re.IGNORECASE)


def exact_scientific_name(raw):
    """Return Genus species [subspecies] or None when the name is not exact."""
    if not raw or not isinstance(raw, str):
        return None
    text = raw.strip().strip(" .;*_")
    text = text.replace("*", " ")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" .;,")
    if not text or _UNCERTAIN_NAME.search(text):
        return None
    parts = text.split()
    if len(parts) < 2:
        return None
    genus, epithet = parts[0], parts[1]
    if not re.fullmatch(r"[A-Za-z][A-Za-z-]*", genus):
        return None
    if not re.fullmatch(r"[A-Za-z][A-Za-z-]*", epithet):
        return None
    canonical = [genus[:1].upper() + genus[1:].lower(), epithet.lower()]
    if len(parts) >= 3 and re.fullmatch(r"[a-z][a-z-]*", parts[2]):
        canonical.append(parts[2].lower())
    return " ".join(canonical)


def _select_inat_taxon(results, scientific):
    target = scientific.lower()
    usable = [t for t in results or [] if isinstance(t, dict) and t.get("rank") in _SPECIES_RANKS]
    exact = [t for t in usable if (t.get("name") or "").lower() == target]
    for taxon in exact:
        if (taxon.get("preferred_common_name") or "").strip():
            return taxon
    if exact:
        return None
    synonyms = [
        t for t in usable
        if (t.get("matched_term") or "").lower() == target and (t.get("name") or "").lower() != target
    ]
    if len(synonyms) == 1 and (synonyms[0].get("preferred_common_name") or "").strip():
        return synonyms[0]
    return None


def _cache_get(key):
    now = time.monotonic()
    with _cache_lock:
        hit = _inat_cache.get(key)
        if not hit or hit[0] <= now:
            _inat_cache.pop(key, None)
            return None
        return hit
    return None


def _cache_put(key, value, ttl):
    with _cache_lock:
        if len(_inat_cache) > 512:
            _inat_cache.clear()
        _inat_cache[key] = (time.monotonic() + ttl, value)


def _fetch_inat_common_name_uncached(scientific):
    """Network lookup. Caller already missed the cache. Failures are cached briefly."""
    key = scientific.lower()
    try:
        resp = http.get(
            _INAT_URL,
            params={"q": scientific, "locale": "en", "per_page": 10},
            headers={
                "User-Agent": "FinFindr/1.0 (marine species common-name lookup)",
                "Accept": "application/json",
            },
            timeout=INAT_TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError):
        app.logger.warning("iNaturalist lookup failed for %s", scientific)
        _cache_put(key, None, INAT_ERROR_TTL)
        return None
    taxon = _select_inat_taxon(payload.get("results") if isinstance(payload, dict) else None, scientific)
    if not taxon:
        _cache_put(key, None, INAT_CACHE_TTL)
        return None
    common = (taxon.get("preferred_common_name") or "").strip()
    if not common:
        _cache_put(key, None, INAT_CACHE_TTL)
        return None
    _cache_put(key, common, INAT_CACHE_TTL)
    return common


def _listed_common_name(scientific, species):
    target = scientific.lower()
    for item in species or []:
        if not isinstance(item, dict):
            continue
        if (item.get("scientific") or "").lower() == target and (item.get("common") or "").strip():
            return item["common"].strip()
    return None


def resolve_common_names(scientific_names, species=None):
    """Curated, then the regional list, then cached iNaturalist, then parallel lookups."""
    found = {}
    pending = []
    seen_pending = set()
    for scientific in scientific_names:
        if not scientific or scientific in found:
            continue
        curated = CURATED_COMMON_NAMES.get(scientific.lower())
        if curated:
            found[scientific] = curated
            continue
        listed = _listed_common_name(scientific, species)
        if listed:
            _cache_put(scientific.lower(), listed, INAT_CACHE_TTL)
            found[scientific] = listed
            continue
        cached = _cache_get(scientific.lower())
        if cached is not None:
            found[scientific] = cached[1]
            continue
        if scientific.lower() not in seen_pending:
            seen_pending.add(scientific.lower())
            pending.append(scientific)
    if not pending:
        return found
    workers = min(8, len(pending))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_fetch_inat_common_name_uncached, name) for name in pending]
        for name, future in zip(pending, futures):
            try:
                found[name] = future.result()
            except Exception:
                app.logger.warning("common-name lookup failed for %s", name, exc_info=True)
                found[name] = None
    return found


def standard_common_name(scientific, species=None):
    """Authoritative common name for an exact scientific name, or None."""
    if not scientific:
        return None
    return resolve_common_names([scientific], species).get(scientific)


def _result_lines(text):
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if "\n" not in normalized and "\\n" in normalized:
        normalized = normalized.replace("\\n", "\n")
    return normalized


def _field_value(text, label):
    prefix = label.lower() + ":"
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.lower().startswith(prefix):
            return stripped.split(":", 1)[1].strip()
    return ""


def _set_field(text, label, value):
    prefix = label.lower() + ":"
    found = False
    out = []
    for line in text.split("\n"):
        if line.strip().lower().startswith(prefix):
            if not found:
                out.append(f"{label}: {value}")
                found = True
        else:
            out.append(line)
    if not found:
        out.insert(0, f"{label}: {value}")
    return "\n".join(out)


def apply_standard_names(result_text, species=None):
    """Replace the model's common name when the scientific name resolves."""
    if not result_text:
        return result_text
    text = _result_lines(result_text)
    scientific_raw = _field_value(text, "Scientific name")
    canonical = exact_scientific_name(scientific_raw)
    if not canonical:
        return text
    text = _set_field(text, "Scientific name", canonical)
    common = standard_common_name(canonical, species)
    if not common:
        return text
    app.logger.info("Standard common name for %s: %s", canonical, common)
    return _set_field(text, "Common name", common)

# Regions are data. Adding a coast means adding a place id or a bounding box here,
# not a new code path. place ids are iNaturalist polygons; boxes cover seas that
# do not have a reliable water-body place.
REGIONS = (
    {
        "id": "sea-of-cortez",
        "label": "Sea of Cortez / La Paz",
        "locality": "the Sea of Cortez (Gulf of California), near La Paz, Mexico",
        "place_id": 122897,
        "swlat": 22.5,
        "swlng": -115.5,
        "nelat": 32.2,
        "nelng": -105.5,
        "curated_fallback": True,
        "lookalikes": True,
    },
    {
        "id": "caribbean",
        "label": "Caribbean",
        "locality": "the Caribbean Sea",
        "swlat": 9.0,
        "swlng": -89.0,
        "nelat": 27.5,
        "nelng": -59.0,
    },
    {
        "id": "hawaii",
        "label": "Hawaii",
        "locality": "Hawaii",
        "place_id": 11,
        "swlat": 18.8,
        "swlng": -178.5,
        "nelat": 28.6,
        "nelng": -154.7,
    },
    {
        "id": "red-sea",
        "label": "Red Sea",
        "locality": "the Red Sea",
        "swlat": 12.0,
        "swlng": 32.0,
        "nelat": 30.0,
        "nelng": 44.0,
    },
    {
        "id": "great-barrier-reef",
        "label": "Great Barrier Reef",
        "locality": "the Great Barrier Reef, Australia",
        "place_id": 50423,
        "swlat": -24.5,
        "swlng": 142.5,
        "nelat": -10.0,
        "nelng": 154.5,
    },
    {
        "id": "mediterranean",
        "label": "Mediterranean",
        "locality": "the Mediterranean Sea",
        "swlat": 30.0,
        "swlng": -6.0,
        "nelat": 46.0,
        "nelng": 36.5,
    },
    {
        "id": "other",
        "label": "Other",
        "locality": "an unspecified coast or ocean",
    },
    {
        "id": "my-location",
        "label": "Use my location",
        "locality": "the photographer's location",
    },
)
_REGIONS_BY_ID = {region["id"]: region for region in REGIONS}

# Actinopterygii, Mollusca, and the catch-all Animalia iconic taxon (sharks,
# rays, corals, echinoderms, crustaceans, and other invertebrates). Birds,
# insects, reptiles, and mammals are separate iconic taxa, so terrestrial
# groups in those classes stay out. The exclusions drop land snails, land
# hermit crabs, land crabs, centipedes, and millipedes that still share an
# included iconic taxon.
MARINE_ICONIC = "Actinopterygii,Mollusca,Animalia"
EXCLUDED_TAXA = "47485,54927,51731,51074,49556,47735"
# Marine groups whose iconic taxon is Reptilia or Mammalia.
SUPPLEMENT_TAXA = (
    372234,  # sea turtles
    152871,  # whales and dolphins
    492346,  # sea snakes
    41736,   # sea lions and fur seals
    41687,   # true seals
    46306,   # manatees and dugongs
)
SPECIES_COUNTS_URL = "https://api.inaturalist.org/v1/observations/species_counts"
_SPECIES_HEADERS = {
    "User-Agent": "FinFindr/1.0 (regional marine species list)",
    "Accept": "application/json",
}
SHORT_LIMIT = 250
# Short lists are per-group quotas, not the global top 250. Observation counts
# are dominated by fishes, which previously pushed sea snakes, urchins, and
# other non-fish groups out of the hint list. Quotas sum to SHORT_LIMIT.
# Non-fish groups are listed first so a token trim keeps them.
SHORT_GROUPS = (
    {"label": "Marine reptiles", "taxon_ids": (372234, 492346), "quota": 20},
    {"label": "Echinoderms", "taxon_ids": (47549,), "quota": 30},
    {"label": "Sharks and rays", "taxon_ids": (196614,), "quota": 25},
    {"label": "Marine mammals", "taxon_ids": (152871, 41736, 41687, 46306), "quota": 20},
    {"label": "Cnidarians", "taxon_ids": (47534,), "quota": 20},
    {"label": "Crustaceans", "taxon_ids": (47186,), "quota": 20, "exclude": "51731,51074"},
    {"label": "Molluscs", "taxon_ids": (47115,), "quota": 25, "exclude": "47485,54927"},
    {"label": "Sponges", "taxon_ids": (48824,), "quota": 10},
    {"label": "Ray-finned fishes", "taxon_ids": (47178,), "quota": 80},
)
SPECIES_PAGE_SIZE = 500
LOCATION_RADIUS_KM = 100
# Leave room for the image and a 256k context window, and stay under the
# higher-priced band that starts at 128k tokens.
MAX_PROMPT_TEXT_TOKENS = 100000
XAI_TEMPERATURE = 0
NL = chr(10)
_SPECIES_LOCK = threading.Lock()
_SPECIES_CACHE = {}
_SPECIES_INFLIGHT = {}

# Feature pairs that were confused in Sea of Cortez tests. Each note names the
# features that select either animal, so mentioning one species does not pull
# the other identification toward it.
CORTEZ_LOOKALIKE_LINES = (
    "Regional look-alikes in the eastern Pacific, including the Sea of Cortez. Decide from the features below, not from which name is more familiar:",
    "- Giant hawkfish (Cirrhitus rivulatus) or a grouper: tufts of cirri on the dorsal-spine tips, thick unbranched lower pectoral rays, and olive bars made of maze-like golden-brown marks edged in black and a thin blue line mean the giant hawkfish. No dorsal-spine cirri, a larger mouth, three spines on the gill cover, and spots or blotches without those blue-edged maze bars mean a grouper. A stout body alone does not decide.",
    "- Blue sea star (Phataria unifascialis), Panamic cushion star (Pentaceraster cumingi), or chocolate-chip star (Nidorellia armata): a small disc with long slender cylindrical arms and no large conical spines means Phataria unifascialis, even when the color is tan, orange, or blue. A thick inflated body with short broad arms and large conical knobs means a cushion star, not Phataria. Among cushion stars, a pale body with very short, almost webbed arms and evenly sized dark-brown spines means Nidorellia armata. A larger cushion star whose knobs are red or orange and often linked in a network means Pentaceraster cumingi. Do not call a slender-armed star a cushion star, and do not call a knobby inflated star Phataria.",
    "- Cortez rainbow wrasse (Thalassoma lucasanum) or Mexican hogfish (Bodianus diplotaenia): a small cigar-shaped fish with horizontal yellow and red stripes, or a blue head with a broad yellow saddle just behind it, is the rainbow wrasse. A pointed hog-like snout with two dark side stripes on a yellow or reddish body, a yellow tail on the reddish fish, or a forehead hump and a yellow mid-body bar on a large fish, is the Mexican hogfish. Yellow with red stripes is not the hogfish. Two dark stripes and a hog snout are not the rainbow wrasse.",
    "- Bullseye puffer (Sphoeroides annulatus) or a porcupinefish (Diodon): long spines that can stand erect mean a porcupinefish, even when the body is spotted. Only short prickles, a more elongated body, and a double concentric oval on the back with narrow yellowish lines mean the bullseye puffer. Spots without long spines are not a porcupinefish. Long spines are not the bullseye puffer. A puffer that lacks the concentric oval is a different puffer, not Sphoeroides annulatus.",
)
EASTERN_PACIFIC_BBOX = (-6.0, -118.0, 33.0, -77.0)


def resolve_region(raw):
    key = (raw or "").strip().lower()
    return _REGIONS_BY_ID.get(key) or _REGIONS_BY_ID["sea-of-cortez"]


def resolve_species_mode(raw=None):
    """short is the default. An explicit short|full value wins over the env var."""
    text = (raw or "").strip().lower()
    if text in ("short", "full"):
        return text
    env = (os.environ.get("SPECIES_LIST_MODE") or "short").strip().lower()
    if env in ("short", "full"):
        return env
    return "short"


def _point_in_bbox(lat, lng, swlat, swlng, nelat, nelng):
    if lat is None or lng is None:
        return False
    return swlat <= lat <= nelat and swlng <= lng <= nelng


def region_matching_point(lat, lng):
    matches = []
    for region in REGIONS:
        if "swlat" not in region:
            continue
        if _point_in_bbox(lat, lng, region["swlat"], region["swlng"], region["nelat"], region["nelng"]):
            area = (region["nelat"] - region["swlat"]) * (region["nelng"] - region["swlng"])
            matches.append((abs(area), region))
    if not matches:
        return None
    matches.sort(key=lambda item: item[0])
    return matches[0][1]


def cortez_lookalikes_apply(region, lat, lng):
    if region.get("lookalikes"):
        return True
    if region["id"] not in ("other", "my-location") or lat is None:
        return False
    swlat, swlng, nelat, nelng = EASTERN_PACIFIC_BBOX
    return _point_in_bbox(lat, lng, swlat, swlng, nelat, nelng)


def species_query_for(region, lat, lng):
    if region["id"] in ("other", "my-location"):
        if lat is None or lng is None:
            return None
        lat_r = round(float(lat), 1)
        lng_r = round(float(lng), 1)
        match = region_matching_point(lat, lng)
        return {
            "key": f"pt:{lat_r:.1f}:{lng_r:.1f}:{LOCATION_RADIUS_KM}",
            "lat": lat_r,
            "lng": lng_r,
            "radius": LOCATION_RADIUS_KM,
            "curated_fallback": bool(match and match.get("curated_fallback")),
        }
    if region.get("place_id"):
        return {
            "key": f"place:{region['place_id']}",
            "place_id": region["place_id"],
            "curated_fallback": bool(region.get("curated_fallback")),
        }
    if "swlat" in region:
        return {
            "key": "bbox:{swlat:.3f}:{swlng:.3f}:{nelat:.3f}:{nelng:.3f}".format(**region),
            "swlat": region["swlat"],
            "swlng": region["swlng"],
            "nelat": region["nelat"],
            "nelng": region["nelng"],
            "curated_fallback": False,
        }
    return None


def apply_location_fallback(region, lat, lng):
    """Use the default region when Use my location produced no coordinates."""
    if region["id"] == "my-location" and (lat is None or lng is None):
        return _REGIONS_BY_ID["sea-of-cortez"], True
    return region, False


def display_region_label(region, lat, lng, location_fallback=False):
    if location_fallback:
        return "Sea of Cortez / La Paz (location unavailable)"
    if region["id"] == "my-location":
        if lat is None:
            return "Use my location"
        match = region_matching_point(lat, lng)
        if match:
            return f"Near you · {match['label']}"
        return f"Near {lat:.2f}, {lng:.2f}"
    if region["id"] == "other":
        if lat is None:
            return "Other"
        match = region_matching_point(lat, lng)
        if match:
            return f"Other · near {match['label']}"
        return f"Other · {lat:.2f}, {lng:.2f}"
    return region["label"]


def locality_text(region, lat, lng, coord_source):
    if region["id"] in ("my-location", "other"):
        if lat is None:
            if region["id"] == "my-location":
                return "an unknown spot (location was requested, but no coordinates were available)"
            return "an unspecified coast or ocean"
        match = region_matching_point(lat, lng)
        where = f"{lat:.4f}, {lng:.4f}"
        if coord_source == "exif":
            prefix = f"the photo's GPS position ({where})"
        elif coord_source == "browser":
            prefix = f"the photographer's current location ({where})"
        else:
            prefix = f"coordinates {where}"
        if match:
            return f"{match['locality']}, based on {prefix}"
        return prefix
    text = region["locality"]
    if lat is not None and "swlat" in region:
        inside = _point_in_bbox(lat, lng, region["swlat"], region["swlng"], region["nelat"], region["nelng"])
        source = "The photo's GPS" if coord_source == "exif" else "A reported position"
        if inside:
            text += f". {source} is {lat:.4f}, {lng:.4f}, inside this region"
        else:
            text += (
                f". {source} is {lat:.4f}, {lng:.4f}, which is outside this region. "
                "Treat the selected region as the locality"
            )
    return text


def _exif_text(value):
    if isinstance(value, bytes):
        value = value.decode("ascii", "ignore")
    return str(value or "").strip().upper()


def _rational_to_float(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    numerator = getattr(value, "numerator", None)
    denominator = getattr(value, "denominator", None)
    if numerator is not None and denominator:
        return float(numerator) / float(denominator)
    if isinstance(value, tuple) and len(value) == 2 and not isinstance(value[0], (tuple, list)):
        try:
            return float(value[0]) / float(value[1] or 1)
        except (TypeError, ValueError, ZeroDivisionError):
            return None
    return None


def _dms_to_degrees(values):
    if not values or not isinstance(values, (tuple, list)):
        return None
    parts = []
    for value in list(values)[:3]:
        number = _rational_to_float(value)
        if number is None:
            return None
        parts.append(number)
    while len(parts) < 3:
        parts.append(0.0)
    return parts[0] + parts[1] / 60.0 + parts[2] / 3600.0


def extract_gps(image_bytes):
    """Return (lat, lng) from photo EXIF, or (None, None)."""
    if not image_bytes:
        return None, None
    img = None
    try:
        img = Image.open(BytesIO(image_bytes))
        gps = img.getexif().get_ifd(0x8825)
        if not gps:
            return None, None
        lat = _dms_to_degrees(gps.get(2))
        lng = _dms_to_degrees(gps.get(4))
        if lat is None or lng is None:
            return None, None
        if _exif_text(gps.get(1)).startswith("S"):
            lat = -abs(lat)
        if _exif_text(gps.get(3)).startswith("W"):
            lng = -abs(lng)
        if abs(lat) > 90 or abs(lng) > 180:
            return None, None
        return lat, lng
    except Exception as exc:
        app.logger.info("No usable GPS in photo: %s", exc)
        return None, None
    finally:
        if img is not None:
            try:
                img.close()
            except Exception:
                pass


def _parse_coord(value, limit):
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or abs(number) > limit:
        return None
    return number


def _form_coordinates():
    lat = _parse_coord(request.form.get("lat"), 90)
    lng = _parse_coord(request.form.get("lng"), 180)
    if lat is None or lng is None:
        return None, None
    return lat, lng


def _choose_coordinates(image_bytes):
    exif_lat, exif_lng = extract_gps(image_bytes)
    if exif_lat is not None and exif_lng is not None:
        return exif_lat, exif_lng, "exif"
    form_lat, form_lng = _form_coordinates()
    if form_lat is not None and form_lng is not None:
        return form_lat, form_lng, "browser"
    return None, None, None


def estimate_text_tokens(text):
    return max(1, (len(text) + 3) // 4)


def assemble_prompt(locality, species, lookalikes, lat, lng):
    lines = [f"You are a marine biologist identifying a marine animal in a photo taken in {locality}."]
    if lat is not None and lng is not None:
        lines.append(f"Approximate coordinates: {lat:.4f}, {lng:.4f}.")
    lines.extend([
        "",
        "Examine the animal carefully. First note the overall body plan "
        "(arms/rays, fins, rounded spiny body, elongated form, hard shell, soft body, etc.).",
        "",
        "Then consider the two most likely groups this could belong to. "
        "Do not lock in the first group — if features could fit either, weigh both.",
        "Example: an elongated animal could be an eel or a sea snake; "
        "a spiny animal could be a starfish or a sea urchin. "
        "Choose the species that best matches the visible features, even if it is not in the first group you thought of.",
        "",
        "Key rules:",
        "- Arms or rays from a central disc means a starfish, not a sea urchin.",
        "- A clearly rounded or globular body with long spines and no arms means a sea urchin.",
        "- Prefer a species known from this region when the visible features fit one.",
        "- The regional species list is a hint, not a menu. Name an unlisted species when the features call for it. Do not replace that animal with a listed species that lacks those features.",
    ])
    if lookalikes:
        lines.append("")
        lines.extend(CORTEZ_LOOKALIKE_LINES)
    if species:
        lines.append("")
        lines.append(
            "Species recorded in this area, as a hint. "
            "If none of them match the visible features, name the species that does, listed or not."
        )
        current_group = None
        for item in species:
            group = item.get("group")
            if group and group != current_group:
                lines.append(f"{group}:")
                current_group = group
            common = item.get("common")
            if common:
                lines.append(f"- {item['scientific']} ({common})")
            else:
                lines.append(f"- {item['scientific']}")
    else:
        lines.append("")
        lines.append("No regional species list was available. Identify the animal from the visible features.")
    lines.extend([
        "",
        "Reply in this exact format, with each field on its own line:",
        "Common name: <name>",
        "Scientific name: <name>",
        "Group: <group>",
        "Alternative group considered: <other possible group or none>",
        "Confidence: <high / medium / low>",
        "Notes: <one cool fact about the species>",
        "If still unsure, list the closest alternatives.",
    ])
    return NL.join(lines)


def prompt_for_species(locality, species, lookalikes, lat, lng):
    prompt = assemble_prompt(locality, species, lookalikes, lat, lng)
    if estimate_text_tokens(prompt) <= MAX_PROMPT_TEXT_TOKENS:
        return prompt, len(species), False
    lo, hi = 0, len(species)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = assemble_prompt(locality, species[:mid], lookalikes, lat, lng)
        if estimate_text_tokens(candidate) <= MAX_PROMPT_TEXT_TOKENS:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    trimmed = species[:best]
    return assemble_prompt(locality, trimmed, lookalikes, lat, lng), len(trimmed), True


def _species_cache_get(key):
    now = time.monotonic()
    with _SPECIES_LOCK:
        hit = _SPECIES_CACHE.get(key)
        if not hit or hit[0] <= now:
            _SPECIES_CACHE.pop(key, None)
            return None
        return hit[1]


def _species_cache_put(key, payload, ttl):
    with _SPECIES_LOCK:
        if len(_SPECIES_CACHE) >= 64 and key not in _SPECIES_CACHE:
            oldest = min(_SPECIES_CACHE, key=lambda item: _SPECIES_CACHE[item][0])
            _SPECIES_CACHE.pop(oldest, None)
        _SPECIES_CACHE[key] = (time.monotonic() + ttl, payload)


def _empty_species():
    return {"species": [], "available": 0, "complete": False, "source": "none"}


def _curated_species_payload():
    rows = []
    seen = set()
    for raw, common in CURATED_COMMON_NAMES.items():
        scientific = exact_scientific_name(raw)
        if not scientific:
            continue
        key = scientific.lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append({"scientific": scientific, "common": common, "count": 0})
    rows.sort(key=lambda item: item["scientific"])
    return {"species": rows, "available": len(rows), "complete": False, "source": "curated"}


def _merge_species_rows(rows):
    best = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        taxon = row.get("taxon") if isinstance(row.get("taxon"), dict) else {}
        if taxon.get("rank") not in _SPECIES_RANKS:
            continue
        scientific = exact_scientific_name(taxon.get("name") or "")
        if not scientific:
            continue
        try:
            count = int(row.get("count") or 0)
        except (TypeError, ValueError):
            count = 0
        common = CURATED_COMMON_NAMES.get(scientific.lower()) or (taxon.get("preferred_common_name") or "").strip() or None
        current = best.get(scientific.lower())
        if current is None or count > current["count"]:
            best[scientific.lower()] = {"scientific": scientific, "common": common, "count": count}
    return sorted(best.values(), key=lambda item: (-item["count"], item["scientific"]))


def assemble_quota_list(group_results):
    """Keep each group's most-observed species, up to that group's quota.

    group_results is ordered dicts of label, quota, and raw iNaturalist rows.
    A rare sea snake stays even when fishes have far more observations.
    """
    chosen = []
    seen = set()
    for group in group_results:
        merged = _merge_species_rows(group.get("rows") or [])
        kept = 0
        for item in merged:
            key = item["scientific"].lower()
            if key in seen:
                continue
            if kept >= group["quota"]:
                break
            copied = dict(item)
            copied["group"] = group["label"]
            chosen.append(copied)
            seen.add(key)
            kept += 1
    return chosen


def _inat_species_page(query, page, per_page, taxon_id, timeout, exclude=None):
    params = {
        "locale": "en",
        "rank": "species",
        "verifiable": "true",
        "quality_grade": "research",
        "page": page,
        "per_page": per_page,
    }
    if taxon_id:
        params["taxon_id"] = taxon_id
        if exclude:
            params["without_taxon_id"] = exclude
    else:
        params["iconic_taxa"] = MARINE_ICONIC
        params["without_taxon_id"] = EXCLUDED_TAXA
    if query.get("place_id"):
        params["place_id"] = query["place_id"]
    elif query.get("radius"):
        params["lat"] = query["lat"]
        params["lng"] = query["lng"]
        params["radius"] = query["radius"]
    else:
        params["swlat"] = query["swlat"]
        params["swlng"] = query["swlng"]
        params["nelat"] = query["nelat"]
        params["nelng"] = query["nelng"]
    try:
        resp = http.get(
            SPECIES_COUNTS_URL,
            params=params,
            headers=_SPECIES_HEADERS,
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError) as exc:
        app.logger.warning("iNaturalist species_counts failed: %s", exc)
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _collect_species_rows(query, taxon_id, per_page, max_pages, deadline, exclude=None):
    rows = []
    total = 0
    got_page = False
    exhausted = False
    for page in range(1, max_pages + 1):
        if page > 1:
            # Full lists are many pages. Pause so a crawl does not burst the API.
            time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
        if time.monotonic() >= deadline:
            break
        remaining = max(0.4, deadline - time.monotonic())
        data = _inat_species_page(
            query, page, per_page, taxon_id, (1.0, min(2.5, remaining)), exclude=exclude
        )
        if data is None:
            break
        got_page = True
        try:
            total = int(data.get("total_results") or 0)
        except (TypeError, ValueError):
            total = 0
        batch = data.get("results") if isinstance(data.get("results"), list) else []
        rows.extend(batch)
        if not batch:
            exhausted = total == 0
            break
        if page * per_page >= total or page * per_page >= 10000:
            exhausted = True
            break
    return rows, total, exhausted and got_page


def _fetch_short_species_list(query, deadline):
    """One page per quota group, in parallel, so a cold request stays short."""
    buckets = {group["label"]: [] for group in SHORT_GROUPS}
    totals = {group["label"]: 0 for group in SHORT_GROUPS}
    failed = {group["label"]: False for group in SHORT_GROUPS}

    def fetch_taxon(group, taxon_id):
        if time.monotonic() >= deadline:
            return group["label"], [], 0, False
        rows, total, done = _collect_species_rows(
            query,
            taxon_id,
            max(group["quota"], 30),
            1,
            deadline,
            exclude=group.get("exclude"),
        )
        return group["label"], rows, total, done

    calls = [(group, taxon_id) for group in SHORT_GROUPS for taxon_id in group["taxon_ids"]]
    with ThreadPoolExecutor(max_workers=len(calls) or 1) as pool:
        futures = [pool.submit(fetch_taxon, group, taxon_id) for group, taxon_id in calls]
        for future in futures:
            label, rows, total, done = future.result()
            buckets[label].extend(rows)
            totals[label] += total
            if not done:
                failed[label] = True

    if not any(buckets.values()):
        if query.get("curated_fallback"):
            return _curated_species_payload()
        return _empty_species()
    species = assemble_quota_list(
        {"label": group["label"], "quota": group["quota"], "rows": buckets[group["label"]]}
        for group in SHORT_GROUPS
    )
    return {
        "species": species,
        "available": sum(totals.values()),
        "complete": not any(failed.values()),
        "source": "inaturalist",
    }


def _fetch_species_list(query, mode, deadline, on_progress):
    if mode == "short":
        payload = _fetch_short_species_list(query, deadline)
        on_progress(payload)
        return payload
    main_rows, main_total, main_done = _collect_species_rows(
        query,
        None,
        SPECIES_PAGE_SIZE,
        20,
        deadline,
    )
    if not main_rows and not main_done:
        payload = _curated_species_payload() if query.get("curated_fallback") else _empty_species()
        on_progress(payload)
        return payload
    on_progress({
        "species": _merge_species_rows(main_rows),
        "available": main_total,
        "complete": False,
        "source": "inaturalist",
    })
    extra_rows = []
    extra_total = 0
    supplements_done = True
    for taxon_id in SUPPLEMENT_TAXA:
        if time.monotonic() >= deadline:
            supplements_done = False
            break
        supplement_rows, supplement_total, supplement_done = _collect_species_rows(
            query,
            taxon_id,
            200,
            5,
            deadline,
        )
        if not supplement_done:
            supplements_done = False
        extra_rows.extend(supplement_rows)
        extra_total += supplement_total
    species = _merge_species_rows(main_rows + extra_rows)
    payload = {
        "species": species,
        "available": main_total + extra_total,
        "complete": bool(main_done and supplements_done),
        "source": "inaturalist",
    }
    on_progress(payload)
    return payload


def load_species_list(query, mode, budget, refresh_incomplete=False, _retry=True):
    if not query:
        return _empty_species()
    key = (mode, query["key"])
    cached = _species_cache_get(key)
    if cached and cached.get("complete"):
        return cached
    if cached and cached.get("species") and not refresh_incomplete:
        return cached

    with _SPECIES_LOCK:
        event = _SPECIES_INFLIGHT.get(key)
        if event is None:
            event = threading.Event()
            _SPECIES_INFLIGHT[key] = event
            owner = True
        else:
            owner = False
    if not owner:
        end = time.monotonic() + budget
        while time.monotonic() < end:
            cached = _species_cache_get(key)
            if cached and cached.get("species") and not refresh_incomplete:
                return cached
            if event.wait(0.25):
                break
        cached = _species_cache_get(key) or _empty_species()
        if refresh_incomplete and _retry and not cached.get("complete"):
            return load_species_list(query, mode, budget, refresh_incomplete=True, _retry=False)
        return cached

    def on_progress(payload):
        if payload.get("complete") and payload.get("source") == "inaturalist":
            ttl = INAT_CACHE_TTL
        elif payload.get("source") == "none":
            ttl = INAT_ERROR_TTL
        else:
            ttl = 120
        _species_cache_put(key, payload, ttl)

    try:
        return _fetch_species_list(query, mode, time.monotonic() + budget, on_progress)
    except Exception:
        app.logger.warning("species list failed for %s", key, exc_info=True)
        payload = _curated_species_payload() if query.get("curated_fallback") else _empty_species()
        on_progress(payload)
        return payload
    finally:
        with _SPECIES_LOCK:
            _SPECIES_INFLIGHT.pop(key, None)
        event.set()


def _prefetch_species_lists():
    """Warm short lists only. Full mode stays on demand. Failures never propagate."""
    try:
        # Spread gunicorn workers so they do not stampede iNaturalist together.
        time.sleep(random.uniform(0.0, 0.5))
        lock_file = None
        try:
            lock_path = os.environ.get("FINFINDR_PREFETCH_LOCK", "/tmp/finfindr-species-prefetch.lock")
            lock_file = open(lock_path, "a")
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        except OSError:
            app.logger.warning("species prefetch lock unavailable; warming without it")
        named = [region for region in REGIONS if region["id"] not in ("other", "my-location")]
        named.sort(key=lambda region: 0 if region["id"] == "sea-of-cortez" else 1)
        for region in named:
            query = species_query_for(region, None, None)
            try:
                load_species_list(query, "short", 40, refresh_incomplete=True)
            except Exception:
                app.logger.warning("species prefetch failed for %s", region["id"], exc_info=True)
            time.sleep(0.2)
    except Exception:
        app.logger.warning("species prefetch stopped", exc_info=True)
    finally:
        if lock_file is not None:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
                lock_file.close()
            except OSError:
                pass


_prefetch_guard = threading.Lock()
_prefetch_pid = None


def _start_species_prefetch():
    """Start one background warm-up per process. Safe after a gunicorn fork."""
    global _prefetch_pid
    if os.environ.get("FINFINDR_PREFETCH", "1") != "1":
        return False
    pid = os.getpid()
    with _prefetch_guard:
        if _prefetch_pid == pid:
            return False
        _prefetch_pid = pid
    threading.Thread(
        target=_prefetch_species_lists,
        name="finfindr-species-prefetch",
        daemon=True,
    ).start()
    return True


def _safe_model_name(name):
    return bool(name) and bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,64}", name))


def resolve_model(raw=None, debug=False):
    """Env XAI_MODEL overrides the flagship. A form value counts only while debug is on."""
    if debug:
        candidate = (raw or "").strip()
        if _safe_model_name(candidate):
            return candidate
    env = (os.environ.get("XAI_MODEL") or "").strip()
    if _safe_model_name(env):
        return env
    return DEFAULT_XAI_MODEL


def _vision_payload(jpeg_bytes, prompt, model):
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    data_url = f"data:image/jpeg;base64,{b64}"
    return {
        "model": model,
        "temperature": XAI_TEMPERATURE,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": data_url, "detail": "high"},
                    {"type": "input_text", "text": prompt},
                ],
            }
        ],
    }


def _extract_text(data):
    text = ""
    if isinstance(data, dict) and data.get("output"):
        for item in data["output"]:
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        text += content.get("text", "")
    if not text and isinstance(data, dict) and "choices" in data:
        text = data["choices"][0].get("message", {}).get("content", "")
    return text


def _prompt_tokens(data, prompt):
    usage = data.get("usage") if isinstance(data, dict) else None
    if isinstance(usage, dict):
        for key in ("input_tokens", "prompt_tokens"):
            value = usage.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, int) and value >= 0:
                return value, "usage"
            if isinstance(value, float) and value >= 0:
                return int(value), "usage"
    return estimate_text_tokens(prompt), "estimate"


def _identification_meta(mode, sent, available, truncated, source, list_ms, model_ms, tokens, token_source, model, image_px):
    return {
        "species_list_mode": mode,
        "species_sent": sent,
        "species_available": available,
        "species_truncated": bool(truncated),
        "list_source": source,
        "prompt_tokens": tokens,
        "prompt_tokens_source": token_source,
        "list_fetch_ms": int(list_ms),
        "model_call_ms": int(model_ms),
        "model": model,
        "image_px": image_px,
    }


def _log_identification(region_id, meta):
    app.logger.info(
        "identify region=%s list_mode=%s species_sent=%s species_available=%s "
        "prompt_tokens=%s prompt_tokens_source=%s list_fetch_ms=%s model_call_ms=%s "
        "list_source=%s truncated=%s model=%s image_px=%s",
        region_id,
        meta["species_list_mode"],
        meta["species_sent"],
        meta["species_available"],
        meta["prompt_tokens"],
        meta["prompt_tokens_source"],
        meta["list_fetch_ms"],
        meta["model_call_ms"],
        meta["list_source"],
        meta["species_truncated"],
        meta.get("model"),
        meta.get("image_px"),
    )


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/identify", methods=["POST"])
def identify():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    try:
        image_bytes = file.read()
    except Exception as exc:
        app.logger.warning("Could not read upload: %s", exc)
        return jsonify({"error": "Please upload a valid photo"}), 400

    prepared = prepare_image(image_bytes)
    if not prepared:
        return jsonify({"error": "Please upload a valid photo"}), 400
    jpeg_bytes, image_w, image_h = prepared
    image_px = f"{image_w}x{image_h}"

    region = resolve_region(request.form.get("region"))
    lat, lng, coord_source = _choose_coordinates(image_bytes)
    region, location_fallback = apply_location_fallback(region, lat, lng)
    region_label = display_region_label(region, lat, lng, location_fallback)
    mode = resolve_species_mode(request.form.get("species_list"))
    debug = (request.form.get("debug") or "").strip() == "1"
    model = resolve_model(request.form.get("model"), debug=debug)

    xai_key = os.environ.get("XAI_API_KEY")
    if not xai_key:
        app.logger.error("XAI_API_KEY is not set")
        return jsonify({
            "error": "Identification is temporarily unavailable. Please try again later.",
            "region_label": region_label,
        }), 500

    started = time.perf_counter()
    species_payload = load_species_list(
        species_query_for(region, lat, lng),
        mode,
        4.0 if mode == "short" else 12.0,
    )
    list_fetch_ms = int(round((time.perf_counter() - started) * 1000))
    species = species_payload.get("species") or []
    locality = locality_text(region, lat, lng, coord_source)
    if location_fallback:
        locality += (
            ". The photographer asked to use their location, but no coordinates were available, "
            "so this default region is used"
        )
    prompt, sent, prompt_trimmed = prompt_for_species(
        locality,
        species,
        cortez_lookalikes_apply(region, lat, lng),
        lat,
        lng,
    )
    available = int(species_payload.get("available") or 0)
    truncated = prompt_trimmed or (mode == "full" and not species_payload.get("complete"))
    source = species_payload.get("source") or "none"

    payload = _vision_payload(jpeg_bytes, prompt, model)
    model_started = time.perf_counter()
    try:
        resp = http.post(
            "https://api.x.ai/v1/responses",
            headers={"Authorization": f"Bearer {xai_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
    except requests.Timeout:
        app.logger.exception("xAI request timed out")
        model_ms = int(round((time.perf_counter() - model_started) * 1000))
        meta = _identification_meta(mode, sent, available, truncated, source, list_fetch_ms, model_ms, None, "unavailable", model, image_px)
        _log_identification(region["id"], meta)
        return jsonify({
            "error": "Identification is taking too long. Please try again.",
            "region_label": region_label,
            "meta": meta,
        }), 504
    except requests.RequestException:
        app.logger.exception("xAI request failed")
        model_ms = int(round((time.perf_counter() - model_started) * 1000))
        meta = _identification_meta(mode, sent, available, truncated, source, list_fetch_ms, model_ms, None, "unavailable", model, image_px)
        _log_identification(region["id"], meta)
        return jsonify({
            "error": "Could not identify this photo. Please try again.",
            "region_label": region_label,
            "meta": meta,
        }), 502

    model_ms = int(round((time.perf_counter() - model_started) * 1000))
    if not resp.ok:
        app.logger.error("xAI HTTP %s: %s", resp.status_code, resp.text[:500])
        meta = _identification_meta(mode, sent, available, truncated, source, list_fetch_ms, model_ms, None, "unavailable", model, image_px)
        _log_identification(region["id"], meta)
        return jsonify({
            "error": "Could not identify this photo. Please try another one.",
            "region_label": region_label,
            "meta": meta,
        }), 502

    try:
        data = resp.json()
    except ValueError:
        app.logger.exception("xAI returned invalid JSON")
        meta = _identification_meta(mode, sent, available, truncated, source, list_fetch_ms, model_ms, None, "unavailable", model, image_px)
        _log_identification(region["id"], meta)
        return jsonify({
            "error": "Could not identify this photo. Please try again.",
            "region_label": region_label,
            "meta": meta,
        }), 502

    tokens, token_source = _prompt_tokens(data, prompt)
    meta = _identification_meta(mode, sent, available, truncated, source, list_fetch_ms, model_ms, tokens, token_source, model, image_px)
    _log_identification(region["id"], meta)
    text = apply_standard_names(_extract_text(data), species)
    return jsonify({
        "result_text": text or "No response",
        "region_label": region_label,
        "meta": meta,
    })


@app.errorhandler(RequestEntityTooLarge)
def too_large(_exc):
    return jsonify({"error": "That photo is too large. Please choose a smaller one."}), 413


@app.errorhandler(Exception)
def unhandled(exc):
    if isinstance(exc, HTTPException):
        return exc
    app.logger.exception("Unhandled error")
    if request.path == "/identify":
        return jsonify({"error": "Something went wrong. Please try again."}), 500
    return "Internal Server Error", 500


@app.before_request
def _warm_species_cache():
    # After a gunicorn fork the import-time thread is gone. Starting again is cheap
    # and does not wait, so a health check is not held up.
    _start_species_prefetch()


_start_species_prefetch()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
