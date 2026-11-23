#!/usr/bin/env python3
"""
FinFindr – Marine species identification
"""

import base64
import contextvars
import fcntl
import hashlib
import json
import logging
import math
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import requests
from flask import Flask, Response, copy_current_request_context, g, jsonify, render_template_string, request
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
DEFAULT_MAX_IMAGE_PX = 1024
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
<link rel="icon" href="/favicon.ico">
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
  .candidates { display: flex; flex-direction: column; gap: 0.75rem; margin-top: 0.75rem; }
  .candidate { border: 1px solid rgba(255,255,255,0.16); border-radius: 12px; overflow: hidden; background: rgba(0,0,0,0.22); }
  .candidate img { width: 100%; max-height: 180px; object-fit: cover; display: block; background: rgba(0,0,0,0.25); }
  .candidate .cap { padding: 0.65rem 0.75rem 0.8rem; }
  .candidate h4 { font-size: 1.05rem; margin-bottom: 0.15rem; }
  .candidate .attr { margin-top: 0.45rem; font-size: 0.75rem; opacity: 0.8; overflow-wrap: anywhere; }
  .candidate a { color: var(--aqua); }
  .unsure-title { color: #fde68a; }
  .discovery-card { margin-top: 0.75rem; padding: 0.9rem 0.85rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.18); background: rgba(255,255,255,0.04); }
  .discovery-card h3 { font-size: 1.15rem; margin-bottom: 0.35rem; }
  .discovery-card p, .discovery-card li { line-height: 1.45; }
  .discovery-card ul { margin: 0.45rem 0 0.2rem 1.1rem; }
  .discovery-card a { color: var(--aqua); }
  .quiet-note { margin-top: 0.55rem; opacity: 0.85; }
  #fileInput { display: block; width: 100%; max-width: 100%; margin: 1rem auto; font-size: 1rem; color: var(--white); }
  #results { max-width: 800px; margin: 0 auto; }
  .result-card { background: rgba(0,0,0,0.35); border-radius: 16px; margin-bottom: 1.25rem; border: 1px solid rgba(255,255,255,0.1); overflow: hidden; }
  .result-card img.preview { width: 100%; max-height: 70vh; object-fit: contain; object-position: center; display: block; background: rgba(0,0,0,0.35); }
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
    .result-card img.preview { max-height: 50vh; }
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
    <label for="discoverySelect">Discovery flags</label>
    <select id="discoverySelect">
      <option value="">Server default</option>
      <option value="1">On</option>
      <option value="0">Off</option>
    </select>
    <label for="excludeObservation">Exclude iNat observation id</label>
    <input id="excludeObservation" type="text" inputmode="numeric" placeholder="Optional" autocomplete="off">
    <label for="verifySelect">Reference check</label>
    <select id="verifySelect">
      <option value="">Server default</option>
      <option value="1">On</option>
      <option value="0">Off</option>
    </select>
    <label for="nocacheSelect">Skip answer cache</label>
    <select id="nocacheSelect">
      <option value="">Use cache</option>
      <option value="1">nocache=1</option>
    </select>
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
const verifySelect = document.getElementById("verifySelect");
const discoverySelect = document.getElementById("discoverySelect");
const excludeObservation = document.getElementById("excludeObservation");
const nocacheSelect = document.getElementById("nocacheSelect");
const locationStatus = document.getElementById("locationStatus");
const MIN_DIMENSION = 32;
const MAX_BYTES = 20 * 1024 * 1024;
const NOT_MARINE_MESSAGE = "FinFindr focuses on marine animal identification only. Please upload a photo of a marine animal.";
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

function applySavedLocation() {
  const saved = readSavedLocation();
  if (!saved) {
    setLocationStatus("Select Use my location to share your position. Photo GPS is still used when the picture has it.");
    return;
  }
  browserLat = saved.lat;
  browserLng = saved.lng;
  setLocationStatus("Using your saved location (" + saved.lat.toFixed(2) + ", " + saved.lng.toFixed(2) + ").");
}

function requestLocation() {
  applySavedLocation();
  if (!navigator.geolocation) {
    setLocationStatus("Location is not available in this browser. Pick a region, or photos will be identified as Sea of Cortez / La Paz. Photo GPS is still used when the picture has it.");
    return;
  }
  setLocationStatus("Finding your location…");
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
  if (regionSelect.value === "my-location") applySavedLocation();
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

function safeHttpUrl(url) {
  try {
    const parsed = new URL(url);
    if (parsed.protocol !== "https:") return "";
    const host = parsed.hostname;
    const allowed = host === "inaturalist.org" || host.endsWith(".inaturalist.org")
      || host === "inaturalist-open-data.s3.amazonaws.com"
      || host.endsWith(".amazonaws.com");
    return allowed ? parsed.href : "";
  } catch (err) {
    return "";
  }
}

function showCandidates(card, candidates, allowHigh) {
  const list = document.createElement("div");
  list.className = "candidates";
  candidates.slice(0, 3).forEach(function (item) {
    const block = document.createElement("article");
    block.className = "candidate";
    const photo = item.reference_photo || {};
    const imgUrl = safeHttpUrl(photo.url || "");
    if (imgUrl) {
      const img = document.createElement("img");
      img.src = imgUrl;
      img.alt = "Reference photo of " + (item.common_name || item.scientific_name || "a candidate");
      block.appendChild(img);
    }
    const cap = document.createElement("div");
    cap.className = "cap";
    const title = document.createElement("h4");
    title.textContent = item.common_name || "Common name unavailable";
    cap.appendChild(title);
    if (item.scientific_name) {
      const sci = document.createElement("div");
      sci.className = "sci";
      sci.setAttribute("translate", "no");
      sci.textContent = item.scientific_name;
      cap.appendChild(sci);
    }
    if (item.off_list) {
      const off = document.createElement("div");
      off.textContent = "Not on the regional list";
      cap.appendChild(off);
    }
    if (item.confidence && item.confidence !== "high") {
      const conf = document.createElement("div");
      conf.className = "confidence";
      conf.textContent = item.confidence + " confidence";
      cap.appendChild(conf);
    } else if (allowHigh && item.confidence === "high") {
      const conf = document.createElement("div");
      conf.className = "confidence";
      conf.textContent = "high confidence";
      cap.appendChild(conf);
    }
    if (item.trait) {
      const trait = document.createElement("div");
      trait.style.marginTop = "0.35rem";
      trait.textContent = "What to check: " + item.trait;
      cap.appendChild(trait);
    }
    if (photo.attribution || photo.license || photo.source_url) {
      const attr = document.createElement("div");
      attr.className = "attr";
      const link = safeHttpUrl(photo.source_url || "");
      if (link) {
        const a = document.createElement("a");
        a.href = link;
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        a.textContent = photo.attribution || "iNaturalist photo";
        attr.appendChild(a);
      } else {
        attr.textContent = photo.attribution || "Reference photo";
      }
      if (photo.license) {
        attr.appendChild(document.createTextNode(" · " + photo.license));
      }
      attr.appendChild(document.createTextNode(" · iNaturalist"));
      cap.appendChild(attr);
    }
    block.appendChild(cap);
    list.appendChild(block);
  });
  card.querySelector(".result-body").appendChild(list);
}

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

function inatSubmitLink() {
  const link = document.createElement("a");
  link.className = "inat-submit";
  link.href = "https://www.inaturalist.org/observations/upload";
  link.target = "_blank";
  link.rel = "noopener";
  link.textContent = "Submit to iNaturalist for expert confirmation";
  return link;
}

function appendNearestRecord(box, discovery) {
  if (discovery.nearest_known_record_km === null || discovery.nearest_known_record_km === undefined || discovery.nearest_known_record_km === "") return;
  const line = document.createElement("p");
  line.className = "nearest-record";
  const source = discovery.nearest_known_record_source || "iNaturalist";
  line.textContent = "Nearest known record: " + discovery.nearest_known_record_km + " km (" + source + ")";
  box.appendChild(line);
}

function appendExternalRange(card, discovery) {
  if (!discovery || !discovery.external_range) return;
  const ext = document.createElement("p");
  ext.className = "external-range";
  const source = discovery.external_range_source || "not checked";
  ext.textContent = "Outside iNaturalist: " + discovery.external_range + " (" + source + ")";
  card.querySelector(".result-body").appendChild(ext);
}

function showNotMarine(card) {
  card.querySelector("h3").textContent = "Marine animals only";
  const sciWrap = card.querySelector(".scientific");
  if (sciWrap) sciWrap.hidden = true;
  const conf = card.querySelector(".confidence");
  if (conf) conf.textContent = "";
  const note = document.createElement("p");
  note.className = "not-marine";
  note.textContent = NOT_MARINE_MESSAGE;
  card.querySelector(".result-body").appendChild(note);
}

function showPhotoRejection(card, data) {
  card.querySelector("h3").textContent = "Can't identify from this photo";
  const sciWrap = card.querySelector(".scientific");
  if (sciWrap) sciWrap.hidden = true;
  const conf = card.querySelector(".confidence");
  if (conf) conf.textContent = "";
  const note = document.createElement("p");
  note.textContent = data.photo_issue || "The animal is not clear enough to identify.";
  card.querySelector(".result-body").appendChild(note);
}

function showDiscovery(card, data) {
  const discovery = data.discovery;
  const body = card.querySelector(".result-body");
  const sciWrap = card.querySelector(".scientific");
  if (sciWrap) sciWrap.hidden = true;
  const conf = card.querySelector(".confidence");
  if (conf) conf.textContent = "";
  const box = document.createElement("div");
  box.className = "discovery-card";
  const heading = document.createElement("h3");
  const lead = document.createElement("p");
  const why = document.createElement("p");
  why.textContent = "Why we think so";
  const list = document.createElement("ul");
  if (discovery.type === "possible_range_extension") {
    card.querySelector("h3").textContent = discovery.common_name || discovery.scientific_name || "Result";
    heading.textContent = "Possible range extension";
    const area = discovery.radius_km ? "within " + discovery.radius_km + " km" : "in the selected region";
    lead.textContent = (discovery.scientific_name || "This species") + " has " + discovery.records_in_region + " research-grade iNaturalist records " + area + ". The identification matches reference photos, so this may be outside the recorded range.";
    if (discovery.external_range) {
      const ext = document.createElement("p");
      ext.className = "external-range";
      const source = discovery.external_range_source || "not checked";
      ext.textContent = "Outside iNaturalist: " + discovery.external_range + " (" + source + ")";
      box.appendChild(ext);
    }
    (discovery.traits || []).forEach(function (trait) {
      const item = document.createElement("li");
      item.textContent = trait;
      list.appendChild(item);
    });
    if (discovery.nearest_records && discovery.nearest_records.length) {
      const near = document.createElement("p");
      near.textContent = "Nearest known records";
      box.appendChild(heading);
      box.appendChild(lead);
      appendNearestRecord(box, discovery);
      if (list.childNodes.length) {
        box.appendChild(why);
        box.appendChild(list);
      }
      box.appendChild(near);
      const records = document.createElement("ul");
      discovery.nearest_records.forEach(function (record) {
        const item = document.createElement("li");
        const link = document.createElement("a");
        link.href = record.url;
        link.target = "_blank";
        link.rel = "noopener";
        const place = record.place || "location not listed";
        link.textContent = record.distance_km + " km — " + place + (record.date ? " (" + record.date + ")" : "");
        item.appendChild(link);
        records.appendChild(item);
      });
      box.appendChild(records);
    } else {
      box.appendChild(heading);
      box.appendChild(lead);
      appendNearestRecord(box, discovery);
      if (list.childNodes.length) {
        box.appendChild(why);
        box.appendChild(list);
      }
    }
  } else {
    card.querySelector("h3").textContent = discovery.genus ? "Closest group: " + discovery.genus : "No matching species";
    heading.textContent = "No matching species";
    lead.textContent = discovery.note || "The photo does not match the reference photos of the closest named species.";
    (discovery.closest_relatives || []).forEach(function (item) {
      const row = document.createElement("li");
      const name = item.common_name ? item.common_name + " (" + item.scientific_name + ")" : item.scientific_name;
      row.textContent = name + (item.trait ? " — " + item.trait : "");
      list.appendChild(row);
    });
    box.appendChild(heading);
    box.appendChild(lead);
    if (list.childNodes.length) {
      box.appendChild(why);
      box.appendChild(list);
    }
  }
  if (discovery.type === "possible_range_extension" && discovery.rule && discovery.rule.text) {
    const rule = document.createElement("p");
    rule.className = "quiet-note";
    rule.textContent = "Range rule: " + discovery.rule.text + ".";
    box.appendChild(rule);
  }
  box.appendChild(inatSubmitLink());
  body.appendChild(box);
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
  if (busy) return;
  if (!fileInput.files || !fileInput.files[0]) return;
  identifyBtn.disabled = true;
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
  if (busy) return;
  const files = event.dataTransfer && event.dataTransfer.files;
  if (!files || !files[0]) return;
  identifyBtn.disabled = true;
  handleFile(files[0]);
});

document.addEventListener("dragover", (event) => event.preventDefault());
document.addEventListener("drop", (event) => event.preventDefault());

async function handleFile(file) {
  if (busy) return;
  busy = true;
  identifyBtn.disabled = true;
  clearClientError();
  let progressTimer = null;
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
      <img class="preview" alt="Photo being identified" style="width:100%; max-height:70vh; object-fit:contain; object-position:center;">
      <div class="result-body">
        <div class="spinner" role="status" aria-label="Identifying photo"></div>
        <h3 class="progress-title">Identifying…</h3>
        <p class="hint progress-note">Usually 30–90 seconds.</p>
        <div class="scientific" hidden>
          <div class="sci-label" hidden>Scientific name</div>
          <div class="sci" translate="no"></div>
        </div>
        <div class="confidence"></div>
        <div class="error"></div>
      </div>
    `;
    results.appendChild(card);
    const progressStarted = Date.now();
    progressTimer = setInterval(function () {
      const title = card.querySelector(".progress-title");
      if (!title) return;
      const elapsed = Date.now() - progressStarted;
      title.textContent = elapsed < 8000 ? "Identifying…" : "Checking reference photos…";
    }, 1000);

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
      if (verifySelect && (verifySelect.value === "0" || verifySelect.value === "1")) {
        formData.append("verify", verifySelect.value);
      }
      if (discoverySelect && (discoverySelect.value === "0" || discoverySelect.value === "1")) {
        formData.append("discovery", discoverySelect.value);
      }
      if (excludeObservation && excludeObservation.value.trim()) {
        formData.append("exclude_observation_id", excludeObservation.value.trim());
      }
    }
    if (nocacheSelect && nocacheSelect.value === "1") {
      formData.append("nocache", "1");
    }

    const resp = await fetch("/identify", { method: "POST", body: formData });
    const parsed = await readJsonResponse(resp);
    clearInterval(progressTimer);
    const progressNote = card.querySelector(".progress-note");
    if (progressNote) progressNote.remove();
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

    if (data.reason === "not_marine") {
      showNotMarine(card);
      addRetry(card);
      return;
    }
    if ((!data.candidates || !data.candidates.length) && (data.photo_quality === "poor" || data.photo_quality === "invalid" || data.message === "Can't identify from this photo.")) {
      showPhotoRejection(card, data);
      addRetry(card);
      return;
    }
    if (data.discovery && (data.discovery.type === "possible_range_extension" || data.discovery.type === "no_match")) {
      showDiscovery(card, data);
      addRetry(card);
      return;
    }
    if (data.certainty !== "confident") {
      if (data.candidates && data.candidates.length) {
        card.querySelector("h3").textContent = data.certainty === "unsure" ? "Not sure yet" : "Likely — check these";
        card.querySelector("h3").classList.add("unsure-title");
        const sciWrapEarly = card.querySelector(".scientific");
        if (sciWrapEarly) sciWrapEarly.hidden = true;
        const confEarly = card.querySelector(".confidence");
        if (confEarly) confEarly.textContent = "";
        showCandidates(card, data.candidates, false);
        if (data.notable_note) {
          const note = document.createElement("p");
          note.className = "quiet-note";
          note.textContent = data.notable_note;
          card.querySelector(".result-body").appendChild(note);
        }
        if (data.discovery && data.discovery.type === "off_list") appendExternalRange(card, data.discovery);
      } else {
        showPhotoRejection(card, data);
      }
      addRetry(card);
      return;
    }
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
    if (confidence && confidence.toLowerCase() === "high") {
      card.querySelector(".confidence").textContent = "high confidence";
    }
    if (notes) {
      const n = document.createElement("div");
      n.style.marginTop = "0.5rem";
      n.textContent = notes;
      card.querySelector(".result-body").appendChild(n);
    }
    if (data.discovery && data.discovery.type === "off_list") appendExternalRange(card, data.discovery);
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
    if (progressTimer) clearInterval(progressTimer);
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



# Connect + read caps. A 1s read timeout was aborting healthy iNaturalist replies.
INAT_TIMEOUT = (3.0, 8.0)
INAT_CACHE_TTL = 24 * 60 * 60
INAT_ERROR_TTL = 60
_INAT_MIN_INTERVAL_DEFAULT = 1.0
_INAT_URL = "https://api.inaturalist.org/v1/taxa"
_SPECIES_RANKS = {"species", "subspecies", "variety", "form"}
_cache_lock = threading.Lock()
_inat_cache = {}
_INAT_JSON_CACHE = {}
_INAT_JSON_LOCK = threading.Lock()
_INAT_PACE_LOCK = threading.Lock()
_INAT_NEXT_AT = 0.0

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
    payload = _inat_get(
        _INAT_URL,
        {"q": scientific, "locale": "en", "per_page": 10},
        INAT_TIMEOUT,
        {
            "User-Agent": "FinFindr/1.0 (marine species common-name lookup)",
            "Accept": "application/json",
        },
    )
    if not payload:
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
        futures = [_submit(pool, _fetch_inat_common_name_uncached, name) for name in pending]
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
XAI_SEED = 0
NOT_MARINE_MESSAGE = (
    "FinFindr focuses on marine animal identification only. "
    "Please upload a photo of a marine animal."
)
# FINFINDR_SCOPE=marine is the default. The names below are the one list a
# marine identification may name. FINFINDR_SCOPE=all turns the gate off.
# Any other value is a comma-separated ancestry list that replaces this one,
# so the scope can widen later without touching the identification pipeline.
MARINE_SCOPE_ANCESTORS = frozenset({
    "actinopterygii",
    "chondrichthyes",
    "elasmobranchii",
    "holocephali",
    "myxini",
    "petromyzontida",
    "mollusca",
    "echinodermata",
    "cnidaria",
    "porifera",
    "ctenophora",
    "crustacea",
    "cetacea",
    "odontoceti",
    "mysticeti",
    "sirenia",
    "pinnipedia",
    "otariidae",
    "phocidae",
    "odobenidae",
    "enhydra",
    "cheloniidae",
    "dermochelyidae",
    "hydrophiinae",
    "laticaudinae",
    "hydrophiidae",
    "hydrophis",
    "laticauda",
    "aipysurus",
    "emydocephalus",
    "pelamis",
    "spheniscidae",
    "diomedeidae",
    "procellariidae",
    "hydrobatidae",
    "pelecanidae",
    "sulidae",
    "fregatidae",
    "phaethontidae",
    "alcidae",
    "laridae",
    "stercorariidae",
    "phalacrocoracidae",
    # Shorebirds. A gull, pelican, or oystercatcher is in scope even on the beach.
    "haematopodidae",
    "scolopacidae",
    "charadriidae",
    "recurvirostridae",
    "ardeidae",
    "threskiornithidae",
    "burhinidae",
    # Marine iguana, polar bear, and sea otter are marine even though their
    # class is otherwise terrestrial.
    "amblyrhynchus",
    "ursus maritimus",
    "enhydra lutris",
    # Marine algae.
    "chlorophyta",
    "rhodophyta",
    "phaeophyceae",
    "ochrophyta",
    "ulvophyceae",
})
# Groups that sit inside an otherwise marine phylum but are not marine.
TERRESTRIAL_SCOPE_EXCEPTIONS = frozenset({
    "stylommatophora",
    "helicidae",
    "coenobitidae",
    "gecarcinidae",
})
LAND_ICONIC_TAXA = frozenset({
    "mammalia",
    "aves",
    "insecta",
    "arachnida",
    "amphibia",
    "reptilia",
    "plantae",
    "fungi",
})
# Traits the photo description is compared with. A color, shape, or pattern
# that cannot belong to the species blocks a confident identification.
SPECIES_TRAITS = {
    "mithrodia bradleyi": "tan or brown mottled cylindrical arms with a fine irregular network, not a red tile lattice",
    "phataria unifascialis": "small disc and long slender cylindrical arms colored tan, orange, or blue, not a red lattice and not a thick knobby cushion",
    "pharia pyramidata": "tan or orange star with a broad arm base, not a red tile lattice",
    "echinaster tenuispina": "slender orange arms with fine spines, not a cream lattice of raised plates",
    "fromia monilis": "red or orange arms covered by a cream lattice of raised plates",
    "pentaceraster cumingi": "thick inflated body with short broad arms and large conical knobs",
    "nidorellia armata": "pale thick body with very short arms and dark brown spines",
}
NL = chr(10)
_REQUEST_STATS = contextvars.ContextVar("finfindr_request_stats", default=None)
_SPECIES_LOCK = threading.Lock()
_SPECIES_CACHE = {}
_SPECIES_WAIT_RESULT = {}
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
    "- Bull shark (Carcharhinus leucas) or silky shark (Carcharhinus falciformis): a stocky body, broad blunt snout, small eyes, and no ridge between the dorsal fins mean the bull shark, Carcharhinus leucas. Rank Carcharhinus leucas first in that case. A slender body, longer rounded snout, and a visible low ridge between the dorsal fins mean the silky shark. Do not name falciformis, and do not rank it first, when the snout is blunt or the body is stocky. Put leucas in the candidate list whenever both are plausible.",
    "- Two-spot octopus (Octopus bimaculatus), Hubbs' octopus (Octopus hubbsorum), or red octopus (Octopus rubescens): a pair of false eye spots (ocelli) under the eyes means bimaculatus. A brown-gray octopus without those eye spots can be hubbsorum. A reddish octopus without eye spots, with a stout body and pale papillae, is Octopus rubescens even when it is not on the regional list. Do not leave rubescens off the off-list lines when the regional octopus species lack that reddish color and papillae.",
    "- California blue dorid (Felimare californiensis) or a rough nerite (Nerita scabricosta): a dark blue mantle with yellow spots or rings is Felimare californiensis. A thick dark shell with rough spiral ridges in the intertidal is Nerita scabricosta. Name the one whose spots or shell ridges you can see, and say that trait explicitly.",
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


def species_query_for(region, lat, lng, radius=None):
    """Coordinates drive the species list anywhere. Presets are the fallback."""
    if lat is not None and lng is not None:
        distance = int(radius or LOCATION_RADIUS_KM)
        lat_r = round(float(lat), 1)
        lng_r = round(float(lng), 1)
        match = region_matching_point(lat, lng)
        return {
            "key": f"pt:{lat_r:.1f}:{lng_r:.1f}:{distance}",
            "lat": lat_r,
            "lng": lng_r,
            "radius": distance,
            "curated_fallback": bool(match and match.get("curated_fallback")),
        }
    if region["id"] in ("other", "my-location"):
        return None
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
                f". {source} is {lat:.4f}, {lng:.4f}, which is outside the selected region. "
                "Use the coordinates as the place the animal was seen"
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


def _form_coordinates(fields=None):
    source = request.form if fields is None else fields
    lat = _parse_coord(source.get("lat"), 90)
    lng = _parse_coord(source.get("lng"), 180)
    if lat is None or lng is None:
        return None, None
    return lat, lng


def _choose_coordinates(image_bytes, fields=None):
    exif_lat, exif_lng = extract_gps(image_bytes)
    if exif_lat is not None and exif_lng is not None:
        return exif_lat, exif_lng, "exif"
    form_lat, form_lng = _form_coordinates(fields)
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
        "- A species with no records in this area can still be the right identification. Never reject a species only because it is unrecorded here.",
        "- If the animal is not in the list, or a better match exists outside it, propose up to 2 off_list species on the Off-list lines. Do not invent a name.",
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
        "Photo quality: <good / poor / invalid>",
        "Photo issue: <none, or a short reason if the photo is poor or not a real animal>",
        "Common name: <name>",
        "Scientific name: <name>",
        "Group: <group>",
        "Alternative group considered: <other possible group or none>",
        "Confidence: <high / medium / low>",
        "Notes: <one cool fact about the species>",
        "Candidate 1: <scientific name> | <common name> | <high/medium/low> | <one trait that separates it from the next candidate>",
        "Candidate 2: <scientific name> | <common name> | <high/medium/low> | <one trait>",
        "Candidate 3: <scientific name or none> | <common name> | <high/medium/low> | <one trait>",
        "Off-list 1: <scientific name or none> | <common name> | <trait that the regional list does not cover>",
        "Off-list 2: <scientific name or none> | <common name> | <trait>",
        "List the best supported species first. Include a look-alike when one is plausible. Use none only when you have no second species.",
        "When the visible animal matches a listed species, set both off-list lines to none.",
        "When it does not, name up to two real species that are not in the regional list. Do not invent a name.",
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


def cache_dir():
    return os.environ.get("FINFINDR_CACHE_DIR", "/tmp/finfindr-cache")


def _cache_key_text(key):
    if isinstance(key, str):
        return key
    try:
        return json.dumps(key, sort_keys=True, default=str)
    except TypeError:
        return str(key)


def _disk_cache_path(name):
    directory = cache_dir()
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, name + ".json")


def _disk_lock(path):
    lock_path = path + ".lock"
    handle = open(lock_path, "a")
    fcntl.flock(handle, fcntl.LOCK_EX)
    return handle


def _disk_read(name):
    path = _disk_cache_path(name)
    if not os.path.isfile(path):
        return {}
    lock = _disk_lock(path)
    try:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return {}
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    if not isinstance(data, dict):
        return {}
    now = time.time()
    fresh = {}
    for key, item in data.items():
        if isinstance(item, dict) and float(item.get("expires") or 0) > now and "value" in item:
            fresh[key] = item
    return fresh


def _disk_write(name, data):
    path = _disk_cache_path(name)
    lock = _disk_lock(path)
    try:
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
        os.replace(temporary, path)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def _disk_cache_get(name, key):
    item = _disk_read(name).get(_cache_key_text(key))
    if not item:
        return None
    return item.get("value")


def _disk_cache_put(name, key, value, ttl):
    try:
        json.dumps(value)
    except TypeError:
        return
    path_name = name
    data = _disk_read(path_name)
    data[_cache_key_text(key)] = {"expires": time.time() + max(1.0, float(ttl)), "value": value}
    if len(data) > 512:
        ordered = sorted(data, key=lambda item: data[item].get("expires") or 0)
        for old in ordered[: len(data) - 512]:
            data.pop(old, None)
    try:
        _disk_write(path_name, data)
    except OSError:
        app.logger.warning("could not persist %s cache", name, exc_info=True)


def clear_persisted_caches():
    """Drop in-memory and on-disk species and taxon caches. Tests use this."""
    with _SPECIES_LOCK:
        _SPECIES_CACHE.clear()
    with _INAT_JSON_LOCK:
        _INAT_JSON_CACHE.clear()
    directory = cache_dir()
    try:
        for name in ("species", "taxa"):
            path = os.path.join(directory, name + ".json")
            if os.path.isfile(path):
                os.remove(path)
    except OSError:
        app.logger.warning("could not clear the disk cache", exc_info=True)


def _species_memory_put(key, payload, ttl):
    with _SPECIES_LOCK:
        if len(_SPECIES_CACHE) >= 64 and key not in _SPECIES_CACHE:
            oldest = min(_SPECIES_CACHE, key=lambda item: _SPECIES_CACHE[item][0])
            _SPECIES_CACHE.pop(oldest, None)
        _SPECIES_CACHE[key] = (time.monotonic() + ttl, payload)


def _species_cache_get(key):
    now = time.monotonic()
    with _SPECIES_LOCK:
        hit = _SPECIES_CACHE.get(key)
        if hit and hit[0] > now:
            return hit[1]
        _SPECIES_CACHE.pop(key, None)
    persisted = _disk_cache_get("species", key)
    if persisted is None:
        return None
    _species_memory_put(key, persisted, 60)
    return persisted


def _species_cache_put(key, payload, ttl):
    _species_memory_put(key, payload, ttl)
    _disk_cache_put("species", key, payload, ttl)


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


def inat_min_interval():
    """Seconds between iNaturalist requests. FINFINDR_INAT_MIN_INTERVAL=0 disables the pause."""
    raw = (os.environ.get("FINFINDR_INAT_MIN_INTERVAL") or "").strip()
    if not raw:
        return _INAT_MIN_INTERVAL_DEFAULT
    try:
        return max(0.0, float(raw))
    except ValueError:
        return _INAT_MIN_INTERVAL_DEFAULT


def _inat_wait():
    """Global pace so every iNaturalist call, including prefetch, stays at or under 1 per second."""
    global _INAT_NEXT_AT
    interval = inat_min_interval()
    if interval <= 0:
        return
    with _INAT_PACE_LOCK:
        now = time.monotonic()
        delay = _INAT_NEXT_AT - now
        if delay > 0:
            time.sleep(delay)
            now = time.monotonic()
        _INAT_NEXT_AT = now + interval


def _inat_cache_key(url, params):
    items = tuple(sorted((str(key), str(value)) for key, value in (params or {}).items()))
    return (url, items)


def _inat_memory_put(key, payload, ttl):
    with _INAT_JSON_LOCK:
        if len(_INAT_JSON_CACHE) > 512 and key not in _INAT_JSON_CACHE:
            _INAT_JSON_CACHE.clear()
        _INAT_JSON_CACHE[key] = (time.monotonic() + ttl, payload)


def _persist_taxon_response(url):
    return "/taxa" in (url or "")


def _inat_cache_get(url, params):
    key = _inat_cache_key(url, params)
    now = time.monotonic()
    with _INAT_JSON_LOCK:
        hit = _INAT_JSON_CACHE.get(key)
        if hit and hit[0] > now:
            return hit[1]
        _INAT_JSON_CACHE.pop(key, None)
    if not _persist_taxon_response(url):
        return None
    persisted = _disk_cache_get("taxa", key)
    if persisted is None:
        return None
    _inat_memory_put(key, persisted, 60)
    return persisted


def _taxon_payload_worth_keeping(payload):
    if isinstance(payload, dict):
        results = payload.get("results")
        if isinstance(results, list):
            return bool(results)
        return bool(payload.get("name") or payload.get("ancestors") or payload.get("iconic_taxon_name"))
    if isinstance(payload, list):
        return bool(payload)
    return False


def _inat_cache_put(url, params, payload):
    key = _inat_cache_key(url, params)
    _inat_memory_put(key, payload, INAT_CACHE_TTL)
    if _persist_taxon_response(url) and _taxon_payload_worth_keeping(payload):
        _disk_cache_put("taxa", key, payload, INAT_CACHE_TTL)


def _retry_after_seconds(response, attempt):
    raw = ""
    if response is not None:
        raw = (response.headers.get("Retry-After") or "").strip()
    try:
        delay = float(raw)
    except ValueError:
        delay = float(attempt)
    return min(8.0, max(0.0, delay))


_CALL_DEADLINE = threading.local()


def set_call_deadline(deadline):
    _CALL_DEADLINE.value = deadline


def clear_call_deadline():
    if hasattr(_CALL_DEADLINE, "value"):
        delattr(_CALL_DEADLINE, "value")


def call_time_left():
    deadline = getattr(_CALL_DEADLINE, "value", None)
    if deadline is None:
        return None
    return deadline - time.monotonic()


def begin_request_stats():
    stats = {"external_calls": 0, "inat_throttled": 0, "lock": threading.Lock()}
    _REQUEST_STATS.set(stats)
    return stats


def _bump_stat(name):
    stats = _REQUEST_STATS.get()
    if not stats:
        return
    with stats["lock"]:
        stats[name] = int(stats.get(name) or 0) + 1


def request_stat(name):
    stats = _REQUEST_STATS.get()
    if not stats:
        return 0
    with stats["lock"]:
        return int(stats.get(name) or 0)


def _run_in_worker(fn, *args):
    """Run fn in the current context so request counters survive a worker thread."""
    ctx = contextvars.copy_context()

    def run(fn=fn, args=args, ctx=ctx):
        return ctx.run(fn, *args)

    return run


def _start_thread(target, name):
    worker = threading.Thread(target=_run_in_worker(target), name=name, daemon=True)
    worker.start()
    return worker


def _submit(pool, fn, *args):
    return pool.submit(_run_in_worker(fn, *args))


def _cap_timeout(timeout):
    """Fit a requests timeout inside the request's hard budget."""
    left = call_time_left()
    if left is None:
        return timeout
    if left < 0.4:
        return None
    if isinstance(timeout, tuple):
        connect, read = timeout
        return (min(float(connect), left), min(float(read), left))
    return min(float(timeout), left)


def _http_get_bounded(url, params, headers, timeout):
    """GET that returns even if the socket keeps trickling past the budget."""
    capped = _cap_timeout(timeout)
    if capped is None:
        return None
    read_cap = capped[1] if isinstance(capped, tuple) else float(capped)
    holder = {}

    def run():
        try:
            holder["resp"] = http.get(url, params=params, headers=headers, timeout=capped)
        except Exception as exc:
            holder["error"] = exc

    worker = _start_thread(run, "finfindr-http")
    worker.join(read_cap + 0.5)
    if worker.is_alive():
        app.logger.warning("outbound wall-clock timeout %s", url)
        return None
    if holder.get("error"):
        raise holder["error"]
    return holder.get("resp")


def _inat_get(url, params, timeout, headers):
    """One paced external GET. Successful JSON is cached for hours. 429s are retried."""
    cached = _inat_cache_get(url, params)
    if cached is not None:
        return cached
    host = url.split("/")[2] if "://" in url else url
    for attempt in range(1, 4):
        if call_time_left() is not None and call_time_left() < 0.4:
            return None
        _inat_wait()
        _bump_stat("external_calls")
        try:
            resp = _http_get_bounded(url, params, headers, timeout)
        except requests.Timeout as exc:
            app.logger.warning("%s read timeout: %s", host, exc)
            return None
        except requests.RequestException as exc:
            app.logger.warning("%s request failed: %s", host, exc)
            return None
        if resp is None:
            return None
        if resp.status_code == 429 and "inaturalist.org" in host:
            _bump_stat("inat_throttled")
        if resp.status_code == 429 and attempt < 3:
            delay = _retry_after_seconds(resp, attempt)
            app.logger.warning("%s 429; retrying in %ss", host, delay)
            left = call_time_left()
            if left is not None:
                delay = min(delay, max(0.0, left - 0.5))
            if delay:
                time.sleep(delay)
            continue
        if resp.status_code == 429:
            app.logger.warning("%s failed: 429 normal_throttling", host)
            return None
        try:
            resp.raise_for_status()
            payload = resp.json()
        except (requests.RequestException, ValueError) as exc:
            app.logger.warning("%s request failed: %s", host, exc)
            return None
        if isinstance(payload, (dict, list)):
            _inat_cache_put(url, params, payload)
            return payload
        return None
    return None


def _inat_species_page(query, page, per_page, taxon_id, timeout, exclude=None):
    params = {
        "locale": "en",
        "rank": "species",
        "verifiable": "true",
        "quality_grade": "research",
        "page": page,
        "per_page": per_page,
    }
    if query.get("not_id"):
        params["not_id"] = query["not_id"]
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
    return _inat_get(SPECIES_COUNTS_URL, params, timeout, _SPECIES_HEADERS)


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
        remaining = max(3.0, deadline - time.monotonic())
        data = _inat_species_page(
            query, page, per_page, taxon_id, (3.0, min(8.0, remaining)), exclude=exclude
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
        if page == max_pages:
            # The caller asked for this many pages (the short-list quota is one
            # page). A successful page is complete even when the taxon has more.
            exhausted = True
            break
    return rows, total, exhausted and got_page


def _fetch_short_species_list(query, deadline):
    """One page per quota group, in parallel, so a cold request stays short.

    A group that has more species than its quota is still complete: the page is
    the top of that quota. Only a failed or timed-out call is retried, and an
    incomplete result is not cached.
    """
    buckets = {group["label"]: [] for group in SHORT_GROUPS}
    totals = {group["label"]: 0 for group in SHORT_GROUPS}
    pending = [(group, taxon_id) for group in SHORT_GROUPS for taxon_id in group["taxon_ids"]]

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

    for _attempt in range(2):
        if not pending or time.monotonic() >= deadline:
            break
        still_pending = []
        with ThreadPoolExecutor(max_workers=1) as pool:
            futures = [_submit(pool, fetch_taxon, group, taxon_id) for group, taxon_id in pending]
            for (group, taxon_id), future in zip(pending, futures):
                try:
                    label, rows, total, done = future.result()
                except Exception:
                    app.logger.warning("species group failed for %s", taxon_id, exc_info=True)
                    label, rows, total, done = group["label"], [], 0, False
                if done:
                    buckets[label].extend(rows)
                    totals[label] += total
                else:
                    still_pending.append((group, taxon_id))
        pending = still_pending

    if pending or not any(buckets.values()):
        # A partial quota list is not usable. Prefer the curated list when this region has one.
        if query.get("curated_fallback"):
            payload = _curated_species_payload()
            payload["complete"] = True
            return payload
        return _empty_species()
    species = assemble_quota_list(
        {"label": group["label"], "quota": group["quota"], "rows": buckets[group["label"]]}
        for group in SHORT_GROUPS
    )
    return {
        "species": species,
        "available": sum(totals.values()),
        "complete": True,
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


def _cache_usable(cached, refresh_incomplete):
    """Incomplete iNaturalist lists are never reused. They made species_sent drift."""
    if not cached:
        return False
    if cached.get("complete"):
        return True
    if refresh_incomplete:
        return False
    return cached.get("source") in ("curated", "none")


def _should_cache_species(payload):
    source = payload.get("source")
    if source == "inaturalist" and not payload.get("complete"):
        return False
    return True


def load_species_list(query, mode, budget, refresh_incomplete=False, _retry=True):
    if not query:
        return _empty_species()
    key = (mode, query["key"])
    cached = _species_cache_get(key)
    if _cache_usable(cached, refresh_incomplete):
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
            if _cache_usable(cached, refresh_incomplete):
                return cached
            if event.wait(0.25):
                break
        cached = _species_cache_get(key)
        if _cache_usable(cached, refresh_incomplete):
            return cached
        with _SPECIES_LOCK:
            shared = _SPECIES_WAIT_RESULT.get(key)
        if shared and (shared.get("complete") or shared.get("source") in ("curated", "none")):
            return shared
        if refresh_incomplete and _retry and not (cached and cached.get("complete")):
            return load_species_list(query, mode, budget, refresh_incomplete=True, _retry=False)
        return shared or cached or _empty_species()

    def on_progress(payload):
        with _SPECIES_LOCK:
            _SPECIES_WAIT_RESULT[key] = payload
        if not _should_cache_species(payload):
            return
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


def prefetch_reference_limit():
    """How many Sea of Cortez reference photos to warm. FINFINDR_PREFETCH_REFS=0 disables it."""
    raw = (os.environ.get("FINFINDR_PREFETCH_REFS") or "").strip()
    if not raw:
        return 30
    if raw.lower() in ("0", "false", "off", "no"):
        return 0
    try:
        return min(80, max(0, int(raw)))
    except ValueError:
        return 24


def _prefetch_reference_photos(species):
    """Cache taxon photos for the common local species so the reference check is not waiting on iNaturalist."""
    limit = prefetch_reference_limit()
    if limit <= 0:
        return
    names = []
    for item in species or []:
        name = (item.get("scientific") or "").strip()
        if name and name not in names:
            names.append(name)
        if len(names) >= limit:
            break
    for name in names:
        try:
            fetch_reference_photos(name, None, None)
        except Exception:
            app.logger.warning("reference prefetch failed for %s", name)


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
        if (os.environ.get("FINFINDR_PREFETCH_ALL") or "").strip() != "1":
            named = [region for region in named if region["id"] == "sea-of-cortez"]
        named.sort(key=lambda region: 0 if region["id"] == "sea-of-cortez" else 1)
        for region in named:
            query = species_query_for(region, None, None)
            try:
                payload = load_species_list(query, "short", 40, refresh_incomplete=True)
                if region["id"] == "sea-of-cortez":
                    _prefetch_reference_photos(payload.get("species") or [])
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


def _vision_payload(jpeg_bytes, prompt, model, extra_images=None, extra_detail="low"):
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    data_url = f"data:image/jpeg;base64,{b64}"
    content = [{"type": "input_image", "image_url": data_url, "detail": "high"}]
    for image_url in extra_images or []:
        if image_url:
            content.append({"type": "input_image", "image_url": image_url, "detail": extra_detail})
    content.append({"type": "input_text", "text": prompt})
    return {
        "model": model,
        "temperature": XAI_TEMPERATURE,
        "seed": resolve_seed(),
        "input": [{"role": "user", "content": content}],
    }


def resolve_seed():
    """XAI_SEED overrides 0. The responses API accepts seed for best-effort determinism."""
    raw = (os.environ.get("XAI_SEED") or "").strip()
    if not raw:
        return XAI_SEED
    try:
        return int(raw)
    except ValueError:
        return XAI_SEED


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


def _env_seconds(name, default, low, high):
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return float(default)
    try:
        value = float(raw)
    except ValueError:
        return float(default)
    return min(float(high), max(float(low), value))


def resolve_fallback_model(primary):
    """Faster model used once if the primary call times out or the server errors.

    XAI_FALLBACK_MODEL defaults to grok-4. An empty value disables the fallback.
    """
    if "XAI_FALLBACK_MODEL" in os.environ:
        raw = os.environ.get("XAI_FALLBACK_MODEL", "").strip()
    else:
        raw = "grok-4"
    if not _safe_model_name(raw) or raw == primary:
        return None
    return raw


def _identification_meta(mode, sent, available, truncated, source, list_ms, model_ms, tokens, token_source, model, image_px, fallback_used=False, attempts=None, verify_used=False, verify_ms=0, reference_fetch_ms=0, total_ms=None, verify_skipped=""):
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
        "fallback_used": bool(fallback_used),
        "attempts": attempts or [],
        "verify_used": bool(verify_used),
        "verify_skipped": verify_skipped or "",
        "verify_ms": int(verify_ms),
        "reference_fetch_ms": int(reference_fetch_ms),
        "total_ms": int(total_ms) if total_ms is not None else None,
    }


def _log_identification(region_id, meta):
    app.logger.info(
        "identify region=%s list_mode=%s species_sent=%s species_available=%s "
        "prompt_tokens=%s prompt_tokens_source=%s list_fetch_ms=%s model_call_ms=%s "
        "list_source=%s truncated=%s model=%s image_px=%s fallback_used=%s attempts=%s "
        "verify_used=%s verify_skipped=%s verify_ms=%s verify_success_rate=%s",
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
        meta.get("fallback_used"),
        meta.get("attempts"),
        meta.get("verify_used"),
        meta.get("verify_skipped"),
        meta.get("verify_ms"),
        meta.get("verify_success_rate"),
    )


def _attempt_record(model, elapsed_ms, ok, error=None):
    record = {"model": model, "ms": int(elapsed_ms), "ok": bool(ok)}
    if error:
        record["error"] = error
    return record


def _call_vision_model_once(jpeg_bytes, prompt, model, timeout, api_key, extra_images=None, extra_detail="low"):
    """One HTTP attempt. A 429 is retried with backoff. It never becomes a 502."""
    started = time.perf_counter()
    payload = _vision_payload(jpeg_bytes, prompt, model, extra_images, extra_detail=extra_detail)
    resp = None
    for attempt in range(1, 4):
        _bump_stat("external_calls")
        try:
            resp = http.post(
                "https://api.x.ai/v1/responses",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=max(1.0, timeout),
            )
        except requests.Timeout:
            elapsed = int(round((time.perf_counter() - started) * 1000))
            app.logger.warning("xAI timeout model=%s after %sms", model, elapsed)
            return {"ok": False, "retryable": True, "model": model, "ms": elapsed, "error": "timeout", "status": 504}
        except requests.RequestException as exc:
            elapsed = int(round((time.perf_counter() - started) * 1000))
            app.logger.warning("xAI connection error model=%s: %s", model, exc)
            return {"ok": False, "retryable": True, "model": model, "ms": elapsed, "error": "connection", "status": 502}
        if resp.status_code == 429 and attempt < 3:
            delay = _retry_after_seconds(resp, attempt) + random.uniform(0.0, 0.25)
            app.logger.warning("xAI 429 model=%s; retrying in %.2fs", model, delay)
            time.sleep(min(delay, 2.0))
            continue
        break
    elapsed = int(round((time.perf_counter() - started) * 1000))
    if resp is not None and resp.status_code == 429:
        app.logger.error("xAI 429 model=%s after retries", model)
        return {"ok": False, "retryable": True, "model": model, "ms": elapsed, "error": "rate_limited", "status": 200}
    if resp.status_code >= 500:
        app.logger.error("xAI HTTP %s model=%s: %s", resp.status_code, model, resp.text[:500])
        return {"ok": False, "retryable": True, "model": model, "ms": elapsed, "error": f"http_{resp.status_code}", "status": 502}
    if not resp.ok:
        app.logger.error("xAI HTTP %s model=%s: %s", resp.status_code, model, resp.text[:500])
        return {"ok": False, "retryable": False, "model": model, "ms": elapsed, "error": f"http_{resp.status_code}", "status": 502}
    try:
        data = resp.json()
    except ValueError:
        app.logger.warning("xAI invalid JSON model=%s", model)
        return {"ok": False, "retryable": True, "model": model, "ms": elapsed, "error": "invalid_json", "status": 502}
    return {"ok": True, "retryable": False, "model": model, "ms": elapsed, "data": data, "status": 200}


_MODEL_CACHE = {}
_MODEL_CACHE_LOCK = threading.Lock()
MODEL_CACHE_TTL = 6 * 60 * 60


def model_cache_enabled():
    """FINFINDR_MODEL_CACHE=0 turns off the image-hash cache. Default is on."""
    return (os.environ.get("FINFINDR_MODEL_CACHE", "1") or "1").strip().lower() not in ("0", "false", "off", "no")


def _model_cache_key(jpeg_bytes, prompt, model, extra_images):
    digest = hashlib.sha256()
    digest.update(jpeg_bytes or b"")
    digest.update(b"|")
    digest.update((model or "").encode())
    digest.update(b"|")
    digest.update(str(XAI_TEMPERATURE).encode())
    digest.update(b"|")
    digest.update(str(resolve_seed()).encode())
    for image in extra_images or []:
        digest.update(b"|")
        digest.update((image or "").encode())
    digest.update(b"|")
    digest.update((prompt or "").encode())
    return digest.hexdigest()


def clear_model_cache():
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()
    try:
        path = os.path.join(cache_dir(), "model.json")
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


def _model_cache_get(key):
    now = time.monotonic()
    with _MODEL_CACHE_LOCK:
        hit = _MODEL_CACHE.get(key)
        if hit and hit[0] > now:
            return hit[1]
        _MODEL_CACHE.pop(key, None)
    persisted = _disk_cache_get("model", key)
    if not isinstance(persisted, dict) or not persisted.get("ok"):
        return None
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE[key] = (time.monotonic() + 60, persisted)
    return persisted


def _model_cache_put(key, result):
    if not isinstance(result, dict) or not result.get("ok"):
        return
    with _MODEL_CACHE_LOCK:
        if len(_MODEL_CACHE) > 128 and key not in _MODEL_CACHE:
            _MODEL_CACHE.clear()
        _MODEL_CACHE[key] = (time.monotonic() + MODEL_CACHE_TTL, result)
    _disk_cache_put("model", key, result, MODEL_CACHE_TTL)


def _call_vision_model(jpeg_bytes, prompt, model, timeout, api_key, extra_images=None, extra_detail="low", use_cache=True):
    """One model attempt, stopped when the wall clock runs out even if the socket keeps trickling."""
    cache_key = _model_cache_key(jpeg_bytes, prompt, model, extra_images)
    if use_cache and model_cache_enabled():
        cached = _model_cache_get(cache_key)
        if cached is not None:
            replay = dict(cached)
            replay["cached"] = True
            return replay
    started = time.perf_counter()
    holder = {}

    def run():
        holder["result"] = _call_vision_model_once(
            jpeg_bytes, prompt, model, timeout, api_key, extra_images, extra_detail
        )

    worker = _start_thread(run, "finfindr-model")
    worker.join(max(0.2, timeout))
    if worker.is_alive():
        elapsed = int(round((time.perf_counter() - started) * 1000))
        app.logger.warning("xAI wall-clock timeout model=%s after %sms", model, elapsed)
        return {"ok": False, "retryable": True, "model": model, "ms": elapsed, "error": "timeout", "status": 504}
    result = holder.get("result") or {
        "ok": False, "retryable": True, "model": model,
        "ms": int(round((time.perf_counter() - started) * 1000)),
        "error": "failed", "status": 502,
    }
    if use_cache and model_cache_enabled():
        _model_cache_put(cache_key, result)
    return result


def plan_model_timeouts(remaining, primary_cap, fallback_cap, verify_reserve, fallback_enabled):
    """Give the primary model its full cap. Do not reserve a fallback up front.

    verify_reserve is unused on purpose: the reference check is trimmed after
    the primary call, and a fallback is started only after a real fast failure.
    fallback_timeout is how long one could run if the primary failed immediately.
    """
    del verify_reserve, fallback_enabled
    slack = 3.0
    primary = min(float(primary_cap), max(0.0, float(remaining) - slack))
    spare = max(0.0, float(remaining) - slack)
    fallback_timeout = min(float(fallback_cap), spare) if spare >= 8.0 else 0.0
    return max(0.0, primary), max(0.0, fallback_timeout)


def should_fallback(result, remaining, fallback_cap):
    """Fall back only after a connection error, 5xx, or an early timeout.

    A timeout counts only when the fallback can still finish. A short clock is
    not itself a reason to switch models.
    """
    if not isinstance(result, dict) or result.get("ok") or not result.get("retryable"):
        return False
    error = str(result.get("error") or "")
    if remaining < 8.0:
        return False
    if error == "connection" or error == "invalid_json" or error.startswith("http_5"):
        return True
    if error == "timeout":
        return remaining >= min(float(fallback_cap), 12.0) + 2.0
    return False


def answer_used_fallback(result, primary_model):
    """True only when the answer we return came from the fallback model.

    A parallel grok-4 row that lost to grok-4.7 is not a fallback. Treating it
    as one was downgrading confident identifications to likely.
    """
    if not isinstance(result, dict) or not result.get("ok"):
        return False
    return (result.get("model") or primary_model) != primary_model


def _record_parallel_fallback(fallback_thread, holder, fallback_model, attempts):
    """Keep the grok-4 attempt on the record even when the primary answer is the one we use."""
    if fallback_thread is None:
        return
    fallback_thread.join(0.05)
    fallback = holder.get("fallback")
    if not fallback:
        attempts.append(_attempt_record(fallback_model, 0, False, "still_running"))
        return
    attempts.append(_attempt_record(
        fallback_model, fallback.get("ms") or 0, fallback.get("ok"), None if fallback.get("ok") else fallback.get("error")
    ))


def _run_model_attempts(jpeg_bytes, prompt, model, api_key, request_started, extra_reserve=0, extra_images=None, use_cache=True):
    """Primary keeps most of its window. Fallback starts in parallel if it runs long.

    grok-4.7 still gets its cap (XAI_PRIMARY_TIMEOUT). If it is still running
    after XAI_FALLBACK_AFTER (default 40s), grok-4 starts beside it. The second
    wait is until the primary deadline, not another full timeout on top of the
    first wait. A fast failure still falls back immediately. Verification uses
    whatever budget remains.
    """
    del extra_reserve
    worker_budget = _env_seconds("FINFINDR_TIME_BUDGET", 110, 30, 115)
    primary_cap = _env_seconds("XAI_PRIMARY_TIMEOUT", 70, 5, 110)
    fallback_cap = _env_seconds("XAI_FALLBACK_TIMEOUT", 20, 5, 40)
    fallback_after = _env_seconds("XAI_FALLBACK_AFTER", 40, 5, 80)
    deadline = request_started + worker_budget
    attempts = []

    def remaining():
        return deadline - time.monotonic()

    def timed_out(name, elapsed):
        return {"ok": False, "retryable": True, "model": name, "ms": int(elapsed), "error": "timeout", "status": 504}

    fallback_model = resolve_fallback_model(model)
    # Do not hold verify time back from the primary. Verification uses what is left.
    primary_timeout = min(primary_cap, max(8.0, remaining() - 3.0))
    holder = {}

    def run_primary():
        holder["primary"] = _call_vision_model(
            jpeg_bytes, prompt, model, primary_timeout, api_key, extra_images, use_cache=use_cache
        )

    primary_started = time.monotonic()
    primary_deadline = primary_started + primary_timeout
    primary_thread = _start_thread(run_primary, "finfindr-primary")
    primary_thread.join(min(fallback_after, max(0.0, primary_deadline - time.monotonic())))
    fallback_thread = None
    if primary_thread.is_alive() and fallback_model and remaining() >= fallback_cap + 8:
        fallback_timeout = min(fallback_cap, max(5.0, remaining() - 3.0))

        def run_fallback():
            holder["fallback"] = _call_vision_model(
                jpeg_bytes, prompt, fallback_model, fallback_timeout, api_key, extra_images, use_cache=use_cache
            )

        fallback_thread = _start_thread(run_fallback, "finfindr-fallback")
    left_primary = primary_deadline - time.monotonic()
    if left_primary > 0:
        primary_thread.join(left_primary)
    if primary_thread.is_alive():
        result = timed_out(model, int(round((time.monotonic() - primary_started) * 1000)))
    else:
        result = holder.get("primary") or timed_out(model, primary_timeout * 1000)
    attempts.append(_attempt_record(model, result["ms"], result["ok"], None if result["ok"] else result.get("error")))
    if result.get("ok"):
        _record_parallel_fallback(fallback_thread, holder, fallback_model, attempts)
        return result, attempts
    if fallback_thread is not None:
        fallback_thread.join(max(0.2, remaining() - 1.0))
        fallback = holder.get("fallback")
        if fallback:
            attempts.append(_attempt_record(
                fallback_model, fallback["ms"], fallback["ok"], None if fallback["ok"] else fallback.get("error")
            ))
            return fallback, attempts
        return result, attempts
    if fallback_model and should_fallback(result, remaining(), fallback_cap):
        fallback_timeout = min(fallback_cap, max(1.0, remaining() - 2.0))
        if fallback_timeout >= 5.0:
            fallback = _call_vision_model(
                jpeg_bytes, prompt, fallback_model, fallback_timeout, api_key, extra_images, use_cache=use_cache
            )
            attempts.append(_attempt_record(
                fallback_model, fallback["ms"], fallback["ok"], None if fallback["ok"] else fallback.get("error")
            ))
            return fallback, attempts
    return result, attempts


_CANDIDATE_LINE = re.compile(r"^candidate\s*\d+\s*:\s*(.+)$", re.IGNORECASE)
_OFFLIST_LINE = re.compile(r"^off-?list\s*\d+\s*:\s*(.+)$", re.IGNORECASE)
_PHOTO_CACHE = {}
_PHOTO_LOCK = threading.Lock()
_TAXA_URL = "https://api.inaturalist.org/v1/taxa"
_OBS_URL = "https://api.inaturalist.org/v1/observations"
SPARSE_SPECIES_MIN = 12
WIDE_RADIUS_KM = 400


def verification_enabled(debug, raw):
    """On by default. A debug verify=0/1 field overrides FINFINDR_VERIFY."""
    if debug and (raw or "").strip() in ("0", "1"):
        return raw.strip() == "1"
    flag = (os.environ.get("FINFINDR_VERIFY", "1") or "1").strip().lower()
    return flag not in ("0", "false", "off", "no")


def verify_timeout_cap():
    """Seconds allowed for the reference-photo model call. FINFINDR_VERIFY_TIMEOUT overrides 12."""
    raw = (os.environ.get("FINFINDR_VERIFY_TIMEOUT") or "").strip()
    if not raw:
        return 12.0
    try:
        value = float(raw)
    except ValueError:
        return 25.0
    return min(40.0, max(5.0, value))


def verify_image_px():
    """Longest edge of the photo sent with the reference check. Default 512."""
    raw = (os.environ.get("FINFINDR_VERIFY_IMAGE_PX") or "").strip()
    if not raw:
        return 512
    try:
        value = int(raw)
    except ValueError:
        return 768
    return min(1536, max(256, value))


def verify_call_budget(request_started):
    """Seconds left for verification. None if fewer than 6 seconds remain.

    A fast primary keeps the configured cap (12s unless FINFINDR_VERIFY_TIMEOUT
    says otherwise). After a long primary, the leftover budget is used instead,
    up to 28s, so the reference check is not cut off at 12s.
    """
    left = _time_left(request_started) - 2.0
    if left < 6.0:
        return None
    cap = verify_timeout_cap()
    budget = _env_seconds("FINFINDR_TIME_BUDGET", 110, 30, 115)
    elapsed = budget - _time_left(request_started)
    if elapsed >= 45.0:
        return min(28.0, left)
    return min(cap, left)


def _shrink_jpeg(jpeg_bytes, limit):
    """Smaller JPEG for the reference check so that call can finish inside the budget."""
    if not jpeg_bytes or limit <= 0:
        return jpeg_bytes
    img = None
    try:
        img = Image.open(BytesIO(jpeg_bytes))
        img.load()
        if max(img.size) <= limit:
            return jpeg_bytes
        scale = limit / float(max(img.size))
        resized = img.resize(
            (
                max(1, int(round(img.size[0] * scale))),
                max(1, int(round(img.size[1] * scale))),
            ),
            Image.Resampling.LANCZOS,
        )
        if resized is not img:
            img.close()
            img = resized
        out = BytesIO()
        img.save(out, format="JPEG", quality=75, optimize=True)
        return out.getvalue()
    except Exception:
        app.logger.warning("Could not shrink the verification image", exc_info=True)
        return jpeg_bytes
    finally:
        if img is not None:
            try:
                img.close()
            except Exception:
                pass


_VERIFY_STATS = {"attempts": 0, "ok": 0}
_VERIFY_STATS_LOCK = threading.Lock()


def _record_verification(ok):
    with _VERIFY_STATS_LOCK:
        _VERIFY_STATS["attempts"] += 1
        if ok:
            _VERIFY_STATS["ok"] += 1
        attempts = _VERIFY_STATS["attempts"]
        rate = _VERIFY_STATS["ok"] / attempts if attempts else None
    return attempts, (None if rate is None else round(rate, 4))


def verification_success_rate():
    with _VERIFY_STATS_LOCK:
        attempts = _VERIFY_STATS["attempts"]
        if not attempts:
            return None
        return round(_VERIFY_STATS["ok"] / attempts, 4)


def _verify_skip_reason(result):
    error = str((result or {}).get("error") or "failed")
    if error == "timeout":
        return "timeout"
    if error == "connection":
        return "connection"
    if error == "invalid_json":
        return "invalid_json"
    return error[:40]


def _time_left(request_started):
    budget = _env_seconds("FINFINDR_TIME_BUDGET", 110, 30, 115)
    return request_started + budget - time.monotonic()


def load_species_for_request(region, lat, lng, mode, exclude_observation_id=None):
    """Use coordinates worldwide. Widen the radius once when the local list is sparse."""
    query = species_query_for(region, lat, lng)
    if query and exclude_observation_id:
        query = dict(query)
        query["not_id"] = str(exclude_observation_id)
        query["key"] = f"{query['key']}:not:{exclude_observation_id}"
    left = call_time_left()
    budget = 20.0 if mode == "short" else 30.0
    reserve = _env_seconds("XAI_PRIMARY_TIMEOUT", 80, 5, 100) + 4.0
    if left is not None:
        budget = min(budget, max(2.0, left - reserve))
    payload = load_species_list(query, mode, budget)
    if not query or not query.get("radius"):
        return payload
    count = len(payload.get("species") or [])
    if count >= SPARSE_SPECIES_MIN or query["radius"] >= WIDE_RADIUS_KM:
        return payload
    if call_time_left() is not None and call_time_left() < reserve:
        return payload
    wider = species_query_for(region, lat, lng, radius=WIDE_RADIUS_KM)
    if exclude_observation_id:
        wider = dict(wider)
        wider["not_id"] = str(exclude_observation_id)
        wider["key"] = f"{wider['key']}:not:{exclude_observation_id}"
    wider_payload = load_species_list(wider, mode, 8.0 if mode == "short" else 12.0)
    if len(wider_payload.get("species") or []) > count:
        return wider_payload
    return payload


def load_species_alongside_model(region, lat, lng, mode, exclude_observation_id=None):
    """Start the regional list beside the model. Wait at most two seconds.

    A cached or prefetched list returns immediately. A slow iNaturalist crawl
    does not delay grok-4.7; this request then uses the curated names and the
    crawl fills the disk cache for the next photo.
    """
    started = time.perf_counter()
    holder = {}

    def run():
        holder["payload"] = load_species_for_request(region, lat, lng, mode, exclude_observation_id)

    worker = _start_thread(run, "finfindr-list")
    worker.join(2.0)
    if "payload" in holder:
        return holder["payload"], int(round((time.perf_counter() - started) * 1000))
    query = species_query_for(region, lat, lng)
    cached = _species_cache_get((mode, query["key"])) if query else None
    if cached and cached.get("species"):
        return cached, int(round((time.perf_counter() - started) * 1000))
    payload = _curated_species_payload() if query and query.get("curated_fallback") else _empty_species()
    return payload, int(round((time.perf_counter() - started) * 1000))


def _candidate_from_parts(parts, off_list=False):
    if not parts or parts[0].lower() in ("", "none"):
        return None
    scientific = exact_scientific_name(parts[0])
    if not scientific:
        return None
    confidence = parts[2].lower() if len(parts) > 2 else "medium"
    if confidence not in ("high", "medium", "low"):
        confidence = "medium"
    trait = parts[3] if len(parts) > 3 else (parts[2] if off_list and len(parts) > 2 and confidence == "medium" and parts[2].lower() not in ("high", "medium", "low") else "")
    if off_list and len(parts) == 3 and parts[2].lower() not in ("high", "medium", "low"):
        confidence = "medium"
        trait = parts[2]
    return {
        "scientific_name": scientific,
        "common_name": parts[1] if len(parts) > 1 and parts[1].lower() != "none" else "",
        "confidence": confidence,
        "trait": trait,
        "off_list": off_list,
        "reference_photo": None,
    }


def parse_model_candidates(text):
    """Top species from the first pass, plus up to two off-list proposals."""
    normalized = _result_lines(text or "")
    found = []
    off_list = []
    seen = set()
    for line in normalized.split("\n"):
        stripped = line.strip()
        off_match = _OFFLIST_LINE.match(stripped)
        match = _CANDIDATE_LINE.match(stripped)
        if off_match:
            item = _candidate_from_parts([part.strip() for part in off_match.group(1).split("|")], off_list=True)
            if item and item["scientific_name"].lower() not in seen and len(off_list) < 2:
                off_list.append(item)
                seen.add(item["scientific_name"].lower())
            continue
        if not match or len(found) >= 3:
            continue
        item = _candidate_from_parts([part.strip() for part in match.group(1).split("|")], off_list=False)
        if not item or item["scientific_name"].lower() in seen:
            continue
        if "off_list" in stripped.lower() or "off-list" in stripped.lower():
            item["off_list"] = True
            if len(off_list) < 2:
                off_list.append(item)
                seen.add(item["scientific_name"].lower())
            continue
        found.append(item)
        seen.add(item["scientific_name"].lower())
    if not found:
        scientific = exact_scientific_name(_field_value(normalized, "Scientific name"))
        if scientific and scientific.lower() not in seen:
            confidence = _field_value(normalized, "Confidence").lower()
            if confidence not in ("high", "medium", "low"):
                confidence = "medium"
            found.append({
                "scientific_name": scientific,
                "common_name": _field_value(normalized, "Common name"),
                "confidence": confidence,
                "trait": "",
                "off_list": False,
                "reference_photo": None,
            })
            seen.add(scientific.lower())
    top = exact_scientific_name(_field_value(normalized, "Scientific name"))
    if top and top.lower() not in seen and len(off_list) < 2:
        off_list.append({
            "scientific_name": top,
            "common_name": _field_value(normalized, "Common name"),
            "confidence": "medium",
            "trait": "",
            "off_list": True,
            "reference_photo": None,
        })
    return found + off_list


def _medium_photo_url(url):
    if not url or not isinstance(url, str):
        return ""
    return (
        url.replace("/square.", "/small.")
        .replace("/square.jpg", "/small.jpg")
        .replace("/medium.", "/small.")
        .replace("/large.", "/small.")
    )


def _photo_record(photo, observation_id=None):
    if not isinstance(photo, dict):
        return None
    url = _medium_photo_url(photo.get("url") or photo.get("medium_url") or "")
    if not url.startswith("https://"):
        return None
    attribution = (photo.get("attribution") or "").strip()
    license_code = (photo.get("license_code") or "").strip()
    license_name = license_code.upper().replace("CC-", "CC ") if license_code else ""
    photo_id = photo.get("id")
    source = f"https://www.inaturalist.org/photos/{photo_id}" if photo_id else "https://www.inaturalist.org"
    return {
        "url": url,
        "attribution": attribution or "iNaturalist contributor",
        "license": license_name,
        "source_url": source,
        "observation_id": observation_id or photo.get("observation_id") or photo.get("native_photo_id"),
    }


def _cache_photos(key, photos, ttl):
    with _PHOTO_LOCK:
        if len(_PHOTO_CACHE) > 256:
            _PHOTO_CACHE.clear()
        _PHOTO_CACHE[key] = (time.monotonic() + ttl, photos)


def _cached_photos(key):
    now = time.monotonic()
    with _PHOTO_LOCK:
        hit = _PHOTO_CACHE.get(key)
        if not hit or hit[0] <= now:
            _PHOTO_CACHE.pop(key, None)
            return None
        return hit[1]


def _inat_json(url, params, timeout):
    return _inat_get(
        url,
        params,
        timeout,
        {
            "User-Agent": "FinFindr/1.0 (reference photo lookup; https://github.com/fred123-eng/Finfindr)",
            "Accept": "application/json",
        },
    )


def confirm_species_name(scientific):
    """True when iNaturalist has this exact species name.

    None means the lookup did not answer (timeout or no time left). Callers
    keep an off-list proposal in that case instead of dropping it.
    """
    if not scientific:
        return False
    if call_time_left() is not None and call_time_left() < 1.0:
        return None
    payload = _inat_json(
        _TAXA_URL,
        {"q": scientific, "rank": "species", "per_page": 5, "locale": "en"},
        INAT_TIMEOUT,
    )
    if payload is None:
        return None
    for item in (payload or {}).get("results") or []:
        if (
            isinstance(item, dict)
            and item.get("rank") in _SPECIES_RANKS
            and (item.get("name") or "").lower() == scientific.lower()
        ):
            return True
    return False


def _prepare_off_list_candidates(candidates, species):
    """Keep off-list names only when they are real taxa and not already in the regional list."""
    listed = {(item.get("scientific") or "").lower() for item in species or [] if isinstance(item, dict)}
    kept = []
    checked = 0
    for item in candidates:
        if not item.get("off_list"):
            kept.append(item)
            continue
        if item["scientific_name"].lower() in listed:
            item["off_list"] = False
            kept.append(item)
            continue
        if checked >= 2:
            continue
        checked += 1
        status = confirm_species_name(item["scientific_name"])
        if status is False:
            continue
        kept.append(item)
    return kept


def _reference_cache_key(scientific, lat, lng, exclude_observation_id):
    exclude = exclude_observation_id or ""
    if lat is not None and lng is not None:
        return f"ref:{scientific.lower()}:{round(float(lat), 1):.1f}:{round(float(lng), 1):.1f}:{exclude}"
    return f"ref:{scientific.lower()}::{exclude}"


def fetch_reference_photos(scientific, lat, lng, exclude_observation_id=None, deadline=None):
    """Research-grade reference photos for one species, preferring the user's area.

    Startup prefetch stores photos without coordinates. A later check reuses
    that cache instead of waiting on iNaturalist again.
    """
    if deadline is not None and time.monotonic() >= deadline:
        return []
    specific = _reference_cache_key(scientific, lat, lng, exclude_observation_id)
    generic = _reference_cache_key(scientific, None, None, exclude_observation_id)
    cached = _cached_photos(specific)
    if cached is None and specific != generic:
        warmed = _cached_photos(generic)
        if warmed:
            return warmed
    elif cached is not None:
        return cached
    taxon_payload = _inat_json(
        _TAXA_URL,
        {"q": scientific, "rank": "species", "per_page": 5, "locale": "en"},
        INAT_TIMEOUT,
    )
    taxon = None
    for item in (taxon_payload or {}).get("results") or []:
        if (
            isinstance(item, dict)
            and item.get("rank") in _SPECIES_RANKS
            and (item.get("name") or "").lower() == scientific.lower()
        ):
            taxon = item
            break
    photos = []
    if taxon:
        for wrapper in taxon.get("taxon_photos") or []:
            photo = wrapper.get("photo") if isinstance(wrapper, dict) else None
            record = _photo_record(photo)
            if record and isinstance(wrapper, dict) and wrapper.get("observation_id"):
                record["observation_id"] = wrapper.get("observation_id")
            if record:
                photos.append(record)
        if not photos:
            record = _photo_record(taxon.get("default_photo"))
            if record:
                photos.append(record)
        if exclude_observation_id:
            photos = [photo for photo in photos if str(photo.get("observation_id") or "") != str(exclude_observation_id)]
        if not photos and lat is not None and lng is not None and (deadline is None or time.monotonic() < deadline):
            obs_payload = _inat_json(
                _OBS_URL,
                {
                    "taxon_id": taxon.get("id"),
                    "quality_grade": "research",
                    "photos": "true",
                    "lat": lat,
                    "lng": lng,
                    "radius": 100,
                    "per_page": 5,
                    "order_by": "votes",
                    "locale": "en",
                },
                (0.6, 1.5),
            )
            nearby = []
            for obs in (obs_payload or {}).get("results") or []:
                if not isinstance(obs, dict):
                    continue
                if exclude_observation_id and str(obs.get("id")) == str(exclude_observation_id):
                    continue
                for photo in obs.get("photos") or []:
                    record = _photo_record(photo, obs.get("id"))
                    if record and str(record.get("observation_id")) != str(exclude_observation_id or ""):
                        nearby.append(record)
                        break
            if nearby:
                photos = nearby + [photo for photo in photos if photo["url"] not in {item["url"] for item in nearby}]
    if exclude_observation_id:
        photos = [photo for photo in photos if str(photo.get("observation_id") or "") != str(exclude_observation_id)]
    photos.sort(key=lambda photo: (str(photo.get("url") or ""), str(photo.get("source_url") or "")))
    photos = photos[:2]
    ttl = INAT_CACHE_TTL if photos else INAT_ERROR_TTL
    _cache_photos(specific, photos, ttl)
    if photos and specific != generic:
        _cache_photos(generic, photos, ttl)
    return photos


def _attach_reference_photos(candidates, lat, lng, exclude_observation_id, deadline):
    def one(candidate):
        if deadline is not None and time.monotonic() >= deadline:
            return candidate
        photos = fetch_reference_photos(
            candidate["scientific_name"], lat, lng, exclude_observation_id, deadline
        )
        if photos:
            candidate["reference_photo"] = photos[0]
            candidate["_reference_urls"] = [photo["url"] for photo in photos[:1]]
        return candidate

    if not candidates:
        return candidates
    with ThreadPoolExecutor(max_workers=min(3, len(candidates))) as pool:
        futures = [_submit(pool, one, candidate) for candidate in candidates]
        attached = []
        for future, candidate in zip(futures, candidates):
            try:
                attached.append(future.result(timeout=2.5))
            except Exception:
                app.logger.warning("reference photo timed out for %s", candidate.get("scientific_name"))
                attached.append(candidate)
        return attached


def _verification_prompt(candidates):
    lines = [
        "The first image is the animal to identify.",
        "Each later image is a research-grade reference photo of one candidate. Compare diagnostic traits you can see. Do not pick a candidate just because it was listed first.",
        "If the photos do not separate the candidates, say you cannot decide.",
        "Compare the leading candidate with the runner-up. Choose the leading one only when a visible diagnostic trait supports it over the other.",
        "If the animal matches none of the reference photos, the decision is NONE_OF_THESE. Do not force the closest name.",
        "NONE_OF_THESE is a valid answer. It means none of these species.",
        "If a clear trait such as color, shape, or pattern disagrees with every reference photo, the decision is NONE_OF_THESE.",
        "",
    ]
    for index, candidate in enumerate(candidates, start=1):
        photo = candidate.get("reference_photo") or {}
        credit = photo.get("attribution") or "iNaturalist"
        lines.append(
            f"Reference {index}: {candidate['scientific_name']} ({candidate.get('common_name') or 'no common name'}). "
            f"Photo: {credit}, {photo.get('license') or 'see iNaturalist'}."
        )
        if candidate.get("trait"):
            lines.append(f"Trait already suggested: {candidate['trait']}")
    lines.extend([
        "",
        "Reply in this exact format:",
        "Decision: <scientific name of the matching candidate, UNSURE, or NONE_OF_THESE>",
        "Certainty: <confident / likely / unsure>",
        "Trait: <the visible trait that separates the choice, or what is missing>",
    ])
    return "\n".join(lines)


def _parse_verification(text, candidates):
    normalized = _result_lines(text or "")
    decision = _field_value(normalized, "Decision")
    certainty = _field_value(normalized, "Certainty").lower()
    trait = _field_value(normalized, "Trait")
    if certainty not in ("confident", "likely", "unsure"):
        certainty = "unsure"
    normalized_decision = (decision or "").upper().replace(" ", "_").replace("-", "_")
    if normalized_decision in ("", "UNSURE", "NO_MATCH", "NONE", "NONE_OF_THESE", "NEITHER"):
        kind = "NO_MATCH" if normalized_decision in ("NO_MATCH", "NONE_OF_THESE", "NONE", "NEITHER") else "UNSURE"
        return {"decision": kind, "certainty": "unsure", "trait": trait, "scientific_name": ""}
    scientific = exact_scientific_name(decision)
    if not scientific:
        return {"decision": "UNSURE", "certainty": "unsure", "trait": trait, "scientific_name": ""}
    known = {item["scientific_name"].lower() for item in candidates}
    if scientific.lower() not in known:
        return {"decision": "UNSURE", "certainty": "unsure", "trait": trait, "scientific_name": scientific}
    return {"decision": scientific, "certainty": certainty, "trait": trait, "scientific_name": scientific}


def _order_candidates(candidates, scientific):
    if not scientific:
        return candidates
    ordered = [item for item in candidates if item["scientific_name"].lower() == scientific.lower()]
    ordered.extend(item for item in candidates if item["scientific_name"].lower() != scientific.lower())
    return ordered


def candidates_far_apart(candidates):
    """A lone high-confidence pick, or a high pick with only a low alternative.

    This used to skip the reference check and mark the answer confident. It no
    longer changes certainty. A wide gap is not a substitute for reference photos.
    """
    if not candidates or candidates[0].get("confidence") != "high":
        return False
    if len(candidates) == 1:
        return True
    return candidates[1].get("confidence") == "low"


_DIAGNOSTIC_TRAIT = re.compile(
    r"\b(?:bars?|barred|stripes?|striped|spots?|spotted|bands?|banded|spines?|"
    r"arms?|fins?|rays?|tails?|snouts?|discs?|knobs?|lattices?|mesh|tiles?|"
    r"scales?|tentacles?|shells?|eyes?|dorsal|pectoral|saddle|hump|cirri|"
    r"prickles?|oval|reticulat\w*|cylindrical|inflated|slender|globular|"
    r"mantles?|rhinophores?|gills?|ocelli|ocellus|papillae|papilla|ridges?|"
    r"shells?|whorls?|apertures?|blunt|stocky)\b",
    re.IGNORECASE,
)


def explicit_diagnostic_trait(trait):
    """A visible feature that separates species. Vague similarity does not count.

    A short phrase such as "5 bars" is explicit. "similar" is not. The old
    8-character cutoff treated "5 bars" as a failed check and then lowered
    every candidate, so a correct rank-1 fish came back all-low.
    """
    text = (trait or "").strip()
    lowered = text.lower().strip(" .")
    if len(lowered) < 4:
        return False
    if re.fullmatch(r"(very |quite |just |looks )?(similar|the same|same|matches|match|alike)( looking| species)?", lowered):
        return False
    if "similar" in lowered and not any(word in lowered for word in ("unlike", "differs", "different", "not ", "instead", "distinct", "rather")):
        return False
    if len(text) >= 8:
        return True
    return bool(_DIAGNOSTIC_TRAIT.search(lowered))


def passing_verification(verification):
    """True only when the reference check names a candidate, is confident, and cites a real trait."""
    if not isinstance(verification, dict):
        return False
    decision = (verification.get("decision") or "").strip()
    scientific = (verification.get("scientific_name") or "").strip()
    if not decision or not scientific or decision.lower() != scientific.lower():
        return False
    if decision.upper().replace(" ", "_") in ("UNSURE", "NO_MATCH", "NO MATCH", "NONE_OF_THESE"):
        return False
    if verification.get("certainty") != "confident":
        return False
    return explicit_diagnostic_trait(verification.get("trait"))


def verification_chose_top(verification, candidates):
    """The reference check named the leading candidate rather than the runner-up or none."""
    if not passing_verification(verification) or not candidates:
        return False
    chosen = (verification.get("scientific_name") or "").lower()
    if chosen != (candidates[0].get("scientific_name") or "").lower():
        return False
    return True


def verification_mismatch(verification):
    """The reference check rejected the candidates, including a clear color mismatch."""
    if not isinstance(verification, dict) or passing_verification(verification):
        return False
    if (verification.get("decision") or "").upper().replace(" ", "_") == "NO_MATCH":
        return True
    trait = (verification.get("trait") or "").lower()
    cues = (
        "does not match", "doesn't match", "do not match", "don't match",
        "no match", "none of the reference", "wrong color", "wrong colour",
        "different color", "different colour", "color does not", "colour does not",
        "main color", "main colour",
    )
    return any(cue in trait for cue in cues)


def cap_unconfirmed_confidence(candidates):
    """A failed reference check must not leave a candidate labeled high."""
    for item in candidates:
        if item.get("confidence") == "high":
            item["confidence"] = "low"
    return candidates


def certainty_for(candidates, verification):
    """Confident only after a passing reference-photo check. Otherwise at most likely."""
    if not candidates:
        return "unsure"
    if passing_verification(verification):
        return "confident"
    if verification is None:
        return "unsure" if candidates[0].get("confidence") == "low" else "likely"
    if verification.get("decision") in ("UNSURE", "NO_MATCH", "") or verification.get("certainty") == "unsure":
        return "unsure"
    return "likely"


_CONFIDENCE_RANK = {"high": 0, "medium": 1, "low": 2}


def sort_candidates(candidates):
    """High before medium before low, keeping the model's order inside one level."""
    ranked = list(enumerate(candidates or []))
    ranked.sort(key=lambda pair: (_CONFIDENCE_RANK.get((pair[1].get("confidence") or "medium"), 1), pair[0]))
    return [item for _, item in ranked]


def public_candidates(candidates, species):
    candidates = sort_candidates(candidates)
    chosen = []
    seen = set()
    in_list = 0
    off_list = 0
    for item in candidates:
        key = (item.get("scientific_name") or "").lower()
        if not key or key in seen:
            continue
        if item.get("off_list"):
            if off_list >= 2:
                continue
            off_list += 1
        else:
            if in_list >= 3:
                continue
            in_list += 1
        chosen.append(item)
        seen.add(key)
    resolved = resolve_common_names([item["scientific_name"] for item in chosen], species)
    cleaned = []
    for item in chosen:
        common = resolved.get(item["scientific_name"]) or item.get("common_name") or ""
        if common.lower() == (item.get("scientific_name") or "").lower():
            common = ""
        photo = item.get("reference_photo") or None
        if photo:
            photo = {
                "url": photo.get("url") or "",
                "attribution": photo.get("attribution") or "",
                "license": photo.get("license") or "",
                "source_url": photo.get("source_url") or "",
            }
        cleaned.append({
            "scientific_name": item["scientific_name"],
            "common_name": common,
            "confidence": item.get("confidence") or "medium",
            "trait": item.get("trait") or "",
            "off_list": bool(item.get("off_list")),
            "reference_photo": photo,
        })
    return cleaned


def discovery_enabled(debug, raw):
    """On by default. A debug discovery=0/1 field overrides FINFINDR_DISCOVERY."""
    if debug and (raw or "").strip() in ("0", "1"):
        return raw.strip() == "1"
    flag = (os.environ.get("FINFINDR_DISCOVERY", "1") or "1").strip().lower()
    return flag not in ("0", "false", "off", "no")


def range_km():
    """Radius for the range-extension rule. FINFINDR_RANGE_KM overrides 100."""
    raw = (os.environ.get("FINFINDR_RANGE_KM") or "").strip()
    if not raw:
        return 100
    try:
        value = float(raw)
    except ValueError:
        return 100
    return min(500, max(1, int(round(value))))


def range_record_limit():
    """Maximum research-grade records inside the radius that still count as unrecorded.

    The default is 0: no research-grade iNaturalist records within FINFINDR_RANGE_KM.
    Set FINFINDR_RANGE_MAX to allow a very small count.
    """
    raw = (os.environ.get("FINFINDR_RANGE_MAX") or "").strip()
    if not raw:
        return 0
    try:
        value = int(raw)
    except ValueError:
        return 0
    return min(10, max(0, value))


def range_rule_text():
    limit = range_record_limit()
    distance = range_km()
    if limit <= 0:
        return f"no research-grade iNaturalist records of the species within {distance} km"
    return f"at most {limit} research-grade iNaturalist records of the species within {distance} km"


def range_rule():
    return {
        "max_records": range_record_limit(),
        "radius_km": range_km(),
        "text": range_rule_text(),
    }


def parse_photo_quality(text):
    value = _field_value(_result_lines(text or ""), "Photo quality").lower()
    if value in ("good", "poor", "invalid"):
        return value
    return "good"


_UNUSABLE_PHOTO = re.compile(
    r"\b(?:blank|empty frame|tiny|too small|no animal|not an animal|not a living|"
    r"not a real animal|no visible animal|drawing|illustration|cartoon|painting|"
    r"nothing visible|no subject)\b",
    re.IGNORECASE,
)


def photo_truly_unusable(quality, issue):
    """Murky or soft photos can still be identified. Blank, tiny, or no-animal frames cannot."""
    if quality != "invalid":
        return False
    return bool(_UNUSABLE_PHOTO.search(issue or ""))


_CLEAR_LAND_ANIMAL = re.compile(
    r"\b(?:domestic cat|house cat|felis catus|domestic dog|canis familiaris|"
    r"puppy|kitten|horse|cow|cattle|sheep|goat|chicken|livestock|pet dog|pet cat)\b",
    re.IGNORECASE,
)


def text_says_not_marine(text):
    """True only for a clear land pet or livestock when no taxon was resolved.

    Phrases such as "on grass" or "not a marine animal" are not enough: a sea
    lion on rocks and a gull on the beach are in scope.
    """
    return bool(_CLEAR_LAND_ANIMAL.search(text or ""))


def mark_unlisted_candidates(candidates, species):
    """A medium or low name that is not in the regional list is an off-list proposal."""
    listed = {(item.get("scientific") or "").lower() for item in species or [] if isinstance(item, dict)}
    promoted = 0
    for item in candidates:
        if item.get("off_list"):
            continue
        if item.get("scientific_name", "").lower() in listed:
            continue
        if (item.get("confidence") or "medium") in ("low", "medium"):
            item["off_list"] = True
            promoted += 1
    return promoted


def app_commit():
    return (os.environ.get("RENDER_GIT_COMMIT") or os.environ.get("GIT_COMMIT") or "").strip()


def apply_verify_rates(meta, attempted, ok, checked, passed, skipped):
    """Per-request rates. A skipped check is 0 with a reason, never null."""
    reason = skipped or ""
    meta["verify_skipped"] = reason
    meta["verify_attempted"] = bool(attempted)
    if attempted and checked:
        meta["verify_pass_rate"] = round(passed / checked, 4)
        meta["verify_pass_reason"] = reason
    else:
        meta["verify_pass_rate"] = 0
        meta["verify_pass_reason"] = reason or "not_run"
    meta["verify_success_rate"] = (1.0 if ok else 0.0) if attempted else 0.0
    meta["commit"] = app_commit()
    return meta


def haversine_km(lat1, lng1, lat2, lng2):
    radius = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(min(1.0, math.sqrt(a)))


def _obs_coordinates(obs):
    geo = (obs.get("geojson") or {}).get("coordinates") or []
    if len(geo) == 2 and geo[0] is not None and geo[1] is not None:
        return float(geo[1]), float(geo[0])
    location = obs.get("location") or ""
    if isinstance(location, str) and "," in location:
        lat_text, lng_text = location.split(",", 1)
        try:
            return float(lat_text), float(lng_text)
        except ValueError:
            return None, None
    return None, None


def count_research_records(scientific, lat, lng, radius, place_id=None, bbox=None, not_id=None):
    """How many research-grade records iNaturalist has in this radius or place. None if the lookup fails."""
    params = {
        "taxon_name": scientific,
        "quality_grade": "research",
        "verifiable": "true",
        "per_page": 1,
        "locale": "en",
    }
    if not_id:
        params["not_id"] = not_id
    if lat is not None and lng is not None:
        params.update({"lat": lat, "lng": lng, "radius": radius})
    elif place_id:
        params["place_id"] = place_id
    elif bbox:
        params.update(bbox)
    else:
        return None
    payload = _inat_json(_OBS_URL, params, INAT_TIMEOUT)
    if not payload or "total_results" not in payload:
        return None
    try:
        return int(payload.get("total_results") or 0)
    except (TypeError, ValueError):
        return None


def nearest_research_records(scientific, lat, lng, limit=3):
    """Nearest research-grade observations, sorted by distance so the order is stable."""
    if lat is None or lng is None:
        return []
    payload = _inat_json(_OBS_URL, {
        "taxon_name": scientific,
        "quality_grade": "research",
        "geo": "true",
        "lat": lat,
        "lng": lng,
        "radius": 500,
        "per_page": 20,
        "order_by": "observed_on",
        "order": "desc",
        "locale": "en",
    }, INAT_TIMEOUT)
    rows = list((payload or {}).get("results") or [])
    if not rows:
        payload = _inat_json(_OBS_URL, {
            "taxon_name": scientific,
            "quality_grade": "research",
            "geo": "true",
            "per_page": 20,
            "order_by": "observed_on",
            "order": "desc",
            "locale": "en",
        }, INAT_TIMEOUT)
        rows = list((payload or {}).get("results") or [])
    found = []
    for obs in rows:
        if not isinstance(obs, dict) or not obs.get("id"):
            continue
        olat, olng = _obs_coordinates(obs)
        if olat is None:
            continue
        found.append({
            "distance_km": round(haversine_km(lat, lng, olat, olng), 1),
            "place": obs.get("place_guess") or "",
            "date": obs.get("observed_on") or "",
            "url": f"https://www.inaturalist.org/observations/{obs['id']}",
        })
    found.sort(key=lambda item: (item["distance_km"], item["url"]))
    return found[:limit]


_COLOR_WORDS = {
    "red": "red", "orange": "orange", "yellow": "yellow", "green": "green",
    "blue": "blue", "purple": "purple", "pink": "pink", "black": "black",
    "white": "white", "brown": "brown", "grey": "gray", "gray": "gray",
    "scarlet": "red", "crimson": "red", "azure": "blue",
}


def _color_set(text):
    words = re.findall(r"[a-z]+", (text or "").lower())
    return {_COLOR_WORDS[word] for word in words if word in _COLOR_WORDS}


def colors_contradict(visible_text, species_text):
    """True when the photo notes name a color the species name does not share."""
    visible = _color_set(visible_text)
    named = _color_set(species_text)
    if not visible or not named:
        return False
    return visible.isdisjoint(named)


def _without_negations(text):
    return re.sub(r"\bnot\b[^,.;]*", " ", text or "", flags=re.IGNORECASE)


def traits_contradict(photo_text, species_text):
    """Color, shape, or pattern in the photo description that the species does not have."""
    if colors_contradict(photo_text, _without_negations(species_text)):
        return True
    photo = (photo_text or "").lower()
    species = (species_text or "").lower()
    positive = _without_negations(species_text).lower()
    lattice = ("lattice", "tile", "red mesh", "mesh pattern")
    denies_lattice = any(phrase in species for phrase in (
        "not a red tile lattice", "not a red lattice", "no red lattice", "not a cream lattice",
    ))
    if any(word in photo for word in lattice) and denies_lattice:
        return True
    photo_slender = any(word in photo for word in ("slender", "cylindrical arm"))
    species_inflated = any(word in positive for word in ("inflated", "short broad", "cushion"))
    species_slender = any(word in positive for word in ("slender", "cylindrical"))
    if photo_slender and species_inflated and not species_slender:
        return True
    photo_inflated = any(word in photo for word in ("inflated", "knobby", "conical knob"))
    if photo_inflated and species_slender and not species_inflated:
        return True
    return False


def scope_setting():
    """FINFINDR_SCOPE selects which ancestry list is in scope. Default is marine."""
    raw = (os.environ.get("FINFINDR_SCOPE") or "marine").strip().lower()
    return raw or "marine"


def allowed_scope_names():
    """None means every animal is in scope. Otherwise a set of ancestry names."""
    mode = scope_setting()
    if mode in ("all", "any", "off"):
        return None
    if mode == "marine":
        return MARINE_SCOPE_ANCESTORS
    names = {part.strip().lower() for part in mode.split(",") if part.strip()}
    return names or MARINE_SCOPE_ANCESTORS


def _ancestor_names(taxon):
    names = set()
    if not isinstance(taxon, dict):
        return names
    for key in ("name", "iconic_taxon_name"):
        if taxon.get(key):
            names.add(str(taxon[key]).strip().lower())
    for ancestor in taxon.get("ancestors") or []:
        if isinstance(ancestor, dict) and ancestor.get("name"):
            names.add(str(ancestor["name"]).strip().lower())
        elif isinstance(ancestor, str) and ancestor.strip():
            names.add(ancestor.strip().lower())
    return names


def ancestry_in_scope(taxon, allowed=None):
    """True, False, or None when the ancestry does not decide.

    A land group inside a marine phylum, such as a land snail, is out of scope
    unless the active scope list names that land group itself.
    """
    if allowed is None:
        allowed = allowed_scope_names()
    if allowed is None:
        return True
    names = _ancestor_names(taxon)
    if not names:
        return None
    land_hits = names & TERRESTRIAL_SCOPE_EXCEPTIONS
    if land_hits and not (land_hits & allowed):
        return False
    if names & allowed:
        return True
    iconic = str((taxon or {}).get("iconic_taxon_name") or "").strip().lower()
    # Mammalia and Aves contain both marine and land species. An iNaturalist
    # search often returns the class and an empty ancestor list. That is not
    # enough to call the animal terrestrial.
    if iconic in ("mammalia", "aves") or names & {"mammalia", "aves"}:
        if not _ancestors_resolved(taxon):
            return None
        return False
    if names & LAND_ICONIC_TAXA:
        return False
    return None


def _ancestors_resolved(taxon):
    """True when the taxon record includes named ancestors, not only a class."""
    if not isinstance(taxon, dict):
        return False
    for ancestor in taxon.get("ancestors") or []:
        if isinstance(ancestor, dict) and ancestor.get("name"):
            return True
        if isinstance(ancestor, str) and ancestor.strip():
            return True
    return False


def _taxon_from_payload(payload, scientific):
    if not isinstance(payload, dict):
        return None
    wanted = (scientific or "").strip().lower()
    results = payload.get("results")
    if isinstance(results, list):
        for item in results:
            if isinstance(item, dict) and (item.get("name") or "").strip().lower() == wanted:
                return item
        if len(results) == 1 and isinstance(results[0], dict):
            return results[0]
        return None
    if (payload.get("name") or "").strip().lower() == wanted or payload.get("ancestors") or payload.get("ancestor_ids"):
        return payload
    return None


def _worms_is_marine(scientific):
    """WoRMS/OBIS isMarine. True, False, or None when the lookup does not answer."""
    if not scientific or (call_time_left() is not None and call_time_left() < 1.5):
        return None
    quoted = requests.utils.quote(scientific)
    payload = _external_json(
        f"https://www.marinespecies.org/rest/AphiaRecordsByName/{quoted}",
        {"like": "false", "marine_only": "false"},
    )
    rows = payload if isinstance(payload, list) else None
    if rows is None and isinstance(payload, dict):
        rows = payload.get("results") or payload.get("data")
    if not isinstance(rows, list):
        return None
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = (row.get("scientificname") or row.get("scientificName") or row.get("name") or "")
        if name and name.lower() != scientific.lower():
            continue
        for key in ("isMarine", "is_marine", "marine"):
            if key not in row or row.get(key) is None:
                continue
            if row.get(key) is True or str(row.get(key)).lower() in ("true", "1"):
                return True
            if row.get(key) is False or str(row.get(key)).lower() in ("false", "0"):
                brackish = row.get("isBrackish", row.get("is_brackish"))
                if brackish is True or str(brackish).lower() in ("true", "1"):
                    return True
                return False
    return None


def fetch_inat_taxon(scientific):
    """Species record, including ancestors from /v1/taxa/{id}.

    The search endpoint often returns iconic_taxon_name and an empty ancestors
    list. The id endpoint has the ancestor names the scope check needs. Both
    responses are cached by _inat_json.
    """
    if not scientific:
        return None
    payload = _inat_json(
        _TAXA_URL,
        {"q": scientific, "rank": "species", "per_page": 5, "locale": "en"},
        INAT_TIMEOUT,
    )
    match = None
    for item in (payload or {}).get("results") or []:
        if (
            isinstance(item, dict)
            and item.get("rank") in _SPECIES_RANKS
            and (item.get("name") or "").lower() == scientific.lower()
        ):
            match = item
            break
    if not match:
        return None
    if _ancestors_resolved(match):
        return match
    taxon_id = match.get("id")
    if not taxon_id or (call_time_left() is not None and call_time_left() < 1.5):
        return match
    full = _inat_json(f"{_TAXA_URL}/{taxon_id}", {}, INAT_TIMEOUT)
    resolved = _taxon_from_payload(full, scientific)
    if resolved and (_ancestors_resolved(resolved) or resolved.get("ancestor_ids")):
        if not resolved.get("iconic_taxon_name"):
            resolved["iconic_taxon_name"] = match.get("iconic_taxon_name")
        return resolved
    return match


def taxon_in_scope(scientific):
    """True in scope, False for a clearly non-marine animal, None if unknown."""
    allowed = allowed_scope_names()
    if allowed is None:
        return True
    taxon = fetch_inat_taxon(scientific)
    if taxon:
        status = ancestry_in_scope(taxon, allowed)
        iconic = str(taxon.get("iconic_taxon_name") or "").strip().lower()
        if iconic in ("mammalia", "aves"):
            if status is True:
                return True
            if status is False and _ancestors_resolved(taxon):
                return False
            return None
        if status is not None:
            return status
    marine = _worms_is_marine(scientific)
    if marine is not None:
        return marine
    return None


def _radius_wkt(lat, lng, radius_km):
    delta_lat = radius_km / 111.0
    cosine = max(0.2, abs(math.cos(math.radians(lat))))
    delta_lng = radius_km / (111.0 * cosine)
    south, north = lat - delta_lat, lat + delta_lat
    west, east = lng - delta_lng, lng + delta_lng
    return f"POLYGON(({west} {south},{east} {south},{east} {north},{west} {north},{west} {south}))"


def _external_json(url, params):
    return _inat_get(
        url,
        params,
        INAT_TIMEOUT,
        {"User-Agent": "FinFindr/1.0 (range evidence; https://github.com/fred123-eng/Finfindr)", "Accept": "application/json"},
    )


def obis_records_nearby(scientific, lat, lng, radius_km):
    """How many OBIS occurrences fall near the point. None when the lookup fails."""
    if lat is None or lng is None or not scientific:
        return None
    payload = _external_json("https://api.obis.org/v3/occurrence", {
        "scientificname": scientific,
        "geometry": _radius_wkt(lat, lng, radius_km),
        "size": 0,
    })
    if not payload or "total" not in payload:
        return None
    try:
        return int(payload.get("total") or 0)
    except (TypeError, ValueError):
        return None


def _lifebase_range_contains(base, scientific, lat, lng):
    """True when a FishBase or SeaLifeBase box contains the point."""
    parts = (scientific or "").split()
    if len(parts) < 2 or lat is None or lng is None:
        return None
    species = _external_json(base + "/species", {
        "genus": parts[0],
        "species": parts[1],
    })
    rows = species.get("data") if isinstance(species, dict) else None
    if rows is None and isinstance(species, dict):
        rows = species.get("results")
    if isinstance(species, list):
        rows = species
    if not rows:
        return None
    spec_code = None
    for row in rows:
        if isinstance(row, dict) and row.get("SpecCode"):
            spec_code = row["SpecCode"]
            break
    if not spec_code:
        return None
    distributions = _external_json(base + "/distributions", {"SpecCode": spec_code})
    boxes = distributions.get("data") if isinstance(distributions, dict) else None
    if isinstance(distributions, list):
        boxes = distributions
    if not boxes:
        return None
    for box in boxes:
        if not isinstance(box, dict):
            continue
        try:
            north = float(box.get("NorthernLat") if box.get("NorthernLat") is not None else box.get("NorthLat"))
            south = float(box.get("SouthernLat") if box.get("SouthernLat") is not None else box.get("SouthLat"))
            west = float(box.get("WesternLong") if box.get("WesternLong") is not None else box.get("WestLong"))
            east = float(box.get("EasternLong") if box.get("EasternLong") is not None else box.get("EastLong"))
        except (TypeError, ValueError):
            continue
        if south <= lat <= north and west <= lng <= east:
            return True
    return False


def fishbase_range_contains(scientific, lat, lng):
    """True when a FishBase distribution box contains the point, False when none do, None if unknown."""
    return _lifebase_range_contains("https://fishbase.ropensci.org", scientific, lat, lng)


def sealifebase_range_contains(scientific, lat, lng):
    """SeaLifeBase distribution for invertebrates. None when the lookup does not answer."""
    return _lifebase_range_contains("https://sealifebase.ropensci.org", scientific, lat, lng)


def external_range_evidence(scientific, lat, lng, radius_km):
    """State and source. Source is OBIS, FishBase, or SeaLifeBase."""
    if call_time_left() is not None and call_time_left() < 3:
        return {"state": "unknown", "source": ""}
    obis = obis_records_nearby(scientific, lat, lng, radius_km)
    fish = fishbase_range_contains(scientific, lat, lng)
    life = None if fish is not None else sealifebase_range_contains(scientific, lat, lng)
    if obis is not None and obis > 0:
        return {"state": "known", "source": "OBIS"}
    if fish is True:
        return {"state": "known", "source": "FishBase"}
    if life is True:
        return {"state": "known", "source": "SeaLifeBase"}
    if obis == 0:
        return {"state": "absent", "source": "OBIS"}
    if fish is False:
        return {"state": "absent", "source": "FishBase"}
    if life is False:
        return {"state": "absent", "source": "SeaLifeBase"}
    return {"state": "unknown", "source": ""}


def external_range_status(scientific, lat, lng, radius_km):
    """'known', 'absent', or 'unknown' from OBIS, FishBase, and SeaLifeBase."""
    return external_range_evidence(scientific, lat, lng, radius_km)["state"]


def _result_text_for(candidate, group, alt, notes, confidence):
    return "\n".join([
        f"Common name: {candidate.get('common_name') or ''}",
        f"Scientific name: {candidate['scientific_name']}",
        f"Group: {group}",
        f"Alternative group considered: {alt or 'none'}",
        f"Confidence: {confidence}",
        f"Notes: {notes}",
    ])


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/favicon.ico")
def favicon():
    return ("", 204)


@app.after_request
def _permissions_policy(response):
    response.headers.setdefault("Permissions-Policy", "geolocation=(self)")
    return response


@app.route("/identify", methods=["GET", "HEAD"])
def identify_get():
    return jsonify({"error": "Use POST to identify a photo."}), 405


def _coerce_field(value):
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if value is None:
        return ""
    return str(value)


def read_identify_request():
    """Form uploads and JSON uploads share one field map and one image path."""
    fields = {}
    payload = request.get_json(silent=True) if request.is_json else None
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in ("image", "image_base64"):
                continue
            fields[key] = _coerce_field(value)
    for key in request.form.keys():
        fields[key] = request.form.get(key) or ""
    saw_image = False
    image_bytes = None
    if "image" in request.files:
        saw_image = True
        try:
            image_bytes = request.files["image"].read()
        except Exception as exc:
            app.logger.warning("Could not read upload: %s", exc)
            image_bytes = b""
    elif isinstance(payload, dict):
        raw = payload.get("image_base64")
        if not isinstance(raw, str):
            raw = payload.get("image") if isinstance(payload.get("image"), str) else ""
        if isinstance(raw, str) and raw.strip():
            saw_image = True
            try:
                image_bytes = base64.b64decode(raw)
            except Exception:
                app.logger.warning("Could not decode JSON image")
                image_bytes = b""
    return saw_image, image_bytes, fields


def stamp_meta(meta, request_started):
    stamped = dict(meta or {})
    stamped["budget_s"] = _env_seconds("FINFINDR_TIME_BUDGET", 110, 30, 115)
    if request_started is None:
        stamped["total_ms"] = 0
    else:
        stamped["total_ms"] = int(round((time.monotonic() - request_started) * 1000))
    stamped["external_calls"] = request_stat("external_calls")
    stamped["inat_throttled"] = request_stat("inat_throttled")
    stamped["commit"] = app_commit()
    return stamped


def _prefer_bull_shark(candidates, text):
    """A blunt snout or stocky body is Carcharhinus leucas, not the silky shark.

    The model was ranking falciformis first even when its own trait said the
    snout was blunt. Reorder when both names are present. If only falciformis
    was named and the text says blunt or stocky, and it does not mention the
    interdorsal ridge, replace that top name with leucas.
    """
    if not candidates:
        return candidates
    blob = (text or "").lower()
    blunt = any(phrase in blob for phrase in ("blunt snout", "stocky", "no ridge", "without a ridge", "no interdorsal"))
    ridge = "interdorsal ridge" in blob or "ridge between the dorsal" in blob
    if not blunt or ridge:
        return candidates
    leucas = None
    rest = []
    for item in candidates:
        if (item.get("scientific_name") or "").lower() == "carcharhinus leucas":
            leucas = item
        else:
            rest.append(item)
    if leucas is not None:
        if rest and _CONFIDENCE_RANK.get(leucas.get("confidence") or "medium", 1) > _CONFIDENCE_RANK.get(rest[0].get("confidence") or "medium", 1):
            leucas["confidence"] = rest[0].get("confidence") or "medium"
        return [leucas] + rest
    top = candidates[0]
    if (top.get("scientific_name") or "").lower() == "carcharhinus falciformis":
        top["scientific_name"] = "Carcharhinus leucas"
        top["common_name"] = top.get("common_name") or "Bull shark"
        if "silky" in (top.get("common_name") or "").lower():
            top["common_name"] = "Bull shark"
    return candidates


def _no_match_result_text(discovery):
    """One wording for a rejected identification. No species is named as the match."""
    names = []
    for item in (discovery or {}).get("closest_relatives") or []:
        scientific = item.get("scientific_name")
        if scientific and scientific not in names:
            names.append(scientific)
    considered = ", ".join(names) if names else "none"
    return "\n".join([
        "Common name:",
        "Scientific name:",
        "Confidence: none",
        "Notes: No matching species. Closest names considered and rejected: " + considered + ".",
    ])


def _offlist_followup_prompt(locality, candidates):
    names = ", ".join(
        item["scientific_name"] for item in candidates if not item.get("off_list")
    ) or "none"
    return "\n".join([
        f"The animal was photographed in {locality}.",
        "The regional candidates were not confirmed. They are low or medium confidence, or the reference check did not accept them.",
        f"Regional candidates already considered: {names}.",
        "Propose up to 2 species that are not on the regional list and that match the visible animal better.",
        "Do not repeat a regional candidate. Do not invent a name.",
        "If the animal is a reddish octopus without eye spots, propose Octopus rubescens when it fits better than the regional octopus species.",
        "If the shark is stocky with a blunt snout, propose Carcharhinus leucas when falciformis was the only shark named.",
        "OFFLIST_FOLLOWUP",
        "Reply with only these lines:",
        "Off-list 1: <scientific name or none> | <common name> | <trait>",
        "Off-list 2: <scientific name or none> | <common name> | <trait>",
    ])


def _needs_offlist_followup(candidates, verification, verify_skipped):
    """Ask again when the regional list did not produce a confirmed identification."""
    if not candidates:
        return False
    if verify_skipped in ("timeout", "no_time"):
        return True
    if any(item.get("off_list") for item in candidates):
        return False
    if verification is not None and not passing_verification(verification):
        return True
    on_list = [item for item in candidates if not item.get("off_list")]
    if not on_list:
        return True
    return all((item.get("confidence") or "medium") in ("low", "medium") for item in on_list)


def _unsure_crash_body():
    return jsonify({
        "result_text": "",
        "certainty": "unsure",
        "photo_quality": "good",
        "candidates": [],
        "discovery": None,
        "message": "Could not identify this photo. Please try again.",
        "meta": apply_verify_rates(
            stamp_meta({"verify_skipped": "error"}, getattr(g, "identify_started", None)),
            False, False, 0, 0, "error",
        ),
    }), 200


def response_wall_seconds():
    """Hard cap for the HTTP response. Whitespace heartbeats must not hold it open longer."""
    return _env_seconds("FINFINDR_TIME_BUDGET", 110, 30, 115) + 5.0


def _wall_timeout_response(started):
    meta = apply_verify_rates(
        stamp_meta({
            "verify_skipped": "wall",
            "model": resolve_model(),
            "fallback_used": False,
            "attempts": [],
        }, started),
        False, False, 0, 0, "wall",
    )
    return jsonify({
        "result_text": "",
        "certainty": "unsure",
        "photo_quality": "good",
        "candidates": [],
        "discovery": None,
        "message": "Identification is taking too long. Please try again.",
        "reason": "timeout",
        "meta": meta,
    })


def _identify_chunks(worker, started, final):
    """Whitespace heartbeats, then one JSON body. Stops at the response wall."""
    wall = response_wall_seconds()
    while worker.is_alive():
        if time.monotonic() - started >= wall:
            payload = _ensure_meta(_wall_timeout_response(started), started)
            yield payload.get_data()
            return
        yield b" \n"
        worker.join(min(5.0, max(0.05, wall - (time.monotonic() - started))))
    if time.monotonic() - started >= wall:
        payload = _ensure_meta(_wall_timeout_response(started), started)
        yield payload.get_data()
        return
    response = _ensure_meta(final(), started)
    payload = response[0] if isinstance(response, tuple) else response
    yield payload.get_data()


def _ensure_meta(response, started):
    """Every identify body carries a meta object. A missing one is filled in."""
    status = None
    resp = response
    if isinstance(response, tuple):
        resp = response[0]
        status = response[1] if len(response) > 1 else 200
    data = None
    try:
        data = resp.get_json(silent=True) if resp is not None else None
    except Exception:
        data = None
    if isinstance(data, dict) and not isinstance(data.get("meta"), dict):
        reason = data.get("reason") or "missing_meta"
        data["meta"] = apply_verify_rates(
            stamp_meta({"verify_skipped": reason}, started),
            False, False, 0, 0, reason,
        )
        resp = jsonify(data)
    if status is None:
        return resp
    return resp, status


@app.route("/health", methods=["GET", "HEAD"])
def health():
    primary = resolve_model()
    fallback = resolve_fallback_model(primary)
    return jsonify({
        "ok": True,
        "commit": app_commit(),
        "model": primary,
        "fallback_model": fallback or "",
    })


@app.route("/identify", methods=["POST"])
def identify():
    """Run identification on a worker thread and send bytes before a proxy gives up.

    A silent response longer than about 30s becomes a Cloudflare 520. Fast
    answers return normally. Longer ones stream whitespace, which JSON parsers
    ignore, and then the JSON body. The stream stops at the budget plus 5
    seconds even if the worker is still going, so a heartbeat cannot hold the
    connection open past about 115 seconds.
    """
    started = time.monotonic()
    holder = {}

    @copy_current_request_context
    def run():
        try:
            holder["out"] = _identify_http()
        except Exception as exc:
            holder["error"] = exc

    worker = threading.Thread(target=run, name="finfindr-identify", daemon=True)
    worker.start()
    worker.join(min(15.0, response_wall_seconds()))
    if not worker.is_alive():
        if holder.get("error"):
            raise holder["error"]
        return _ensure_meta(holder.get("out") or _unsure_crash_body(), started)

    def generate():
        yield from _identify_chunks(worker, started, lambda: holder.get("out") or _unsure_crash_body())

    return Response(
        generate(),
        status=200,
        mimetype="application/json",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache, no-transform",
        },
    )


def _identify_http():
    g.identify_started = time.monotonic()
    begin_request_stats()
    saw_image, image_bytes, fields = read_identify_request()
    if not saw_image:
        return jsonify({
            "error": "No image uploaded",
            "meta": stamp_meta({}, g.identify_started),
        }), 400

    prepared = prepare_image(image_bytes)
    if not prepared:
        return jsonify({
            "error": "Please upload a valid photo",
            "meta": stamp_meta({}, g.identify_started),
        }), 400
    jpeg_bytes, image_w, image_h = prepared
    image_px = f"{image_w}x{image_h}"

    region = resolve_region(fields.get("region"))
    lat, lng, coord_source = _choose_coordinates(image_bytes, fields)
    region, location_fallback = apply_location_fallback(region, lat, lng)
    region_label = display_region_label(region, lat, lng, location_fallback)
    mode = resolve_species_mode(fields.get("species_list"))
    debug = (fields.get("debug") or "").strip() == "1"
    model = resolve_model(fields.get("model"), debug=debug)
    verify = verification_enabled(debug, fields.get("verify"))
    discover = discovery_enabled(debug, fields.get("discovery"))
    exclude_observation = (fields.get("exclude_observation_id") or "").strip()
    use_cache = (fields.get("nocache") or "").strip().lower() not in ("1", "true", "yes", "on")

    xai_key = os.environ.get("XAI_API_KEY")
    if not xai_key:
        app.logger.error("XAI_API_KEY is not set")
        return jsonify({
            "error": "Identification is temporarily unavailable. Please try again later.",
            "region_label": region_label,
            "meta": stamp_meta({}, g.identify_started),
        }), 500

    request_started = g.identify_started
    set_call_deadline(request_started + _env_seconds("FINFINDR_TIME_BUDGET", 110, 30, 115))
    species_payload, list_fetch_ms = load_species_alongside_model(
        region, lat, lng, mode, exclude_observation
    )
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

    result, attempts = _run_model_attempts(
        jpeg_bytes, prompt, model, xai_key, request_started,
        extra_reserve=verify_timeout_cap() if verify else 0,
        use_cache=use_cache,
    )
    model_ms = sum(item["ms"] for item in attempts)
    answered = result.get("model") or model
    fallback_used = answer_used_fallback(result, model)
    if not result.get("ok"):
        if not result.get("retryable"):
            status = 502
            message = "Could not identify this photo. Please try another one."
            body = {"error": message, "region_label": region_label}
        else:
            status = 200
            if result.get("error") == "rate_limited":
                message = "Could not identify this photo. Please try again."
                reason = "rate_limited"
            elif result.get("error") == "timeout":
                message = "Identification is taking too long. Please try again."
                reason = "timeout"
            else:
                message = "Could not identify this photo. Please try again."
                reason = ""
            body = {
                "result_text": "",
                "region_label": region_label,
                "certainty": "unsure",
                "photo_quality": "good",
                "candidates": [],
                "discovery": None,
                "message": message,
            }
            if reason:
                body["reason"] = reason
        total_ms = int(round((time.monotonic() - request_started) * 1000))
        meta = _identification_meta(
            mode, sent, available, truncated, source, list_fetch_ms, model_ms,
            None, "unavailable", answered, image_px, fallback_used, attempts,
            False, 0, 0, total_ms, "model_timeout" if result.get("error") == "timeout" else "model_failed",
        )
        apply_verify_rates(meta, False, False, 0, 0, meta["verify_skipped"])
        meta = stamp_meta(meta, request_started)
        _log_identification(region["id"], meta)
        body["meta"] = meta
        return jsonify(body), status

    data = result["data"]
    tokens, token_source = _prompt_tokens(data, prompt)
    text = apply_standard_names(_extract_text(data), species)
    photo_quality = parse_photo_quality(text)
    photo_issue = _field_value(text, "Photo issue")
    parsed_candidates = _prefer_bull_shark(parse_model_candidates(text), text)
    named = parsed_candidates[0]["scientific_name"] if parsed_candidates else ""
    taxon_status = taxon_in_scope(named) if named else None
    # Text is only a land-animal veto when iNaturalist did not resolve a taxon.
    if taxon_status is False or (taxon_status is None and text_says_not_marine("\n".join([text, photo_issue]))):
        meta = _identification_meta(
            mode, sent, available, truncated, source, list_fetch_ms, model_ms,
            tokens, token_source, answered, image_px, fallback_used, attempts,
            False, 0, 0, None, "not_marine",
        )
        meta["photo_quality"] = photo_quality
        meta["evidence"] = ["not_marine"]
        apply_verify_rates(meta, False, False, 0, 0, "not_marine")
        meta = stamp_meta(meta, request_started)
        _log_identification(region["id"], meta)
        return jsonify({
            "result_text": "",
            "region_label": region_label,
            "certainty": "unsure",
            "photo_quality": photo_quality,
            "reason": "not_marine",
            "message": NOT_MARINE_MESSAGE,
            "candidates": [],
            "discovery": None,
            "meta": meta,
        })
    if photo_truly_unusable(photo_quality, photo_issue):
        total_ms = int(round((time.monotonic() - request_started) * 1000))
        meta = _identification_meta(
            mode, sent, available, truncated, source, list_fetch_ms, model_ms,
            tokens, token_source, answered, image_px, fallback_used, attempts,
            False, 0, 0, total_ms, "photo_unusable",
        )
        meta["photo_quality"] = photo_quality
        meta["evidence"] = [f"photo_{photo_quality}"]
        meta["range_km"] = range_km()
        meta["range_max"] = range_record_limit()
        meta["range_rule"] = range_rule_text()
        apply_verify_rates(meta, False, False, 0, 0, "photo_unusable")
        meta = stamp_meta(meta, request_started)
        _log_identification(region["id"], meta)
        return jsonify({
            "result_text": text,
            "region_label": region_label,
            "certainty": "unsure",
            "photo_quality": photo_quality,
            "photo_issue": photo_issue or "The animal is not clear enough to identify.",
            "message": "Can't identify from this photo.",
            "candidates": [],
            "discovery": None,
            "meta": meta,
        })
    off_list_proposed = sum(1 for item in parsed_candidates if item.get("off_list"))
    candidates = _prepare_off_list_candidates(parsed_candidates, species)
    off_list_kept = sum(1 for item in candidates if item.get("off_list"))
    app.logger.info("off_list_proposed=%s off_list_kept=%s", off_list_proposed, off_list_kept)
    if candidates and taxon_in_scope(candidates[0]["scientific_name"]) is False:
        meta = _identification_meta(
            mode, sent, available, truncated, source, list_fetch_ms, model_ms,
            tokens, token_source, answered, image_px, fallback_used, attempts,
            False, 0, 0, None, "not_marine",
        )
        meta["photo_quality"] = photo_quality
        meta["off_list_proposed"] = off_list_proposed
        meta["off_list_kept"] = off_list_kept
        meta["evidence"] = ["not_marine"]
        apply_verify_rates(meta, False, False, 0, 0, "not_marine")
        meta = stamp_meta(meta, request_started)
        _log_identification(region["id"], meta)
        return jsonify({
            "result_text": "",
            "region_label": region_label,
            "certainty": "unsure",
            "photo_quality": photo_quality,
            "reason": "not_marine",
            "message": NOT_MARINE_MESSAGE,
            "candidates": [],
            "discovery": None,
            "meta": meta,
        })
    candidates = [
        item for item in candidates
        if taxon_in_scope(item["scientific_name"]) is not False
    ]
    verify_used = False
    verify_attempted = False
    verify_ms = 0
    reference_fetch_ms = 0
    verify_skipped = ""
    verification = None
    if not verify:
        verify_skipped = "disabled"
    elif not candidates:
        verify_skipped = "no_candidates"
    elif verify_call_budget(request_started) is None:
        verify_skipped = "no_time"
    else:
        fetch_started = time.perf_counter()
        fetch_cap = min(3.0, _time_left(request_started) - 6.0)
        reference_urls = []
        if fetch_cap < 0.4:
            verify_skipped = "no_time"
        else:
            fetch_deadline = time.monotonic() + fetch_cap
            candidates = _attach_reference_photos(
                candidates, lat, lng, exclude_observation, fetch_deadline
            )
            photo_order = [item for item in candidates if item.get("off_list")] + [item for item in candidates if not item.get("off_list")]
            for candidate in photo_order:
                reference_urls.extend((candidate.get("_reference_urls") or [])[:1])
            reference_urls = reference_urls[:2]
        reference_fetch_ms = int(round((time.perf_counter() - fetch_started) * 1000))
        call_timeout = verify_call_budget(request_started)
        if verify_skipped == "no_time":
            pass
        elif not reference_urls:
            verify_skipped = "no_photos"
        elif call_timeout is None:
            verify_skipped = "no_time"
        else:
            verify_attempted = True
            verify_started = time.perf_counter()
            verify_result = _call_vision_model(
                _shrink_jpeg(jpeg_bytes, verify_image_px()),
                _verification_prompt(candidates),
                answered,
                call_timeout,
                xai_key,
                reference_urls,
                extra_detail="low",
            )
            verify_ms = int(round((time.perf_counter() - verify_started) * 1000))
            _record_verification(bool(verify_result.get("ok")))
            if verify_result.get("ok"):
                verify_used = True
                verification = _parse_verification(_extract_text(verify_result["data"]), candidates)
                if verification_mismatch(verification):
                    verification = {
                        "decision": "NO_MATCH",
                        "certainty": "unsure",
                        "trait": verification.get("trait") or "",
                        "scientific_name": "",
                    }
                if verification.get("trait"):
                    for candidate in candidates:
                        if candidate["scientific_name"].lower() == (verification.get("scientific_name") or "").lower():
                            candidate["trait"] = verification["trait"]
                if verification.get("decision") not in ("", "UNSURE", "NO_MATCH"):
                    candidates = _order_candidates(candidates, verification["scientific_name"])
                    group = _field_value(text, "Group")
                    alt = _field_value(text, "Alternative group considered")
                    notes = _field_value(text, "Notes")
                    text = apply_standard_names(
                        _result_text_for(candidates[0], group, alt, notes, candidates[0].get("confidence") or "medium"),
                        species,
                    )
            else:
                verify_skipped = _verify_skip_reason(verify_result)
    for candidate in candidates:
        candidate.pop("_reference_urls", None)
    if _needs_offlist_followup(candidates, verification, verify_skipped) and _time_left(request_started) >= 12:
        follow_timeout = min(16.0, max(5.0, _time_left(request_started) - 4.0))
        follow = _call_vision_model(
            jpeg_bytes,
            _offlist_followup_prompt(locality, candidates),
            model if not fallback_used else answered,
            follow_timeout,
            xai_key,
            use_cache=use_cache,
        )
        attempts.append(_attempt_record(
            model if not fallback_used else answered,
            follow.get("ms") or 0,
            follow.get("ok"),
            None if follow.get("ok") else follow.get("error"),
        ))
        if follow.get("ok"):
            proposed = [
                item for item in parse_model_candidates(_extract_text(follow["data"]))
                if item.get("off_list")
            ]
            off_list_proposed += len(proposed)
            prepared = _prepare_off_list_candidates(proposed, species)
            prepared = [
                item for item in prepared
                if item.get("off_list") and taxon_in_scope(item["scientific_name"]) is not False
            ]
            seen_names = {(item.get("scientific_name") or "").lower() for item in candidates}
            fresh = []
            for item in prepared:
                if item["scientific_name"].lower() in seen_names:
                    continue
                fresh.append(item)
                seen_names.add(item["scientific_name"].lower())
            if fresh and not passing_verification(verification):
                candidates = fresh + candidates
            else:
                candidates.extend(fresh)
            off_list_kept = sum(1 for item in candidates if item.get("off_list"))
    verification_disagreed = bool(
        verify_used and verification is not None and not passing_verification(verification)
    )
    certainty = certainty_for(candidates, verification if passing_verification(verification) else None)
    if verification_disagreed:
        cap_unconfirmed_confidence(candidates)
        certainty = "unsure"
    shown = public_candidates(candidates, species)
    notes_text = _field_value(text, "Notes")
    known_traits = SPECIES_TRAITS.get((shown[0].get("scientific_name") if shown else "").lower(), "") if shown else ""
    trait_conflict = bool(shown) and traits_contradict(
        f"{shown[0].get('trait') or ''} {notes_text}",
        f"{shown[0].get('common_name') or ''} {known_traits}",
    )
    discovery = None
    notable_note = ""
    evidence = [f"photo_{photo_quality}"]
    if verify_used:
        evidence.append("verification_used")
    if verification and verification.get("decision") == "NO_MATCH":
        evidence.append("verification_no_match")
    discovery_checked = False
    record_count = None
    focus = shown[0]["scientific_name"] if shown else ""
    rejected_all = bool(
        verify_used and verification and (
            verification.get("decision") == "NO_MATCH" or verification_mismatch(verification)
        )
    )
    no_match = trait_conflict or rejected_all
    color_reject = False
    if isinstance(verification, dict):
        trait_line = (verification.get("trait") or "").lower()
        color_reject = any(cue in trait_line for cue in (
            "does not match", "doesn't match", "wrong color", "wrong colour",
            "different color", "different colour", "color does not", "colour does not",
        ))
    contradicts = trait_conflict or color_reject
    if isinstance(verification, dict) and (verification.get("decision") or "").upper() == "NO_MATCH":
        trait_line = (verification.get("trait") or "").lower()
        if any(word in trait_line for word in ("lattice", "contradict", "does not match", "doesn't match", "wrong ", "not the")):
            contradicts = True
    if no_match and shown and not contradicts:
        # The top candidate is the closest name we have. Saying "no match"
        # while still ranking that species first is a false reject.
        no_match = False
        rejected_all = False
    if no_match:
        certainty = "unsure"
        cap_unconfirmed_confidence(candidates)
        shown = public_candidates(candidates, species)
    strong_id = (
        not no_match
        and not fallback_used
        and not photo_truly_unusable(photo_quality, photo_issue)
        and certainty == "confident"
        and verify_used
        and passing_verification(verification)
        and verification_chose_top(verification, candidates)
        and verification.get("scientific_name")
        and focus
        and verification["scientific_name"].lower() == focus.lower()
    )
    rule = range_rule()
    if discover and no_match and photo_quality == "good":
        relatives = []
        for item in shown:
            genus = item["scientific_name"].split()[0]
            relatives.append({
                "scientific_name": item["scientific_name"],
                "common_name": item.get("common_name") or "",
                "genus": genus,
                "trait": item.get("trait") or (verification or {}).get("trait") or "",
            })
        genus = relatives[0]["genus"] if relatives else ""
        discovery = {
            "type": "no_match",
            "closest_relatives": relatives,
            "genus": genus,
            "family": "",
            "note": "The photo does not match the reference photos of the closest named species. A genus-level guess is not a species identification.",
            "rule": rule,
        }
        certainty = "unsure"
        evidence.append("no_species_match")
    elif discover and focus and photo_quality == "good" and lat is not None and lng is not None and _time_left(request_started) >= 3:
        record_count = count_research_records(
            focus, lat, lng, rule["radius_km"], not_id=exclude_observation or None
        )
        discovery_checked = record_count is not None
        evidence.append(f"range_rule={rule['text']}")
        if record_count is not None:
            evidence.append(f"records_in_radius={record_count}")
        unrecorded = record_count is not None and record_count <= rule["max_records"]
        off_list_focus = False
        if focus and record_count is not None:
            listed_names = {(item.get("scientific") or "").lower() for item in species}
            listed_here = focus.lower() in listed_names
            for item in shown:
                name = (item.get("scientific_name") or "").lower()
                if name in listed_names and name != focus.lower():
                    item["off_list"] = False
                if name != focus.lower():
                    continue
                if record_count > rule["max_records"] or (listed_here and record_count > 0):
                    item["off_list"] = False
                elif (
                    record_count <= rule["max_records"]
                    and not listed_here
                    and ((item.get("confidence") or "medium") in ("low", "medium") or item.get("off_list"))
                ):
                    if not item.get("off_list"):
                        off_list_proposed += 1
                    item["off_list"] = True
        off_list_focus = bool(shown and shown[0].get("off_list"))
        range_ready = strong_id or off_list_focus
        range_evidence = external_range_evidence(focus, lat, lng, rule["radius_km"]) if unrecorded and range_ready else {"state": "", "source": ""}
        range_status = range_evidence.get("state") or ""
        if range_status:
            evidence.append(f"external_range={range_status}")
            if range_evidence.get("source"):
                evidence.append(f"external_range_source={range_evidence['source']}")
        if unrecorded and range_ready and range_status == "absent":
            if _time_left(request_started) >= 2:
                nearest = nearest_research_records(focus, lat, lng)
            else:
                nearest = []
            traits = [item.get("trait") for item in shown if item.get("trait")]
            if verification and verification.get("trait"):
                traits = [verification["trait"]] + [trait for trait in traits if trait != verification["trait"]]
            discovery = {
                "type": "possible_range_extension",
                "scientific_name": focus,
                "common_name": shown[0].get("common_name") or "",
                "records_in_region": record_count,
                "radius_km": rule["radius_km"],
                "nearest_records": nearest,
                "nearest_known_record_km": nearest[0]["distance_km"] if nearest else None,
                "nearest_known_record_source": "iNaturalist" if nearest else "",
                "traits": [trait for trait in traits if trait][:3],
                "rule": rule,
                "external_range": range_status or "unknown",
                "external_range_source": range_evidence.get("source") or "",
            }
            evidence.append("verification_agreed" if strong_id else "off_list_range_unverified")
            if not strong_id and certainty == "confident":
                certainty = "likely"
        elif off_list_focus:
            discovery = {
                "type": "off_list",
                "scientific_name": focus,
                "common_name": shown[0].get("common_name") or "",
                "records_in_region": record_count,
                "radius_km": rule["radius_km"],
                "nearest_records": [],
                "traits": [item.get("trait") for item in shown if item.get("trait")][:3],
                "rule": rule,
                "external_range": range_status or "unknown",
                "external_range_source": range_evidence.get("source") or "",
                "note": "This species is not on the regional list.",
            }
            evidence.append("off_list_kept")
        elif unrecorded and strong_id and range_status == "unknown":
            certainty = "unsure"
            notable_note = (
                f"{shown[0].get('common_name') or focus} has {record_count} research-grade "
                f"iNaturalist records within {rule['radius_km']} km, but outside range evidence "
                "could not be checked. This is not flagged as a range extension."
            )
            evidence.append("range_evidence_unavailable")
        elif unrecorded and not strong_id and not off_list_focus:
            notable_note = (
                f"{shown[0].get('common_name') or focus} has {record_count} research-grade "
                f"iNaturalist records within {rule['radius_km']} km. That would be notable if the identification is confirmed, "
                "but this photo is not strong enough to flag a range extension."
            )
            evidence.append("notable_if_confirmed")
    focus_off_list = bool(shown and shown[0].get("off_list"))
    if focus_off_list and not (discovery and discovery.get("type") == "possible_range_extension"):
        certainty = "unsure"
    total_ms = int(round((time.monotonic() - request_started) * 1000))
    meta = _identification_meta(
        mode, sent, available, truncated, source, list_fetch_ms, model_ms,
        tokens, token_source, answered, image_px, fallback_used, attempts,
        verify_used, verify_ms, reference_fetch_ms, total_ms, verify_skipped,
    )
    meta["photo_quality"] = photo_quality
    meta["discovery_checked"] = discovery_checked
    if fallback_used and certainty == "confident":
        certainty = "likely"
        if discovery and discovery.get("type") == "possible_range_extension":
            discovery = None
            evidence.append("fallback_not_confident")
    if not (verify_used and passing_verification(verification)):
        if certainty == "confident":
            certainty = "likely" if shown else "unsure"
        if (
            discovery
            and discovery.get("type") == "possible_range_extension"
            and "off_list_range_unverified" not in evidence
        ):
            discovery = None
            evidence.append("discovery_blocked_without_verification")
    if discovery and discovery.get("type") == "no_match" and not (rejected_all or trait_conflict):
        discovery = None
    checked = len(candidates) if verify_attempted else 0
    passed = 1 if verify_used and passing_verification(verification) and verification_chose_top(verification, candidates) else 0
    verified_id = bool(
        verify_used and passing_verification(verification) and verification_chose_top(verification, candidates) and not fallback_used
    )
    if verified_id and not no_match:
        certainty = "confident"
        for item in shown[:1]:
            item["confidence"] = "high"
    elif photo_quality == "poor" and not photo_truly_unusable(photo_quality, photo_issue):
        if certainty == "confident":
            certainty = "likely"
        for item in shown:
            item["confidence"] = {"high": "medium", "medium": "low", "low": "low"}.get(item.get("confidence") or "medium", "medium")
        evidence.append("quality_lowered_confidence")
    elif certainty != "confident":
        for item in shown:
            if item.get("confidence") == "high":
                item["confidence"] = "medium"
    if no_match or (discovery and discovery.get("type") == "no_match"):
        text = _no_match_result_text(discovery)
        certainty = "unsure"
    elif shown and text:
        text = _set_field(text, "Confidence", shown[0].get("confidence") or "medium")
    off_list_kept = sum(1 for item in shown if item.get("off_list"))
    if verify_skipped in ("timeout", "connection", "no_time") and certainty == "confident":
        certainty = "likely"
    meta["evidence"] = evidence
    meta["range_km"] = rule["radius_km"]
    meta["range_max"] = rule["max_records"]
    meta["range_rule"] = rule["text"]
    meta["verify_succeeded"] = verified_id
    apply_verify_rates(meta, verify_attempted, verify_used, checked, passed, verify_skipped)
    meta["verify_timeout_s"] = verify_timeout_cap()
    meta["off_list_proposed"] = off_list_proposed
    meta["off_list_kept"] = off_list_kept
    meta = stamp_meta(meta, request_started)
    _log_identification(region["id"], meta)
    return jsonify({
        "result_text": text or "No response",
        "region_label": region_label,
        "certainty": certainty,
        "photo_quality": photo_quality,
        "candidates": [] if no_match else shown,
        "discovery": discovery,
        "notable_note": notable_note,
        "meta": meta,
    })


def _error_body(message):
    body = {"error": message}
    if request.path == "/identify":
        started = getattr(g, "identify_started", None)
        body["meta"] = stamp_meta({}, started)
    return body


@app.errorhandler(RequestEntityTooLarge)
def too_large(_exc):
    return jsonify(_error_body("That photo is too large. Please choose a smaller one.")), 413


@app.errorhandler(HTTPException)
def http_exception(exc):
    return jsonify(_error_body(exc.description or "Request failed")), exc.code


@app.errorhandler(Exception)
def unhandled(exc):
    app.logger.exception("Unhandled error on %s", request.path)
    if isinstance(exc, HTTPException):
        return jsonify(_error_body(exc.description or "Request failed")), exc.code
    return jsonify(_error_body("Something went wrong. Please try again.")), 500


@app.teardown_request
def _clear_request_deadline(_exc):
    clear_call_deadline()


@app.before_request
def _warm_species_cache():
    # After a gunicorn fork the import-time thread is gone. Starting again is cheap
    # and does not wait, so a health check is not held up.
    _start_species_prefetch()


_start_species_prefetch()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
