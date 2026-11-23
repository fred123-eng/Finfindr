#!/usr/bin/env python3
"""
FinFindr – Marine species identification
"""

import base64
import logging
import os
import re
import threading
import time
from io import BytesIO

import requests
from flask import Flask, jsonify, render_template_string, request
from PIL import Image, ImageOps
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
# Phone photos are often 8–15MB; reject anything larger before it ties up the worker.
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
# Cap decoded pixels so a huge upload cannot exhaust a small instance.
Image.MAX_IMAGE_PIXELS = 50_000_000

MIN_DIMENSION = 32
MAX_DIMENSION = 2048
JPEG_QUALITY = 85

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
  }
</style>
</head>
<body>
<header>
  <h1>FinFindr</h1>
  <p>Upload a photo · Get the species</p>
</header>

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
const MIN_DIMENSION = 32;
const MAX_BYTES = 20 * 1024 * 1024;
let busy = false;
let validateToken = 0;
let dragDepth = 0;

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
    return { ok: false, message: friendlyMessage(data && data.error) };
  }
  return { ok: true, data: data };
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

    const resp = await fetch("/identify", { method: "POST", body: formData });
    const parsed = await readJsonResponse(resp);
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


def prepare_image(image_bytes):
    """Return JPEG bytes for a usable photo, or None when the upload is not one."""
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

        longest = max(img.size)
        if longest > MAX_DIMENSION:
            scale = MAX_DIMENSION / float(longest)
            new_size = (
                max(1, min(MAX_DIMENSION, int(round(img.size[0] * scale)))),
                max(1, min(MAX_DIMENSION, int(round(img.size[1] * scale)))),
            )
            resized = img.resize(new_size, Image.Resampling.LANCZOS)
            img.close()
            img = resized

        out = BytesIO()
        img.save(out, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        return out.getvalue()
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


def _fetch_inat_common_name(scientific):
    key = scientific.lower()
    cached = _cache_get(key)
    if cached is not None:
        return cached[1]
    try:
        resp = requests.get(
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


def standard_common_name(scientific):
    """Authoritative common name for an exact scientific name, or None."""
    if not scientific:
        return None
    curated = CURATED_COMMON_NAMES.get(scientific.lower())
    if curated:
        return curated
    return _fetch_inat_common_name(scientific)


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


def apply_standard_names(result_text):
    """Replace the model's common name when the scientific name resolves."""
    if not result_text:
        return result_text
    text = _result_lines(result_text)
    scientific_raw = _field_value(text, "Scientific name")
    canonical = exact_scientific_name(scientific_raw)
    if not canonical:
        return text
    text = _set_field(text, "Scientific name", canonical)
    common = standard_common_name(canonical)
    if not common:
        return text
    app.logger.info("Standard common name for %s: %s", canonical, common)
    return _set_field(text, "Common name", common)

def _vision_payload(jpeg_bytes):
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    data_url = f"data:image/jpeg;base64,{b64}"
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
    return payload


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

    jpeg_bytes = prepare_image(image_bytes)
    if not jpeg_bytes:
        return jsonify({"error": "Please upload a valid photo"}), 400

    xai_key = os.environ.get("XAI_API_KEY")
    if not xai_key:
        app.logger.error("XAI_API_KEY is not set")
        return jsonify({"error": "Identification is temporarily unavailable. Please try again later."}), 500

    payload = _vision_payload(jpeg_bytes)
    try:
        resp = requests.post(
            "https://api.x.ai/v1/responses",
            headers={"Authorization": f"Bearer {xai_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=60
        )
    except requests.Timeout:
        app.logger.exception("xAI request timed out")
        return jsonify({"error": "Identification is taking too long. Please try again."}), 504
    except requests.RequestException:
        app.logger.exception("xAI request failed")
        return jsonify({"error": "Could not identify this photo. Please try again."}), 502

    if not resp.ok:
        app.logger.error("xAI HTTP %s: %s", resp.status_code, resp.text[:500])
        return jsonify({"error": "Could not identify this photo. Please try another one."}), 502

    try:
        data = resp.json()
    except ValueError:
        app.logger.exception("xAI returned invalid JSON")
        return jsonify({"error": "Could not identify this photo. Please try again."}), 502

    text = apply_standard_names(_extract_text(data))
    return jsonify({"result_text": text or "No response"})


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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
