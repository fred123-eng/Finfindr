"""Route and prompt checks for regional identification. No live xAI or iNaturalist calls."""

import base64
import io
import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ["FINFINDR_PREFETCH"] = "0"
os.environ.pop("SPECIES_LIST_MODE", None)

import requests
from PIL import Image

import finfindr_server
import scripts.eval_region as eval_region


def epithet(index):
    number = index + 1
    chars = []
    while number:
        number, remainder = divmod(number - 1, 26)
        chars.append(chr(ord("a") + remainder))
    return "sp" + "".join(reversed(chars))


def jpeg_bytes(width=80, height=60, gps=None):
    image = Image.new("RGB", (width, height), (20, 90, 140))
    buffer = io.BytesIO()
    if gps is None:
        image.save(buffer, format="JPEG")
        return buffer.getvalue()
    lat, lng = gps
    exif = Image.Exif()
    gps_ifd = exif.get_ifd(0x8825)

    def dms(value):
        absolute = abs(value)
        degrees = int(absolute)
        minutes_float = (absolute - degrees) * 60
        minutes = int(minutes_float)
        seconds = round((minutes_float - minutes) * 60, 4)
        return (degrees, minutes, seconds)

    gps_ifd[1] = "S" if lat < 0 else "N"
    gps_ifd[2] = dms(lat)
    gps_ifd[3] = "W" if lng < 0 else "E"
    gps_ifd[4] = dms(lng)
    image.save(buffer, format="JPEG", exif=exif)
    return buffer.getvalue()


class FakeResponse:
    def __init__(self, payload, status=200):
        self.status_code = status
        self.ok = 200 <= status < 300
        self._payload = payload
        self.text = json.dumps(payload)
        self.headers = {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if not self.ok:
            raise requests.HTTPError(response=self)


def _one_species(name, common, count):
    return {
        "total_results": 1,
        "results": [
            {
                "count": count,
                "taxon": {
                    "name": name,
                    "rank": "species",
                    "preferred_common_name": common,
                },
            }
        ],
    }


def species_payload(params):
    per_page = int(params.get("per_page") or 1)
    page = int(params.get("page") or 1)
    taxon_id = params.get("taxon_id")
    if taxon_id:
        taxon_id = int(taxon_id)
        if taxon_id == 492346:
            return _one_species("Hydrophis platurus", "Yellow-bellied Sea Snake", 2)
        if taxon_id == 372234:
            return _one_species("Chelonia mydas", "not the curated name", 40)
        if taxon_id == 47549:
            return _one_species("Phataria unifascialis", "not the curated name", 5)
        if taxon_id != 47178:
            return _one_species(f"Groupus {epithet(taxon_id)}", f"Group {taxon_id}", 3)
    total = 600
    start = (page - 1) * per_page
    results = []
    for index in range(start, min(start + per_page, total)):
        results.append(
            {
                "count": total - index,
                "taxon": {
                    "name": f"Fishus {epithet(index)}",
                    "rank": "species",
                    "preferred_common_name": f"Fish {index}",
                },
            }
        )
    if page == 1 and results:
        results[0] = {
            "count": 10000,
            "taxon": {
                "name": "Thalassoma lucasanum",
                "rank": "species",
                "preferred_common_name": "wrong common",
            },
        }
    return {"total_results": total, "results": results}


class IdentifyTests(unittest.TestCase):
    def setUp(self):
        os.environ["XAI_API_KEY"] = "test-key"
        os.environ.pop("SPECIES_LIST_MODE", None)
        os.environ.pop("XAI_MODEL", None)
        os.environ.pop("XAI_FALLBACK_MODEL", None)
        os.environ.pop("XAI_PRIMARY_TIMEOUT", None)
        os.environ.pop("XAI_FALLBACK_TIMEOUT", None)
        os.environ.pop("FINFINDR_TIME_BUDGET", None)
        os.environ.pop("MAX_IMAGE_PX", None)
        os.environ.pop("FINFINDR_DISCOVERY", None)
        os.environ.pop("FINFINDR_RANGE_KM", None)
        os.environ.pop("FINFINDR_RANGE_MAX", None)
        os.environ.pop("FINFINDR_VERIFY", None)
        os.environ.pop("FINFINDR_VERIFY_TIMEOUT", None)
        os.environ.pop("FINFINDR_VERIFY_IMAGE_PX", None)
        os.environ.pop("FINFINDR_LOOKALIKE_PASS", None)
        os.environ.pop("FINFINDR_NOT_IN_REGION", None)
        os.environ.pop("FINFINDR_FIRST_GUESS", None)
        os.environ.pop("FINFINDR_FIRST_GUESS_TIMEOUT", None)
        os.environ.pop("FINFINDR_FIRST_GUESS_MODEL", None)
        os.environ["FINFINDR_INAT_MIN_INTERVAL"] = "0"
        os.environ["FINFINDR_MODEL_CACHE"] = "0"
        self.cache_dir = tempfile.mkdtemp(prefix="finfindr-cache-")
        os.environ["FINFINDR_CACHE_DIR"] = self.cache_dir
        finfindr_server._SPECIES_CACHE.clear()
        finfindr_server._INAT_JSON_CACHE.clear()
        finfindr_server.clear_persisted_caches()
        finfindr_server.clear_model_cache()
        finfindr_server._INAT_NEXT_AT = 0.0
        finfindr_server._SPECIES_INFLIGHT.clear()
        finfindr_server._inat_cache.clear()
        finfindr_server._PHOTO_CACHE.clear()
        with finfindr_server._VERIFY_STATS_LOCK:
            finfindr_server._VERIFY_STATS["attempts"] = 0
            finfindr_server._VERIFY_STATS["ok"] = 0
        self.inat_handler = None
        self.client = finfindr_server.app.test_client()
        self.captured = {}
        self.fail_timeout = set()
        self.fail_status = {}
        self.species_calls = []
        self._calls_lock = threading.Lock()
        self.get_patch = patch("finfindr_server.http.get", side_effect=self.fake_get)
        self.post_patch = patch("finfindr_server.http.post", side_effect=self.fake_post)
        self.get_patch.start()
        self.post_patch.start()

    def tearDown(self):
        self.get_patch.stop()
        self.post_patch.stop()
        shutil.rmtree(getattr(self, "cache_dir", ""), ignore_errors=True)

    def fake_get(self, url, params=None, **kwargs):
        params = params or {}
        self.captured.setdefault("gets", []).append(str(url))
        if "species_counts" in url:
            with self._calls_lock:
                self.species_calls.append(params)
            return FakeResponse(species_payload(params))
        if self.inat_handler:
            handled = self.inat_handler(url, params)
            if handled is not None:
                return handled
        return FakeResponse({"results": []})

    def fake_post(self, url, json=None, **kwargs):
        self.captured["url"] = url
        self.captured["json"] = json
        model = (json or {}).get("model")
        prompt = ""
        try:
            content = (json or {})["input"][0]["content"]
            prompt = next(part.get("text", "") for part in content if part.get("type") == "input_text")
        except (KeyError, IndexError, StopIteration, TypeError):
            prompt = ""
        self.captured.setdefault("calls", []).append({
            "model": model,
            "timeout": kwargs.get("timeout"),
            "temperature": (json or {}).get("temperature"),
            "seed": (json or {}).get("seed"),
            "verify": "The first image is the animal" in prompt,
        })
        if model in self.fail_timeout:
            raise requests.Timeout("slow")
        if model in self.fail_status:
            return FakeResponse({"error": "upstream"}, status=self.fail_status[model])
        if self.captured.get("delay"):
            time.sleep(self.captured["delay"])
        by_model = self.captured.get("model_text_by_model") or {}
        if finfindr_server.LOOKALIKE_MARKER in prompt:
            self.captured.setdefault("lookalike_calls", []).append({"model": model, "timeout": kwargs.get("timeout"), "prompt": prompt})
            if self.captured.get("lookalike_raises") == "timeout":
                raise requests.Timeout("lookalike slow")
            if self.captured.get("lookalike_raises") == "error":
                return FakeResponse({"error": "upstream"}, status=503)
            text = self.captured.get("lookalike_text", "")
            body = {
                "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
                "usage": {"input_tokens": 1500, "output_tokens": 120, "total_tokens": 1620},
            }
            return FakeResponse(body)
        if finfindr_server.FIRST_GUESS_COMPARE_MARKER in prompt:
            with self._calls_lock:
                self.captured.setdefault("first_guess_compare_calls", []).append(
                    {"model": model, "timeout": kwargs.get("timeout"), "prompt": prompt}
                )
            if self.captured.get("first_guess_compare_raises") == "timeout":
                raise requests.Timeout("first guess compare slow")
            if self.captured.get("first_guess_compare_raises") == "error":
                return FakeResponse({"error": "upstream"}, status=503)
            text = self.captured.get("first_guess_compare_text", "")
            body = {
                "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
                "usage": {"input_tokens": 400, "output_tokens": 80, "total_tokens": 480},
            }
            return FakeResponse(body)
        if finfindr_server.FIRST_GUESS_MARKER in prompt:
            with self._calls_lock:
                self.captured.setdefault("first_guess_calls", []).append(
                    {"model": model, "timeout": kwargs.get("timeout"), "prompt": prompt}
                )
            if self.captured.get("first_guess_raises") == "timeout":
                raise requests.Timeout("first guess slow")
            if self.captured.get("first_guess_raises") == "error":
                return FakeResponse({"error": "upstream"}, status=503)
            text = self.captured.get("first_guess_text", "")
            body = {
                "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
                "usage": {"input_tokens": 800, "output_tokens": 60, "total_tokens": 860},
            }
            return FakeResponse(body)
        if "OFFLIST_FOLLOWUP" in prompt and self.captured.get("offlist_text"):
            text = self.captured["offlist_text"]
        elif model in by_model and "The first image is the animal" not in prompt:
            text = by_model[model]
        elif "The first image is the animal" in prompt and self.captured.get("verify_raises") == "timeout":
            raise requests.Timeout("verify slow")
        elif "The first image is the animal" in prompt and self.captured.get("verify_text"):
            text = self.captured["verify_text"]
        else:
            text = self.captured.get(
                "model_text",
                "Common name: Wrong\nScientific name: Thalassoma lucasanum\n"
                "Group: wrasse\nAlternative group considered: none\n"
                "Confidence: high\nNotes: A reef wrasse.",
            )
        body = {
            "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
            "usage": {"input_tokens": 4321, "output_tokens": 30, "total_tokens": 4351},
        }
        return FakeResponse(body)

    def post_photo(self, **fields):
        data = {"image": (io.BytesIO(jpeg_bytes()), "reef.jpg"), "sync": "1"}
        data.update(fields)
        return self.client.post("/identify", data=data, content_type="multipart/form-data")

    def prompt(self):
        content = self.captured["json"]["input"][0]["content"]
        return next(part["text"] for part in content if part["type"] == "input_text")

    def test_index_has_region_selector_and_existing_upload_ui(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        page = response.get_data(as_text=True)
        self.assertIn("Sea of Cortez / La Paz", page)
        self.assertIn("Caribbean", page)
        self.assertIn("Use my location", page)
        self.assertIn("Great Barrier Reef", page)
        self.assertIn('value="my-location"', page)
        self.assertIn("Drop a photo here", page)
        self.assertIn("Try another photo", page)
        self.assertIn("spinner", page)
        self.assertIn("finfindr.region", page)
        self.assertIn("function resultLines", page)
        self.assertIn("Testing: species list size", page)
        self.assertIn('id="speciesListOptions"', page)
        self.assertIn('class="test-options" hidden', page)
        self.assertIn('params.get("debug") === "1"', page)
        self.assertIn('id="modelInput"', page)
        self.assertIn("Checking reference photos", page)
        self.assertIn("Usually 30–90 seconds", page)
        self.assertIn("object-fit:contain", page)
        self.assertIn("function applySavedLocation", page)
        self.assertIn('if (regionSelect.value === "my-location") applySavedLocation();', page)
        self.assertIn("Nearest known record:", page)
        self.assertIn("inat-submit", page)
        self.assertIn('rel="icon"', page)
        self.assertIn("Common name unavailable", page)

    def test_missing_and_invalid_photos_stay_json(self):
        missing = self.client.post("/identify", data={})
        self.assertEqual(missing.status_code, 400)
        self.assertIn("error", missing.get_json())

        tiny = self.client.post(
            "/identify",
            data={"image": (io.BytesIO(jpeg_bytes(8, 8)), "tiny.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(tiny.status_code, 400)
        self.assertEqual(tiny.get_json()["error"], "Please upload a valid photo")
        self.assertFalse(self.captured)

    def test_default_region_prompt_newlines_temperature_and_meta(self):
        response = self.post_photo()
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        prompt = self.prompt()
        self.assertIn("\n", prompt)
        self.assertNotIn("\\n", prompt)
        self.assertIn("Sea of Cortez", prompt)
        self.assertIn("Cirrhitus rivulatus", prompt)
        self.assertIn("Pentaceraster cumingi", prompt)
        self.assertIn("Nidorellia armata", prompt)
        self.assertIn("Thalassoma lucasanum", prompt)
        self.assertIn("Bodianus diplotaenia", prompt)
        self.assertIn("Sphoeroides annulatus", prompt)
        self.assertIn("Thalassoma lucasanum (Cortez Rainbow Wrasse)", prompt)
        self.assertIn("Chelonia mydas (Green Sea Turtle)", prompt)
        self.assertIn("Hydrophis platurus", prompt)
        self.assertIn("Phataria unifascialis", prompt)
        self.assertIn("slender cylindrical arms", prompt)
        self.assertIn("thick inflated body", prompt)
        self.assertIn("hint, not a menu", prompt)
        self.assertIn("propose up to 2 off_list species", prompt)
        self.assertEqual(self.captured["json"]["temperature"], 0)
        self.assertEqual(self.captured["json"]["seed"], 0)
        self.assertEqual({call["model"] for call in self.captured["calls"] if not call.get("verify")}, {"grok-4", "grok-4.7"})
        self.assertEqual(body["meta"]["final_source"], "both")
        self.assertTrue(body["meta"]["agreement"])
        self.assertEqual(body["meta"]["models_run"], ["grok-4", "grok-4.7"])
        self.assertEqual(body["meta"]["image_px"], "80x60")
        self.assertEqual(body["certainty"], "likely")
        self.assertEqual(body["meta"]["verify_skipped"], "no_photos")
        self.assertFalse(body["meta"]["verify_used"])
        self.assertFalse(body["meta"]["verify_succeeded"])
        self.assertEqual(body["candidates"][0]["scientific_name"], "Thalassoma lucasanum")
        self.assertEqual(body["region_label"], "Sea of Cortez / La Paz")
        self.assertIn("Cortez Rainbow Wrasse", body["result_text"])
        self.assertNotIn("Wrong", body["result_text"])
        meta = body["meta"]
        self.assertEqual(meta["species_list_mode"], "short")
        self.assertLessEqual(meta["species_sent"], 250)
        self.assertGreater(meta["species_available"], meta["species_sent"])
        self.assertFalse(meta["species_truncated"])
        self.assertEqual(meta["list_source"], "inaturalist")
        self.assertEqual(meta["prompt_tokens"], 4321)
        self.assertEqual(meta["prompt_tokens_source"], "usage")
        self.assertGreaterEqual(meta["list_fetch_ms"], 0)
        self.assertGreaterEqual(meta["model_call_ms"], 0)
        fish_calls = [call for call in self.species_calls if int(call.get("taxon_id") or 0) == 47178]
        self.assertTrue(fish_calls)
        self.assertEqual(fish_calls[0]["place_id"], 122897)
        self.assertLessEqual(int(fish_calls[0]["per_page"]), 80)
        snake_calls = [call for call in self.species_calls if int(call.get("taxon_id") or 0) == 492346]
        self.assertTrue(snake_calls)

    def test_caribbean_skips_cortez_lookalikes_and_uses_its_bounds(self):
        response = self.post_photo(region="caribbean")
        self.assertEqual(response.status_code, 200)
        prompt = self.prompt()
        self.assertIn("Caribbean Sea", prompt)
        self.assertNotIn("Cirrhitus rivulatus", prompt)
        self.assertNotIn("Sphoeroides annulatus", prompt)
        self.assertNotIn("slender cylindrical arms", prompt)
        self.assertEqual(response.get_json()["region_label"], "Caribbean")
        located = [call for call in self.species_calls if "swlat" in call]
        self.assertTrue(located)
        self.assertEqual(float(located[0]["swlat"]), 9.0)
        self.assertNotIn("place_id", located[0])

    def test_species_list_full_pages_and_env_default(self):
        response = self.post_photo(species_list="full")
        body = response.get_json()
        self.assertEqual(body["meta"]["species_list_mode"], "full")
        self.assertGreater(body["meta"]["species_sent"], 250)
        self.assertFalse(body["meta"]["species_truncated"])
        main_calls = [call for call in self.species_calls if "taxon_id" not in call]
        self.assertTrue(any(int(call["page"]) == 2 for call in main_calls))
        self.assertTrue(all(int(call["per_page"]) == 500 for call in main_calls))

        finfindr_server._SPECIES_CACHE.clear()
        self.species_calls.clear()
        os.environ["SPECIES_LIST_MODE"] = "full"
        try:
            response = self.post_photo(region="hawaii")
        finally:
            os.environ.pop("SPECIES_LIST_MODE", None)
        self.assertEqual(response.get_json()["meta"]["species_list_mode"], "full")
        self.assertEqual(response.get_json()["region_label"], "Hawaii")
        main = next(call for call in self.species_calls if "taxon_id" not in call)
        self.assertEqual(main["place_id"], 11)

        finfindr_server._SPECIES_CACHE.clear()
        response = self.post_photo(species_list="nope")
        self.assertEqual(response.get_json()["meta"]["species_list_mode"], "short")
        plain = self.post_photo(species_list="none", models="grok-4")
        plain_body = plain.get_json()
        self.assertEqual(plain_body["meta"]["species_list_mode"], "short")
        self.assertNotIn("nature_data", plain_body["meta"])
        self.assertEqual(plain_body["meta"]["models_run"], ["grok-4", "grok-4.7"])

    def test_my_location_uses_browser_or_exif_coordinates(self):
        response = self.post_photo(region="my-location", lat="21.30", lng="-157.80")
        self.assertEqual(response.status_code, 200)
        prompt = self.prompt()
        self.assertIn("21.3000", prompt)
        self.assertIn("-157.8000", prompt)
        self.assertIn("Hawaii", response.get_json()["region_label"])
        located = next(call for call in self.species_calls if "radius" in call)
        self.assertEqual(float(located["lat"]), 21.3)
        self.assertEqual(float(located["lng"]), -157.8)
        self.assertEqual(int(located["radius"]), 100)

        finfindr_server._SPECIES_CACHE.clear()
        self.species_calls.clear()
        photo = jpeg_bytes(gps=(24.1426, -110.3128))
        found = finfindr_server.extract_gps(photo)
        self.assertAlmostEqual(found[0], 24.1426, places=2)
        self.assertAlmostEqual(found[1], -110.3128, places=2)
        response = self.client.post(
            "/identify",
            data={"image": (io.BytesIO(photo), "lapaz.jpg"), "region": "my-location", "sync": "1"},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Sea of Cortez", response.get_json()["region_label"])
        self.assertIn("Cirrhitus rivulatus", self.prompt())
        located = next(call for call in self.species_calls if "radius" in call)
        self.assertNotIn("place_id", located)

    def test_selected_region_wins_when_photo_gps_is_elsewhere(self):
        photo = jpeg_bytes(gps=(21.3, -157.8))
        response = self.client.post(
            "/identify",
            data={"image": (io.BytesIO(photo), "hawaii.jpg"), "region": "sea-of-cortez", "sync": "1"},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.get_json()["region_label"], "Sea of Cortez / La Paz")
        prompt = self.prompt()
        self.assertIn("outside the selected region", prompt)
        self.assertIn("coordinates as the place", prompt)
        self.assertIn("Cirrhitus rivulatus", prompt)
        located = next(call for call in self.species_calls if "radius" in call)
        self.assertAlmostEqual(float(located["lat"]), 21.3, places=1)
        self.assertAlmostEqual(float(located["lng"]), -157.8, places=1)

    def test_inaturalist_failure_falls_back_without_blocking_identification(self):
        self.get_patch.stop()
        with patch("finfindr_server.http.get", side_effect=requests.ConnectionError("down")):
            cortez = self.post_photo(region="sea-of-cortez")
            cortez_prompt = self.prompt()
            caribbean = self.post_photo(region="caribbean")
            caribbean_prompt = self.prompt()
        self.get_patch.start()
        self.assertEqual(cortez.status_code, 200)
        self.assertIn("Phataria unifascialis", cortez_prompt)
        self.assertEqual(cortez.get_json()["meta"]["list_source"], "curated")
        self.assertIn("Cirrhitus rivulatus", cortez_prompt)
        self.assertEqual(caribbean.status_code, 200)
        self.assertIn("No regional species list was available", caribbean_prompt)
        self.assertEqual(caribbean.get_json()["meta"]["list_source"], "none")
        self.assertEqual(caribbean.get_json()["meta"]["species_sent"], 0)

    def test_literal_newlines_from_the_model_still_resolve_common_names(self):
        self.captured["model_text"] = (
            "Common name: Wrong\\nScientific name: Thalassoma lucasanum\\n"
            "Group: wrasse\\nAlternative group considered: none\\n"
            "Confidence: high\\nNotes: A reef wrasse."
        )
        response = self.post_photo()
        text = response.get_json()["result_text"]
        self.assertIn("Cortez Rainbow Wrasse", text)
        self.assertIn("\n", text)
        self.assertNotIn("\\n", text)

    def test_token_estimate_when_usage_is_missing(self):
        def post_without_usage(url, json=None, **kwargs):
            self.captured["json"] = json
            return FakeResponse(
                {"output": [{"type": "message", "content": [{"type": "output_text", "text": "Common name: Fish"}]}]}
            )

        self.post_patch.stop()
        with patch("finfindr_server.http.post", side_effect=post_without_usage):
            response = self.post_photo(region="red-sea")
        self.post_patch.start()
        meta = response.get_json()["meta"]
        self.assertEqual(meta["prompt_tokens_source"], "estimate")
        self.assertEqual(meta["prompt_tokens"], finfindr_server.estimate_text_tokens(self.prompt()))
        self.assertEqual(response.get_json()["region_label"], "Red Sea")

    def test_prompt_stays_within_token_budget(self):
        species = [
            {"scientific": f"Genus {'longname' * 8}{index}", "common": "common " * 12, "count": 10000 - index}
            for index in range(8000)
        ]
        prompt, sent, trimmed = finfindr_server.prompt_for_species("the test sea", species, False, None, None)
        self.assertTrue(trimmed)
        self.assertLess(sent, 8000)
        self.assertLessEqual(finfindr_server.estimate_text_tokens(prompt), finfindr_server.MAX_PROMPT_TEXT_TOKENS)
        self.assertNotIn("\\n", prompt)

    def test_other_without_coordinates_has_no_species_list(self):
        response = self.post_photo(region="other")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["region_label"], "Other")
        self.assertEqual(response.get_json()["meta"]["species_sent"], 0)
        self.assertEqual(response.get_json()["meta"]["list_source"], "none")
        self.assertIn("unspecified coast or ocean", self.prompt())
        self.assertNotIn("Cirrhitus rivulatus", self.prompt())
        self.assertEqual(self.species_calls, [])

    def test_short_quotas_keep_uncommon_groups(self):
        fish = [
            {"count": 5000, "taxon": {"name": "Fishus alpha", "rank": "species", "preferred_common_name": "Alpha"}},
            {"count": 4000, "taxon": {"name": "Fishus beta", "rank": "species", "preferred_common_name": "Beta"}},
            {"count": 3000, "taxon": {"name": "Fishus gamma", "rank": "species", "preferred_common_name": "Gamma"}},
        ]
        snakes = [
            {
                "count": 2,
                "taxon": {
                    "name": "Hydrophis platurus",
                    "rank": "species",
                    "preferred_common_name": "Yellow-bellied Sea Snake",
                },
            }
        ]
        species = finfindr_server.assemble_quota_list([
            {"label": "Ray-finned fishes", "quota": 2, "rows": fish},
            {"label": "Marine reptiles", "quota": 5, "rows": snakes},
        ])
        names = [item["scientific"] for item in species]
        self.assertEqual(names.count("Fishus alpha") + names.count("Fishus beta"), 2)
        self.assertNotIn("Fishus gamma", names)
        self.assertIn("Hydrophis platurus", names)
        self.assertEqual(sum(group["quota"] for group in finfindr_server.SHORT_GROUPS), finfindr_server.SHORT_LIMIT)

    def test_location_without_coordinates_uses_the_default_region(self):
        response = self.post_photo(region="my-location")
        body = response.get_json()
        self.assertEqual(body["region_label"], "Sea of Cortez / La Paz (location unavailable)")
        prompt = self.prompt()
        self.assertIn("Sea of Cortez", prompt)
        self.assertIn("default region", prompt)
        self.assertNotIn("unknown spot", prompt)
        self.assertIn("Hydrophis platurus", prompt)
        self.assertIn("Cirrhitus rivulatus", prompt)
        self.assertTrue(any(call.get("place_id") == 122897 for call in self.species_calls))

    def test_model_and_image_size_are_configurable(self):
        wide = jpeg_bytes(2000, 1000)
        response = self.client.post(
            "/identify",
            data={"image": (io.BytesIO(wide), "wide.jpg"), "sync": "1"},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.get_json()["meta"]["image_px"], "1024x512")
        self.assertIn("grok-4.7", {call["model"] for call in self.captured["calls"]})

        os.environ["MAX_IMAGE_PX"] = "800"
        os.environ["XAI_MODEL"] = "grok-4.5"
        try:
            ignored = self.post_photo(model="grok-4.3")
            self.assertEqual(ignored.get_json()["meta"]["models_run"], ["grok-4", "grok-4.5"])
            self.assertEqual(ignored.get_json()["meta"]["model"], "grok-4.5")
            chosen = self.post_photo(model="grok-4.3", debug="1")
            self.assertEqual(self.captured["json"]["model"], "grok-4.3")
            self.assertEqual(chosen.get_json()["meta"]["model"], "grok-4.3")
            rejected = self.post_photo(model="not a model", debug="1")
            # Both grok-4 and XAI_MODEL run as a pair; which POST is captured
            # last is not guaranteed. The invalid debug model must still fall
            # back to grok-4.5 as the answering model.
            self.assertIn("grok-4.5", {call["model"] for call in self.captured["calls"]})
            self.assertEqual(rejected.get_json()["meta"]["model"], "grok-4.5")
            sized = self.client.post(
                "/identify",
                data={"image": (io.BytesIO(wide), "wide.jpg"), "sync": "1"},
                content_type="multipart/form-data",
            )
            self.assertEqual(sized.get_json()["meta"]["image_px"], "800x400")
        finally:
            os.environ.pop("MAX_IMAGE_PX", None)
            os.environ.pop("XAI_MODEL", None)

        os.environ["XAI_MODEL"] = "bad model"
        try:
            fallback = self.post_photo()
            self.assertEqual(fallback.get_json()["meta"]["model"], "grok-4.7")
        finally:
            os.environ.pop("XAI_MODEL", None)

    def test_common_names_fetch_together_and_cache(self):
        calls = []

        def slow_get(url, params=None, **kwargs):
            calls.append(params["q"])
            time.sleep(0.25)
            name = params["q"]
            epithet = name.split()[-1]
            return FakeResponse({
                "results": [{
                    "name": name,
                    "rank": "species",
                    "preferred_common_name": "Common " + epithet,
                }]
            })

        self.get_patch.stop()
        finfindr_server._inat_cache.clear()
        try:
            with patch("finfindr_server.http.get", side_effect=slow_get):
                started = time.perf_counter()
                found = finfindr_server.resolve_common_names([
                    "Thalassoma lucasanum",
                    "Zanclus imaginatus",
                    "Abudefduf fictus",
                ])
                elapsed = time.perf_counter() - started
        finally:
            self.get_patch.start()
        self.assertLess(elapsed, 0.45)
        self.assertEqual(found["Thalassoma lucasanum"], "Cortez Rainbow Wrasse")
        self.assertEqual(found["Zanclus imaginatus"], "Common imaginatus")
        self.assertEqual(found["Abudefduf fictus"], "Common fictus")
        self.assertNotIn("Thalassoma lucasanum", calls)
        listed = finfindr_server.resolve_common_names(
            ["Fishus alpha"],
            species=[{"scientific": "Fishus alpha", "common": "Alpha Fish"}],
        )
        self.assertEqual(listed["Fishus alpha"], "Alpha Fish")
        started = time.perf_counter()
        cached = finfindr_server.resolve_common_names(["Zanclus imaginatus"])
        self.assertLess(time.perf_counter() - started, 0.1)
        self.assertEqual(cached["Zanclus imaginatus"], "Common imaginatus")

    def test_slow_model_timeout_keeps_the_fast_candidates(self):
        self.fail_timeout = {"grok-4.7"}
        response = self.post_photo()
        body = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["meta"]["final_source"], "grok-4")
        self.assertFalse(body["meta"]["agreement"])
        self.assertFalse(body["meta"]["fallback_used"])
        by_model = {item["model"]: item for item in body["meta"]["attempts"]}
        self.assertFalse(by_model["grok-4.7"]["ok"])
        self.assertEqual(by_model["grok-4.7"]["error"], "timeout")
        self.assertTrue(by_model["grok-4"]["ok"])
        slow = next(call for call in self.captured["calls"] if call["model"] == "grok-4.7")
        fast = next(call for call in self.captured["calls"] if call["model"] == "grok-4")
        self.assertGreaterEqual(slow["timeout"], 150)
        self.assertLessEqual(slow["timeout"], 180)
        self.assertGreaterEqual(fast["timeout"], 150)
        self.assertLessEqual(fast["timeout"], 180)
        self.assertEqual(fast["temperature"], 0)
        self.assertEqual(fast["seed"], 0)
        self.assertIn("Cortez Rainbow Wrasse", body["result_text"])

    def test_both_models_timing_out_stays_json(self):
        self.fail_timeout = {"grok-4.7", "grok-4"}
        response = self.post_photo()
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertNotIn("error", body)
        self.assertIn("taking too long", body["message"])
        self.assertEqual(body["certainty"], "unsure")
        self.assertEqual(body["candidates"], [])
        self.assertFalse(body["meta"]["fallback_used"])
        self.assertEqual(len(body["meta"]["attempts"]), 2)
        self.assertEqual(body["meta"]["verify_pass_rate"], 0)
        self.assertEqual(body["meta"]["verify_success_rate"], 0)

    def test_one_model_http_400_does_not_drop_the_other(self):
        self.fail_status = {"grok-4.7": 400}
        response = self.post_photo()
        body = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("error", body)
        self.assertEqual(body["meta"]["final_source"], "grok-4")
        self.assertFalse(body["meta"]["model_candidates"]["grok-4.7"]["ok"])

    def test_single_model_list_can_replace_the_pair(self):
        os.environ["XAI_MODELS"] = "grok-4.7"
        self.fail_timeout = {"grok-4.7"}
        try:
            response = self.post_photo()
        finally:
            os.environ.pop("XAI_MODELS", None)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.captured["calls"]), 1)
        self.assertFalse(response.get_json()["meta"]["fallback_used"])
        self.assertEqual(response.get_json()["certainty"], "unsure")
        self.assertNotIn("error", response.get_json())
        self.assertIn("taking too long", response.get_json()["message"])

    def test_short_list_is_stable_and_incomplete_lists_are_not_cached(self):
        query = finfindr_server.species_query_for(
            finfindr_server._REGIONS_BY_ID["sea-of-cortez"], None, None
        )
        first = finfindr_server.load_species_list(query, "short", 5)
        self.assertTrue(first["complete"])
        calls = len(self.species_calls)
        second = finfindr_server.load_species_list(query, "short", 5)
        self.assertEqual(len(self.species_calls), calls)
        self.assertEqual(
            [item["scientific"] for item in first["species"]],
            [item["scientific"] for item in second["species"]],
        )

        def flaky(url, params=None, **kwargs):
            if int((params or {}).get("taxon_id") or 0) == 47178:
                raise requests.ConnectionError("fish down")
            return self.fake_get(url, params, **kwargs)

        self.get_patch.stop()
        finfindr_server._SPECIES_CACHE.clear()
        finfindr_server.clear_persisted_caches()
        finfindr_server._SPECIES_WAIT_RESULT.clear()
        finfindr_server._INAT_JSON_CACHE.clear()
        seen = {"n": 0}

        def counting(url, params=None, **kwargs):
            seen["n"] += 1
            return flaky(url, params, **kwargs)

        try:
            with patch("finfindr_server.http.get", side_effect=counting):
                partial = finfindr_server.load_species_list(query, "short", 5)
                self.assertTrue(partial["complete"])
                self.assertEqual(partial["source"], "curated")
                names = [item["scientific"] for item in partial["species"]]
                self.assertIn("Thalassoma lucasanum", names)
                self.assertIn("Hydrophis platurus", names)
                cached = finfindr_server._species_cache_get(("short", query["key"]))
                self.assertTrue(cached and cached.get("source") == "curated")
                before = seen["n"]
                finfindr_server.load_species_list(query, "short", 5)
                self.assertEqual(seen["n"], before)
        finally:
            self.get_patch.start()

    def test_prefetch_warms_short_lists_and_survives_errors(self):
        finfindr_server._SPECIES_CACHE.clear()
        finfindr_server._prefetch_species_lists()
        warmed = finfindr_server._species_cache_get(("short", "place:122897"))
        self.assertTrue(warmed and warmed.get("species"))
        self.assertIsNone(finfindr_server._species_cache_get(("full", "place:122897")))

        self.get_patch.stop()
        try:
            with patch("finfindr_server.http.get", side_effect=requests.ConnectionError("down")):
                finfindr_server._prefetch_species_lists()
        finally:
            self.get_patch.start()

    def test_prefetch_starts_once_per_process(self):
        os.environ["FINFINDR_PREFETCH"] = "1"
        finfindr_server._prefetch_pid = None
        try:
            with patch("finfindr_server.threading.Thread") as thread_cls:
                self.assertTrue(finfindr_server._start_species_prefetch())
                self.assertFalse(finfindr_server._start_species_prefetch())
                self.assertEqual(thread_cls.call_count, 1)
        finally:
            os.environ["FINFINDR_PREFETCH"] = "0"
            finfindr_server._prefetch_pid = None

    def test_candidates_and_verification_toggle(self):
        text = (
            "Common name: Cortez damselfish\n"
            "Scientific name: Stegastes rectifraenum\n"
            "Confidence: high\n"
            "Candidate 1: Stegastes rectifraenum | Cortez damselfish | high | yellow tail on a dark body\n"
            "Candidate 2: Microspathodon dorsalis | Giant damselfish | medium | taller body and a bump on the forehead\n"
        )
        candidates = finfindr_server.parse_model_candidates(text)
        self.assertEqual(candidates[0]["scientific_name"], "Stegastes rectifraenum")
        self.assertEqual(candidates[1]["scientific_name"], "Microspathodon dorsalis")
        self.assertEqual(finfindr_server.certainty_for(candidates, None), "likely")
        unsure = finfindr_server._parse_verification("Decision: UNSURE\nCertainty: unsure\nTrait: the tail is hidden", candidates)
        self.assertEqual(finfindr_server.certainty_for(candidates, unsure), "unsure")
        agreed = finfindr_server._parse_verification(
            "Decision: Microspathodon dorsalis\nCertainty: confident\nTrait: bump on the forehead",
            candidates,
        )
        self.assertEqual(finfindr_server.certainty_for(candidates, agreed), "confident")
        self.assertFalse(finfindr_server.candidates_far_apart(candidates))
        self.assertTrue(finfindr_server.candidates_far_apart(candidates[:1]))
        shown = finfindr_server.public_candidates(
            [{"scientific_name": "Microspathodon dorsalis", "common_name": "", "confidence": "medium", "trait": "forehead bump", "reference_photo": None}],
            None,
        )
        self.assertEqual(shown[0]["common_name"], "Giant Damselfish")
        self.assertEqual(shown[0]["scientific_name"], "Microspathodon dorsalis")
        self.assertTrue(finfindr_server.verification_enabled(False, None))
        self.assertFalse(finfindr_server.verification_enabled(True, "0"))
        os.environ["FINFINDR_VERIFY"] = "0"
        try:
            self.assertFalse(finfindr_server.verification_enabled(False, "1"))
            self.assertTrue(finfindr_server.verification_enabled(True, "1"))
        finally:
            os.environ.pop("FINFINDR_VERIFY", None)

    def test_reference_photos_skip_the_same_observation(self):
        def fake_get(url, params=None, **kwargs):
            if url.endswith("/taxa"):
                return FakeResponse({"results": [{
                    "id": 9,
                    "name": "Sphoeroides annulatus",
                    "rank": "species",
                    "default_photo": {
                        "id": 1,
                        "url": "https://inaturalist-open-data.s3.amazonaws.com/photos/1/square.jpg",
                        "attribution": "Sam Example",
                        "license_code": "cc-by",
                        "observation_id": 50,
                    },
                }]})
            return FakeResponse({"results": [{
                "id": 50,
                "photos": [{
                    "id": 1,
                    "url": "https://inaturalist-open-data.s3.amazonaws.com/photos/1/square.jpg",
                    "attribution": "Sam Example",
                    "license_code": "cc-by",
                }],
            }, {
                "id": 77,
                "photos": [{
                    "id": 2,
                    "url": "https://static.inaturalist.org/photos/2/square.jpg",
                    "attribution": "Alex Example",
                    "license_code": "cc-by-nc",
                }],
            }]})

        finfindr_server._PHOTO_CACHE.clear()
        with patch("finfindr_server.http.get", side_effect=fake_get):
            photos = finfindr_server.fetch_reference_photos("Sphoeroides annulatus", 24.1, -110.3, "50")
        self.assertTrue(photos)
        self.assertEqual(photos[0]["attribution"], "Alex Example")
        self.assertIn("medium.jpg", photos[0]["url"])
        self.assertEqual(photos[0]["license"], "CC BY-NC")
        self.assertTrue(photos[0]["source_url"].startswith("https://www.inaturalist.org/photos/"))

    def test_favicon_and_identify_errors_are_not_html(self):
        favicon = self.client.get("/favicon.ico")
        self.assertEqual(favicon.status_code, 204)
        self.assertNotIn(b"<html", favicon.data.lower())
        self.assertIn("geolocation=(self)", favicon.headers.get("Permissions-Policy", ""))
        self.assertEqual(self.client.head("/favicon.ico").status_code, 204)
        with patch("finfindr_server.prepare_image", side_effect=RuntimeError("boom")):
            response = self.post_photo()
        self.assertEqual(response.status_code, 500)
        self.assertTrue(response.is_json)
        self.assertEqual(response.get_json()["error"], "Something went wrong. Please try again.")

    def test_missing_api_key_is_json(self):
        os.environ.pop("XAI_API_KEY", None)
        response = self.post_photo()
        self.assertEqual(response.status_code, 500)
        body = response.get_json()
        self.assertIn("error", body)
        self.assertEqual(body["region_label"], "Sea of Cortez / La Paz")
        self.assertNotIn("json", self.captured)

    def test_photo_quality_blocks_discovery(self):
        self.captured["model_text"] = (
            "Photo quality: invalid\n"
            "Photo issue: it is a drawing, not a living animal\n"
            "Scientific name: Thalassoma lucasanum\n"
            "Confidence: high\n"
        )
        response = self.post_photo(lat="24.2", lng="-110.3")
        body = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["photo_quality"], "invalid")
        self.assertEqual(body["message"], "Can't identify from this photo.")
        self.assertIsNone(body["discovery"])
        self.assertEqual(body["candidates"], [])
        self.assertEqual(body["certainty"], "unsure")
        self.assertIn("photo_invalid", body["meta"]["evidence"])
        self.assertEqual(body["meta"]["budget_s"], 240)
        self.assertIsInstance(body["meta"]["total_ms"], int)
        self.assertIn("external_calls", body["meta"])
        self.assertIn("inat_throttled", body["meta"])
        self.assertNotEqual(body.get("reason"), "not_marine")
        prompt = self.prompt()
        self.assertIn("Never reject a species only because it is unrecorded", prompt)
        self.assertIn("Photo quality:", prompt)

    def test_range_extension_requires_no_records_within_100_km_and_verification(self):
        self.obs_calls = []

        def handler(url, params):
            if "obis.org" in url:
                return FakeResponse({"total": 0, "results": []})
            if "fishbase" in url:
                return FakeResponse({"data": []})
            if "observations" in url:
                self.obs_calls.append(params)
                if str(params.get("per_page")) == "1":
                    return FakeResponse({"total_results": 0, "results": []})
                return FakeResponse({"total_results": 1, "results": [{
                    "id": 321,
                    "geojson": {"coordinates": [-109.2, 26.4]},
                    "place_guess": "Bahia de los Angeles",
                    "observed_on": "2019-06-01",
                }]})
            if "taxa" in url:
                name = params.get("q") or "Zebrasoma xanthurum"
                return FakeResponse({"results": [{
                    "name": name,
                    "rank": "species",
                    "preferred_common_name": "Yellowtail tang",
                    "default_photo": {
                        "id": 4,
                        "url": "https://inaturalist-open-data.s3.amazonaws.com/photos/4/square.jpg",
                        "attribution": "Riley Example",
                        "license_code": "cc-by",
                    },
                }]})
            return None

        self.inat_handler = handler
        self.captured["model_text"] = (
            "Photo quality: good\n"
            "Photo issue: none\n"
            "Common name: Yellowtail tang\n"
            "Scientific name: Zebrasoma xanthurum\n"
            "Group: surgeonfish\n"
            "Alternative group considered: none\n"
            "Confidence: high\n"
            "Notes: A surgeonfish.\n"
            "Candidate 1: Zebrasoma xanthurum | Yellowtail tang | high | yellow tail\n"
            "Candidate 2: Acanthurus nigricans | Whitecheek surgeonfish | medium | pale cheek\n"
        )
        self.captured["verify_text"] = (
            "Decision: Zebrasoma xanthurum\n"
            "Certainty: confident\n"
            "Trait: yellow tail on a round body\n"
        )
        response = self.post_photo(lat="24.2", lng="-110.3", region="sea-of-cortez")
        body = response.get_json()
        self.assertEqual(response.status_code, 200)
        discovery = body["discovery"]
        self.assertEqual(discovery["type"], "possible_range_extension")
        self.assertEqual(discovery["scientific_name"], "Zebrasoma xanthurum")
        self.assertEqual(discovery["records_in_region"], 0)
        self.assertEqual(discovery["radius_km"], 100)
        self.assertEqual(discovery["rule"]["max_records"], 0)
        self.assertIn("within 100 km", discovery["rule"]["text"])
        self.assertTrue(discovery["traits"])
        self.assertEqual(discovery["nearest_records"][0]["place"], "Bahia de los Angeles")
        self.assertGreater(discovery["nearest_records"][0]["distance_km"], 0)
        self.assertIn("https://www.inaturalist.org/observations/321", discovery["nearest_records"][0]["url"])
        self.assertEqual(body["meta"]["range_km"], 100)
        self.assertEqual(body["meta"]["range_max"], 0)
        count_calls = [call for call in self.obs_calls if str(call.get("per_page")) == "1"]
        self.assertTrue(count_calls)
        self.assertEqual(int(float(count_calls[0]["radius"])), 100)
        page = self.client.get("/").get_data(as_text=True)
        self.assertIn("Possible range extension", page)
        self.assertIn("Submit to iNaturalist for expert confirmation", page)
        self.assertIn("Can't identify from this photo", page)
        self.assertIn("Range rule:", page)

    def test_records_inside_the_radius_are_not_a_range_extension(self):
        def handler(url, params):
            if "observations" in url and str(params.get("per_page")) == "1":
                return FakeResponse({"total_results": 6, "results": []})
            if "taxa" in url:
                return FakeResponse({"results": [{
                    "name": params.get("q") or "Zebrasoma xanthurum",
                    "rank": "species",
                    "preferred_common_name": "Yellowtail tang",
                    "default_photo": {
                        "id": 4,
                        "url": "https://inaturalist-open-data.s3.amazonaws.com/photos/4/square.jpg",
                        "attribution": "Riley Example",
                        "license_code": "cc-by",
                    },
                }]})
            return None

        self.inat_handler = handler
        self.captured["model_text"] = (
            "Photo quality: good\n"
            "Scientific name: Zebrasoma xanthurum\n"
            "Confidence: high\n"
            "Candidate 1: Zebrasoma xanthurum | Yellowtail tang | high | yellow tail\n"
            "Candidate 2: Acanthurus nigricans | Whitecheek surgeonfish | medium | pale cheek\n"
        )
        self.captured["verify_text"] = (
            "Decision: Zebrasoma xanthurum\nCertainty: confident\nTrait: yellow tail\n"
        )
        body = self.post_photo(lat="24.2", lng="-110.3").get_json()
        self.assertIsNone(body["discovery"])
        self.assertEqual(body["certainty"], "confident")
        self.assertEqual(body["notable_note"], "")

    def test_weak_evidence_does_not_flag_a_range_extension(self):
        def handler(url, params):
            if "observations" in url and str(params.get("per_page")) == "1":
                return FakeResponse({"total_results": 0, "results": []})
            return None

        self.inat_handler = handler
        self.captured["model_text"] = (
            "Photo quality: good\n"
            "Common name: Yellowtail tang\n"
            "Scientific name: Zebrasoma xanthurum\n"
            "Confidence: high\n"
            "Notes: A surgeonfish.\n"
        )
        body = self.post_photo(lat="24.2", lng="-110.3").get_json()
        self.assertIsNone(body["discovery"])
        self.assertEqual(body["certainty"], "likely")
        self.assertIn("notable if the identification is confirmed", body["notable_note"])
        self.assertTrue(body["candidates"])

    def test_no_match_does_not_force_a_species(self):
        def handler(url, params):
            if "taxa" in url:
                return FakeResponse({"results": [{
                    "name": params.get("q") or "Zebrasoma xanthurum",
                    "rank": "species",
                    "preferred_common_name": "Yellowtail tang",
                    "default_photo": {
                        "id": 4,
                        "url": "https://inaturalist-open-data.s3.amazonaws.com/photos/4/square.jpg",
                        "attribution": "Riley Example",
                        "license_code": "cc-by",
                    },
                }]})
            return None

        self.inat_handler = handler
        self.captured["model_text"] = (
            "Photo quality: good\n"
            "Scientific name: Zebrasoma xanthurum\n"
            "Confidence: medium\n"
            "Candidate 1: Zebrasoma xanthurum | Yellowtail tang | medium | yellow tail\n"
            "Candidate 2: Acanthurus nigricans | Whitecheek surgeonfish | medium | pale cheek\n"
        )
        self.captured["verify_text"] = (
            "Decision: NO_MATCH\nCertainty: unsure\nTrait: the tail spine is different from both\n"
        )
        body = self.post_photo(lat="24.2", lng="-110.3").get_json()
        self.assertIsNone(body["discovery"])
        self.assertTrue(body["candidates"])
        self.assertEqual(body["candidates"][0]["scientific_name"], "Zebrasoma xanthurum")
        self.assertEqual(body["certainty"], "unsure")
        self.assertIsNone(body["message"] if "message" in body else None)

    def test_discovery_can_be_turned_off(self):
        def handler(url, params):
            if "observations" in url and str(params.get("per_page")) == "1":
                return FakeResponse({"total_results": 0, "results": []})
            if "taxa" in url:
                return FakeResponse({"results": [{
                    "name": params.get("q") or "Zebrasoma xanthurum",
                    "rank": "species",
                    "default_photo": {
                        "id": 4,
                        "url": "https://static.inaturalist.org/photos/4/square.jpg",
                        "attribution": "Riley Example",
                        "license_code": "cc-by",
                    },
                }]})
            return None

        self.inat_handler = handler
        self.captured["model_text"] = (
            "Photo quality: good\n"
            "Scientific name: Zebrasoma xanthurum\n"
            "Confidence: high\n"
            "Candidate 1: Zebrasoma xanthurum | Yellowtail tang | high | yellow tail\n"
            "Candidate 2: Acanthurus nigricans | Whitecheek surgeonfish | medium | pale cheek\n"
        )
        self.captured["verify_text"] = (
            "Decision: Zebrasoma xanthurum\nCertainty: confident\nTrait: yellow tail\n"
        )
        body = self.post_photo(lat="24.2", lng="-110.3", debug="1", discovery="0").get_json()
        self.assertIsNone(body["discovery"])
        self.assertEqual(body["certainty"], "confident")
        self.assertFalse(body["meta"]["discovery_checked"])

    def test_range_radius_follows_the_environment(self):
        os.environ["FINFINDR_RANGE_KM"] = "40"
        os.environ["FINFINDR_RANGE_MAX"] = "0"
        self.obs_calls = []

        def handler(url, params):
            if "observations" in url:
                self.obs_calls.append(params)
                return FakeResponse({"total_results": 3, "results": []})
            return None

        self.inat_handler = handler
        try:
            self.assertEqual(finfindr_server.range_km(), 40)
            self.assertEqual(finfindr_server.range_record_limit(), 0)
            self.post_photo(lat="24.2", lng="-110.3")
        finally:
            os.environ.pop("FINFINDR_RANGE_KM", None)
            os.environ.pop("FINFINDR_RANGE_MAX", None)
        count_calls = [call for call in self.obs_calls if str(call.get("per_page")) == "1"]
        self.assertTrue(count_calls)
        self.assertEqual(int(float(count_calls[0]["radius"])), 40)


    def _taxon_photos(self, url, params):
        if "taxa" in url:
            return FakeResponse({"results": [{
                "name": params.get("q") or "Zebrasoma xanthurum",
                "rank": "species",
                "preferred_common_name": "Yellowtail tang",
                "default_photo": {
                    "id": 4,
                    "url": "https://inaturalist-open-data.s3.amazonaws.com/photos/4/square.jpg",
                    "attribution": "Riley Example",
                    "license_code": "cc-by",
                },
            }]})
        if "observations" in url and str(params.get("per_page")) == "1":
            return FakeResponse({"total_results": 4, "results": []})
        return None

    def test_confident_requires_a_passing_reference_check(self):
        """No path may return confident unless verification ran and agreed."""
        lone = (
            "Photo quality: good\n"
            "Common name: Yellowtail tang\n"
            "Scientific name: Zebrasoma xanthurum\n"
            "Confidence: high\n"
            "Notes: A surgeonfish.\n"
        )
        self.captured["model_text"] = lone
        self.inat_handler = None
        skipped = self.post_photo(lat="24.2", lng="-110.3").get_json()
        self.assertNotEqual(skipped["certainty"], "confident")
        self.assertEqual(skipped["certainty"], "likely")
        self.assertFalse(skipped["meta"]["verify_used"])
        self.assertEqual(skipped["meta"]["verify_skipped"], "no_photos")
        self.assertIsNone(skipped["discovery"])

        self.inat_handler = self._taxon_photos
        self.captured["verify_text"] = (
            "Decision: Zebrasoma xanthurum\nCertainty: confident\nTrait: yellow tail\n"
        )
        finfindr_server._PHOTO_CACHE.clear()
        finfindr_server._INAT_JSON_CACHE.clear()
        agreed = self.post_photo(lat="24.2", lng="-110.3").get_json()
        self.assertEqual(agreed["certainty"], "confident")
        self.assertEqual(agreed["meta"]["verify_pass_via"], "confident")
        self.assertTrue(agreed["meta"]["verify_used"])
        self.assertTrue(agreed["meta"]["verify_succeeded"])
        self.assertEqual(agreed["meta"]["verify_success_rate"], 1)
        self.assertEqual(agreed["meta"]["verify_timeout_s"], 60)
        self.assertEqual(agreed["meta"]["verify_skipped"], "")

        self.captured["verify_text"] = (
            "Decision: Zebrasoma xanthurum\nCertainty: likely\nTrait: yellow tail\n"
        )
        finfindr_server._PHOTO_CACHE.clear()
        hedged = self.post_photo(lat="24.2", lng="-110.3").get_json()
        self.assertEqual(hedged["certainty"], "confident")
        self.assertEqual(hedged["meta"]["verify_pass_via"], "likely")
        self.assertTrue(hedged["meta"]["verify_used"])
        self.assertTrue(hedged["meta"]["verify_succeeded"])
        self.assertIsNone(hedged["discovery"])
        self.assertEqual(hedged["meta"]["verify_certainty"].lower(), "likely")

        os.environ["FINFINDR_VERIFY_TIMEOUT"] = "18"
        self.captured["verify_raises"] = "timeout"
        finfindr_server._PHOTO_CACHE.clear()
        try:
            timed = self.post_photo(lat="24.2", lng="-110.3").get_json()
        finally:
            os.environ.pop("FINFINDR_VERIFY_TIMEOUT", None)
            self.captured.pop("verify_raises", None)
        self.assertEqual(timed["meta"]["verify_skipped"], "timeout")
        self.assertTrue(timed["meta"]["verify_attempted"])
        self.assertFalse(timed["meta"]["verify_used"])
        self.assertNotEqual(timed["certainty"], "confident")
        self.assertEqual(timed["meta"]["verify_pass_via"], "")
        self.assertIsNone(timed["discovery"])
        verify_calls = [call for call in self.captured["calls"] if call.get("verify")]
        self.assertTrue(verify_calls)
        self.assertGreaterEqual(verify_calls[-1]["timeout"], 18)

        wide = Image.new("RGB", (1200, 800), (10, 40, 80))
        buffer = io.BytesIO()
        wide.save(buffer, format="JPEG")
        smaller = finfindr_server._shrink_jpeg(buffer.getvalue(), 768)
        with Image.open(io.BytesIO(smaller)) as image:
            self.assertLessEqual(max(image.size), 768)
        self.assertFalse(finfindr_server.passing_verification(None))
        self.assertFalse(finfindr_server.passing_verification({"decision": "UNSURE", "certainty": "confident", "scientific_name": ""}))
        self.assertTrue(finfindr_server.candidates_far_apart([{
            "scientific_name": "Zebrasoma xanthurum",
            "confidence": "high",
        }]))
        self.assertEqual(
            finfindr_server.certainty_for(
                [{"scientific_name": "Zebrasoma xanthurum", "confidence": "high"}],
                None,
            ),
            "likely",
        )


    def test_discovery_stays_on_when_the_host_does_not_set_it(self):
        os.environ.pop("FINFINDR_DISCOVERY", None)
        self.assertTrue(finfindr_server.discovery_enabled(False, None))
        self.assertTrue(finfindr_server.discovery_enabled(False, ""))
        os.environ["FINFINDR_DISCOVERY"] = "0"
        try:
            self.assertFalse(finfindr_server.discovery_enabled(False, None))
            self.assertTrue(finfindr_server.discovery_enabled(True, "1"))
        finally:
            os.environ.pop("FINFINDR_DISCOVERY", None)

    def test_failed_verification_is_unsure_and_color_mismatch_is_no_match(self):
        def handler(url, params):
            if "taxa" in url:
                return FakeResponse({"results": [{
                    "name": params.get("q") or "Phataria unifascialis",
                    "rank": "species",
                    "preferred_common_name": "Blue Sea Star",
                    "default_photo": {
                        "id": 4,
                        "url": "https://inaturalist-open-data.s3.amazonaws.com/photos/4/square.jpg",
                        "attribution": "Riley Example",
                        "license_code": "cc-by",
                    },
                }]})
            return None

        self.inat_handler = handler
        self.captured["model_text"] = (
            "Photo quality: good\n"
            "Scientific name: Phataria unifascialis\n"
            "Common name: Blue Sea Star\n"
            "Confidence: high\n"
            "Candidate 1: Phataria unifascialis | Blue Sea Star | high | blue body\n"
            "Candidate 2: Pentaceraster cumingi | Panamic cushion star | medium | thick body\n"
        )
        self.captured["verify_text"] = (
            "Decision: UNSURE\nCertainty: unsure\nTrait: the arms are hidden\n"
        )
        unsure = self.post_photo(lat="24.2", lng="-110.3").get_json()
        self.assertNotEqual(unsure["certainty"], "confident")
        self.assertEqual(unsure["meta"]["verify_pass_via"], "")
        self.assertIsNone(unsure["discovery"])
        self.assertNotEqual(unsure["candidates"][0]["confidence"], "high")

        self.captured["verify_text"] = (
            "Decision: Phataria unifascialis\n"
            "Certainty: unsure\n"
            "Trait: the main color does not match the blue reference\n"
        )
        finfindr_server._PHOTO_CACHE.clear()
        mismatch = self.post_photo(lat="24.2", lng="-110.3").get_json()
        self.assertEqual(mismatch["certainty"], "unsure")
        self.assertEqual(mismatch["discovery"]["type"], "no_match")

    def test_off_list_species_can_be_a_range_extension_but_not_a_local_confident_match(self):
        def handler(url, params):
            if "obis.org" in url:
                return FakeResponse({"total": 0, "results": []})
            if "fishbase" in url:
                return FakeResponse({"data": []})
            if "taxa" in url:
                name = params.get("q") or ""
                if name.lower().startswith("notareal"):
                    return FakeResponse({"results": []})
                return FakeResponse({"results": [{
                    "name": name,
                    "rank": "species",
                    "preferred_common_name": "Red octopus",
                    "default_photo": {
                        "id": 8,
                        "url": "https://inaturalist-open-data.s3.amazonaws.com/photos/8/square.jpg",
                        "attribution": "Riley Example",
                        "license_code": "cc-by",
                    },
                }]})
            if "observations" in url and str(params.get("per_page")) == "1":
                return FakeResponse({"total_results": self.record_count, "results": []})
            if "observations" in url:
                return FakeResponse({"total_results": 1, "results": [{
                    "id": 88,
                    "geojson": {"coordinates": [-122.4, 37.8]},
                    "place_guess": "Monterey",
                    "observed_on": "2021-04-01",
                }]})
            return None

        self.inat_handler = handler
        self.record_count = 0
        self.captured["model_text"] = (
            "Photo quality: good\n"
            "Scientific name: Octopus rubescens\n"
            "Common name: Red octopus\n"
            "Confidence: medium\n"
            "Candidate 1: Phataria unifascialis | Blue Sea Star | medium | blue body\n"
            "Off-list 1: Octopus rubescens | Red octopus | reddish body and eight arms\n"
            "Off-list 2: Notareal animalis | Invented | not a real animal\n"
        )
        self.captured["verify_text"] = (
            "Decision: Octopus rubescens\nCertainty: confident\nTrait: eight arms and a reddish body\n"
        )
        body = self.post_photo(lat="24.2", lng="-110.3").get_json()
        self.assertEqual(body["discovery"]["type"], "possible_range_extension")
        self.assertEqual(body["discovery"]["scientific_name"], "Octopus rubescens")
        self.assertIsNotNone(body["discovery"]["nearest_known_record_km"])
        self.assertEqual(body["discovery"]["nearest_known_record_source"], "iNaturalist")
        self.assertTrue(body["discovery"]["nearest_records"])
        self.assertNotIn("Notareal animalis", json.dumps(body["candidates"]))

        self.record_count = 12
        finfindr_server._PHOTO_CACHE.clear()
        finfindr_server._INAT_JSON_CACHE.clear()
        local = self.post_photo(lat="24.2", lng="-110.3").get_json()
        self.assertIsNone(local["discovery"])
        self.assertEqual(local["certainty"], "confident")
        self.assertFalse(local["candidates"][0]["off_list"])
        self.assertGreaterEqual(body["meta"]["off_list_proposed"], 1)
        self.assertGreaterEqual(body["meta"]["off_list_kept"], 1)

    def test_identify_returns_inside_the_time_budget(self):
        def slow_post(*args, **kwargs):
            time.sleep(2)
            raise requests.Timeout("still going")

        started = time.perf_counter()
        with patch("finfindr_server.http.post", side_effect=slow_post):
            result = finfindr_server._call_vision_model(b"jpeg", "prompt", "grok-4.7", 0.4, "key")
        elapsed = time.perf_counter() - started
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "timeout")
        self.assertLess(elapsed, 1.5)

        response = self.post_photo()
        body = response.get_json()
        self.assertIn("total_ms", body["meta"])
        self.assertIsInstance(body["meta"]["total_ms"], int)
        self.assertLess(body["meta"]["total_ms"], 240000)
        self.assertEqual(body["meta"]["budget_s"], 240)


    def test_inat_throttle_and_retry_after(self):
        os.environ["FINFINDR_INAT_MIN_INTERVAL"] = "0.2"
        finfindr_server._INAT_NEXT_AT = 0.0
        calls = {"n": 0}

        def paced(url, params=None, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                response = FakeResponse({}, status=429)
                response.headers = {"Retry-After": "0"}
                return response
            return FakeResponse({"total_results": 3, "results": []})

        started = time.perf_counter()
        finfindr_server.begin_request_stats()
        try:
            with patch("finfindr_server.http.get", side_effect=paced):
                payload = finfindr_server._inat_get(
                    "https://api.inaturalist.org/v1/observations",
                    {"taxon_name": "Abudefduf troschelii", "per_page": 1},
                    (3, 8),
                    {"Accept": "application/json"},
                )
                finfindr_server._inat_get(
                    "https://api.inaturalist.org/v1/taxa",
                    {"q": "Abudefduf troschelii"},
                    (3, 8),
                    {"Accept": "application/json"},
                )
        finally:
            os.environ["FINFINDR_INAT_MIN_INTERVAL"] = "0"
        self.assertEqual(payload["total_results"], 3)
        self.assertGreaterEqual(finfindr_server.request_stat("inat_throttled"), 1)
        self.assertGreaterEqual(finfindr_server.request_stat("external_calls"), 1)
        self.assertGreaterEqual(calls["n"], 3)
        self.assertGreaterEqual(time.perf_counter() - started, 0.2)

    def test_exclude_observation_id_is_sent_as_not_id(self):
        response = self.post_photo(lat="24.2", lng="-110.3", exclude_observation_id="4242")
        self.assertEqual(response.status_code, 200)
        listed = [call for call in self.species_calls if str(call.get("not_id") or "") == "4242"]
        self.assertTrue(listed)

    def test_infrastructure_failure_keeps_the_model_confidence(self):
        self.captured["model_text"] = (
            "Photo quality: good\n"
            "Scientific name: Abudefduf troschelii\n"
            "Common name: Panamic sergeant major\n"
            "Confidence: high\n"
            "Candidate 1: Abudefduf troschelii | Panamic sergeant major | high | five black bars\n"
            "Candidate 2: Abudefduf saxatilis | Sergeant major | medium | a yellow back\n"
        )
        self.inat_handler = self._taxon_photos
        self.captured["verify_raises"] = "timeout"
        try:
            body = self.post_photo(lat="24.2", lng="-110.3").get_json()
        finally:
            self.captured.pop("verify_raises", None)
        self.assertEqual(body["meta"]["verify_skipped"], "timeout")
        self.assertNotEqual(body["certainty"], "confident")
        self.assertNotEqual(body["candidates"][0]["confidence"], "high")
        self.assertEqual(body["meta"]["verify_pass_via"], "")
        self.assertIsNone(body["discovery"])
        self.assertEqual(body["candidates"][0]["scientific_name"], "Abudefduf troschelii")

    def test_color_contradiction_becomes_no_match(self):
        self.captured["model_text"] = (
            "Photo quality: good\n"
            "Scientific name: Phataria unifascialis\n"
            "Common name: Blue Sea Star\n"
            "Confidence: high\n"
            "Notes: A red mesh star.\n"
            "Candidate 1: Phataria unifascialis | Blue Sea Star | high | red mesh pattern\n"
        )
        body = self.post_photo().get_json()
        self.assertEqual(body["discovery"]["type"], "no_match")
        self.assertEqual(body["certainty"], "unsure")
        self.assertTrue(finfindr_server.colors_contradict("red mesh pattern", "Blue Sea Star"))
        self.assertFalse(finfindr_server.colors_contradict("blue arms", "Blue Sea Star"))

    def test_verifier_choice_stays_ahead_of_a_higher_model_confidence(self):
        ordered = [
            {"scientific_name": "Arctocephalus townsendi", "common_name": "Guadalupe Fur Seal", "confidence": "medium", "trait": "pointed snout"},
            {"scientific_name": "Zalophus californianus", "common_name": "California Sea Lion", "confidence": "high", "trait": "sleek coat"},
        ]
        shown = finfindr_server.public_candidates(ordered, None, preserve_order=True)
        self.assertEqual(shown[0]["scientific_name"], "Arctocephalus townsendi")
        resorted = finfindr_server.public_candidates(ordered, None)
        self.assertEqual(resorted[0]["scientific_name"], "Zalophus californianus")

    def test_candidates_are_sorted_by_confidence(self):
        shown = finfindr_server.public_candidates([
            {"scientific_name": "Aaa aaa", "common_name": "Low fish", "confidence": "low", "trait": "plain"},
            {"scientific_name": "Bbb bbb", "common_name": "Medium fish", "confidence": "medium", "trait": "barred"},
        ], None)
        self.assertEqual([item["scientific_name"] for item in shown], ["Bbb bbb", "Aaa aaa"])

    def test_fallback_model_cannot_be_confident(self):
        self.fail_timeout = {"grok-4.7"}
        self.inat_handler = self._taxon_photos
        self.captured["model_text"] = (
            "Photo quality: good\n"
            "Scientific name: Thalassoma lucasanum\n"
            "Confidence: high\n"
            "Candidate 1: Thalassoma lucasanum | Cortez Rainbow Wrasse | high | yellow tail\n"
            "Candidate 2: Bodianus diplotaenia | Mexican hogfish | medium | red head\n"
        )
        self.captured["verify_text"] = (
            "Decision: Thalassoma lucasanum\nCertainty: confident\nTrait: a yellow tail unlike the red head\n"
        )
        body = self.post_photo(lat="24.2", lng="-110.3").get_json()
        self.assertEqual(body["meta"]["final_source"], "grok-4")
        self.assertFalse(body["meta"]["fallback_used"])
        self.assertEqual(body["certainty"], "confident")
        self.assertFalse(finfindr_server.passing_verification({
            "decision": "Thalassoma lucasanum",
            "scientific_name": "Thalassoma lucasanum",
            "certainty": "confident",
            "trait": "similar",
        }))

    def test_get_identify_is_json_405(self):
        response = self.client.get("/identify")
        self.assertEqual(response.status_code, 405)
        self.assertTrue(response.is_json)
        self.assertIn("error", response.get_json())

    def test_both_models_wait_up_to_the_client_timeout(self):
        self.assertEqual(finfindr_server.model_call_timeout(), 180)
        self.assertEqual(finfindr_server.job_wall_seconds(), 240)


    def _felis_taxon(self, url, params):
        if "taxa" in url:
            return FakeResponse({"results": [{
                "id": 118552,
                "name": "Felis catus",
                "rank": "species",
                "iconic_taxon_name": "Mammalia",
                "preferred_common_name": "Domestic Cat",
                "ancestors": [
                    {"name": "Animalia", "rank": "kingdom"},
                    {"name": "Chordata", "rank": "phylum"},
                    {"name": "Mammalia", "rank": "class"},
                    {"name": "Carnivora", "rank": "order"},
                    {"name": "Felidae", "rank": "family"},
                ],
            }]})
        return None

    def _assert_not_marine(self, body):
        self.assertEqual(body["reason"], "not_marine")
        self.assertEqual(
            body["message"],
            "FinFindr focuses on marine animal identification only. Please upload a photo of a marine animal.",
        )
        self.assertEqual(body["candidates"], [])
        self.assertIsNone(body["discovery"])
        self.assertNotEqual(body["certainty"], "confident")
        blob = json.dumps({key: value for key, value in body.items() if key != "meta"})
        self.assertNotIn("Felis", blob)
        self.assertNotIn("catus", blob)
        self.assertNotIn("Domestic", blob)
        self.assertNotIn("high", blob.lower())
        self.assertEqual(body["meta"]["budget_s"], 240)
        self.assertIsInstance(body["meta"]["total_ms"], int)
        self.assertEqual(body["meta"]["verify_pass_rate"], 0)
        self.assertEqual(body["meta"]["verify_success_rate"], 0)
        self.assertEqual(body["meta"]["verify_pass_reason"], "not_marine")

    def test_felis_catus_on_list_and_off_list_are_not_marine(self):
        page = self.client.get("/").get_data(as_text=True)
        self.assertIn("FinFindr focuses on marine animal identification only. Please upload a photo of a marine animal.", page)
        self.assertIn('data.reason === "not_marine"', page)
        self.assertIn("function showNotMarine", page)
        self.assertIn("Can't identify from this photo", page)
        self.inat_handler = self._felis_taxon
        self.captured["model_text"] = (
            "Photo quality: good\n"
            "Common name: Domestic Cat\n"
            "Scientific name: Felis catus\n"
            "Confidence: high\n"
            "Candidate 1: Felis catus | Domestic Cat | high | pointed ears and whiskers\n"
        )
        on_list = self.post_photo().get_json()
        self._assert_not_marine(on_list)
        self.captured["model_text"] = (
            "Photo quality: good\n"
            "Common name: Domestic Cat\n"
            "Scientific name: Felis catus\n"
            "Confidence: high\n"
            "Off-list 1: Felis catus | Domestic Cat | whiskers and a short muzzle\n"
        )
        off_list = self.post_photo(region="sea-of-cortez").get_json()
        self._assert_not_marine(off_list)
        self.assertIn("actinopterygii", finfindr_server.MARINE_SCOPE_ANCESTORS)
        self.assertIn("mollusca", finfindr_server.MARINE_SCOPE_ANCESTORS)
        self.assertIn("echinodermata", finfindr_server.MARINE_SCOPE_ANCESTORS)
        self.assertIn("cnidaria", finfindr_server.MARINE_SCOPE_ANCESTORS)
        self.assertIn("crustacea", finfindr_server.MARINE_SCOPE_ANCESTORS)
        self.assertIn("cheloniidae", finfindr_server.MARINE_SCOPE_ANCESTORS)
        self.assertEqual(finfindr_server.scope_setting(), "marine")
        os.environ["FINFINDR_SCOPE"] = "all"
        try:
            self.assertIsNone(finfindr_server.allowed_scope_names())
            self.assertTrue(finfindr_server.taxon_in_scope("Felis catus"))
        finally:
            os.environ.pop("FINFINDR_SCOPE", None)

    def test_fromia_lattice_is_not_a_confident_mithrodia(self):
        self.inat_handler = self._taxon_photos
        self.captured["model_text"] = (
            "Photo quality: good\n"
            "Common name: Bradley's Sea Star\n"
            "Scientific name: Mithrodia bradleyi\n"
            "Confidence: high\n"
            "Notes: A red sea star with a cream lattice across the arms.\n"
            "Candidate 1: Mithrodia bradleyi | Bradley's Sea Star | high | red lattice of raised plates\n"
            "Candidate 2: Phataria unifascialis | Blue Sea Star | medium | slender blue arms\n"
        )
        self.captured["verify_text"] = (
            "Decision: Mithrodia bradleyi\n"
            "Certainty: confident\n"
            "Trait: red lattice of raised plates\n"
        )
        body = self.post_photo(lat="24.2", lng="-110.3").get_json()
        self.assertNotEqual(body["certainty"], "confident")
        self.assertEqual(body["discovery"]["type"], "no_match")
        self.assertEqual(body["candidates"], [])
        self.assertIn("Confidence: none", body["result_text"])
        self.assertNotIn("Mithrodia", body["result_text"].split("Notes:")[0])
        finfindr_server._PHOTO_CACHE.clear()
        finfindr_server._INAT_JSON_CACHE.clear()
        self.captured["verify_text"] = (
            "Decision: NONE_OF_THESE\nCertainty: unsure\nTrait: the red lattice matches none of these\n"
        )
        none = self.post_photo(lat="24.2", lng="-110.3").get_json()
        self.assertEqual(none["discovery"]["type"], "no_match")
        self.assertNotEqual(none["certainty"], "confident")

    def test_abudefduf_short_diagnostic_trait_can_be_confident(self):
        """A rank-1 Gulf sergeant major must not collapse to all-low.

        The 8-character trait cutoff treated "5 bars" as a failed reference
        check, and that failure lowered every candidate and set certainty to
        unsure. A short explicit trait still passes. A fallback answer still
        cannot be confident.
        """
        self.assertTrue(finfindr_server.explicit_diagnostic_trait("5 bars"))
        self.assertFalse(finfindr_server.explicit_diagnostic_trait("similar"))
        self.inat_handler = self._taxon_photos
        self.captured["model_text"] = (
            "Photo quality: good\n"
            "Scientific name: Abudefduf troschelii\n"
            "Common name: Panamic sergeant major\n"
            "Confidence: high\n"
            "Notes: A silver damselfish with black bars.\n"
            "Candidate 1: Abudefduf troschelii | Panamic sergeant major | high | 5 bars\n"
            "Candidate 2: Abudefduf saxatilis | Sergeant major | medium | a yellow back\n"
        )
        self.captured["verify_text"] = (
            "Decision: Abudefduf troschelii\nCertainty: confident\nTrait: 5 bars\n"
        )
        body = self.post_photo(lat="24.2", lng="-110.3").get_json()
        self.assertEqual(body["certainty"], "confident")
        self.assertEqual(body["candidates"][0]["scientific_name"], "Abudefduf troschelii")
        self.assertEqual(body["candidates"][0]["confidence"], "high")
        self.assertNotEqual(body["candidates"][1]["confidence"], "high")
        self.assertTrue(body["meta"]["verify_succeeded"])
        self.assertEqual(body["meta"]["verify_pass_rate"], 0.5)
        self.assertIsNotNone(body["meta"]["verify_success_rate"])
        self.assertFalse(body["meta"]["fallback_used"])

    def test_form_and_json_posts_share_one_pipeline(self):
        image = Image.new("RGB", (120, 40), (8, 40, 90))
        exif = Image.Exif()
        exif[274] = 6
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", exif=exif)
        photo = buffer.getvalue()
        form = self.client.post(
            "/identify",
            data={"image": (io.BytesIO(photo), "reef.jpg"), "region": "sea-of-cortez", "sync": "1"},
            content_type="multipart/form-data",
        )
        form_payload = self.captured["json"]
        form_body = form.get_json()
        self.captured["calls"] = []
        js = self.client.post("/identify", json={
            "image_base64": base64.b64encode(photo).decode("ascii"),
            "region": "sea-of-cortez",
            "sync": "1",
        })
        js_body = js.get_json()
        self.assertEqual(form.status_code, 200)
        self.assertEqual(js.status_code, 200)
        self.assertEqual(form_payload["temperature"], 0)
        self.assertEqual(self.captured["json"]["temperature"], 0)
        self.assertEqual(form_payload["seed"], 0)
        self.assertEqual(self.captured["json"]["seed"], 0)

        def model_image(payload):
            content = payload["input"][0]["content"]
            return next(part["image_url"] for part in content if part["type"] == "input_image")

        self.assertEqual(model_image(form_payload), model_image(self.captured["json"]))
        encoded = model_image(form_payload).split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as sent:
            self.assertGreater(sent.size[1], sent.size[0])
        self.assertEqual(form_body["certainty"], js_body["certainty"])
        self.assertEqual(form_body["candidates"], js_body["candidates"])
        self.assertEqual(form_body["discovery"], js_body["discovery"])
        self.assertEqual(form_body.get("message"), js_body.get("message"))
        self.assertEqual(form_body["meta"]["image_px"], js_body["meta"]["image_px"])
        self.assertEqual(form_body["meta"]["image_px"], "40x120")

    def test_species_and_taxon_caches_survive_a_memory_clear(self):
        query = finfindr_server.species_query_for(
            finfindr_server._REGIONS_BY_ID["sea-of-cortez"], None, None
        )
        first = finfindr_server.load_species_list(query, "short", 5)
        calls = len(self.species_calls)
        finfindr_server._SPECIES_CACHE.clear()
        second = finfindr_server.load_species_list(query, "short", 5)
        self.assertEqual(len(self.species_calls), calls)
        self.assertEqual(second["species"][0]["scientific"], first["species"][0]["scientific"])
        self.inat_handler = self._felis_taxon
        self.assertEqual(finfindr_server.fetch_inat_taxon("Felis catus")["iconic_taxon_name"], "Mammalia")
        finfindr_server._INAT_JSON_CACHE.clear()
        self.inat_handler = None
        again = finfindr_server.fetch_inat_taxon("Felis catus")
        self.assertEqual(again["iconic_taxon_name"], "Mammalia")


    def test_murky_photo_cannot_override_a_verified_identification(self):
        self.inat_handler = self._taxon_photos
        self.captured["model_text"] = (
            "Photo quality: poor\n"
            "Photo issue: murky water and soft focus\n"
            "Scientific name: Abudefduf troschelii\n"
            "Common name: Panamic sergeant major\n"
            "Confidence: high\n"
            "Candidate 1: Abudefduf troschelii | Panamic sergeant major | high | 5 bars\n"
            "Candidate 2: Abudefduf saxatilis | Sergeant major | medium | a yellow back\n"
        )
        self.captured["verify_text"] = (
            "Decision: Abudefduf troschelii\nCertainty: confident\nTrait: 5 bars\n"
        )
        body = self.post_photo(lat="24.2", lng="-110.3").get_json()
        self.assertEqual(body["certainty"], "confident")
        self.assertEqual(body["candidates"][0]["scientific_name"], "Abudefduf troschelii")
        self.assertEqual(body["candidates"][0]["confidence"], "high")
        self.assertIn("Confidence: high", body["result_text"])
        self.assertNotIn("quality_lowered_confidence", body["meta"]["evidence"])
        self.assertTrue(finfindr_server.photo_is_usable("poor", "murky water and soft focus"))
        self.assertFalse(finfindr_server.photo_is_usable("invalid", "too blurry to see the animal"))
        self.assertFalse(finfindr_server.photo_is_usable("invalid", "blank frame"))
        self.assertGreater(body["meta"]["verify_pass_rate"], 0)
        self.assertNotIn("Can't identify", body.get("message") or "")
        self.assertTrue(all(call["temperature"] == 0 and call["seed"] == 0 for call in self.captured["calls"]))
        same = self.client.post("/identify", json={
            "image_base64": base64.b64encode(jpeg_bytes()).decode("ascii"),
            "region": "sea-of-cortez",
            "lat": "24.2",
            "lng": "-110.3",
            "nocache": "1",
            "sync": "1",
        }).get_json()
        self.assertEqual(same["candidates"][0]["scientific_name"], body["candidates"][0]["scientific_name"])
        self.assertEqual(same["certainty"], body["certainty"])
        self.assertEqual(same.get("message"), body.get("message"))
        page = self.client.get("/").get_data(as_text=True)
        self.assertIn("!data.candidates || !data.candidates.length", page)

    def test_verify_timeout_keeps_a_poor_primary_identification(self):
        self.inat_handler = self._taxon_photos
        self.captured["verify_raises"] = "timeout"
        self.captured["model_text"] = (
            "Photo quality: poor\n"
            "Photo issue: murky water and soft focus\n"
            "Scientific name: Carcharhinus leucas\n"
            "Common name: Bull shark\n"
            "Confidence: high\n"
            "Candidate 1: Carcharhinus leucas | Bull shark | high | broad blunt snout\n"
            "Candidate 2: Carcharhinus falciformis | Silky shark | medium | slender body\n"
        )
        body = self.post_photo(lat="24.2", lng="-110.3").get_json()
        self.assertEqual(body["meta"]["verify_skipped"], "timeout")
        self.assertEqual(body["candidates"][0]["scientific_name"], "Carcharhinus leucas")
        self.assertNotEqual(body["candidates"][0]["confidence"], "high")
        self.assertNotEqual(body["certainty"], "confident")
        self.assertNotIn("quality_lowered_confidence", body["meta"]["evidence"])
        self.assertNotIn("Can't identify", body.get("message") or "")
        self.assertIsNone(body["discovery"])

    def test_too_blurry_stops_before_an_identification(self):
        self.captured["model_text"] = (
            "Photo quality: invalid\n"
            "Photo issue: too blurry to see the animal\n"
            "Scientific name: Abudefduf troschelii\n"
            "Confidence: high\n"
            "Candidate 1: Abudefduf troschelii | Panamic sergeant major | high | 5 bars\n"
        )
        body = self.post_photo().get_json()
        self.assertEqual(body["message"], "Can't identify from this photo.")
        self.assertEqual(body["candidates"], [])
        self.assertEqual(body["result_text"], "")
        self.assertIsNone(body["discovery"])
        self.assertIsInstance(body["meta"], dict)
        self.assertNotIn("troschelii", json.dumps({k: v for k, v in body.items() if k != "meta"}))

    def test_murky_photo_does_not_change_confidence(self):
        self.captured["model_text"] = (
            "Photo quality: poor\n"
            "Photo issue: murky water and soft focus\n"
            "Scientific name: Abudefduf troschelii\n"
            "Common name: Panamic sergeant major\n"
            "Confidence: medium\n"
            "Candidate 1: Abudefduf troschelii | Panamic sergeant major | medium | 5 bars\n"
        )
        body = self.post_photo().get_json()
        self.assertEqual(body["candidates"][0]["scientific_name"], "Abudefduf troschelii")
        self.assertEqual(body["candidates"][0]["confidence"], "medium")
        self.assertIn("Confidence: medium", body["result_text"])
        self.assertNotIn("quality_lowered_confidence", body["meta"]["evidence"])
        self.assertNotIn("Can't identify", body.get("message") or "")

    def test_cat_phrase_is_not_marine_before_the_quality_gate(self):
        self.captured["model_text"] = (
            "Photo quality: poor\n"
            "Photo issue: domestic cat on grass, not a marine animal\n"
            "Scientific name: none\n"
            "Confidence: high\n"
        )
        form = self.post_photo().get_json()
        js = self.client.post("/identify", json={
            "image_base64": base64.b64encode(jpeg_bytes()).decode("ascii"),
            "region": "sea-of-cortez",
            "nocache": "1",
            "sync": "1",
        }).get_json()
        for body in (form, js):
            self.assertEqual(body["reason"], "not_marine")
            self.assertEqual(
                body["message"],
                "FinFindr focuses on marine animal identification only. Please upload a photo of a marine animal.",
            )
            self.assertEqual(body["candidates"], [])
            self.assertNotIn("Can't identify", body["message"])

    def test_unlisted_medium_winner_is_a_range_extension(self):
        def handler(url, params):
            if "obis.org" in url:
                return FakeResponse({"total": 0})
            if "fishbase" in url:
                return FakeResponse({"data": []})
            if "taxa" in url:
                name = params.get("q") or "Octopus rubescens"
                return FakeResponse({"results": [{
                    "name": name,
                    "rank": "species",
                    "iconic_taxon_name": "Mollusca",
                    "ancestors": [{"name": "Mollusca", "rank": "phylum"}],
                }]})
            if "observations" in url and str(params.get("per_page")) == "1":
                return FakeResponse({"total_results": 0, "results": []})
            return None

        self.inat_handler = handler
        self.captured["model_text"] = (
            "Photo quality: good\n"
            "Scientific name: Octopus rubescens\n"
            "Common name: Red octopus\n"
            "Confidence: medium\n"
            "Candidate 1: Octopus rubescens | Red octopus | medium | reddish body without eye spots\n"
        )
        body = self.post_photo(lat="24.2", lng="-110.3").get_json()
        self.assertGreaterEqual(body["meta"]["off_list_proposed"], 1)
        self.assertGreaterEqual(body["meta"]["off_list_kept"], 1)
        self.assertTrue(body["candidates"][0]["off_list"])
        self.assertEqual(body["discovery"]["type"], "possible_range_extension")
        self.assertEqual(body["discovery"]["scientific_name"], "Octopus rubescens")
        self.assertEqual(body["discovery"]["external_range"], "absent")
        self.assertNotEqual(body["certainty"], "confident")
        self.assertNotIn("not strong enough", body.get("notable_note") or "")

    def test_no_match_does_not_fire_when_the_top_candidate_is_the_closest(self):
        self.fail_timeout = {"grok-4.7"}
        self.inat_handler = self._taxon_photos
        self.captured["model_text"] = (
            "Photo quality: good\n"
            "Scientific name: Nerita scabricosta\n"
            "Common name: Rough nerite\n"
            "Confidence: high\n"
            "Candidate 1: Nerita scabricosta | Rough nerite | high | rough spiral ridges\n"
        )
        self.captured["verify_text"] = (
            "Decision: NO_MATCH\nCertainty: unsure\nTrait: the shell ridges are hard to see\n"
        )
        body = self.post_photo(lat="24.2", lng="-110.3").get_json()
        self.assertEqual(body["meta"]["final_source"], "grok-4")
        self.assertFalse(body["meta"]["fallback_used"])
        self.assertNotEqual((body.get("discovery") or {}).get("type"), "no_match")
        self.assertEqual(body["candidates"][0]["scientific_name"], "Nerita scabricosta")

    def test_parallel_models_are_both_recorded(self):
        self.captured["model_text"] = (
            "Photo quality: good\n"
            "Scientific name: Octopus hubbsorum\n"
            "Common name: Hubbs' octopus\n"
            "Confidence: medium\n"
            "Candidate 1: Octopus hubbsorum | Hubbs' octopus | medium | brown body without eye spots\n"
            "Candidate 2: Octopus bimaculatus | Two-spot octopus | low | eye spots\n"
        )
        body = self.post_photo().get_json()
        self.assertEqual(body["candidates"][0]["scientific_name"], "Octopus hubbsorum")
        self.assertFalse(body["meta"]["fallback_used"])

    def test_nocache_skips_the_model_reply_cache(self):
        os.environ["FINFINDR_MODEL_CACHE"] = "1"
        try:
            self.post_photo()
            cached_calls = len(self.captured["calls"])
            self.post_photo()
            self.assertEqual(len(self.captured["calls"]), cached_calls)
            self.post_photo(nocache="1")
            self.assertGreater(len(self.captured["calls"]), cached_calls)
        finally:
            os.environ["FINFINDR_MODEL_CACHE"] = "0"
            finfindr_server.clear_model_cache()

    def test_model_cache_repeats_the_same_answer(self):
        os.environ["FINFINDR_MODEL_CACHE"] = "1"
        try:
            first = self.post_photo()
            calls = len(self.captured["calls"])
            second = self.post_photo()
        finally:
            os.environ["FINFINDR_MODEL_CACHE"] = "0"
            finfindr_server.clear_model_cache()
        self.assertEqual(len(self.captured["calls"]), calls)
        self.assertEqual(first.get_json()["certainty"], second.get_json()["certainty"])
        self.assertEqual(
            first.get_json()["candidates"][0]["scientific_name"],
            second.get_json()["candidates"][0]["scientific_name"],
        )

    def test_off_list_followup_keeps_rubescens_when_verify_times_out(self):
        def handler(url, params):
            if "obis.org" in url:
                return FakeResponse({"total": 0, "results": []})
            if "fishbase" in url:
                return FakeResponse({"data": []})
            if "taxa" in url:
                name = params.get("q") or "Octopus rubescens"
                return FakeResponse({"results": [{
                    "name": name,
                    "rank": "species",
                    "iconic_taxon_name": "Mollusca",
                    "preferred_common_name": "Red octopus",
                    "ancestors": [
                        {"name": "Mollusca", "rank": "phylum"},
                        {"name": "Octopus", "rank": "genus"},
                    ],
                    "default_photo": {
                        "id": 8,
                        "url": "https://inaturalist-open-data.s3.amazonaws.com/photos/8/square.jpg",
                        "attribution": "Riley Example",
                        "license_code": "cc-by",
                    },
                }]})
            if "observations" in url and str(params.get("per_page")) == "1":
                return FakeResponse({"total_results": 0, "results": []})
            if "observations" in url:
                return FakeResponse({"total_results": 1, "results": [{
                    "id": 88,
                    "geojson": {"coordinates": [-122.4, 37.8]},
                    "place_guess": "Monterey",
                    "observed_on": "2021-04-01",
                }]})
            return None

        self.inat_handler = handler
        self.captured["verify_raises"] = "timeout"
        shared = (
            "Photo quality: good\n"
            "Confidence: medium\n"
            "Candidate 1: Octopus bimaculatus | Two-spot octopus | medium | faint side spots\n"
            "Candidate 2: Octopus hubbsorum | Hubbs' octopus | low | brown body\n"
        )
        other = (
            "Photo quality: good\n"
            "Confidence: medium\n"
            "Candidate 1: Octopus hubbsorum | Hubbs' octopus | medium | brown body\n"
            "Candidate 2: Octopus bimaculatus | Two-spot octopus | low | faint side spots\n"
        )
        self.captured["model_text_by_model"] = {"grok-4": shared, "grok-4.7": other}
        self.captured["model_text"] = shared
        self.captured["offlist_text"] = (
            "Off-list 1: Octopus rubescens | Red octopus | reddish body without eye spots\n"
            "Off-list 2: none\n"
        )
        body = self.post_photo(lat="24.2", lng="-110.3", region="sea-of-cortez").get_json()
        self.assertGreaterEqual(body["meta"]["off_list_proposed"], 1)
        self.assertGreaterEqual(body["meta"]["off_list_kept"], 1)
        self.assertEqual(body["discovery"]["type"], "possible_range_extension")
        self.assertEqual(body["discovery"]["scientific_name"], "Octopus rubescens")
        self.assertNotEqual(body["certainty"], "confident")
        self.assertEqual(body["meta"]["verify_skipped"], "timeout")
        self.assertEqual(body["meta"]["verify_pass_rate"], 0)
        self.assertEqual(body["meta"]["verify_success_rate"], 0)
        self.assertIn("OFFLIST_FOLLOWUP", self.prompt())

    def test_health_and_commit_meta(self):
        os.environ["RENDER_GIT_COMMIT"] = "abc123"
        try:
            health = self.client.get("/health")
            body = health.get_json()
            identified = self.post_photo().get_json()
        finally:
            os.environ.pop("RENDER_GIT_COMMIT", None)
        self.assertEqual(health.status_code, 200)
        self.assertTrue(body["ok"])
        self.assertEqual(body["commit"], "abc123")
        self.assertEqual(body["model"], "grok-4.7")
        self.assertEqual(body["models"], ["grok-4", "grok-4.7"])
        self.assertEqual(body["fallback_model"], "")
        self.assertEqual(identified["meta"]["commit"], "abc123")
        self.assertIn("verify_pass_rate", identified["meta"])
        self.assertIn("verify_success_rate", identified["meta"])
        skipped = {}
        finfindr_server.apply_verify_rates(skipped, False, False, 0, 0, "no_time")
        self.assertEqual(skipped["verify_pass_rate"], 0)
        self.assertEqual(skipped["verify_pass_reason"], "no_time")
        self.assertEqual(skipped["verify_success_rate"], 0.0)

    def test_prompt_names_leucas_and_rubescens(self):
        self.post_photo(region="sea-of-cortez")
        prompt = self.prompt()
        self.assertIn("Carcharhinus leucas", prompt)
        self.assertIn("Octopus rubescens", prompt)
        self.assertIn("Felimare californiensis", prompt)
        self.assertIn("Nerita scabricosta", prompt)


    def test_scope_keeps_sea_lion_gull_and_oystercatcher(self):
        def taxon(name, iconic, ancestors):
            return {
                "name": name,
                "rank": "species",
                "iconic_taxon_name": iconic,
                "ancestors": [{"name": item, "rank": "family"} for item in ancestors],
            }

        cases = (
            ("Zalophus californianus", "Mammalia", ["Otariidae", "Pinnipedia"]),
            ("Larus heermanni", "Aves", ["Laridae"]),
            ("Haematopus palliatus", "Aves", ["Haematopodidae"]),
            ("Ursus maritimus", "Mammalia", ["Ursidae"]),
        )
        for name, iconic, ancestors in cases:
            self.assertTrue(
                finfindr_server.ancestry_in_scope(taxon(name, iconic, ancestors)),
                name,
            )
        self.assertFalse(finfindr_server.text_says_not_marine("hauled out on rocks, not a marine mammal in the water"))
        self.assertTrue(finfindr_server.text_says_not_marine("domestic cat on grass"))
        self.inat_handler = lambda url, params: FakeResponse({"results": [{
            "name": "Zalophus californianus",
            "rank": "species",
            "iconic_taxon_name": "Mammalia",
            "ancestors": [
                {"name": "Mammalia", "rank": "class"},
                {"name": "Otariidae", "rank": "family"},
                {"name": "Pinnipedia", "rank": "order"},
            ],
        }]}) if "taxa" in url else None
        self.captured["model_text"] = (
            "Photo quality: good\n"
            "Photo issue: hauled out on rocks\n"
            "Scientific name: Zalophus californianus\n"
            "Common name: California sea lion\n"
            "Confidence: high\n"
            "Candidate 1: Zalophus californianus | California sea lion | high | external ear flaps\n"
        )
        body = self.post_photo().get_json()
        self.assertNotEqual(body.get("reason"), "not_marine")
        self.assertEqual(body["candidates"][0]["scientific_name"], "Zalophus californianus")

    def test_records_in_the_region_are_not_off_list(self):
        def handler(url, params):
            if "observations" in url and str(params.get("per_page")) == "1":
                return FakeResponse({"total_results": 29, "results": []})
            if "taxa" in url:
                return FakeResponse({"results": [{
                    "name": params.get("q") or "Felimare californiensis",
                    "rank": "species",
                    "iconic_taxon_name": "Mollusca",
                    "ancestors": [{"name": "Mollusca", "rank": "phylum"}],
                }]})
            return None

        self.inat_handler = handler
        self.captured["model_text"] = (
            "Photo quality: good\n"
            "Scientific name: Felimare californiensis\n"
            "Common name: California chromodorid\n"
            "Confidence: medium\n"
            "Candidate 1: Felimare californiensis | California chromodorid | medium | blue mantle with yellow spots\n"
            "Off-list 1: Felimare californiensis | California chromodorid | blue mantle\n"
        )
        body = self.post_photo(lat="24.2", lng="-110.3").get_json()
        self.assertFalse(body["candidates"][0]["off_list"])
        self.assertNotEqual((body.get("discovery") or {}).get("type"), "possible_range_extension")

    def test_upstream_429_is_retried_and_not_a_502(self):
        queue = {"left": 1}

        def flaky(url, json=None, **kwargs):
            if queue["left"]:
                queue["left"] -= 1
                response = FakeResponse({"error": "slow down"}, status=429)
                response.headers = {"Retry-After": "0"}
                return response
            return self.fake_post(url, json=json, **kwargs)

        self.post_patch.stop()
        try:
            with patch("finfindr_server.http.post", side_effect=flaky):
                response = self.post_photo()
        finally:
            self.post_patch.start()
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("error", response.get_json())
        self.assertEqual(queue["left"], 0)

    def test_exhausted_429_is_rate_limited_with_meta(self):
        def always(url, json=None, **kwargs):
            response = FakeResponse({"error": "slow down"}, status=429)
            response.headers = {"Retry-After": "0"}
            return response

        self.post_patch.stop()
        try:
            with patch("finfindr_server.http.post", side_effect=always):
                response = self.post_photo()
        finally:
            self.post_patch.start()
        body = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["reason"], "rate_limited")
        self.assertNotIn("error", body)
        self.assertIsInstance(body["meta"], dict)
        self.assertIn("total_ms", body["meta"])

    def test_agreement_plus_inconclusive_verify_can_be_confident(self):
        self.inat_handler = self._taxon_photos
        self.captured["model_text"] = (
            "Photo quality: good\n"
            "Scientific name: Zalophus californianus\n"
            "Common name: California sea lion\n"
            "Confidence: high\n"
            "Candidate 1: Zalophus californianus | California sea lion | high | external ear flaps\n"
            "Candidate 2: Arctocephalus townsendi | Guadalupe fur seal | low | a pointed snout\n"
        )
        self.captured["verify_text"] = (
            "Decision: UNSURE\nCertainty: unsure\nTrait: the reference photos do not separate them\n"
        )
        body = self.post_photo(lat="24.2", lng="-110.3").get_json()
        self.assertTrue(body["meta"]["agreement"])
        self.assertNotEqual(body["certainty"], "confident")
        self.assertEqual(body["meta"]["verify_pass_via"], "")
        self.assertEqual(body["candidates"][0]["scientific_name"], "Zalophus californianus")
        self.captured["verify_text"] = (
            "Decision: NO_MATCH\nCertainty: unsure\nTrait: the ear flaps do not match these seals\n"
        )
        finfindr_server._PHOTO_CACHE.clear()
        blocked = self.post_photo(lat="24.2", lng="-110.3").get_json()
        self.assertNotEqual(blocked["certainty"], "confident")

    def test_likely_passes_only_for_the_leading_species(self):
        self.assertFalse(finfindr_server.likely_names_the_leader(
            {"decision": "UNSURE", "certainty": "likely", "scientific_name": "", "trait": "yellow tail"},
            "Zebrasoma xanthurum",
        ))
        self.assertFalse(finfindr_server.likely_names_the_leader(
            {"decision": "NONE_OF_THESE", "certainty": "likely", "scientific_name": "", "trait": "yellow tail"},
            "Zebrasoma xanthurum",
        ))
        self.assertFalse(finfindr_server.likely_names_the_leader(
            {"decision": "Abudefduf saxatilis", "certainty": "likely", "scientific_name": "Abudefduf saxatilis", "trait": "a yellow back"},
            "Abudefduf troschelii",
        ))
        self.assertFalse(finfindr_server.likely_names_the_leader(
            {"decision": "Zebrasoma xanthurum", "certainty": "likely", "scientific_name": "Zebrasoma xanthurum", "trait": "the color does not match"},
            "Zebrasoma xanthurum",
        ))
        self.assertTrue(finfindr_server.likely_names_the_leader(
            {"decision": "Zebrasoma xanthurum", "certainty": "likely", "scientific_name": "Zebrasoma xanthurum", "trait": "yellow tail"},
            "Zebrasoma xanthurum",
        ))
        os.environ["FINFINDR_VERIFY_ACCEPT_LIKELY"] = "0"
        try:
            self.assertFalse(finfindr_server.likely_names_the_leader(
                {"decision": "Zebrasoma xanthurum", "certainty": "likely", "scientific_name": "Zebrasoma xanthurum", "trait": "yellow tail"},
                "Zebrasoma xanthurum",
            ))
        finally:
            os.environ.pop("FINFINDR_VERIFY_ACCEPT_LIKELY", None)
        prompt = finfindr_server._verification_prompt([], "the Sea of Cortez")
        self.assertIn("confident: a diagnostic trait is clearly visible and the runner-up is ruled out.", prompt)
        self.assertIn("likely: a consistent diagnostic trait is visible, but the runner-up is not fully excluded.", prompt)
        self.assertIn("unsure: no diagnostic trait is visible, or the photo is ambiguous.", prompt)
        self.post_photo()
        self.assertIn("high: a diagnostic trait is clearly visible and look-alikes are excluded.", self.prompt())
        self.assertIn("medium: the photo is consistent with the species, but some look-alikes are not excluded.", self.prompt())
        self.assertIn("low: the evidence is weak or only partly visible.", self.prompt())
        parsed = finfindr_server.parse_model_candidates(
            "Off-list 1: Octopus rubescens | Red octopus | low | reddish body without eye spots\n"
        )
        self.assertEqual(parsed[0]["confidence"], "low")
        self.assertEqual(parsed[0]["trait"], "reddish body without eye spots")

    def test_disagreement_shows_both_species_until_verify_separates_them(self):
        self.captured["model_text_by_model"] = {
            "grok-4": (
                "Photo quality: good\n"
                "Scientific name: Abudefduf troschelii\n"
                "Confidence: high\n"
                "Candidate 1: Abudefduf troschelii | Panamic sergeant major | high | 5 bars\n"
            ),
            "grok-4.7": (
                "Photo quality: good\n"
                "Scientific name: Abudefduf saxatilis\n"
                "Confidence: high\n"
                "Candidate 1: Abudefduf saxatilis | Sergeant major | high | a yellow back\n"
            ),
        }
        body = self.post_photo().get_json()
        names = [item["scientific_name"] for item in body["candidates"]]
        self.assertEqual(names[:2], ["Abudefduf troschelii", "Abudefduf saxatilis"])
        self.assertFalse(body["meta"]["agreement"])
        self.assertNotEqual(body["certainty"], "confident")
        self.assertFalse(body["meta"]["fallback_used"])
        self.assertIn(body["meta"]["final_source"], ("grok-4", "grok-4.7", "both"))

    def test_both_model_replies_are_kept(self):
        body = self.post_photo().get_json()
        self.assertEqual(body["meta"]["models_run"], ["grok-4", "grok-4.7"])
        for name in body["meta"]["models_run"]:
            self.assertIn("reply", body["meta"]["model_candidates"][name])
        self.assertIn("verify_reply", body["meta"])
        self.assertIn("verify_why", body["meta"])
        self.assertIn("verify_pass_via", body["meta"])

    def test_long_primary_gives_verify_the_time_that_is_left(self):
        started = time.monotonic() - 70
        seconds = finfindr_server.verify_call_budget(started)
        self.assertGreaterEqual(seconds, 45)
        quick = finfindr_server.verify_call_budget(time.monotonic())
        self.assertGreaterEqual(quick, 45)

    def test_unresolved_mammal_or_bird_is_not_rejected(self):
        sea_lion = {
            "name": "Zalophus californianus",
            "iconic_taxon_name": "Mammalia",
            "ancestors": [],
        }
        gull = {"name": "Larus heermanni", "iconic_taxon_name": "Aves", "ancestors": []}
        self.assertIsNone(finfindr_server.ancestry_in_scope(sea_lion))
        self.assertIsNone(finfindr_server.ancestry_in_scope(gull))
        cat = {
            "name": "Felis catus",
            "iconic_taxon_name": "Mammalia",
            "ancestors": [{"name": "Felidae", "rank": "family"}],
        }
        self.assertIs(finfindr_server.ancestry_in_scope(cat), False)

        def handler(url, params):
            if str(url).rstrip("/").endswith("/taxa/123"):
                return FakeResponse({"results": [{
                    "id": 123,
                    "name": "Zalophus californianus",
                    "rank": "species",
                    "iconic_taxon_name": "Mammalia",
                    "ancestor_ids": [40151],
                    "ancestors": [
                        {"name": "Mammalia", "rank": "class"},
                        {"name": "Otariidae", "rank": "family"},
                    ],
                }]})
            if "taxa" in url:
                return FakeResponse({"results": [{
                    "id": 123,
                    "name": "Zalophus californianus",
                    "rank": "species",
                    "iconic_taxon_name": "Mammalia",
                    "ancestors": [],
                }]})
            return None

        self.inat_handler = handler
        self.assertTrue(finfindr_server.taxon_in_scope("Zalophus californianus"))
        self.captured["model_text"] = (
            "Photo quality: good\n"
            "Photo issue: hauled out on rocks\n"
            "Scientific name: Zalophus californianus\n"
            "Common name: California sea lion\n"
            "Confidence: high\n"
            "Candidate 1: Zalophus californianus | California sea lion | high | external ear flaps\n"
        )
        body = self.post_photo().get_json()
        self.assertNotEqual(body.get("reason"), "not_marine")
        self.assertIsInstance(body["meta"], dict)

    def test_off_list_discovery_is_present_when_the_outside_range_is_known(self):
        def handler(url, params):
            if "obis.org" in url:
                return FakeResponse({"total": 4})
            if "observations" in url and str(params.get("per_page")) == "1":
                return FakeResponse({"total_results": 0, "results": []})
            if "taxa" in url:
                return FakeResponse({"results": [{
                    "name": params.get("q") or "Octopus rubescens",
                    "rank": "species",
                    "iconic_taxon_name": "Mollusca",
                    "ancestors": [{"name": "Mollusca", "rank": "phylum"}],
                }]})
            return None

        self.inat_handler = handler
        self.captured["model_text"] = (
            "Photo quality: good\n"
            "Scientific name: Octopus rubescens\n"
            "Common name: Red octopus\n"
            "Confidence: medium\n"
            "Candidate 1: Octopus rubescens | Red octopus | medium | reddish body without eye spots\n"
        )
        body = self.post_photo(lat="24.2", lng="-110.3").get_json()
        self.assertTrue(body["candidates"][0]["off_list"])
        self.assertIsInstance(body["discovery"], dict)
        self.assertEqual(body["discovery"]["external_range"], "known")
        self.assertEqual(body["discovery"]["external_range_source"], "OBIS")
        self.assertIn("Outside iNaturalist:", self.client.get("/").get_data(as_text=True))

    def test_blunt_snout_ranks_the_bull_shark_ahead_of_the_silky(self):
        self.captured["model_text"] = (
            "Photo quality: good\n"
            "Scientific name: Carcharhinus falciformis\n"
            "Common name: Silky shark\n"
            "Confidence: medium\n"
            "Notes: stocky body and a blunt snout\n"
            "Candidate 1: Carcharhinus falciformis | Silky shark | medium | blunt snout\n"
            "Candidate 2: Carcharhinus leucas | Bull shark | medium | small eyes\n"
        )
        body = self.post_photo().get_json()
        self.assertEqual(body["candidates"][0]["scientific_name"], "Carcharhinus falciformis")
        self.assertIn("Rank Carcharhinus leucas first", "\n".join(finfindr_server.CORTEZ_LOOKALIKE_LINES))

    def test_prefetched_reference_photo_is_reused(self):
        finfindr_server._cache_photos(
            "ref:abudefduf troschelii::",
            [{
                "url": "https://example.test/troschelii.jpg",
                "attribution": "Prefetch",
                "license": "CC BY",
                "source_url": "https://www.inaturalist.org/photos/1",
            }],
            3600,
        )
        self.inat_handler = lambda url, params: FakeResponse({"results": []})
        self.captured["model_text"] = (
            "Photo quality: good\n"
            "Scientific name: Abudefduf troschelii\n"
            "Common name: Panamic sergeant major\n"
            "Confidence: high\n"
            "Candidate 1: Abudefduf troschelii | Panamic sergeant major | high | 5 bars\n"
        )
        self.captured["verify_text"] = (
            "Decision: Abudefduf troschelii\nCertainty: confident\nTrait: 5 bars\n"
        )
        body = self.post_photo(lat="24.2", lng="-110.3").get_json()
        self.assertTrue(body["meta"]["verify_used"])
        self.assertEqual(body["certainty"], "confident")
        urls = json.dumps(self.captured["calls"])
        self.assertIn("https://example.test/troschelii.jpg", json.dumps(self.captured.get("json")))

    def test_stream_stops_at_the_response_wall(self):
        class Stuck(object):
            def is_alive(self):
                return True

            def join(self, timeout=None):
                time.sleep(0.02)

        started = time.monotonic()
        with finfindr_server.app.test_request_context("/identify", method="POST"):
            with patch.object(finfindr_server, "response_wall_seconds", return_value=0.12):
                raw = b"".join(finfindr_server._identify_chunks(Stuck(), started, lambda: None))
        body = json.loads(raw.decode().strip())
        self.assertEqual(body["reason"], "timeout")
        self.assertIsInstance(body["meta"], dict)
        self.assertIn("try again", body["message"].lower())
        self.assertLess(time.monotonic() - started, 2.0)

    def test_async_job_returns_an_id_then_the_result(self):
        response = self.client.post(
            "/identify",
            data={"image": (io.BytesIO(jpeg_bytes()), "reef.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 202)
        job = response.get_json()
        self.assertTrue(job["job_id"])
        deadline = time.time() + 8
        status = None
        while time.time() < deadline:
            status = self.client.get("/identify/status/" + job["job_id"]).get_json()
            if status["status"] in ("done", "timeout"):
                break
            time.sleep(0.05)
        self.assertEqual(status["status"], "done")
        self.assertEqual(status["result"]["candidates"][0]["scientific_name"], "Thalassoma lucasanum")
        self.assertIsInstance(status["result"]["meta"], dict)
        self.assertNotEqual(status["result"]["certainty"], "confident")

    def test_identify_meta_is_never_null(self):
        missing = self.client.post("/identify", data={}).get_json()
        self.assertIsInstance(missing["meta"], dict)
        body = self.post_photo().get_json()
        self.assertIsInstance(body["meta"], dict)
        self.assertIsNotNone(body["meta"])

    def test_primary_window_is_not_shrunk_by_verify_hold(self):
        os.environ["XAI_PRIMARY_TIMEOUT"] = "95"
        try:
            self.fail_timeout = {"grok-4.7"}
            self.post_photo()
        finally:
            os.environ.pop("XAI_PRIMARY_TIMEOUT", None)
            self.fail_timeout = set()
        slow = next(call for call in self.captured["calls"] if call["model"] == "grok-4.7")
        self.assertGreaterEqual(slow["timeout"], 150)
        self.assertLessEqual(slow["timeout"], 180)

    def test_fromia_contradiction_is_still_no_match(self):
        self.inat_handler = self._taxon_photos
        self.captured["model_text"] = (
            "Photo quality: good\n"
            "Scientific name: Echinaster tenuispina\n"
            "Common name: Thin-spined sea star\n"
            "Confidence: medium\n"
            "Notes: A red lattice star.\n"
            "Candidate 1: Echinaster tenuispina | Thin-spined sea star | medium | red lattice of raised plates\n"
        )
        self.captured["verify_text"] = (
            "Decision: NO_MATCH\nCertainty: unsure\nTrait: the red lattice does not match these stars\n"
        )
        body = self.post_photo(lat="24.2", lng="-110.3").get_json()
        self.assertEqual(body["discovery"]["type"], "no_match")
        self.assertNotEqual(body["certainty"], "confident")
        self.assertEqual(body["candidates"], [])
        self.assertIn("Confidence: none", body["result_text"])
        self.assertNotIn("Confidence: medium", body["result_text"])
        self.assertIn("No matching species", body["result_text"])

    def test_debug_species_list_none_skips_nature_data(self):
        self.captured["model_text"] = (
            "Photo quality: good\n"
            "Scientific name: Thalassoma lucasanum\n"
            "Common name: Cortez rainbow wrasse\n"
            "Confidence: high\n"
            "Candidate 1: Thalassoma lucasanum | Cortez rainbow wrasse | high | yellow tail\n"
        )
        with self.assertLogs("finfindr_server", level="INFO") as logs:
            body = self.post_photo(
                debug="1",
                species_list="none",
                models="grok-4,grok-4.7",
                lat="24.2",
                lng="-110.3",
                region="sea-of-cortez",
            ).get_json()
        self.assertEqual(body["meta"]["species_list_mode"], "none")
        self.assertEqual(body["meta"]["nature_data"], "off")
        self.assertEqual(body["meta"]["models_run"], ["grok-4", "grok-4.7"])
        self.assertEqual(body["meta"]["species_sent"], 0)
        self.assertFalse(body["meta"]["verify_used"])
        self.assertEqual(body["meta"]["verify_skipped"], "disabled")
        self.assertIsNone(body["discovery"])
        self.assertIn("No regional species list was available", self.prompt())
        nature = ("inaturalist.org", "obis.org", "fishbase", "sealifebase", "marinespecies.org")
        for url in self.captured.get("gets") or []:
            self.assertFalse(any(host in url for host in nature), url)
        logged = "\n".join(logs.output)
        self.assertIn("nature_data=off", logged)
        self.assertNotIn("api.inaturalist.org", logged)
        self.assertNotIn("api.obis.org", logged)
        self.assertNotIn("fishbase.ropensci.org", logged)
        only = self.post_photo(debug="1", species_list="none", models="grok-4").get_json()
        self.assertEqual(only["meta"]["models_run"], ["grok-4"])
        self.assertEqual(only["meta"]["nature_data"], "off")


    # FF-16: a cold cache waits for the full iNaturalist list instead of the 2s fallback.
    def _slow_species_get(self, delay):
        def slow(url, params=None, **kwargs):
            if "species_counts" in url:
                time.sleep(delay)
            return self.fake_get(url, params, **kwargs)
        return slow

    def _ff16_join_list_threads(self):
        # A legacy 2s fallback leaves the crawl running; finish it before the http patch stops.
        for thread in threading.enumerate():
            if thread.name in ("finfindr-list", "finfindr-http"):
                thread.join(30)

    def _ff16_post(self, delay):
        self.get_patch.stop()
        self.get_patch = patch("finfindr_server.http.get", side_effect=self._slow_species_get(delay))
        self.get_patch.start()
        return self.post_photo(region="sea-of-cortez").get_json()

    def test_ff16_cold_start_waits_for_inat_list_not_curated(self):
        os.environ.pop("FINFINDR_LIST_WAIT", None)
        with self.assertLogs("finfindr_server", level="INFO") as logs:
            body = self._ff16_post(0.25)
        meta = body["meta"]
        self.assertEqual(meta["list_source"], "inaturalist")
        self.assertGreaterEqual(meta["list_fetch_ms"], 2500)
        self.assertGreater(meta["species_sent"], 0)
        self.assertNotEqual(meta["species_sent"], len(finfindr_server._curated_species_payload()["species"]))
        lines = [line for line in logs.output if "species_list job=" in line]
        self.assertEqual(len(lines), 1)
        self.assertIn("list_source=inaturalist", lines[0])
        self.assertIn("list_fetch_ms=", lines[0])
        self.assertIn("waited_out=False", lines[0])
        self.assertTrue(any("list_source=inaturalist" in line and "list_fetch_ms=" in line
                            for line in logs.output if line.startswith("INFO:finfindr_server:identify ")))

    def test_ff16_legacy_flag_keeps_the_two_second_fallback(self):
        os.environ["FINFINDR_LIST_WAIT"] = "2"
        try:
            body = self._ff16_post(0.25)
        finally:
            os.environ.pop("FINFINDR_LIST_WAIT", None)
            self._ff16_join_list_threads()
        self.assertEqual(body["meta"]["list_source"], "curated")
        self.assertLess(body["meta"]["list_fetch_ms"], 2600)

    def test_ff16_warm_cache_is_near_zero_and_same_list(self):
        os.environ.pop("FINFINDR_LIST_WAIT", None)
        cold = self._ff16_post(0.25)
        cold_prompt = self.prompt()
        calls_after_cold = len(self.species_calls)
        warm = self.post_photo(region="sea-of-cortez").get_json()
        self.assertEqual(warm["meta"]["list_source"], "inaturalist")
        self.assertLess(warm["meta"]["list_fetch_ms"], 200)
        self.assertEqual(len(self.species_calls), calls_after_cold)
        self.assertEqual(warm["meta"]["species_sent"], cold["meta"]["species_sent"])
        self.assertEqual(self.prompt(), cold_prompt)

    def test_ff16_waited_list_matches_a_fast_warm_path_list(self):
        os.environ.pop("FINFINDR_LIST_WAIT", None)
        self._ff16_post(0.0)
        fast_prompt = self.prompt()
        fast_calls = [dict(item) for item in self.species_calls]
        finfindr_server._SPECIES_CACHE.clear()
        finfindr_server._INAT_JSON_CACHE.clear()
        finfindr_server.clear_persisted_caches()
        self.species_calls.clear()
        slow = self._ff16_post(0.25)
        self.assertEqual(slow["meta"]["list_source"], "inaturalist")
        self.assertEqual(self.prompt(), fast_prompt)
        key = lambda item: json.dumps(item, sort_keys=True, default=str)
        self.assertEqual(sorted(map(key, self.species_calls)), sorted(map(key, fast_calls)))

    def test_ff16_wait_is_bounded_then_falls_back(self):
        os.environ["FINFINDR_LIST_WAIT"] = "3"
        try:
            started = time.monotonic()
            body = self._ff16_post(1.0)
            elapsed = time.monotonic() - started
        finally:
            os.environ.pop("FINFINDR_LIST_WAIT", None)
            self._ff16_join_list_threads()
        self.assertEqual(body["meta"]["list_source"], "curated")
        self.assertLess(elapsed, 3 + finfindr_server.LIST_WAIT_GRACE + 2)

    def test_ff16_async_job_record_has_list_fields(self):
        os.environ.pop("FINFINDR_LIST_WAIT", None)
        self.get_patch.stop()
        self.get_patch = patch("finfindr_server.http.get", side_effect=self._slow_species_get(0.25))
        self.get_patch.start()
        response = self.client.post(
            "/identify",
            data={"image": (io.BytesIO(jpeg_bytes()), "reef.jpg"), "region": "sea-of-cortez"},
            content_type="multipart/form-data",
        )
        job_id = response.get_json()["job_id"]
        deadline = time.monotonic() + 30
        job = {}
        while time.monotonic() < deadline:
            job = self.client.get("/identify/status/" + job_id).get_json()
            if job.get("status") not in ("pending", "running"):
                break
            time.sleep(0.1)
        meta = job["result"]["meta"]
        self.assertEqual(meta["list_source"], "inaturalist")
        self.assertGreaterEqual(meta["list_fetch_ms"], 2500)

    def test_ff16_client_meta_includes_list_wait_and_waited_out(self):
        # list_wait_s is the measured wait, not the configured FINFINDR_LIST_WAIT bound.
        os.environ.pop("FINFINDR_LIST_WAIT", None)
        with self.assertLogs("finfindr_server", level="INFO") as logs:
            cold = self._ff16_post(0.25)
        meta = cold["meta"]
        bound = finfindr_server.species_list_wait_seconds()
        self.assertEqual(bound, 60.0)
        self.assertEqual(meta["list_wait_s"], round(meta["list_fetch_ms"] / 1000.0, 3))
        self.assertGreaterEqual(meta["list_wait_s"], 2.5)
        self.assertLess(meta["list_wait_s"], bound)
        self.assertNotEqual(meta["list_wait_s"], bound)
        self.assertFalse(meta["waited_out"])
        self.assertEqual(meta["list_source"], "inaturalist")
        self.assertGreaterEqual(meta["list_fetch_ms"], 2500)
        lines = [line for line in logs.output if "species_list job=" in line]
        self.assertEqual(len(lines), 1)
        self.assertIn("list_wait_s=%s" % meta["list_wait_s"], lines[0])
        self.assertIn("waited_out=%s" % meta["waited_out"], lines[0])
        self.assertNotIn("list_wait_s=60.0", lines[0])

        warm = self.post_photo(region="sea-of-cortez").get_json()
        self.assertEqual(warm["meta"]["list_wait_s"], round(warm["meta"]["list_fetch_ms"] / 1000.0, 3))
        self.assertLess(warm["meta"]["list_wait_s"], 0.2)
        self.assertNotEqual(warm["meta"]["list_wait_s"], bound)
        self.assertFalse(warm["meta"]["waited_out"])
        self.assertLess(warm["meta"]["list_fetch_ms"], 200)

        # A crawl that is still running after the wait+grace bound is the
        # waited_out=True path already logged by _log_species_wait.
        payload = finfindr_server._curated_species_payload()
        with patch.object(
            finfindr_server,
            "load_species_alongside_model",
            return_value=(payload, 3100, 3.0, True),
        ):
            body = self.post_photo(region="sea-of-cortez").get_json()
        self.assertTrue(body["meta"]["waited_out"])
        self.assertEqual(body["meta"]["list_wait_s"], 3.0)
        self.assertEqual(body["meta"]["list_fetch_ms"], 3100)
        self.assertEqual(body["meta"]["list_source"], "curated")

    def test_list_wait_s_logs_measured_wait_not_the_configured_bound(self):
        os.environ.pop("FINFINDR_LIST_WAIT", None)
        region = finfindr_server.resolve_region("sea-of-cortez")

        def slow_load(*args, **kwargs):
            time.sleep(0.2)
            return finfindr_server._curated_species_payload()

        with self.assertLogs("finfindr_server", level="INFO") as logs:
            with patch.object(finfindr_server, "load_species_for_request", side_effect=slow_load):
                payload, fetch_ms, list_wait_s, waited_out = finfindr_server.load_species_alongside_model(
                    region, None, None, "short",
                )
        self.assertFalse(waited_out)
        self.assertGreaterEqual(fetch_ms, 200)
        self.assertEqual(list_wait_s, round(fetch_ms / 1000.0, 3))
        self.assertGreaterEqual(list_wait_s, 0.2)
        self.assertLess(list_wait_s, 2.0)
        self.assertNotEqual(list_wait_s, finfindr_server.species_list_wait_seconds())
        lines = [line for line in logs.output if "species_list job=" in line]
        self.assertEqual(len(lines), 1)
        self.assertIn("list_wait_s=%s" % list_wait_s, lines[0])
        self.assertNotIn("list_wait_s=60.0", lines[0])


    # --- LA-v1 look-alike pass: end-to-end through /identify (mocked model) ---

    LA_FIRST_PASS = (
        "Photo quality: good\nPhoto issue: none\n"
        "Common name: Cortez damselfish\nScientific name: Stegastes rectifraenum\n"
        "Group: damselfish\nAlternative group considered: none\nConfidence: medium\n"
        "Notes: A reef damselfish.\n"
        "Candidate 1: Stegastes rectifraenum | Cortez damselfish | medium | dark oval body with a yellow tail\n"
        "Candidate 2: Microspathodon dorsalis | Giant damselfish | low | larger body with scattered blue spots\n"
    )

    @staticmethod
    def la_reply(decision, chosen, features=None, extra=None):
        features = features if features is not None else [
            {"scientific_name": "Stegastes rectifraenum", "diagnostic_feature": "bright yellow caudal fin", "visible": True},
            {"scientific_name": "Microspathodon dorsalis", "diagnostic_feature": "scattered blue spots on the flanks", "visible": True},
        ]
        payload = {"decision": decision, "chosen": chosen, "candidates": features, "reason": "test"}
        payload.update(extra or {})
        return json.dumps(payload)

    def la_post(self, **fields):
        self.captured["model_text"] = self.LA_FIRST_PASS
        base = {"debug": "1", "verify": "0", "models": "grok-4.7", "discovery": "0"}
        base.update(fields)
        return self.post_photo(**base).get_json()

    @staticmethod
    def stable(body):
        out = {k: v for k, v in body.items() if k != "meta"}
        return json.dumps(out, sort_keys=True)

    def test_lookalike_flag_off_means_no_behaviour_change(self):
        os.environ.pop("FINFINDR_LOOKALIKE_PASS", None)
        self.assertFalse(finfindr_server.lookalike_enabled())
        baseline = self.la_post()
        self.assertNotIn("lookalike", baseline["meta"])
        self.assertFalse(self.captured.get("lookalike_calls"))
        # A switch reply is waiting, but with the flag off it must never be requested.
        self.captured["lookalike_text"] = self.la_reply("switch", "Microspathodon dorsalis")
        for value in (None, "0", "false", "off"):
            if value is None:
                os.environ.pop("FINFINDR_LOOKALIKE_PASS", None)
            else:
                os.environ["FINFINDR_LOOKALIKE_PASS"] = value
            try:
                body = self.la_post()
            finally:
                os.environ.pop("FINFINDR_LOOKALIKE_PASS", None)
            self.assertEqual(self.stable(body), self.stable(baseline))
            self.assertNotIn("lookalike", body["meta"])
            self.assertEqual(len(body["meta"]["attempts"]), len(baseline["meta"]["attempts"]))
        self.assertFalse(self.captured.get("lookalike_calls"))
        self.assertEqual(baseline["certainty"], "likely")
        self.assertEqual(baseline["candidates"][0]["scientific_name"], "Stegastes rectifraenum")

    def test_lookalike_switch_to_listed_candidate_at_likely(self):
        self.captured["lookalike_text"] = self.la_reply("switch", "Microspathodon dorsalis", extra={"certainty": "confident"})
        body = self.la_post(lookalike="1")
        self.assertEqual(len(self.captured["lookalike_calls"]), 1)
        self.assertEqual(self.captured["lookalike_calls"][0]["model"], "grok-4.7")
        self.assertIn("Stegastes rectifraenum", self.captured["lookalike_calls"][0]["prompt"])
        self.assertIn("Microspathodon dorsalis", self.captured["lookalike_calls"][0]["prompt"])
        self.assertEqual(body["candidates"][0]["scientific_name"], "Microspathodon dorsalis")
        self.assertEqual(body["certainty"], "likely")  # a "confident" in the reply is ignored
        self.assertIn("Microspathodon dorsalis", body["result_text"])
        la = body["meta"]["lookalike"]
        self.assertEqual(la["outcome"], "switch")
        self.assertEqual(la["certainty_final"], "likely")
        self.assertEqual(la["prompt_tokens"], 1500)
        self.assertEqual(la["completion_tokens"], 120)

    def test_lookalike_keep_leaves_answer_unchanged(self):
        off = self.la_post()
        self.captured["lookalike_text"] = self.la_reply("keep", "Stegastes rectifraenum")
        on = self.la_post(lookalike="1")
        self.assertEqual(on["meta"]["lookalike"]["outcome"], "keep")
        self.assertEqual(self.stable(on), self.stable(off))

    def test_lookalike_uncited_switch_becomes_uncertain(self):
        self.captured["lookalike_text"] = self.la_reply("switch", "Microspathodon dorsalis", features=[])
        body = self.la_post(lookalike="1")
        self.assertEqual(body["certainty"], "unsure")
        self.assertEqual(body["meta"]["lookalike"]["outcome"], "uncertain")
        self.assertEqual(body["meta"]["lookalike"]["reason"], "uncited_feature")

    def test_lookalike_unlisted_choice_becomes_uncertain(self):
        feats = [
            {"scientific_name": "Stegastes rectifraenum", "diagnostic_feature": "bright yellow caudal fin", "visible": True},
            {"scientific_name": "Microspathodon dorsalis", "diagnostic_feature": "scattered blue spots", "visible": True},
            {"scientific_name": "Holacanthus passer", "diagnostic_feature": "white vertical bar", "visible": True},
        ]
        self.captured["lookalike_text"] = self.la_reply("switch", "Holacanthus passer", features=feats)
        body = self.la_post(lookalike="1")
        self.assertEqual(body["certainty"], "unsure")
        self.assertNotIn("Holacanthus passer", [c["scientific_name"] for c in body["candidates"]])
        self.assertEqual(body["meta"]["lookalike"]["reason"], "unlisted_choice")

    def test_lookalike_timeout_and_error_fall_back_to_original(self):
        off = self.la_post()
        for mode in ("timeout", "error"):
            self.captured["lookalike_raises"] = mode
            self.captured["lookalike_text"] = self.la_reply("switch", "Microspathodon dorsalis")
            body = self.la_post(lookalike="1")
            self.assertEqual(self.stable(body), self.stable(off), mode)
            self.assertEqual(body["meta"]["lookalike"]["outcome"], "fallback")
            self.assertEqual(body["meta"]["lookalike"]["reason"], "timeout" if mode == "timeout" else "call_failed")
        timeouts = [call["timeout"] for call in self.captured["lookalike_calls"]]
        self.assertTrue(all(t is not None and t <= finfindr_server.model_call_timeout() for t in timeouts))

    def test_lookalike_malformed_json_falls_back_to_original(self):
        off = self.la_post()
        for reply in ("", "not json at all", '{"decision": "switch", "chosen": ', "[1, 2, 3]", '{"decision": "maybe"}'):
            self.captured["lookalike_text"] = reply
            body = self.la_post(lookalike="1")
            self.assertEqual(self.stable(body), self.stable(off), reply)
            self.assertEqual(body["meta"]["lookalike"]["outcome"], "fallback", reply)



    def test_lookalike_does_not_run_when_usable_or_marine_gate_stops(self):
        """Gates stay usable -> marine -> identify. LA runs only after both pass."""
        os.environ["FINFINDR_LOOKALIKE_PASS"] = "1"
        try:
            self.captured["lookalike_text"] = self.la_reply("switch", "Microspathodon dorsalis")
            self.captured["model_text"] = (
                "Photo quality: invalid\n"
                "Photo issue: too blurry to see the animal\n"
                "Scientific name: Stegastes rectifraenum\n"
                "Confidence: medium\n"
                "Candidate 1: Stegastes rectifraenum | Cortez damselfish | medium | yellow tail\n"
                "Candidate 2: Microspathodon dorsalis | Giant damselfish | low | blue spots\n"
            )
            unusable = self.post_photo(debug="1", verify="0", models="grok-4.7", lookalike="1", discovery="0").get_json()
            self.assertEqual(unusable["message"], "Can't identify from this photo.")
            self.assertFalse(self.captured.get("lookalike_calls"))
            self.assertNotIn("lookalike", unusable["meta"])

            self.captured["lookalike_calls"] = []
            self.inat_handler = self._felis_taxon
            self.captured["model_text"] = (
                "Photo quality: good\n"
                "Common name: Domestic Cat\n"
                "Scientific name: Felis catus\n"
                "Confidence: high\n"
                "Candidate 1: Felis catus | Domestic Cat | high | pointed ears and whiskers\n"
                "Candidate 2: Felis silvestris | Wildcat | low | striped flanks\n"
            )
            marine = self.post_photo(debug="1", verify="0", models="grok-4.7", lookalike="1", discovery="0").get_json()
            self._assert_not_marine(marine)
            self.assertFalse(self.captured.get("lookalike_calls"))
            self.assertNotIn("lookalike", marine["meta"])
        finally:
            os.environ.pop("FINFINDR_LOOKALIKE_PASS", None)


    def _freshwater_handler(self, scientific, common, ancestors, iconic="Actinopterygii"):
        def handler(url, params):
            if "marinespecies.org" in url:
                return FakeResponse([{
                    "scientificname": scientific,
                    "isMarine": False,
                    "isFreshwater": True,
                    "isBrackish": False,
                    "isTerrestrial": False,
                }])
            if "taxa" in url:
                name = params.get("q") or scientific
                if (name or "").lower() != scientific.lower() and "taxa/" not in url:
                    return FakeResponse({"results": []})
                return FakeResponse({"results": [{
                    "id": 1,
                    "name": scientific,
                    "rank": "species",
                    "iconic_taxon_name": iconic,
                    "preferred_common_name": common,
                    "ancestors": ancestors,
                }]})
            return None
        return handler

    def _land_handler(self, scientific, common, iconic, ancestors):
        def handler(url, params):
            if "taxa" in url:
                return FakeResponse({"results": [{
                    "id": 2,
                    "name": scientific,
                    "rank": "species",
                    "iconic_taxon_name": iconic,
                    "preferred_common_name": common,
                    "ancestors": ancestors,
                }]})
            if "marinespecies.org" in url:
                return FakeResponse([{
                    "scientificname": scientific,
                    "isMarine": False,
                    "isFreshwater": False,
                    "isBrackish": False,
                    "isTerrestrial": True,
                }])
            return None
        return handler

    def _post_named(self, scientific, common, trait, region="sea-of-cortez", **fields):
        self.captured["model_text"] = (
            "Photo quality: good\n"
            f"Common name: {common}\n"
            f"Scientific name: {scientific}\n"
            "Confidence: medium\n"
            f"Candidate 1: {scientific} | {common} | medium | {trait}\n"
        )
        data = {"region": region, "sync": "1", "debug": "1", "verify": "0", "discovery": "0"}
        data.update(fields)
        return self.post_photo(**data).get_json()

    def test_ff27_flag_off_is_identical_and_omits_the_new_field(self):
        os.environ.pop("FINFINDR_NOT_IN_REGION", None)
        self.assertFalse(finfindr_server.not_in_region_enabled())
        off = self.post_photo(region="sea-of-cortez", debug="1", verify="0", discovery="0", not_in_region="0").get_json()
        default = self.post_photo(region="sea-of-cortez", debug="1", verify="0", discovery="0").get_json()
        self.assertNotIn("not_in_region", off)
        self.assertNotIn("not_in_region", default)
        self.assertEqual(off["certainty"], default["certainty"])
        self.assertEqual(off["candidates"][0]["scientific_name"], "Thalassoma lucasanum")
        self.assertEqual(default["candidates"][0]["scientific_name"], "Thalassoma lucasanum")
        self.assertEqual(off["reason"] if "reason" in off else None, default.get("reason"))
        self.assertEqual(off["discovery"], default["discovery"])
        self.assertNotIn("not_in_region", off["meta"].get("evidence") or [])

    def test_ff27_freshwater_fish_are_named_and_flagged_in_the_selected_region(self):
        os.environ["FINFINDR_NOT_IN_REGION"] = "1"
        cases = [
            ("Carassius auratus", "Goldfish", "Cyprinidae", "orange body and no dorsal spines"),
            ("Micropterus salmoides", "Largemouth Bass", "Centrarchidae", "a wide mouth and a dark lateral stripe"),
            ("Lepomis gibbosus", "Pumpkinseed", "Centrarchidae", "orange spots and a red ear flap"),
        ]
        try:
            page = self.client.get("/").get_data(as_text=True)
            self.assertIn("function showNotInRegion", page)
            self.assertIn("not-in-region", page)
            for scientific, common, family, trait in cases:
                self.inat_handler = self._freshwater_handler(scientific, common, [
                    {"name": "Actinopterygii", "rank": "class"},
                    {"name": family, "rank": "family"},
                ])
                body = self._post_named(scientific, common, trait)
                self.assertNotEqual(body.get("reason"), "not_marine", scientific)
                self.assertNotEqual(body["certainty"], "unsure", scientific)
                self.assertEqual(body["certainty"], "likely", scientific)
                self.assertEqual(body["candidates"][0]["scientific_name"], scientific)
                self.assertEqual(body["region_label"], "Sea of Cortez / La Paz")
                flag = body["not_in_region"]
                self.assertEqual(flag["region"], "Sea of Cortez / La Paz")
                self.assertEqual(flag["text"], "not found in Sea of Cortez / La Paz")
                self.assertIn("not found in Sea of Cortez", flag["text"])
                self.assertEqual(
                    flag["display"],
                    f"{common} ({scientific}) — not found in Sea of Cortez / La Paz.",
                )
                self.assertIsNone(body["discovery"])
                self.assertIn("not_in_region", body["meta"]["evidence"])
                self.assertNotIn("FinFindr focuses on marine animal identification only.", body.get("message") or "")
        finally:
            os.environ.pop("FINFINDR_NOT_IN_REGION", None)

    def test_ff27_land_animals_stay_not_marine_when_the_flag_is_on(self):
        os.environ["FINFINDR_NOT_IN_REGION"] = "1"
        cases = [
            ("Felis catus", "Domestic Cat", "Mammalia", "Felidae", "pointed ears and whiskers"),
            ("Canis familiaris", "Domestic Dog", "Mammalia", "Canidae", "a muzzle and floppy ears"),
            ("Cardinalis cardinalis", "Northern Cardinal", "Aves", "Cardinalidae", "a red crest"),
            ("Sceloporus occidentalis", "Western Fence Lizard", "Reptilia", "Phrynosomatidae", "keeled scales"),
            ("Caracara cheriway", "Crested Caracara", "Aves", "Falconidae", "a bare orange face"),
        ]
        try:
            for scientific, common, iconic, family, trait in cases:
                self.inat_handler = self._land_handler(scientific, common, iconic, [
                    {"name": iconic, "rank": "class"},
                    {"name": family, "rank": "family"},
                ])
                body = self._post_named(scientific, common, trait)
                self._assert_not_marine(body)
                self.assertNotIn("not_in_region", body, scientific)
        finally:
            os.environ.pop("FINFINDR_NOT_IN_REGION", None)

    def test_ff27_in_region_marine_answer_is_unchanged_when_the_flag_is_on(self):
        os.environ.pop("FINFINDR_NOT_IN_REGION", None)
        off = self.post_photo(region="sea-of-cortez", debug="1", verify="0", discovery="0").get_json()
        os.environ["FINFINDR_NOT_IN_REGION"] = "1"
        try:
            on = self.post_photo(region="sea-of-cortez", debug="1", verify="0", discovery="0").get_json()
        finally:
            os.environ.pop("FINFINDR_NOT_IN_REGION", None)
        self.assertEqual(on["certainty"], off["certainty"])
        self.assertEqual(on["candidates"][0]["scientific_name"], off["candidates"][0]["scientific_name"])
        self.assertEqual(on["candidates"][0]["scientific_name"], "Thalassoma lucasanum")
        self.assertEqual(on["discovery"], off["discovery"])
        self.assertEqual(on.get("reason"), off.get("reason"))
        self.assertEqual(on["region_label"], off["region_label"])
        self.assertNotIn("not_in_region", on)
        self.assertNotIn("not_in_region", off)
        self.assertNotIn("not_in_region", on["meta"].get("evidence") or [])

    def test_ff27_marine_off_list_species_are_not_given_the_inland_flag(self):
        def handler(url, params):
            if "marinespecies.org" in url:
                return FakeResponse([{
                    "scientificname": "Octopus rubescens",
                    "isMarine": True,
                    "isFreshwater": False,
                    "isBrackish": False,
                    "isTerrestrial": False,
                }])
            if "obis.org" in url:
                return FakeResponse({"total": 0})
            if "fishbase" in url:
                return FakeResponse({"data": []})
            if "taxa" in url:
                return FakeResponse({"results": [{
                    "name": params.get("q") or "Octopus rubescens",
                    "rank": "species",
                    "iconic_taxon_name": "Mollusca",
                    "ancestors": [{"name": "Mollusca", "rank": "phylum"}],
                }]})
            if "observations" in url and str(params.get("per_page")) == "1":
                return FakeResponse({"total_results": 0, "results": []})
            return None

        self.inat_handler = handler
        self.captured["model_text"] = (
            "Photo quality: good\n"
            "Scientific name: Octopus rubescens\n"
            "Common name: Red octopus\n"
            "Confidence: medium\n"
            "Candidate 1: Octopus rubescens | Red octopus | medium | reddish body without eye spots\n"
        )
        os.environ["FINFINDR_NOT_IN_REGION"] = "1"
        try:
            body = self.post_photo(region="sea-of-cortez", lat="24.2", lng="-110.3", debug="1", verify="0", discovery="0").get_json()
        finally:
            os.environ.pop("FINFINDR_NOT_IN_REGION", None)
        self.assertNotIn("not_in_region", body)
        self.assertEqual(body["candidates"][0]["scientific_name"], "Octopus rubescens")
        self.assertNotEqual(body.get("reason"), "not_marine")

    def test_ff27_flag_uses_the_effective_region_name(self):
        os.environ["FINFINDR_NOT_IN_REGION"] = "1"
        self.inat_handler = self._freshwater_handler(
            "Carassius auratus", "Goldfish",
            [{"name": "Actinopterygii", "rank": "class"}, {"name": "Cyprinidae", "rank": "family"}],
        )
        try:
            caribbean = self._post_named(
                "Carassius auratus", "Goldfish", "orange body and no dorsal spines",
                region="caribbean",
            )
            self.assertEqual(caribbean["not_in_region"]["region"], "Caribbean")
            self.assertEqual(caribbean["not_in_region"]["text"], "not found in Caribbean")
            self.assertEqual(
                caribbean["not_in_region"]["display"],
                "Goldfish (Carassius auratus) — not found in Caribbean.",
            )
            inland = self._post_named(
                "Carassius auratus", "Goldfish", "orange body and no dorsal spines",
                region="other", lat="30.70", lng="-96.47",
            )
            self.assertNotIn("not_in_region", inland)
            self.assertEqual(inland["region_label"], "Other · 30.70, -96.47")
            decision = inland["meta"]["not_in_region_decision"]
            self.assertFalse(decision["fired"])
            self.assertEqual(decision["reason"], "no_real_region_name")
        finally:
            os.environ.pop("FINFINDR_NOT_IN_REGION", None)

    def test_ff27_picked_region_is_named_even_when_gps_is_elsewhere(self):
        """Fred: the note names the region the user selected, not the GPS list."""
        os.environ["FINFINDR_NOT_IN_REGION"] = "1"
        self.inat_handler = self._freshwater_handler(
            "Carassius auratus", "Goldfish",
            [{"name": "Actinopterygii", "rank": "class"}, {"name": "Cyprinidae", "rank": "family"}],
        )
        try:
            body = self._post_named(
                "Carassius auratus", "Goldfish", "orange body and no dorsal spines",
                region="sea-of-cortez", lat="18.0", lng="-66.0",
            )
            self.assertEqual(body["region_label"], "Sea of Cortez / La Paz")
            self.assertEqual(body["not_in_region"]["region"], "Sea of Cortez / La Paz")
            self.assertEqual(body["not_in_region"]["text"], "not found in Sea of Cortez / La Paz")
            self.assertEqual(
                body["not_in_region"]["display"],
                "Goldfish (Carassius auratus) — not found in Sea of Cortez / La Paz.",
            )
            self.assertEqual(body["not_in_region"]["region"], body["region_label"])
        finally:
            os.environ.pop("FINFINDR_NOT_IN_REGION", None)

    def test_ff27_unpicked_region_uses_the_gps_list_region(self):
        """When no named region was picked, the note names the GPS-built list's region."""
        os.environ["FINFINDR_NOT_IN_REGION"] = "1"
        self.inat_handler = self._freshwater_handler(
            "Carassius auratus", "Goldfish",
            [{"name": "Actinopterygii", "rank": "class"}, {"name": "Cyprinidae", "rank": "family"}],
        )
        try:
            body = self._post_named(
                "Carassius auratus", "Goldfish", "orange body and no dorsal spines",
                region="other", lat="24.2", lng="-110.3",
            )
            self.assertEqual(body["not_in_region"]["region"], "Sea of Cortez / La Paz")
            self.assertEqual(body["not_in_region"]["text"], "not found in Sea of Cortez / La Paz")
            self.assertEqual(
                body["not_in_region"]["display"],
                "Goldfish (Carassius auratus) — not found in Sea of Cortez / La Paz.",
            )
        finally:
            os.environ.pop("FINFINDR_NOT_IN_REGION", None)

    def test_ff27_does_not_flag_unsure_or_unusable_photos(self):
        os.environ["FINFINDR_NOT_IN_REGION"] = "1"
        self.inat_handler = self._freshwater_handler(
            "Carassius auratus", "Goldfish",
            [{"name": "Actinopterygii", "rank": "class"}, {"name": "Cyprinidae", "rank": "family"}],
        )
        try:
            self.captured["model_text"] = (
                "Photo quality: invalid\n"
                "Photo issue: too blurry to see the animal\n"
                "Common name: Goldfish\n"
                "Scientific name: Carassius auratus\n"
                "Confidence: medium\n"
                "Candidate 1: Carassius auratus | Goldfish | medium | orange body\n"
            )
            unusable = self.post_photo(region="sea-of-cortez", debug="1", verify="0", discovery="0").get_json()
            self.assertEqual(unusable["message"], "Can't identify from this photo.")
            self.assertNotIn("not_in_region", unusable)
        finally:
            os.environ.pop("FINFINDR_NOT_IN_REGION", None)

    def test_ff27_poor_quality_usable_goldfish_is_likely_with_the_note(self):
        """A poor but usable goldfish still gets the likely not-in-region note."""
        os.environ["FINFINDR_NOT_IN_REGION"] = "1"
        self.inat_handler = self._freshwater_handler(
            "Carassius auratus", "Goldfish",
            [{"name": "Actinopterygii", "rank": "class"}, {"name": "Cyprinidae", "rank": "family"}],
        )
        try:
            self.assertTrue(finfindr_server.photo_is_usable("poor", "murky water and soft focus"))
            self.captured["model_text"] = (
                "Photo quality: poor\n"
                "Photo issue: murky water and soft focus\n"
                "Common name: Goldfish\n"
                "Scientific name: Carassius auratus\n"
                "Confidence: medium\n"
                "Candidate 1: Carassius auratus | Goldfish | medium | orange body and no dorsal spines\n"
            )
            body = self.post_photo(
                region="sea-of-cortez", debug="1", verify="0", discovery="0",
            ).get_json()
            self.assertNotEqual(body.get("reason"), "not_marine")
            self.assertEqual(body["photo_quality"], "poor")
            self.assertEqual(body["certainty"], "likely")
            self.assertEqual(body["candidates"][0]["scientific_name"], "Carassius auratus")
            self.assertEqual(body["region_label"], "Sea of Cortez / La Paz")
            self.assertEqual(
                body["not_in_region"]["display"],
                "Goldfish (Carassius auratus) — not found in Sea of Cortez / La Paz.",
            )
            self.assertEqual(body["not_in_region"]["region"], body["region_label"])
            self.assertEqual(body["not_in_region"]["text"], "not found in Sea of Cortez / La Paz")
            decision = body["meta"]["not_in_region_decision"]
            self.assertTrue(decision["fired"])
            self.assertEqual(decision["reason"], "attached")
            self.assertIsNone(decision["records_in_radius"])
        finally:
            os.environ.pop("FINFINDR_NOT_IN_REGION", None)

    def _heron_handler(self):
        return self._combo_wildlife_handler(
            "Ardea herodias",
            "Great Blue Heron",
            iconic="Aves",
            ancestors=[{"name": "Aves", "rank": "class"}, {"name": "Ardeidae", "rank": "family"}],
            worms={"isMarine": False, "isFreshwater": True, "isBrackish": False, "isTerrestrial": True},
        )

    def _combo_wildlife_handler(self, scientific, common, iconic, ancestors, worms, record_total=None, photos=False):
        def handler(url, params):
            if "marinespecies.org" in url:
                row = {"scientificname": scientific}
                row.update(worms)
                return FakeResponse([row])
            if "taxa" in url:
                name = params.get("q") or scientific
                if (name or "").lower() != scientific.lower() and "taxa/" not in url:
                    return FakeResponse({"results": []})
                taxon = {
                    "id": 11,
                    "name": scientific,
                    "rank": "species",
                    "iconic_taxon_name": iconic,
                    "preferred_common_name": common,
                    "ancestors": ancestors,
                }
                if photos:
                    taxon["default_photo"] = {
                        "id": 11,
                        "url": "https://inaturalist-open-data.s3.amazonaws.com/photos/11/square.jpg",
                        "attribution": "Riley Example",
                        "license_code": "cc-by",
                    }
                return FakeResponse({"results": [taxon]})
            if "observations" in url and str(params.get("per_page")) == "1" and record_total is not None:
                return FakeResponse({"total_results": record_total, "results": []})
            return None
        return handler

    def test_ff27_v2_in_region_heron_gets_no_note(self):
        os.environ["FINFINDR_NOT_IN_REGION"] = "1"
        self.inat_handler = self._heron_handler()
        try:
            body = self._post_named(
                "Ardea herodias", "Great Blue Heron", "a long dagger bill and an S-curved neck",
            )
            self.assertNotEqual(body.get("reason"), "not_marine")
            self.assertEqual(body["candidates"][0]["scientific_name"], "Ardea herodias")
            self.assertNotIn("not_in_region", body)
            decision = body["meta"]["not_in_region_decision"]
            self.assertFalse(decision["fired"])
            self.assertIn(decision["reason"], ("worms_terrestrial", "not_freshwater_fish"))
            self.assertTrue(decision["worms_terrestrial"])
        finally:
            os.environ.pop("FINFINDR_NOT_IN_REGION", None)

    def test_ff27_v2_caracara_gets_no_note(self):
        os.environ["FINFINDR_NOT_IN_REGION"] = "1"
        self.inat_handler = self._land_handler(
            "Caracara cheriway", "Crested Caracara", "Aves",
            [{"name": "Aves", "rank": "class"}, {"name": "Falconidae", "rank": "family"}],
        )
        try:
            body = self._post_named(
                "Caracara cheriway", "Crested Caracara", "a bare orange face",
            )
            self.assertNotIn("not_in_region", body)
            if body.get("reason") == "not_marine":
                self._assert_not_marine(body)
            else:
                decision = body["meta"]["not_in_region_decision"]
                self.assertFalse(decision["fired"])
                self.assertIn(decision["reason"], ("worms_terrestrial", "not_freshwater_fish"))
        finally:
            os.environ.pop("FINFINDR_NOT_IN_REGION", None)

    def test_ff27_v2_goldfish_and_sunfish_in_cortez_with_no_gps_get_the_note(self):
        os.environ["FINFINDR_NOT_IN_REGION"] = "1"
        cases = [
            ("Carassius auratus", "Goldfish", "Cyprinidae", "orange body and no dorsal spines"),
            ("Lepomis gibbosus", "Pumpkinseed", "Centrarchidae", "orange spots and a red ear flap"),
        ]
        try:
            for scientific, common, family, trait in cases:
                self.inat_handler = self._freshwater_handler(scientific, common, [
                    {"name": "Actinopterygii", "rank": "class"},
                    {"name": family, "rank": "family"},
                ])
                body = self._post_named(scientific, common, trait, region="sea-of-cortez")
                self.assertEqual(
                    body["not_in_region"]["display"],
                    f"{common} ({scientific}) — not found in Sea of Cortez / La Paz.",
                    scientific,
                )
                decision = body["meta"]["not_in_region_decision"]
                self.assertTrue(decision["fired"], scientific)
                self.assertEqual(decision["reason"], "attached", scientific)
                self.assertIsNone(decision["records_in_radius"], scientific)
                self.assertIsNone(decision["gps_in_region"], scientific)
        finally:
            os.environ.pop("FINFINDR_NOT_IN_REGION", None)

    def test_ff27_v2_picked_cortez_with_inland_gps_and_records_still_gets_the_note(self):
        os.environ["FINFINDR_NOT_IN_REGION"] = "1"
        self.inat_handler = self._combo_wildlife_handler(
            "Carassius auratus", "Goldfish", "Actinopterygii",
            [{"name": "Actinopterygii", "rank": "class"}, {"name": "Cyprinidae", "rank": "family"}],
            {"isMarine": False, "isFreshwater": True, "isBrackish": False, "isTerrestrial": False},
            record_total=89,
        )
        try:
            body = self._post_named(
                "Carassius auratus", "Goldfish", "orange body and no dorsal spines",
                region="sea-of-cortez", lat="30.70", lng="-96.47", discovery="1",
            )
            self.assertEqual(
                body["not_in_region"]["display"],
                "Goldfish (Carassius auratus) — not found in Sea of Cortez / La Paz.",
            )
            decision = body["meta"]["not_in_region_decision"]
            self.assertTrue(decision["fired"])
            self.assertEqual(decision["reason"], "attached")
            self.assertEqual(decision["records_in_radius"], 89)
            self.assertFalse(decision["gps_in_region"])
        finally:
            os.environ.pop("FINFINDR_NOT_IN_REGION", None)

    def test_ff27_v2_gps_inside_cortez_with_records_gets_no_note(self):
        os.environ["FINFINDR_NOT_IN_REGION"] = "1"
        self.inat_handler = self._combo_wildlife_handler(
            "Carassius auratus", "Goldfish", "Actinopterygii",
            [{"name": "Actinopterygii", "rank": "class"}, {"name": "Cyprinidae", "rank": "family"}],
            {"isMarine": False, "isFreshwater": True, "isBrackish": False, "isTerrestrial": False},
            record_total=34,
        )
        try:
            body = self._post_named(
                "Carassius auratus", "Goldfish", "orange body and no dorsal spines",
                region="sea-of-cortez", lat="24.2", lng="-110.3", discovery="1",
            )
            self.assertNotIn("not_in_region", body)
            self.assertEqual(body["candidates"][0]["scientific_name"], "Carassius auratus")
            decision = body["meta"]["not_in_region_decision"]
            self.assertFalse(decision["fired"])
            self.assertEqual(decision["reason"], "records_in_region")
            self.assertEqual(decision["records_in_radius"], 34)
            self.assertTrue(decision["gps_in_region"])
        finally:
            os.environ.pop("FINFINDR_NOT_IN_REGION", None)

    def test_ff27_v2_does_not_cap_a_confident_answer(self):
        os.environ["FINFINDR_NOT_IN_REGION"] = "1"
        self.inat_handler = self._combo_wildlife_handler(
            "Carassius auratus", "Goldfish", "Actinopterygii",
            [{"name": "Actinopterygii", "rank": "class"}, {"name": "Cyprinidae", "rank": "family"}],
            {"isMarine": False, "isFreshwater": True, "isBrackish": False, "isTerrestrial": False},
            photos=True,
        )
        self.captured["model_text"] = (
            "Photo quality: good\n"
            "Common name: Goldfish\n"
            "Scientific name: Carassius auratus\n"
            "Confidence: high\n"
            "Candidate 1: Carassius auratus | Goldfish | high | orange body and no dorsal spines\n"
        )
        self.captured["verify_text"] = (
            "Decision: Carassius auratus\nCertainty: confident\nTrait: orange body and no dorsal spines\n"
        )
        try:
            body = self.post_photo(
                region="sea-of-cortez", debug="1", verify="1", discovery="0", model="grok-4.7",
            ).get_json()
            self.assertEqual(body["certainty"], "confident")
            self.assertEqual(body["candidates"][0]["confidence"], "high")
            self.assertEqual(
                body["not_in_region"]["display"],
                "Goldfish (Carassius auratus) — not found in Sea of Cortez / La Paz.",
            )
            self.assertTrue(body["meta"]["verify_succeeded"])
            decision = body["meta"]["not_in_region_decision"]
            self.assertTrue(decision["fired"])
            self.assertEqual(decision["reason"], "attached")
        finally:
            os.environ.pop("FINFINDR_NOT_IN_REGION", None)

    def test_ff27_v2_unconfirmed_high_stays_likely(self):
        os.environ["FINFINDR_NOT_IN_REGION"] = "1"
        self.inat_handler = self._freshwater_handler(
            "Carassius auratus", "Goldfish",
            [{"name": "Actinopterygii", "rank": "class"}, {"name": "Cyprinidae", "rank": "family"}],
        )
        try:
            self.captured["model_text"] = (
                "Photo quality: good\n"
                "Common name: Goldfish\n"
                "Scientific name: Carassius auratus\n"
                "Confidence: high\n"
                "Candidate 1: Carassius auratus | Goldfish | high | orange body and no dorsal spines\n"
            )
            body = self.post_photo(
                region="sea-of-cortez", debug="1", verify="0", discovery="0",
            ).get_json()
            self.assertEqual(body["certainty"], "likely")
            self.assertEqual(body["candidates"][0]["confidence"], "medium")
            self.assertIn("not_in_region", body)
        finally:
            os.environ.pop("FINFINDR_NOT_IN_REGION", None)


    def _ff30_guess_text(self, scientific, common, trait, confidence="medium"):
        return (
            f"Common name: {common}\n"
            f"Scientific name: {scientific}\n"
            f"Confidence: {confidence}\n"
            f"Trait: {trait}\n"
        )

    def _ff30_compare_text(self, decision, chosen, features, reason="visible features decide it"):
        return json.dumps({
            "decision": decision,
            "chosen": chosen,
            "cited_features": features,
            "reason": reason,
        })

    def _ff30_snook_model_text(self):
        return (
            "Photo quality: good\n"
            "Common name: White Snook\n"
            "Scientific name: Centropomus viridis\n"
            "Confidence: medium\n"
            "Candidate 1: Centropomus viridis | White Snook | medium | a dark lateral line to the tail\n"
            "Candidate 2: Cynoscion parvipinnis | Shortfin corvina | low | a spotted side\n"
        )

    def _micropterus_and_snook_handler(self):
        micro = self._combo_wildlife_handler(
            "Micropterus salmoides",
            "Largemouth Bass",
            "Actinopterygii",
            [
                {"name": "Actinopterygii", "rank": "class"},
                {"name": "Centrarchidae", "rank": "family"},
            ],
            {"isMarine": False, "isFreshwater": True, "isBrackish": False, "isTerrestrial": False},
        )

        def handler(url, params):
            params = params or {}
            if "species_counts" in url:
                payload = species_payload(params)
                if int(params.get("page") or 1) == 1:
                    payload["results"].insert(0, {
                        "count": 9000,
                        "taxon": {
                            "name": "Centropomus viridis",
                            "rank": "species",
                            "preferred_common_name": "White Snook",
                        },
                    })
                return FakeResponse(payload)
            if "marinespecies.org" in url:
                if "micropterus" in (url or "").lower():
                    return micro(url, params)
                return FakeResponse([{
                    "scientificname": "Centropomus viridis",
                    "isMarine": True,
                    "isFreshwater": False,
                    "isBrackish": True,
                    "isTerrestrial": False,
                }])
            if "taxa" in url:
                name = (params.get("q") or "").lower()
                if "micropterus" in name or "salmoides" in name:
                    return micro(url, params)
                return FakeResponse({"results": [{
                    "id": 12,
                    "name": "Centropomus viridis",
                    "rank": "species",
                    "iconic_taxon_name": "Actinopterygii",
                    "preferred_common_name": "White Snook",
                    "ancestors": [{"name": "Actinopterygii", "rank": "class"}],
                }]})
            return micro(url, params)

        return handler

    def _ff30_post(self, **fields):
        data = {
            "region": "sea-of-cortez",
            "sync": "1",
            "debug": "1",
            "verify": "0",
            "discovery": "0",
            "models": "grok-4.7",
            "first_guess": "1",
            "not_in_region": "1",
        }
        data.update(fields)
        return self.post_photo(**data).get_json()

    def test_ff30_flag_off_means_no_change(self):
        os.environ.pop("FINFINDR_FIRST_GUESS", None)
        self.assertFalse(finfindr_server.first_guess_enabled())
        self.captured["first_guess_text"] = self._ff30_guess_text(
            "Micropterus salmoides", "Largemouth Bass", "a large mouth and a deep bass body",
        )
        self.captured["first_guess_compare_text"] = self._ff30_compare_text(
            "hold", "Micropterus salmoides", "a large mouth and a deep bass body",
        )
        baseline = self.post_photo(
            region="sea-of-cortez", debug="1", verify="0", discovery="0", models="grok-4.7", first_guess="0",
        ).get_json()
        self.assertNotIn("first_guess", baseline["meta"])
        self.assertFalse(self.captured.get("first_guess_calls"))
        self.assertFalse(self.captured.get("first_guess_compare_calls"))
        for value in (None, "0", "false", "off"):
            if value is None:
                os.environ.pop("FINFINDR_FIRST_GUESS", None)
            else:
                os.environ["FINFINDR_FIRST_GUESS"] = value
            try:
                body = self.post_photo(
                    region="sea-of-cortez", debug="1", verify="0", discovery="0", models="grok-4.7",
                ).get_json()
            finally:
                os.environ.pop("FINFINDR_FIRST_GUESS", None)
            self.assertEqual(self.stable(body), self.stable(baseline), value)
            self.assertNotIn("first_guess", body["meta"])
        self.assertFalse(self.captured.get("first_guess_calls"))
        self.assertFalse(self.captured.get("first_guess_compare_calls"))
        self.assertEqual(baseline["candidates"][0]["scientific_name"], "Thalassoma lucasanum")

    def test_ff30_on_list_or_agree_skips_second_call(self):
        os.environ["FINFINDR_FIRST_GUESS"] = "1"
        try:
            self.captured["first_guess_text"] = self._ff30_guess_text(
                "Thalassoma lucasanum", "Cortez Rainbow Wrasse", "a yellow tail and a steep wrasse head",
            )
            self.captured["first_guess_compare_text"] = self._ff30_compare_text(
                "hold", "Thalassoma lucasanum", "should not be requested",
            )
            on_list = self._ff30_post(not_in_region="0")
            self.assertEqual(on_list["candidates"][0]["scientific_name"], "Thalassoma lucasanum")
            self.assertEqual(on_list["certainty"], "likely")
            self.assertEqual(on_list["meta"]["first_guess"]["outcome"], "agree")
            self.assertTrue(on_list["meta"]["first_guess"]["on_list"])
            self.assertTrue(on_list["meta"]["first_guess"]["agrees"])
            self.assertTrue(self.captured.get("first_guess_calls"))
            self.assertFalse(self.captured.get("first_guess_compare_calls"))
            prompt = self.captured["first_guess_calls"][0]["prompt"]
            self.assertIn(finfindr_server.FIRST_GUESS_MARKER, prompt)
            self.assertNotIn("marine biologist", prompt.lower())
            self.assertNotIn("Sea of Cortez", prompt)
            self.assertNotIn("Prefer a species known from this region", prompt)

            self.captured["first_guess_calls"] = []
            self.captured["first_guess_compare_calls"] = []
            self.inat_handler = self._freshwater_handler(
                "Carassius auratus",
                "Goldfish",
                [{"name": "Actinopterygii", "rank": "class"}, {"name": "Cyprinidae", "rank": "family"}],
            )
            self.captured["model_text"] = (
                "Photo quality: good\n"
                "Common name: Goldfish\n"
                "Scientific name: Carassius auratus\n"
                "Confidence: medium\n"
                "Candidate 1: Carassius auratus | Goldfish | medium | a deep orange body and no dorsal spines\n"
            )
            self.captured["first_guess_text"] = self._ff30_guess_text(
                "Carassius auratus", "Goldfish", "a deep orange body and no dorsal spines",
            )
            agree = self._ff30_post()
            self.assertEqual(agree["candidates"][0]["scientific_name"], "Carassius auratus")
            self.assertEqual(agree["meta"]["first_guess"]["outcome"], "agree")
            self.assertTrue(agree["meta"]["first_guess"]["agrees"])
            self.assertFalse(agree["meta"]["first_guess"]["on_list"])
            self.assertFalse(self.captured.get("first_guess_compare_calls"))
            self.assertIn("not_in_region", agree)
        finally:
            os.environ.pop("FINFINDR_FIRST_GUESS", None)

    def test_ff30_off_list_freshwater_guess_holds_and_gets_the_note(self):
        os.environ["FINFINDR_FIRST_GUESS"] = "1"
        os.environ["FINFINDR_NOT_IN_REGION"] = "1"
        self.inat_handler = self._micropterus_and_snook_handler()
        try:
            self.captured["model_text"] = self._ff30_snook_model_text()
            self.captured["first_guess_text"] = self._ff30_guess_text(
                "Micropterus salmoides",
                "Largemouth Bass",
                "a large mouth and a deep freshwater bass body",
            )
            self.captured["first_guess_compare_text"] = self._ff30_compare_text(
                "hold",
                "Micropterus salmoides",
                "a large mouth and a deep freshwater bass body, not a snook lateral line",
            )
            body = self._ff30_post()
            self.assertEqual(body["candidates"][0]["scientific_name"], "Micropterus salmoides")
            self.assertEqual(body["certainty"], "likely")
            self.assertIn("not_in_region", body)
            self.assertIn("Sea of Cortez", body["not_in_region"]["display"])
            self.assertIn("Largemouth Bass", body["not_in_region"]["display"])
            fg = body["meta"]["first_guess"]
            self.assertEqual(fg["outcome"], "hold")
            self.assertFalse(fg["on_list"])
            self.assertFalse(fg["agrees"])
            self.assertTrue(self.captured.get("first_guess_compare_calls"))
            compare_prompt = self.captured["first_guess_compare_calls"][0]["prompt"]
            self.assertIn(finfindr_server.FIRST_GUESS_COMPARE_MARKER, compare_prompt)
            self.assertNotIn("Sea of Cortez", compare_prompt)
            self.assertNotIn("marine biologist", compare_prompt.lower())
        finally:
            os.environ.pop("FINFINDR_FIRST_GUESS", None)
            os.environ.pop("FINFINDR_NOT_IN_REGION", None)

    def test_ff30_lookalike_switch_needs_cited_features(self):
        os.environ["FINFINDR_FIRST_GUESS"] = "1"
        self.inat_handler = self._micropterus_and_snook_handler()
        try:
            self.captured["model_text"] = self._ff30_snook_model_text()
            self.captured["first_guess_text"] = self._ff30_guess_text(
                "Micropterus salmoides",
                "Largemouth Bass",
                "a large mouth and a deep freshwater bass body",
            )
            self.captured["first_guess_compare_text"] = self._ff30_compare_text(
                "lookalike",
                "Centropomus viridis",
                "a dark lateral line that runs to the edge of the tail",
            )
            cited = self._ff30_post()
            self.assertEqual(cited["candidates"][0]["scientific_name"], "Centropomus viridis")
            self.assertEqual(cited["certainty"], "likely")
            self.assertNotIn("not_in_region", cited)
            self.assertEqual(cited["meta"]["first_guess"]["outcome"], "lookalike")
            self.assertIn("lateral line", cited["meta"]["first_guess"]["cited_features"])

            self.captured["first_guess_compare_text"] = self._ff30_compare_text(
                "lookalike", "Centropomus viridis", "similar",
            )
            uncited = self._ff30_post()
            self.assertEqual(uncited["candidates"][0]["scientific_name"], "Centropomus viridis")
            self.assertEqual(uncited["certainty"], "likely")
            self.assertEqual(uncited["meta"]["first_guess"]["outcome"], "unsure")
            self.assertEqual(uncited["meta"]["first_guess"]["reason"], "uncited_feature")
        finally:
            os.environ.pop("FINFINDR_FIRST_GUESS", None)

    def test_ff30_cant_tell_keeps_the_normal_result(self):
        os.environ["FINFINDR_FIRST_GUESS"] = "1"
        self.inat_handler = self._micropterus_and_snook_handler()
        try:
            self.captured["model_text"] = self._ff30_snook_model_text()
            self.captured["first_guess_text"] = self._ff30_guess_text(
                "Micropterus salmoides",
                "Largemouth Bass",
                "a large mouth and a deep freshwater bass body",
            )
            self.captured["first_guess_compare_text"] = self._ff30_compare_text(
                "unsure", "", "", "the photo cannot separate them",
            )
            body = self._ff30_post()
            self.assertEqual(body["candidates"][0]["scientific_name"], "Centropomus viridis")
            self.assertEqual(body["certainty"], "likely")
            self.assertEqual(body["meta"]["first_guess"]["outcome"], "unsure")
            self.assertEqual(body["meta"]["first_guess"]["reason"], "model_unsure")
            self.assertEqual(body["meta"]["first_guess"]["certainty_after"], "likely")
        finally:
            os.environ.pop("FINFINDR_FIRST_GUESS", None)

    def test_ff30_off_list_disagreement_unsure_keeps_the_normal_result(self):
        """415594578-style: off-list first guess + model_unsure keeps likely and the normal name."""
        os.environ["FINFINDR_FIRST_GUESS"] = "1"

        def handler(url, params):
            params = params or {}
            if "species_counts" in url:
                payload = species_payload(params)
                if int(params.get("page") or 1) == 1:
                    payload["results"].insert(0, {
                        "count": 9000,
                        "taxon": {
                            "name": "Abudefduf troschelii",
                            "rank": "species",
                            "preferred_common_name": "Panamic Sergeant Major",
                        },
                    })
                return FakeResponse(payload)
            return None

        self.inat_handler = handler
        try:
            self.captured["model_text"] = (
                "Photo quality: good\n"
                "Common name: Panamic Sergeant Major\n"
                "Scientific name: Abudefduf troschelii\n"
                "Confidence: medium\n"
                "Candidate 1: Abudefduf troschelii | Panamic Sergeant Major | medium | five black bars on a yellow back\n"
                "Candidate 2: Abudefduf declivifrons | Mexican Nightsergeant | low | a darker head\n"
            )
            self.captured["first_guess_text"] = self._ff30_guess_text(
                "Abudefduf saxatilis",
                "Sergeant major",
                "a silvery oval body with five bold black vertical bars",
            )
            self.captured["first_guess_compare_text"] = self._ff30_compare_text(
                "unsure", "", "", "the photo cannot separate the two sergeant majors",
            )
            body = self._ff30_post()
            self.assertEqual(body["candidates"][0]["scientific_name"], "Abudefduf troschelii")
            self.assertEqual(body["certainty"], "likely")
            self.assertNotIn("not_in_region", body)
            fg = body["meta"]["first_guess"]
            self.assertEqual(fg["scientific_name"], "Abudefduf saxatilis")
            self.assertFalse(fg["on_list"])
            self.assertFalse(fg["agrees"])
            self.assertEqual(fg["outcome"], "unsure")
            self.assertEqual(fg["reason"], "model_unsure")
            self.assertEqual(fg["certainty_before"], "likely")
            self.assertEqual(fg["certainty_after"], "likely")
            self.assertTrue(self.captured.get("first_guess_compare_calls"))
        finally:
            os.environ.pop("FINFINDR_FIRST_GUESS", None)

    def test_ff30_never_raises_certainty(self):
        os.environ["FINFINDR_FIRST_GUESS"] = "1"
        os.environ["FINFINDR_NOT_IN_REGION"] = "1"
        self.inat_handler = self._micropterus_and_snook_handler()
        try:
            self.captured["model_text"] = self._ff30_snook_model_text()
            self.captured["first_guess_text"] = self._ff30_guess_text(
                "Micropterus salmoides",
                "Largemouth Bass",
                "a large mouth and a deep freshwater bass body",
                confidence="high",
            )
            self.captured["first_guess_compare_text"] = self._ff30_compare_text(
                "hold",
                "Micropterus salmoides",
                "a large mouth and a deep freshwater bass body",
            )
            body = self._ff30_post()
            self.assertEqual(body["candidates"][0]["scientific_name"], "Micropterus salmoides")
            self.assertEqual(body["certainty"], "likely")
            self.assertNotEqual(body["certainty"], "confident")
            self.assertEqual(body["meta"]["first_guess"]["outcome"], "hold")
            self.assertEqual(body["meta"]["first_guess"]["certainty_after"], "likely")
        finally:
            os.environ.pop("FINFINDR_FIRST_GUESS", None)
            os.environ.pop("FINFINDR_NOT_IN_REGION", None)

    def test_ff30_time_budget_skip_keeps_the_normal_result(self):
        os.environ["FINFINDR_FIRST_GUESS"] = "1"
        self.inat_handler = self._micropterus_and_snook_handler()
        self.captured["model_text"] = self._ff30_snook_model_text()
        self.captured["first_guess_text"] = self._ff30_guess_text(
            "Micropterus salmoides",
            "Largemouth Bass",
            "a large mouth and a deep freshwater bass body",
        )
        self.captured["first_guess_compare_text"] = self._ff30_compare_text(
            "hold", "Micropterus salmoides", "a large mouth and a deep freshwater bass body",
        )
        try:
            with patch("finfindr_server._time_left", return_value=3.0):
                body = self._ff30_post()
            self.assertEqual(body["candidates"][0]["scientific_name"], "Centropomus viridis")
            self.assertEqual(body["certainty"], "likely")
            self.assertEqual(body["meta"]["first_guess"]["outcome"], "skip")
            self.assertEqual(body["meta"]["first_guess"]["reason"], "no_time")
            self.assertFalse(self.captured.get("first_guess_calls"))
            self.assertFalse(self.captured.get("first_guess_compare_calls"))
            self.assertNotIn("not_in_region", body)
        finally:
            os.environ.pop("FINFINDR_FIRST_GUESS", None)




class NotInRegionRuleTests(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("FINFINDR_NOT_IN_REGION", None)

    def test_flag_is_off_by_default(self):
        os.environ.pop("FINFINDR_NOT_IN_REGION", None)
        self.assertFalse(finfindr_server.not_in_region_enabled())
        for value in ("0", "false", "off", ""):
            os.environ["FINFINDR_NOT_IN_REGION"] = value
            self.assertFalse(finfindr_server.not_in_region_enabled(), value)
        os.environ["FINFINDR_NOT_IN_REGION"] = "1"
        self.assertTrue(finfindr_server.not_in_region_enabled())
        self.assertTrue(finfindr_server.not_in_region_enabled(debug=True, raw="1"))
        os.environ["FINFINDR_NOT_IN_REGION"] = "1"
        self.assertFalse(finfindr_server.not_in_region_enabled(debug=True, raw="0"))

    def test_display_text_names_the_species_and_region(self):
        self.assertEqual(
            finfindr_server.not_in_region_display(
                "Pumpkinseed", "Lepomis gibbosus", "Sea of Cortez / La Paz"
            ),
            "Pumpkinseed (Lepomis gibbosus) — not found in Sea of Cortez / La Paz.",
        )
        self.assertIn(
            "not found in Sea of Cortez",
            finfindr_server.not_in_region_text("Sea of Cortez / La Paz"),
        )

    def test_effective_region_follows_the_picked_region(self):
        cortez = finfindr_server._REGIONS_BY_ID["sea-of-cortez"]
        other = finfindr_server._REGIONS_BY_ID["other"]
        my_loc = finfindr_server._REGIONS_BY_ID["my-location"]
        self.assertEqual(
            finfindr_server.effective_region_label(cortez, None, None),
            "Sea of Cortez / La Paz",
        )
        self.assertEqual(
            finfindr_server.effective_region_label(other, 30.70, -96.47),
            "Other · 30.70, -96.47",
        )
        self.assertEqual(
            finfindr_server.effective_region_label(cortez, 24.2, -110.3),
            "Sea of Cortez / La Paz",
        )
        # Named picker wins over GPS in another ocean region.
        self.assertEqual(
            finfindr_server.effective_region_label(cortez, 18.0, -66.0),
            "Sea of Cortez / La Paz",
        )
        self.assertEqual(
            finfindr_server.display_region_label(cortez, 18.0, -66.0),
            "Sea of Cortez / La Paz",
        )
        # No named region picked: use the GPS-built list's region.
        self.assertEqual(
            finfindr_server.effective_region_label(other, 24.2, -110.3),
            "Sea of Cortez / La Paz",
        )
        self.assertEqual(
            finfindr_server.effective_region_label(my_loc, 24.2, -110.3),
            "Sea of Cortez / La Paz",
        )
        self.assertEqual(
            finfindr_server.effective_region_label(my_loc, None, None, location_fallback=True),
            "Sea of Cortez / La Paz (location unavailable)",
        )

    def test_freshwater_check_requires_positive_inland_evidence(self):
        self.assertTrue(finfindr_server.taxon_is_freshwater_or_inland.__doc__)
        self.assertFalse(finfindr_server.taxon_is_freshwater_or_inland(""))
        with patch("finfindr_server._worms_habitat", return_value={"marine": True, "freshwater": False, "brackish": False, "terrestrial": False}):
            with patch("finfindr_server.fetch_inat_taxon") as fetch:
                self.assertFalse(finfindr_server.taxon_is_freshwater_or_inland("Thalassoma lucasanum"))
                fetch.assert_not_called()
        goldfish = {
            "name": "Carassius auratus",
            "iconic_taxon_name": "Actinopterygii",
            "ancestors": [{"name": "Actinopterygii", "rank": "class"}, {"name": "Cyprinidae", "rank": "family"}],
        }
        with patch("finfindr_server._worms_habitat", return_value={"marine": False, "freshwater": True, "brackish": False, "terrestrial": False}):
            with patch("finfindr_server.fetch_inat_taxon", return_value=goldfish):
                self.assertTrue(finfindr_server.taxon_is_freshwater_or_inland("Carassius auratus"))
        with patch("finfindr_server._worms_habitat", return_value={"marine": False, "freshwater": True, "brackish": False, "terrestrial": True}):
            with patch("finfindr_server.fetch_inat_taxon") as fetch:
                self.assertFalse(finfindr_server.taxon_is_freshwater_or_inland("Caracara cheriway"))
                fetch.assert_not_called()
        heron = {
            "name": "Ardea herodias",
            "iconic_taxon_name": "Aves",
            "ancestors": [{"name": "Aves", "rank": "class"}, {"name": "Ardeidae", "rank": "family"}],
        }
        with patch("finfindr_server._worms_habitat", return_value={"marine": False, "freshwater": True, "brackish": False, "terrestrial": True}):
            with patch("finfindr_server.fetch_inat_taxon", return_value=heron):
                self.assertFalse(finfindr_server.taxon_is_freshwater_or_inland("Ardea herodias"))
        with patch("finfindr_server._worms_habitat", return_value=None):
            with patch("finfindr_server.fetch_inat_taxon", return_value=None):
                self.assertFalse(finfindr_server.taxon_is_freshwater_or_inland("Unknownus species"))
            with patch("finfindr_server.fetch_inat_taxon", return_value=goldfish):
                self.assertTrue(finfindr_server.taxon_is_freshwater_or_inland("Carassius auratus"))

    def test_real_note_region_name_rejects_coordinate_labels(self):
        cortez = finfindr_server._REGIONS_BY_ID["sea-of-cortez"]
        other = finfindr_server._REGIONS_BY_ID["other"]
        self.assertEqual(
            finfindr_server.real_note_region_name(cortez, None, None),
            "Sea of Cortez / La Paz",
        )
        self.assertIsNone(finfindr_server.real_note_region_name(other, 30.70, -96.47))
        self.assertEqual(
            finfindr_server.real_note_region_name(other, 24.2, -110.3),
            "Sea of Cortez / La Paz",
        )
        self.assertFalse(finfindr_server.gps_in_note_region(cortez, 30.70, -96.47))
        self.assertTrue(finfindr_server.gps_in_note_region(cortez, 24.2, -110.3))
        self.assertIsNone(finfindr_server.gps_in_note_region(cortez, None, None))
        self.assertFalse(finfindr_server.records_block_not_in_region(cortez, 24.2, -110.3, None))
        self.assertFalse(finfindr_server.records_block_not_in_region(cortez, 24.2, -110.3, 0))
        self.assertFalse(finfindr_server.records_block_not_in_region(cortez, 30.70, -96.47, 89))
        self.assertTrue(finfindr_server.records_block_not_in_region(cortez, 24.2, -110.3, 34))



class LookalikeRuleTests(unittest.TestCase):
    """LA-v1 rules as pure functions. No model calls."""

    LISTED = [
        {"scientific_name": "Octopus hubbsorum", "confidence": "medium"},
        {"scientific_name": "Octopus bimaculatus", "confidence": "low"},
        {"scientific_name": "Octopus digueti", "confidence": "low"},
    ]

    def feats(self, visible=True, feature="paired ocelli below the eyes"):
        return [{"scientific_name": c["scientific_name"], "diagnostic_feature": feature, "visible": visible} for c in self.LISTED]

    def decide(self, payload, certainty="likely"):
        return finfindr_server.decide_lookalike(payload, self.LISTED, certainty)

    def tearDown(self):
        os.environ.pop("FINFINDR_LOOKALIKE_PASS", None)

    def test_flag_is_off_by_default_and_named(self):
        os.environ.pop("FINFINDR_LOOKALIKE_PASS", None)
        self.assertFalse(finfindr_server.lookalike_enabled())
        self.assertFalse(finfindr_server.lookalike_enabled(False, "1"))  # form field needs debug
        self.assertTrue(finfindr_server.lookalike_enabled(True, "1"))
        os.environ["FINFINDR_LOOKALIKE_PASS"] = "1"
        self.assertTrue(finfindr_server.lookalike_enabled())
        self.assertFalse(finfindr_server.lookalike_enabled(True, "0"))

    def test_never_raises_certainty(self):
        order = {"unsure": 0, "likely": 1, "confident": 2}
        payloads = [
            {"decision": "keep", "chosen": "Octopus hubbsorum", "candidates": self.feats(), "certainty": "confident"},
            {"decision": "switch", "chosen": "Octopus bimaculatus", "candidates": self.feats(), "certainty": "confident"},
            {"decision": "uncertain"},
            None,
            {"decision": "switch", "chosen": "Octopus bimaculatus", "candidates": []},
        ]
        for before in ("unsure", "likely", "confident"):
            for payload in payloads:
                out = self.decide(payload, before)
                self.assertLessEqual(order[out["certainty"]], order[before], (before, payload))
        switched = self.decide(payloads[1], "confident")
        self.assertEqual(switched["outcome"], "switch")
        self.assertEqual(switched["certainty"], "likely")
        self.assertEqual(finfindr_server.cap_certainty("confident", "likely"), "likely")
        self.assertEqual(finfindr_server.cap_certainty("likely", "confident"), "likely")
        self.assertEqual(finfindr_server.cap_certainty("unsure", "likely"), "unsure")

    def test_pass_does_not_run_on_unsure_answers(self):
        with patch.object(finfindr_server, "_call_vision_model") as call:
            meta, cands, certainty = finfindr_server.run_lookalike_pass(
                b"jpeg", self.LISTED, "unsure", "", "key", time.monotonic()
            )
        call.assert_not_called()
        self.assertEqual((meta["outcome"], meta["reason"], certainty), ("skipped", "not_named", "unsure"))
        with patch.object(finfindr_server, "_call_vision_model") as call:
            meta, _, certainty = finfindr_server.run_lookalike_pass(
                b"jpeg", self.LISTED[:1], "likely", "", "key", time.monotonic()
            )
        call.assert_not_called()
        self.assertEqual(meta["reason"], "single_candidate")

    def test_uncited_switch_becomes_uncertain(self):
        cases = [
            {"decision": "switch", "chosen": "Octopus bimaculatus"},
            {"decision": "switch", "chosen": "Octopus bimaculatus", "candidates": self.feats(feature="")},
            {"decision": "switch", "chosen": "Octopus bimaculatus", "candidates": self.feats(feature="similar")},
            {"decision": "switch", "chosen": "Octopus bimaculatus", "candidates": self.feats()[:2]},
            {"decision": "keep", "chosen": "Octopus hubbsorum", "candidates": self.feats()[1:]},
        ]
        for payload in cases:
            out = self.decide(payload)
            self.assertEqual((out["outcome"], out["certainty"], out["reason"]), ("uncertain", "unsure", "uncited_feature"), payload)
        hidden = self.decide({"decision": "switch", "chosen": "Octopus bimaculatus", "candidates": self.feats(visible=False)})
        self.assertEqual((hidden["outcome"], hidden["reason"]), ("uncertain", "feature_not_visible"))

    def test_only_listed_candidates_are_allowed(self):
        for chosen in ("Octopus vulgaris", "Holacanthus passer", "octopus", "UNSURE"):
            out = self.decide({"decision": "switch", "chosen": chosen, "candidates": self.feats()})
            self.assertEqual((out["outcome"], out["certainty"]), ("uncertain", "unsure"), chosen)
            self.assertEqual(out["scientific_name"], "Octopus hubbsorum")
        mixed = self.decide({"decision": "keep", "chosen": "Octopus bimaculatus", "candidates": self.feats()})
        self.assertEqual((mixed["outcome"], mixed["reason"]), ("uncertain", "inconsistent_keep"))
        ok = self.decide({"decision": "switch", "chosen": "octopus BIMACULATUS", "candidates": self.feats()})
        self.assertEqual((ok["outcome"], ok["scientific_name"]), ("switch", "Octopus bimaculatus"))
        same = self.decide({"decision": "switch", "chosen": "Octopus hubbsorum", "candidates": self.feats()})
        self.assertEqual(same["outcome"], "keep")

    def test_malformed_json(self):
        parse = finfindr_server.parse_lookalike_reply
        for text in ("", "   ", "keep", "{", '{"decision": "keep",}', "[]", '"keep"', "null"):
            self.assertIsNone(parse(text), text)
            out = self.decide(parse(text))
            self.assertEqual((out["outcome"], out["certainty"], out["scientific_name"]), ("fallback", "likely", "Octopus hubbsorum"))
        fenced = parse('Here you go:\n```json\n{"decision": "uncertain"}\n```')
        self.assertEqual(fenced, {"decision": "uncertain"})
        self.assertEqual(self.decide({"decision": 7})["outcome"], "fallback")

    def test_timeout_and_error_fallback(self):
        cases = [
            {"ok": False, "retryable": True, "error": "timeout", "ms": 5},
            {"ok": False, "retryable": True, "error": "http_503", "ms": 5},
            {"ok": False, "retryable": True, "error": "invalid_json", "ms": 5},
        ]
        for reply in cases:
            with patch.object(finfindr_server, "_call_vision_model", return_value=reply) as call:
                meta, cands, certainty = finfindr_server.run_lookalike_pass(
                    b"jpeg", self.LISTED, "likely", "", "key", time.monotonic()
                )
            self.assertEqual(meta["outcome"], "fallback")
            self.assertEqual(meta["reason"], "timeout" if reply["error"] == "timeout" else "call_failed")
            self.assertEqual(certainty, "likely")
            self.assertIs(cands, self.LISTED)
            timeout = call.call_args[0][3]
            self.assertLessEqual(timeout, finfindr_server.model_call_timeout())
        with patch.object(finfindr_server, "_call_vision_model", side_effect=RuntimeError("boom")):
            meta, cands, certainty = finfindr_server.run_lookalike_pass(
                b"jpeg", self.LISTED, "likely", "", "key", time.monotonic()
            )
        self.assertEqual((meta["outcome"], meta["reason"], certainty), ("fallback", "call_failed", "likely"))
        with patch.object(finfindr_server, "_time_left", return_value=3.0), \
                patch.object(finfindr_server, "_call_vision_model") as call:
            meta, _, certainty = finfindr_server.run_lookalike_pass(
                b"jpeg", self.LISTED, "likely", "", "key", time.monotonic()
            )
        call.assert_not_called()
        self.assertEqual((meta["reason"], certainty), ("no_time", "likely"))


class EvalScoreTests(unittest.TestCase):
    def test_default_targets_are_the_stricter_gates(self):
        targets, path = eval_region.load_targets()
        self.assertTrue(path.endswith("eval_targets.json"))
        self.assertEqual(targets["confident_precision"]["min"], 0.99)
        self.assertEqual(targets["false_accepts_per_100"]["max"], 1)
        self.assertEqual(targets["correct_reject_rate"]["min"], 0.98)
        self.assertEqual(targets["false_reject_rate"]["max"], 0.01)
        self.assertEqual(targets["decoy_false_flag_rate"]["max"], 0.02)
        self.assertEqual(targets["discovery_precision"]["min"], 0.99)
        self.assertEqual(targets["discovery_recall"]["min"], 0.95)
        self.assertEqual(targets["range_rule"]["radius_km"], 100)
        self.assertEqual(targets["range_rule"]["max_records"], 0)

    def test_manifest_reads_optional_range_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "qa.csv"
            csv_path.write_text(
                "photo_path,true_taxon,true_common,expected_outcome,lat,lng,region,source_url,license,key_confidence,notes,range_sources,nearest_known_record_km,nearest_known_record_source,expected_traits\n"
                "a.jpg,Zebrasoma xanthurum,Yellowtail tang,range_extension,24.2,-110.3,sea-of-cortez,https://example.test/a,cc-by,verified,local,inat;obis,180,obis,yellow tail;round body\n",
                encoding="utf-8",
            )
            items, _folder = eval_region.load_manifest(csv_path)
            json_path = Path(directory) / "qa.json"
            json_path.write_text(json.dumps([{
                "photo_path": "a.jpg",
                "true_taxon": "Zebrasoma xanthurum",
                "true_common": "Yellowtail tang",
                "expected_outcome": "range_extension",
                "lat": 24.2,
                "lng": -110.3,
                "region": "sea-of-cortez",
                "key_confidence": "verified",
                "range_sources": "inat;obis",
                "nearest_known_record_km": 180,
                "nearest_known_record_source": "obis",
                "expected_traits": "yellow tail;round body",
            }]), encoding="utf-8")
            again, _folder = eval_region.load_manifest(json_path)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["range_sources"], ["inat", "obis"])
        self.assertEqual(items[0]["nearest_known_record_km"], 180)
        self.assertEqual(items[0]["nearest_known_record_source"], "obis")
        self.assertEqual(items[0]["expected_traits"], ["yellow tail", "round body"])
        self.assertEqual(again[0]["expected_outcome"], "range_extension")
        self.assertEqual(again[0]["key_confidence"], "verified")

    def _item(self, **overrides):
        item = {
            "photo_path": "a.jpg",
            "true_taxon": "Zebrasoma xanthurum",
            "true_common": "Yellowtail tang",
            "expected_outcome": "match",
            "lat": 24.2,
            "lng": -110.3,
            "region": "sea-of-cortez",
            "source_url": "",
            "license": "",
            "key_confidence": "verified",
            "notes": "",
            "range_sources": ["inat"],
            "nearest_known_record_km": None,
            "nearest_known_record_source": "",
            "expected_traits": [],
        }
        item.update(overrides)
        return item

    def _body(self, **overrides):
        body = {
            "certainty": "confident",
            "photo_quality": "good",
            "candidates": [{
                "scientific_name": "Zebrasoma xanthurum",
                "common_name": "Yellowtail tang",
                "trait": "yellow tail",
            }],
            "discovery": None,
            "meta": {"range_km": 100, "range_max": 0, "range_rule": finfindr_server.range_rule_text(), "evidence": []},
        }
        body.update(overrides)
        return body

    def test_verified_scores_pass_and_uncertain_stays_separate(self):
        range_body = self._body(discovery={
            "type": "possible_range_extension",
            "scientific_name": "Zebrasoma xanthurum",
            "common_name": "Yellowtail tang",
            "traits": ["yellow tail", "round body"],
            "nearest_records": [{"distance_km": 175, "url": "https://www.inaturalist.org/observations/1"}],
        })
        rows = [
            eval_region.answer_from_response(self._body(), self._item(), 1, 1000),
            eval_region.answer_from_response(
                {"photo_quality": "invalid", "message": "Can't identify from this photo.", "candidates": [], "discovery": None},
                self._item(photo_path="bad.jpg", expected_outcome="reject", true_taxon="", true_common="", range_sources=[]),
                1,
                400,
            ),
            eval_region.answer_from_response(
                range_body,
                self._item(
                    photo_path="range.jpg",
                    expected_outcome="range_extension",
                    nearest_known_record_km=180,
                    nearest_known_record_source="obis",
                    expected_traits=["yellow tail", "round body"],
                    range_sources=["inat", "obis"],
                ),
                1,
                1500,
            ),
            eval_region.answer_from_response(
                self._body(),
                self._item(photo_path="decoy.jpg", notes="look-alike of a western Pacific tang"),
                1,
                900,
            ),
            eval_region.answer_from_response(
                self._body(candidates=[{
                    "scientific_name": "Acanthurus nigricans",
                    "common_name": "Whitecheek surgeonfish",
                    "trait": "pale cheek",
                }]),
                self._item(photo_path="unsure-photo.jpg", key_confidence="uncertain"),
                1,
                800,
            ),
        ]
        targets, path = eval_region.load_targets()
        report = eval_region.build_report(rows, "qa", targets, path, 1)
        self.assertEqual(report["targets_result"]["overall"], "pass")
        verified = report["verified"]
        self.assertEqual(verified["confident_precision"], 1)
        self.assertEqual(verified["false_accepts_per_100"], 0)
        self.assertEqual(verified["correct_reject_rate"], 1)
        self.assertEqual(verified["false_reject_rate"], 0)
        self.assertEqual(verified["decoy_false_flag_rate"], 0)
        self.assertEqual(verified["discovery_precision"], 1)
        self.assertEqual(verified["discovery_recall"], 1)
        self.assertEqual(verified["headline_accuracy"], 1)
        self.assertEqual(verified["nearest_record_error"]["mean_abs_km"], 5)
        self.assertEqual(verified["trait_overlap"]["mean_overlap"], 1)
        self.assertIn("inat", verified["by_range_source"])
        self.assertIn("obis", verified["by_range_source"])
        self.assertEqual(verified["by_range_source"]["obis"]["nearest_record_error"]["mean_abs_km"], 5)
        self.assertEqual(report["uncertain"]["confident_precision"], 0)
        self.assertEqual(report["uncertain"]["false_accepts_per_100"], 100)
        rendered = eval_region.markdown_report(report)
        self.assertIn("| confident_precision | >= 99% | 100.0% | pass |", rendered)
        self.assertIn("Range rule:", rendered)

    def test_wrong_discovery_flag_is_a_false_accept_and_fails_precision(self):
        flagged = self._body(discovery={
            "type": "possible_range_extension",
            "scientific_name": "Zebrasoma xanthurum",
            "common_name": "Yellowtail tang",
            "traits": ["yellow tail"],
            "nearest_records": [{"distance_km": 40, "url": "https://www.inaturalist.org/observations/9"}],
        })
        row = eval_region.answer_from_response(
            flagged,
            self._item(notes="decoy look-alike", nearest_known_record_km=20),
            1,
            1000,
        )
        self.assertTrue(row["false_accept"])
        self.assertIn("not a range extension", row["false_accept_reason"])
        self.assertEqual(row["headline_credit"], 0)
        self.assertFalse(row["flag_fully_correct"])
        targets, path = eval_region.load_targets()
        report = eval_region.build_report([row], "qa", targets, path, 1)
        self.assertEqual(report["targets_result"]["overall"], "fail")
        self.assertEqual(report["verified"]["discovery_precision"], 0)
        self.assertEqual(report["targets_result"]["checks"]["decoy_false_flag_rate"]["result"], "fail")
        self.assertEqual(len(report["verified"]["false_accepts"]), 1)

    def test_incomplete_range_flag_is_not_fully_correct(self):
        flagged = self._body(discovery={
            "type": "possible_range_extension",
            "scientific_name": "Zebrasoma xanthurum",
            "common_name": "Yellowtail tang",
            "traits": [],
            "nearest_records": [],
        })
        row = eval_region.answer_from_response(
            flagged,
            self._item(photo_path="range.jpg", expected_outcome="range_extension", expected_traits=["yellow tail"]),
            1,
            1000,
        )
        self.assertFalse(row["flag_fully_correct"])
        self.assertEqual(row["headline_credit"], 0)
        targets, path = eval_region.load_targets()
        report = eval_region.build_report([row], "qa", targets, path, 1)
        self.assertEqual(report["verified"]["discovery_recall"], 1)
        self.assertEqual(report["verified"]["discovery_precision"], 0)

    def test_discovery_flip_between_runs_is_reported(self):
        quiet = eval_region.answer_from_response(self._body(), self._item(photo_path="same.jpg", expected_outcome="range_extension"), 1, 10)
        flagged = eval_region.answer_from_response(
            self._body(discovery={
                "type": "possible_range_extension",
                "scientific_name": "Zebrasoma xanthurum",
                "common_name": "Yellowtail tang",
                "traits": ["yellow tail"],
                "nearest_records": [{"distance_km": 10, "url": "https://www.inaturalist.org/observations/2"}],
            }),
            self._item(photo_path="same.jpg", expected_outcome="range_extension", expected_traits=["yellow tail"]),
            2,
            12,
        )
        targets, path = eval_region.load_targets()
        report = eval_region.build_report([quiet, flagged], "qa", targets, path, 2)
        self.assertEqual(report["verified"]["discovery_flip_rate"], 1)
        self.assertEqual(len(report["verified"]["discovery_flips"]), 1)
        self.assertEqual(report["verified"]["discovery_precision"], 1)
        self.assertLess(report["verified"]["consistency"], 1)

    def test_synonyms_errors_and_server_range_rule(self):
        self.assertTrue(eval_region.names_equivalent("Thalassoma lucasanum", "Thalassoma lucasanum", []))
        self.assertTrue(eval_region.names_equivalent(
            "Thalassoma grammaticum", "Thalassoma lucasanum", ["Thalassoma grammaticum"]
        ))
        self.assertEqual(eval_region.observation_id_from_source(
            "https://www.inaturalist.org/observations/4242"
        ), "4242")
        item = self._item(synonyms=["Abudefduf declivifrons"], source_url="https://www.inaturalist.org/observations/99")
        row = eval_region.answer_from_response(
            self._body(candidates=[{
                "scientific_name": "Abudefduf declivifrons",
                "common_name": "Night sergeant",
                "trait": "dark bars",
            }], meta={"range_km": 100, "range_max": 0, "range_rule": "no records"}),
            item,
            1,
            10,
            status=500,
        )
        row["klass"] = "error"
        row["errored"] = True
        self.assertTrue(row["species_correct"])
        targets, path = eval_region.load_targets()
        good = eval_region.answer_from_response(self._body(), self._item(), 1, 10)
        good["server_range_km"] = 100
        good["server_range_max"] = 0
        report = eval_region.build_report([good, row], "qa", targets, path, 1)
        self.assertGreaterEqual(report["verified"]["errored"], 1)
        self.assertEqual(report["targets_result"]["checks"]["range_rule"]["result"], "pass")
        self.assertGreater(report["verified"]["false_reject_rate"], 0)
        outcome = eval_region.retry_outcome(502, 200, False)
        self.assertTrue(outcome["first_attempt_error"])
        self.assertTrue(outcome["retried_ok"])
        self.assertEqual(outcome["first_attempt_status"], 502)
        quiet = eval_region.retry_outcome(200, 200, False)
        self.assertFalse(quiet["first_attempt_error"])
        self.assertFalse(quiet["retried_ok"])
        failed = eval_region.retry_outcome(503, 503, True)
        self.assertTrue(failed["first_attempt_error"])
        self.assertFalse(failed["retried_ok"])
        row["first_attempt_error"] = True
        row["retried_ok"] = True
        report = eval_region.build_report([good, row], "qa", targets, path, 1)
        self.assertEqual(report["first_attempt_errors"], 1)
        self.assertEqual(report["retried_ok"], 1)
        self.assertIn("First-attempt errors: 1", eval_region.markdown_report(report))




class FirstGuessRuleTests(unittest.TestCase):
    """FF-30 rules as pure functions. No model calls."""

    def tearDown(self):
        os.environ.pop("FINFINDR_FIRST_GUESS", None)
        os.environ.pop("FINFINDR_FIRST_GUESS_TIMEOUT", None)

    def test_flag_is_off_by_default(self):
        os.environ.pop("FINFINDR_FIRST_GUESS", None)
        self.assertFalse(finfindr_server.first_guess_enabled())
        self.assertFalse(finfindr_server.first_guess_enabled(False, "1"))
        self.assertTrue(finfindr_server.first_guess_enabled(True, "1"))
        self.assertFalse(finfindr_server.first_guess_enabled(True, "0"))
        os.environ["FINFINDR_FIRST_GUESS"] = "1"
        self.assertTrue(finfindr_server.first_guess_enabled())

    def test_prompt_has_no_region_or_marine_priors(self):
        prompt = finfindr_server.first_guess_prompt()
        self.assertIn(finfindr_server.FIRST_GUESS_MARKER, prompt)
        lowered = prompt.lower()
        self.assertNotIn("marine biologist", lowered)
        self.assertNotIn("marine animal", lowered)
        self.assertNotIn("sea of cortez", lowered)
        self.assertNotIn("prefer a species known from this region", lowered)
        self.assertNotIn("regional species list", lowered)
        self.assertIn("visible features only", lowered)
        compare = finfindr_server.first_guess_compare_prompt(
            {"scientific_name": "Micropterus salmoides", "common_name": "Largemouth Bass", "trait": "a large mouth"},
            {"scientific_name": "Centropomus viridis", "common_name": "White Snook", "trait": "a dark lateral line"},
        )
        self.assertIn(finfindr_server.FIRST_GUESS_COMPARE_MARKER, compare)
        self.assertNotIn("Sea of Cortez", compare)
        self.assertNotIn("marine biologist", compare.lower())
        self.assertIn("Micropterus salmoides", compare)
        self.assertIn("Centropomus viridis", compare)

    def test_first_guess_certainty_is_never_confident(self):
        self.assertEqual(finfindr_server.first_guess_certainty("high"), "likely")
        self.assertEqual(finfindr_server.first_guess_certainty("medium"), "likely")
        self.assertEqual(finfindr_server.first_guess_certainty("low"), "unsure")

    def test_first_guess_timeout_default_is_60(self):
        os.environ.pop("FINFINDR_FIRST_GUESS_TIMEOUT", None)
        self.assertEqual(finfindr_server.first_guess_timeout(), 60)

    def test_decide_hold_lookalike_unsure_and_fallback(self):
        hold = finfindr_server.decide_first_guess_compare(
            {
                "decision": "hold",
                "chosen": "Micropterus salmoides",
                "cited_features": "a large mouth and a deep bass body",
            },
            "Micropterus salmoides",
            "Centropomus viridis",
            "a large mouth",
            "likely",
        )
        self.assertEqual(hold["outcome"], "hold")
        self.assertEqual(hold["scientific_name"], "Micropterus salmoides")
        self.assertEqual(hold["certainty"], "likely")

        # A first-guess "confident" cap is still likely.
        capped = finfindr_server.decide_first_guess_compare(
            {
                "decision": "hold",
                "chosen": "Micropterus salmoides",
                "cited_features": "a large mouth and a deep bass body",
            },
            "Micropterus salmoides",
            "Centropomus viridis",
            "",
            "confident",
        )
        self.assertEqual(capped["certainty"], "likely")

        lookalike = finfindr_server.decide_first_guess_compare(
            {
                "decision": "lookalike",
                "chosen": "Centropomus viridis",
                "cited_features": "a dark lateral line to the tail",
            },
            "Micropterus salmoides",
            "Centropomus viridis",
            "a large mouth",
            "likely",
        )
        self.assertEqual(lookalike["outcome"], "lookalike")
        self.assertEqual(lookalike["scientific_name"], "Centropomus viridis")

        uncited = finfindr_server.decide_first_guess_compare(
            {"decision": "lookalike", "chosen": "Centropomus viridis", "cited_features": "similar"},
            "Micropterus salmoides",
            "Centropomus viridis",
            "a large mouth",
            "likely",
        )
        self.assertEqual(uncited["outcome"], "unsure")
        self.assertEqual(uncited["reason"], "uncited_feature")
        self.assertEqual(uncited["scientific_name"], "Centropomus viridis")
        self.assertEqual(uncited["certainty"], "likely")

        unsure = finfindr_server.decide_first_guess_compare(
            {"decision": "unsure", "chosen": "", "cited_features": ""},
            "Micropterus salmoides",
            "Centropomus viridis",
            "a large mouth",
            "likely",
        )
        self.assertEqual(unsure["outcome"], "unsure")
        self.assertEqual(unsure["scientific_name"], "Centropomus viridis")
        self.assertEqual(unsure["certainty"], "likely")

        already = finfindr_server.decide_first_guess_compare(
            {"decision": "unsure", "chosen": "", "cited_features": ""},
            "Micropterus salmoides",
            "Centropomus viridis",
            "a large mouth",
            "likely",
            normal_certainty="unsure",
        )
        self.assertEqual(already["outcome"], "unsure")
        self.assertEqual(already["certainty"], "unsure")

        fallback = finfindr_server.decide_first_guess_compare(
            None, "Micropterus salmoides", "Centropomus viridis", "a large mouth", "likely",
        )
        self.assertEqual(fallback["outcome"], "fallback")
        self.assertEqual(fallback["scientific_name"], "Centropomus viridis")

    def test_parse_first_guess_reply_and_name_agreement(self):
        parsed = finfindr_server.parse_first_guess_reply(
            "Common name: Largemouth Bass\n"
            "Scientific name: Micropterus salmoides\n"
            "Confidence: high\n"
            "Trait: a large mouth\n"
        )
        self.assertEqual(parsed["scientific_name"], "Micropterus salmoides")
        self.assertEqual(parsed["confidence"], "high")
        self.assertTrue(finfindr_server.scientific_names_agree("Micropterus salmoides", "micropterus salmoides"))
        self.assertFalse(finfindr_server.scientific_names_agree("Micropterus salmoides", "Centropomus viridis"))
        self.assertIsNone(finfindr_server.parse_first_guess_reply("Confidence: high\nTrait: a fin\n"))


if __name__ == "__main__":
    unittest.main()


FF28_SPECIES = [
    ("Gymnothorax castaneus", "Panamic green moray"),
    ("Muraena lentiginosa", "Jewel moray"),
    ("Larus heermanni", "Heermann's Gull"),
    ("Leucophaeus atricilla", "Laughing Gull"),
    ("Phataria unifascialis", "Blue Sea Star"),
    ("Pharia pyramidata", "Pyramid Sea Star"),
    ("Johnrandallia nigrirostris", "Black-nosed Butterflyfish"),
    ("Zanclus cornutus", "Moorish Idol"),
]


def ff28_list():
    return [{"scientific": name, "common": common} for name, common in FF28_SPECIES]


def ff28_cand(name, common, confidence, rank=None, off_list=False, trait="plain"):
    item = {
        "scientific_name": name, "common_name": common, "confidence": confidence,
        "trait": trait, "off_list": off_list,
        "proposed_by": [{"model": "grok-4.7", "rank": rank, "confidence": confidence}] if rank else [],
    }
    return item


class UnsurePromoteTests(unittest.TestCase):
    """FF-28 v2 (FINFINDR_UNSURE_PROMOTE). No live calls. Reuses IdentifyTests' fakes, not its tests."""

    fake_get = IdentifyTests.fake_get
    fake_post = IdentifyTests.fake_post
    post_photo = IdentifyTests.post_photo

    MORAY_TEXT = (
        "Photo quality: good\n"
        "Photo issue: none\n"
        "Common name: Panamic green moray\n"
        "Scientific name: Gymnothorax castaneus\n"
        "Group: ray-finned fishes (moray eels)\n"
        "Alternative group considered: sea snake\n"
        "Confidence: medium\n"
        "Notes: Morays have a second, hidden set of jaws.\n"
        "Candidate 1: Gymnothorax castaneus | Panamic green moray | medium | uniform brown body, large moray head, pale bluish eye\n"
        "Candidate 2: Muraena lentiginosa | Jewel moray | low | would show pale spots, which are not visible here\n"
    )
    GULL_TEXT = (
        "Photo quality: poor\n"
        "Photo issue: soft focus\n"
        "Common name: Heermann's Gull\n"
        "Scientific name: Larus heermanni\n"
        "Group: seabird (gull)\n"
        "Alternative group considered: none\n"
        "Confidence: medium\n"
        "Notes: Nests on Isla Rasa.\n"
        "Candidate 1: Larus heermanni | Heermann's Gull | medium | uniform sooty-gray body, red bill, white trailing edge\n"
        "Candidate 2: Leucophaeus atricilla | Laughing Gull | low | white below rather than sooty gray\n"
    )
    # Stored replies from QA cold1 on bf7e441 (eval100/runs/ff28v2-bf7e441-cold1).
    MORAY_COLD1_TEXT = (
        "Photo quality: good\n"
        "Photo issue: none\n"
        "Common name: Panamic green moray\n"
        "Scientific name: Gymnothorax castaneus\n"
        "Group: moray eel\n"
        "Alternative group considered: sea snake\n"
        "Confidence: medium\n"
        "Notes: Morays have a second set of pharyngeal jaws that shoot forward to pull prey down the throat.\n"
        "Candidate 1: Gymnothorax castaneus | Panamic green moray | medium | plain chestnut-brown body, large gape, and no pectoral fins\n"
        "Candidate 2: Muraena lentiginosa | Jewel moray | low | would show dense white jewel-like spots, absent here\n"
        "Candidate 3: none | none | none | none\n"
    )
    BUTTERFLY_410627272_TEXT = (
        "Photo quality: good\n"
        "Photo issue: none\n"
        "Common name: Black-nosed Butterflyfish\n"
        "Scientific name: Johnrandallia nigrirostris\n"
        "Group: Ray-finned fish (butterflyfish)\n"
        "Alternative group considered: Moorish idol\n"
        "Confidence: medium\n"
        "Notes: Often called the barberfish because it cleans parasites from larger fishes at reef stations.\n"
        "Candidate 1: Johnrandallia nigrirostris | Black-nosed Butterflyfish | medium | compressed yellow body, dark snout, and a long white dorsal filament without bold vertical bands\n"
        "Candidate 2: Zanclus cornutus | Moorish Idol | low | also has a long dorsal streamer, but shows strong vertical black-white-yellow bands and a tubular snout\n"
        "Candidate 3: none | none | low | none\n"
    )
    GULL_OFFLIST = "Off-list 1: Leucophaeus fuliginosus | Lava Gull | low | dark gray plumage, bill usually black\nOff-list 2: none\n"

    def setUp(self):
        IdentifyTests.setUp(self)
        os.environ.pop("FINFINDR_UNSURE_PROMOTE", None)
        self.species_patch = patch(f"{__name__}.species_payload", side_effect=self._species)
        self._real_species = species_payload
        self.species_patch.start()

    def tearDown(self):
        self.species_patch.stop()
        os.environ.pop("FINFINDR_UNSURE_PROMOTE", None)
        IdentifyTests.tearDown(self)

    def _species(self, params):
        payload = self._real_species(params)
        if params.get("taxon_id") and int(params["taxon_id"]) != 47178:
            return payload
        if int(params.get("page") or 1) == 1:
            named = [
                {"count": 20000 - i, "taxon": {"name": n, "rank": "species", "preferred_common_name": c}}
                for i, (n, c) in enumerate(FF28_SPECIES)
            ]
            payload["results"] = named + payload["results"][len(named):]
        return payload

    def _taxa(self, url, params):
        if "taxa" in url and params.get("q"):
            return FakeResponse({"results": [{"name": params["q"], "rank": "species"}]})
        return None

    def _post(self, **extra):
        self.inat_handler = self._taxa
        fields = dict(debug="1", verify="0", region="sea-of-cortez", lat="24.2", lng="-110.3")
        fields.update(extra)
        return self.post_photo(**fields).get_json()

    # --- pure functions ---------------------------------------------------

    def test_flag_default_off_and_debug_override(self):
        self.assertFalse(finfindr_server.unsure_promote_enabled())
        os.environ["FINFINDR_UNSURE_PROMOTE"] = "1"
        self.assertTrue(finfindr_server.unsure_promote_enabled())
        self.assertFalse(finfindr_server.unsure_promote_enabled(True, "0"))
        os.environ.pop("FINFINDR_UNSURE_PROMOTE")
        self.assertTrue(finfindr_server.unsure_promote_enabled(True, "1"))
        self.assertFalse(finfindr_server.unsure_promote_enabled(False, "1"))

    def test_p1_reads_certainty_from_shown_model_rank1(self):
        lava = ff28_cand("Leucophaeus fuliginosus", "Lava Gull", "low", off_list=True)
        gull = ff28_cand("Larus heermanni", "Heermann's Gull", "medium", rank=1)
        laugh = ff28_cand("Leucophaeus atricilla", "Laughing Gull", "low", rank=2)
        candidates = [lava, gull, laugh]
        self.assertEqual(finfindr_server.certainty_for(candidates, None), "unsure")
        shown = finfindr_server.public_candidates(candidates, ff28_list())
        self.assertEqual(shown[0]["scientific_name"], "Larus heermanni")
        certainty, decision = finfindr_server.ff28_p1_certainty(candidates, shown, "unsure", None, ff28_list())
        self.assertEqual(certainty, "likely")
        self.assertEqual(decision["path"], "P1")

    def test_p1_never_lowers_and_never_confident(self):
        high = ff28_cand("Larus heermanni", "Heermann's Gull", "high", rank=1)
        shown = finfindr_server.public_candidates([high], ff28_list())
        for current in ("likely", "confident"):
            certainty, decision = finfindr_server.ff28_p1_certainty([high], shown, current, None, ff28_list())
            self.assertEqual(certainty, current)
            self.assertEqual(decision["path"], "none")
        passing = {"decision": "Larus heermanni", "certainty": "confident", "scientific_name": "Larus heermanni"}
        with patch("finfindr_server.passing_verification", return_value=True):
            certainty, _ = finfindr_server.ff28_p1_certainty([high], shown, "unsure", passing, ff28_list())
        self.assertEqual(certainty, "likely")
        low = ff28_cand("Larus heermanni", "Heermann's Gull", "low", rank=1)
        shown_low = finfindr_server.public_candidates([low], ff28_list())
        certainty, decision = finfindr_server.ff28_p1_certainty([low], shown_low, "unsure", None, ff28_list())
        self.assertEqual(certainty, "unsure")
        self.assertEqual(decision["reason"], "rank1_not_stronger")

    def test_p1_gates(self):
        gull = ff28_cand("Larus heermanni", "Heermann's Gull", "medium", rank=1)
        laugh = ff28_cand("Leucophaeus atricilla", "Laughing Gull", "medium", rank=2)
        shown = finfindr_server.public_candidates([gull], ff28_list())
        p1 = finfindr_server.ff28_p1_certainty
        self.assertEqual(p1([gull], shown, "unsure", None, ff28_list(), no_match=True)[1]["reason"], "no_match")
        self.assertEqual(p1([gull], [], "unsure", None, ff28_list())[1]["reason"], "no_candidates")
        self.assertEqual(p1([gull], shown, "unsure", None, [])[1]["reason"], "top_off_list")
        other_top = finfindr_server.public_candidates([laugh], ff28_list())
        self.assertEqual(p1([gull, laugh], other_top, "unsure", None, ff28_list())[1]["reason"], "top_not_model_rank1")
        with patch("finfindr_server.passing_verification", return_value=False):
            self.assertEqual(
                p1([gull], shown, "unsure", {"decision": "UNSURE"}, ff28_list())[1]["reason"], "verification_disagreed"
            )

    def _moray_shown(self, **changes):
        top = ff28_cand("Gymnothorax castaneus", "Panamic green moray", "medium", rank=1,
                        trait="uniform brown body, large moray head")
        second = ff28_cand("Muraena lentiginosa", "Jewel moray", "low", rank=2)
        top.update(changes.get("top", {}))
        second.update(changes.get("second", {}))
        candidates = [top, second]
        shown = finfindr_server.public_candidates(candidates, ff28_list(), preserve_order=True)
        return shown, candidates

    def test_p2_clears_common_name_color_clash_under_full_gate(self):
        shown, candidates = self._moray_shown()
        text = "uniform brown body, large moray head Morays have a second set of jaws."
        cleared, decision = finfindr_server.ff28_p2_clears_clash(shown, candidates, ff28_list(), "good", text)
        self.assertTrue(cleared)
        self.assertEqual(decision["path"], "P2")
        self.assertTrue(decision["clash_common_name"])
        self.assertFalse(decision["card_present"])

    def test_p2_blocked_by_each_gate(self):
        text = "uniform brown body, large moray head"
        clear = finfindr_server.ff28_p2_clears_clash
        shown, cands = self._moray_shown()
        self.assertEqual(clear(shown, cands, ff28_list(), "poor", text)[1]["reason"], "photo_not_good")
        self.assertEqual(clear(shown, cands, [], "good", text)[1]["reason"], "top_off_list")
        shown, cands = self._moray_shown(top={"confidence": "low"})
        self.assertEqual(clear(shown, cands, ff28_list(), "good", text)[1]["reason"], "top_not_medium_plus")
        shown, cands = self._moray_shown(second={"confidence": "medium"})
        self.assertEqual(clear(shown, cands, ff28_list(), "good", text)[1]["reason"], "second_not_low")
        shown, cands = self._moray_shown(top={"proposed_by": []})
        self.assertEqual(clear(shown, cands, ff28_list(), "good", text)[1]["reason"], "top_not_model_rank1")
        shown, cands = self._moray_shown()
        self.assertEqual(clear(shown, cands, ff28_list(), "good", "long body with a small head")[1]["reason"],
                         "clash_not_common_name_only")
        # A species with a SPECIES_TRAITS card is never cleared, even if the card agrees.
        self.assertTrue(finfindr_server.SPECIES_TRAITS.get("phataria unifascialis"))
        star = ff28_cand("Phataria unifascialis", "Blue Sea Star", "medium", rank=1, trait="red mesh pattern")
        star_shown = finfindr_server.public_candidates([star], ff28_list(), preserve_order=True)
        self.assertEqual(clear(star_shown, [star], ff28_list(), "good", "red mesh pattern")[1]["reason"],
                         "trait_card_present")
        self.assertEqual(clear([], [], ff28_list(), "good", text)[1]["reason"], "no_candidates")

    # --- route ------------------------------------------------------------

    def test_route_flag_off_is_unchanged_for_moray_and_gull(self):
        self.captured["model_text"] = self.MORAY_TEXT
        body = self._post()
        self.assertEqual(body["certainty"], "unsure")
        self.assertEqual(body["candidates"], [])
        self.assertNotIn("ff28_decision", body["meta"])
        self.assertNotIn("unsure_promoted_common_name_clash", body["meta"]["evidence"])
        self.captured["model_text"] = self.GULL_TEXT
        self.captured["offlist_text"] = self.GULL_OFFLIST
        body = self._post()
        self.assertEqual(body["certainty"], "unsure")
        self.assertEqual(body["candidates"][0]["scientific_name"], "Larus heermanni")
        self.assertNotIn("ff28_decision", body["meta"])

    def test_route_p2_promotes_moray_to_likely_without_record_count(self):
        os.environ["FINFINDR_UNSURE_PROMOTE"] = "1"
        self.captured["model_text"] = self.MORAY_TEXT
        body = self._post()
        self.assertEqual(body["certainty"], "likely")
        self.assertEqual(body["candidates"][0]["scientific_name"], "Gymnothorax castaneus")
        self.assertNotEqual(body["candidates"][0]["confidence"], "high")
        self.assertIn("unsure_promoted_common_name_clash", body["meta"]["evidence"])
        self.assertEqual(body["meta"]["ff28_decision"]["path"], "P2")
        self.assertEqual(body["meta"]["ff28_decision"]["certainty_final"], "likely")
        self.assertIsNone(body["discovery"])
        self.assertFalse(body["meta"]["discovery_checked"])
        self.assertFalse(any("observations" in url and "species_counts" not in url
                             for url in self.captured.get("gets", [])))
        # No extra model call: one identify call plus nothing for FF-28.
        self.assertFalse(any(call["verify"] for call in self.captured["calls"]))

    def test_p2_v3_blocks_multi_color_clash_410627272(self):
        top = ff28_cand("Johnrandallia nigrirostris", "Black-nosed Butterflyfish", "medium", rank=1,
                        trait="compressed yellow body, dark snout, and a long white dorsal filament without bold vertical bands")
        second = ff28_cand("Zanclus cornutus", "Moorish Idol", "low", rank=2)
        shown = finfindr_server.public_candidates([top, second], ff28_list(), preserve_order=True)
        text = f"{top['trait']} Often called the barberfish because it cleans parasites from larger fishes at reef stations."
        cleared, decision = finfindr_server.ff28_p2_clears_clash(shown, [top, second], ff28_list(), "good", text)
        self.assertFalse(cleared)
        self.assertEqual(decision["reason"], "clash_not_single_color")
        self.assertEqual(decision["visible_colors"], ["white", "yellow"])

    def test_p2_v3_keeps_single_color_moray_414705526(self):
        shown, candidates = self._moray_shown(top={"trait": "plain chestnut-brown body, large gape, and no pectoral fins"})
        text = "plain chestnut-brown body, large gape, and no pectoral fins Morays have a second set of pharyngeal jaws."
        cleared, decision = finfindr_server.ff28_p2_clears_clash(shown, candidates, ff28_list(), "good", text)
        self.assertTrue(cleared)
        self.assertEqual(decision["visible_colors"], ["brown"])

    def test_route_v3_410627272_stays_unsure_and_moray_414705526_promoted(self):
        os.environ["FINFINDR_UNSURE_PROMOTE"] = "1"
        self.captured["model_text"] = self.BUTTERFLY_410627272_TEXT
        body = self._post()
        self.assertEqual(body["certainty"], "unsure")
        self.assertNotIn("unsure_promoted_common_name_clash", body["meta"]["evidence"])
        self.assertEqual(body["meta"]["ff28_decision"]["path"], "none")
        self.assertEqual(body["meta"]["ff28_decision"]["reason"], "clash_not_single_color")
        self.captured["model_text"] = self.MORAY_COLD1_TEXT
        body = self._post()
        self.assertEqual(body["certainty"], "likely")
        self.assertEqual(body["candidates"][0]["scientific_name"], "Gymnothorax castaneus")
        self.assertEqual(body["meta"]["ff28_decision"]["path"], "P2")

    def test_route_p1_promotes_gull_to_likely(self):
        os.environ["FINFINDR_UNSURE_PROMOTE"] = "1"
        self.captured["model_text"] = self.GULL_TEXT
        self.captured["offlist_text"] = self.GULL_OFFLIST
        body = self._post()
        self.assertEqual(body["candidates"][0]["scientific_name"], "Larus heermanni")
        self.assertEqual(body["certainty"], "likely")
        self.assertEqual(body["meta"]["ff28_decision"]["path"], "P1")
        self.assertIn("unsure_promoted_model_rank1", body["meta"]["evidence"])

    def test_route_debug_zero_overrides_env(self):
        os.environ["FINFINDR_UNSURE_PROMOTE"] = "1"
        self.captured["model_text"] = self.MORAY_TEXT
        body = self._post(unsure_promote="0")
        self.assertEqual(body["certainty"], "unsure")
        self.assertNotIn("ff28_decision", body["meta"])

    def test_route_never_downgrades_or_upgrades_answered_photos(self):
        texts = [
            None,  # default high Thalassoma lucasanum
            self.GULL_TEXT.replace("Photo quality: poor", "Photo quality: good"),
            "Photo quality: good\nScientific name: Phataria unifascialis\nCommon name: Blue Sea Star\n"
            "Confidence: high\nNotes: A red mesh star.\n"
            "Candidate 1: Phataria unifascialis | Blue Sea Star | high | red mesh pattern\n",
            "Photo quality: poor\nScientific name: Larus heermanni\nConfidence: low\nNotes: blurry.\n"
            "Candidate 1: Larus heermanni | Heermann's Gull | low | gray body\n"
            "Candidate 2: Leucophaeus atricilla | Laughing Gull | low | white below\n",
        ]
        for text in texts:
            if text is None:
                self.captured.pop("model_text", None)
            else:
                self.captured["model_text"] = text
            self.captured.pop("offlist_text", None)
            os.environ.pop("FINFINDR_UNSURE_PROMOTE", None)
            off = self._post()
            os.environ["FINFINDR_UNSURE_PROMOTE"] = "1"
            on = self._post()
            order = finfindr_server._CERTAINTY_ORDER
            self.assertGreaterEqual(order[on["certainty"]], order[off["certainty"]])
            self.assertNotEqual(on["certainty"], "confident" if off["certainty"] != "confident" else "")
            self.assertEqual(on["certainty"], off["certainty"], msg=(text or "default")[:60])
            self.assertEqual(
                [c["scientific_name"] for c in on["candidates"]],
                [c["scientific_name"] for c in off["candidates"]],
            )
