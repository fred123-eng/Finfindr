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
            self.assertEqual(self.captured["json"]["model"], "grok-4.5")
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
        self.assertIn("small.jpg", photos[0]["url"])
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
        self.assertEqual(hedged["certainty"], "unsure")
        self.assertTrue(hedged["meta"]["verify_used"])
        self.assertFalse(hedged["meta"]["verify_succeeded"])
        self.assertIsNone(hedged["discovery"])

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
        self.assertEqual(unsure["certainty"], "unsure")
        self.assertIsNone(unsure["discovery"])
        self.assertTrue(unsure["candidates"])
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
        self.assertEqual(body["certainty"], "likely")
        self.assertNotEqual(body["candidates"][0]["confidence"], "high")
        self.assertNotEqual(body["candidates"][0]["confidence"], "low")
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

    def test_model_timeouts_leave_room_for_fallback_and_verification(self):
        primary, fallback = finfindr_server.plan_model_timeouts(110, 80, 25, 25, True)
        self.assertGreaterEqual(primary, 80)
        self.assertLessEqual(primary + 3, 110)
        self.assertGreater(fallback, 0)
        self.assertFalse(finfindr_server.should_fallback(
            {"ok": False, "retryable": True, "error": "timeout"}, 10, 25
        ))
        self.assertTrue(finfindr_server.should_fallback(
            {"ok": False, "retryable": True, "error": "timeout"}, 40, 25
        ))
        self.assertTrue(finfindr_server.should_fallback(
            {"ok": False, "retryable": True, "error": "connection"}, 30, 25
        ))
        self.assertTrue(finfindr_server.should_fallback(
            {"ok": False, "retryable": True, "error": "http_502"}, 20, 25
        ))
        self.assertFalse(finfindr_server.should_fallback(
            {"ok": False, "retryable": False, "error": "http_400"}, 100, 25
        ))


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
        self.assertEqual(body["candidates"][0]["confidence"], "medium")
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

    def test_parallel_fallback_is_recorded_without_replacing_the_primary(self):
        attempts = []
        holder = {"fallback": {"ok": True, "ms": 1200, "model": "grok-4"}}

        class Finished(object):
            def join(self, timeout):
                return None

        finfindr_server._record_parallel_fallback(Finished(), holder, "grok-4", attempts)
        self.assertEqual(attempts[0]["model"], "grok-4")
        self.assertTrue(attempts[0]["ok"])
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
        self.captured["model_text"] = (
            "Photo quality: good\n"
            "Scientific name: Octopus bimaculatus\n"
            "Common name: Two-spot octopus\n"
            "Confidence: medium\n"
            "Candidate 1: Octopus bimaculatus | Two-spot octopus | medium | faint side spots\n"
            "Candidate 2: Octopus hubbsorum | Hubbs' octopus | low | brown body\n"
        )
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

    def test_parallel_grok4_does_not_count_as_fallback_used(self):
        self.assertFalse(finfindr_server.answer_used_fallback(
            {"ok": True, "model": "grok-4.7"}, "grok-4.7"
        ))
        self.assertTrue(finfindr_server.answer_used_fallback(
            {"ok": True, "model": "grok-4"}, "grok-4.7"
        ))
        self.assertFalse(finfindr_server.answer_used_fallback(
            {"ok": False, "model": "grok-4"}, "grok-4.7"
        ))

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
        self.assertEqual(body["candidates"][0]["scientific_name"], "Carcharhinus leucas")
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


if __name__ == "__main__":
    unittest.main()
