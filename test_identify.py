"""Route and prompt checks for regional identification. No live xAI or iNaturalist calls."""

import io
import json
import os
import threading
import time
import unittest
from unittest.mock import patch

os.environ["FINFINDR_PREFETCH"] = "0"
os.environ.pop("SPECIES_LIST_MODE", None)

import requests
from PIL import Image

import finfindr_server


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
        finfindr_server._SPECIES_CACHE.clear()
        finfindr_server._SPECIES_INFLIGHT.clear()
        finfindr_server._inat_cache.clear()
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

    def fake_get(self, url, params=None, **kwargs):
        params = params or {}
        if "species_counts" in url:
            with self._calls_lock:
                self.species_calls.append(params)
            return FakeResponse(species_payload(params))
        return FakeResponse({"results": []})

    def fake_post(self, url, json=None, **kwargs):
        self.captured["url"] = url
        self.captured["json"] = json
        self.captured.setdefault("calls", []).append({
            "model": (json or {}).get("model"),
            "timeout": kwargs.get("timeout"),
        })
        model = (json or {}).get("model")
        if model in self.fail_timeout:
            raise requests.Timeout("slow")
        if model in self.fail_status:
            return FakeResponse({"error": "upstream"}, status=self.fail_status[model])
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
        data = {"image": (io.BytesIO(jpeg_bytes()), "reef.jpg")}
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
        self.assertEqual(self.captured["json"]["temperature"], 0)
        self.assertEqual(self.captured["json"]["model"], "grok-4.7")
        self.assertEqual(body["meta"]["model"], "grok-4.7")
        self.assertEqual(body["meta"]["image_px"], "80x60")
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
            data={"image": (io.BytesIO(photo), "lapaz.jpg"), "region": "my-location"},
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
            data={"image": (io.BytesIO(photo), "hawaii.jpg"), "region": "sea-of-cortez"},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.get_json()["region_label"], "Sea of Cortez / La Paz")
        prompt = self.prompt()
        self.assertIn("outside this region", prompt)
        self.assertIn("Cirrhitus rivulatus", prompt)
        located = next(call for call in self.species_calls if call.get("place_id"))
        self.assertEqual(located["place_id"], 122897)

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
            data={"image": (io.BytesIO(wide), "wide.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.get_json()["meta"]["image_px"], "1536x768")
        self.assertEqual(self.captured["json"]["model"], "grok-4.7")

        os.environ["MAX_IMAGE_PX"] = "800"
        os.environ["XAI_MODEL"] = "grok-4.5"
        try:
            ignored = self.post_photo(model="grok-4.3")
            self.assertEqual(self.captured["json"]["model"], "grok-4.5")
            self.assertEqual(ignored.get_json()["meta"]["model"], "grok-4.5")
            chosen = self.post_photo(model="grok-4.3", debug="1")
            self.assertEqual(self.captured["json"]["model"], "grok-4.3")
            self.assertEqual(chosen.get_json()["meta"]["model"], "grok-4.3")
            rejected = self.post_photo(model="not a model", debug="1")
            self.assertEqual(self.captured["json"]["model"], "grok-4.5")
            sized = self.client.post(
                "/identify",
                data={"image": (io.BytesIO(wide), "wide.jpg")},
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

    def test_primary_timeout_falls_back_within_budget(self):
        self.fail_timeout = {"grok-4.7"}
        response = self.post_photo()
        body = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["meta"]["model"], "grok-4")
        self.assertTrue(body["meta"]["fallback_used"])
        self.assertEqual([item["model"] for item in body["meta"]["attempts"]], ["grok-4.7", "grok-4"])
        self.assertFalse(body["meta"]["attempts"][0]["ok"])
        self.assertEqual(body["meta"]["attempts"][0]["error"], "timeout")
        self.assertTrue(body["meta"]["attempts"][1]["ok"])
        self.assertGreaterEqual(self.captured["calls"][0]["timeout"], 70)
        self.assertLessEqual(self.captured["calls"][1]["timeout"], 25)
        self.assertIn("Cortez Rainbow Wrasse", body["result_text"])

    def test_both_models_timing_out_stays_json(self):
        self.fail_timeout = {"grok-4.7", "grok-4"}
        response = self.post_photo()
        self.assertEqual(response.status_code, 504)
        body = response.get_json()
        self.assertIn("taking too long", body["error"])
        self.assertTrue(body["meta"]["fallback_used"])
        self.assertEqual(len(body["meta"]["attempts"]), 2)

    def test_client_error_does_not_fall_back(self):
        self.fail_status = {"grok-4.7": 400}
        response = self.post_photo()
        self.assertEqual(response.status_code, 502)
        self.assertEqual(len(self.captured["calls"]), 1)
        self.assertFalse(response.get_json()["meta"]["fallback_used"])
        self.assertIn("try another", response.get_json()["error"])

    def test_fallback_model_can_be_disabled(self):
        os.environ["XAI_FALLBACK_MODEL"] = ""
        self.fail_timeout = {"grok-4.7"}
        try:
            response = self.post_photo()
        finally:
            os.environ.pop("XAI_FALLBACK_MODEL", None)
        self.assertEqual(response.status_code, 504)
        self.assertEqual(len(self.captured["calls"]), 1)
        self.assertFalse(response.get_json()["meta"]["fallback_used"])

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
        finfindr_server._SPECIES_WAIT_RESULT.clear()
        seen = {"n": 0}

        def counting(url, params=None, **kwargs):
            seen["n"] += 1
            return flaky(url, params, **kwargs)

        try:
            with patch("finfindr_server.http.get", side_effect=counting):
                partial = finfindr_server.load_species_list(query, "short", 5)
                self.assertFalse(partial["complete"])
                names = [item["scientific"] for item in partial["species"]]
                self.assertNotIn("Thalassoma lucasanum", names)
                self.assertIn("Hydrophis platurus", names)
                self.assertIsNone(finfindr_server._species_cache_get(("short", "place:122897")))
                before = seen["n"]
                finfindr_server.load_species_list(query, "short", 5)
                self.assertGreater(seen["n"], before)
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

    def test_missing_api_key_is_json(self):
        os.environ.pop("XAI_API_KEY", None)
        response = self.post_photo()
        self.assertEqual(response.status_code, 500)
        body = response.get_json()
        self.assertIn("error", body)
        self.assertEqual(body["region_label"], "Sea of Cortez / La Paz")
        self.assertNotIn("json", self.captured)


if __name__ == "__main__":
    unittest.main()
