"""Route and prompt checks for regional identification. No live xAI or iNaturalist calls."""

import io
import json
import os
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


def species_payload(params):
    per_page = int(params.get("per_page") or 1)
    page = int(params.get("page") or 1)
    if params.get("taxon_id"):
        return {
            "total_results": 1,
            "results": [
                {
                    "count": 9000,
                    "taxon": {
                        "name": "Chelonia mydas",
                        "rank": "species",
                        "preferred_common_name": "not the curated name",
                    },
                }
            ],
        }
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
        finfindr_server._SPECIES_CACHE.clear()
        finfindr_server._SPECIES_INFLIGHT.clear()
        finfindr_server._inat_cache.clear()
        self.client = finfindr_server.app.test_client()
        self.captured = {}
        self.species_calls = []
        self.get_patch = patch("finfindr_server.requests.get", side_effect=self.fake_get)
        self.post_patch = patch("finfindr_server.requests.post", side_effect=self.fake_post)
        self.get_patch.start()
        self.post_patch.start()

    def tearDown(self):
        self.get_patch.stop()
        self.post_patch.stop()

    def fake_get(self, url, params=None, **kwargs):
        params = params or {}
        if "species_counts" in url:
            self.species_calls.append(params)
            return FakeResponse(species_payload(params))
        return FakeResponse({"results": []})

    def fake_post(self, url, json=None, **kwargs):
        self.captured["url"] = url
        self.captured["json"] = json
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
        self.assertEqual(self.captured["json"]["temperature"], 0)
        self.assertEqual(self.captured["json"]["model"], "grok-4")
        self.assertEqual(body["region_label"], "Sea of Cortez / La Paz")
        self.assertIn("Cortez Rainbow Wrasse", body["result_text"])
        self.assertNotIn("Wrong", body["result_text"])
        meta = body["meta"]
        self.assertEqual(meta["species_list_mode"], "short")
        self.assertEqual(meta["species_sent"], 250)
        self.assertEqual(meta["species_available"], 606)
        self.assertFalse(meta["species_truncated"])
        self.assertEqual(meta["list_source"], "inaturalist")
        self.assertEqual(meta["prompt_tokens"], 4321)
        self.assertEqual(meta["prompt_tokens_source"], "usage")
        self.assertGreaterEqual(meta["list_fetch_ms"], 0)
        self.assertGreaterEqual(meta["model_call_ms"], 0)
        main_calls = [call for call in self.species_calls if "taxon_id" not in call]
        self.assertTrue(main_calls)
        self.assertEqual(main_calls[0]["place_id"], 122897)
        self.assertEqual(int(main_calls[0]["per_page"]), 300)
        self.assertTrue(all(int(call.get("page") or 1) == 1 for call in main_calls))

    def test_caribbean_skips_cortez_lookalikes_and_uses_its_bounds(self):
        response = self.post_photo(region="caribbean")
        self.assertEqual(response.status_code, 200)
        prompt = self.prompt()
        self.assertIn("Caribbean Sea", prompt)
        self.assertNotIn("Cirrhitus rivulatus", prompt)
        self.assertNotIn("Sphoeroides annulatus", prompt)
        self.assertEqual(response.get_json()["region_label"], "Caribbean")
        main = next(call for call in self.species_calls if "taxon_id" not in call)
        self.assertEqual(float(main["swlat"]), 9.0)
        self.assertNotIn("place_id", main)

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
        main = next(call for call in self.species_calls if "taxon_id" not in call)
        self.assertEqual(float(main["lat"]), 21.3)
        self.assertEqual(float(main["lng"]), -157.8)
        self.assertEqual(int(main["radius"]), 100)

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
        main = next(call for call in self.species_calls if "taxon_id" not in call)
        self.assertIn("radius", main)
        self.assertNotIn("place_id", main)

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
        main = next(call for call in self.species_calls if "taxon_id" not in call)
        self.assertEqual(main["place_id"], 122897)

    def test_inaturalist_failure_falls_back_without_blocking_identification(self):
        self.get_patch.stop()
        with patch("finfindr_server.requests.get", side_effect=requests.ConnectionError("down")):
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
        with patch("finfindr_server.requests.post", side_effect=post_without_usage):
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
