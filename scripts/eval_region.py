#!/usr/bin/env python3
"""Score FinFindr on a labeled manifest, or on research-grade iNaturalist photos.

Manifest columns (CSV or JSON). Optional columns are used when present:
  photo_path, true_taxon, true_common, expected_outcome, lat, lng, region,
  source_url, license, key_confidence, notes, range_sources,
  nearest_known_record_km, nearest_known_record_source, expected_traits

expected_outcome is match, range_extension, no_match, or reject.
key_confidence is verified or uncertain. Strict pass/fail uses verified rows.
range_sources is a semicolon list such as inat;obis;fishbase.
expected_traits is a semicolon or comma list.

The range-extension rule is the app rule: no research-grade iNaturalist records
of the species within FINFINDR_RANGE_KM (default 100). FINFINDR_RANGE_MAX
(default 0) is the largest count that still counts as unrecorded.

Example:
  python3 scripts/eval_region.py --manifest qa.json --runs 3 \\
    --base-url http://127.0.0.1:5000 --targets scripts/eval_targets.json \\
    --json eval.json --markdown eval.md
"""

import argparse
import csv
import json
import math
import os
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("FINFINDR_PREFETCH", "0")

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import finfindr_server

USER_AGENT = "FinFindr-eval/1.0 (marine ID evaluation; https://github.com/fred123-eng/Finfindr)"
INAT_MIN_INTERVAL = 1.05
DEFAULT_TARGETS_PATH = Path(__file__).with_name("eval_targets.json")
OUTCOMES = ("match", "range_extension", "no_match", "reject")
RANGE_FLAG = "possible_range_extension"

DEFAULT_TARGETS = {
    "confident_precision": {"min": 0.99},
    "false_accepts_per_100": {"max": 1},
    "correct_reject_rate": {"min": 0.98},
    "false_reject_rate": {"max": 0.05},
    "decoy_false_flag_rate": {"max": 0.02},
    "discovery_precision": {"min": 0.99},
    "discovery_recall": {"min": 0.80},
    "range_rule": {
        "max_records": 0,
        "radius_km": 100,
        "text": "no research-grade iNaturalist records of the species within 100 km",
    },
}


class RateLimiter:
    def __init__(self, interval):
        self.interval = interval
        self._last = 0.0

    def wait(self):
        delay = self.interval - (time.monotonic() - self._last)
        if delay > 0:
            time.sleep(delay)
        self._last = time.monotonic()


def inat_get(session, limiter, url, params):
    limiter.wait()
    response = session.get(url, params=params, timeout=(5, 30))
    response.raise_for_status()
    return response.json()


def region_params(args):
    if args.region:
        region = finfindr_server._REGIONS_BY_ID.get(args.region)
        if not region:
            known = ", ".join(sorted(finfindr_server._REGIONS_BY_ID))
            raise SystemExit(f"Unknown region {args.region!r}. Known: {known}")
        if region.get("place_id"):
            return {"place_id": region["place_id"]}, region["label"]
        return {
            "swlat": region["swlat"],
            "swlng": region["swlng"],
            "nelat": region["nelat"],
            "nelng": region["nelng"],
        }, region["label"]
    if args.lat is None or args.lng is None:
        raise SystemExit("Provide --region or both --lat and --lng")
    return {"lat": args.lat, "lng": args.lng, "radius": args.radius}, f"{args.lat:.3f}, {args.lng:.3f}"


def sample_observations(session, limiter, params, count, seed):
    """One research-grade photo per species, spread across the area."""
    payload = inat_get(session, limiter, "https://api.inaturalist.org/v1/observations", {
        **params,
        "quality_grade": "research",
        "photos": "true",
        "identified": "true",
        "per_page": 200,
        "order_by": "votes",
        "iconic_taxa": "Actinopterygii,Mollusca,Animalia",
        "locale": "en",
    })
    by_species = {}
    for obs in payload.get("results") or []:
        taxon = obs.get("taxon") or {}
        if taxon.get("rank") != "species":
            continue
        name = taxon.get("name")
        photos = [photo for photo in obs.get("photos") or [] if photo.get("url")]
        if not name or not photos:
            continue
        by_species.setdefault(name, []).append(obs)
    names = list(by_species)
    random.Random(seed).shuffle(names)
    chosen = []
    for name in names:
        chosen.append(by_species[name][0])
        if len(chosen) >= count:
            break
    return chosen


def photo_bytes(session, limiter, url):
    medium = url.replace("/square.", "/medium.").replace("/square.jpg", "/medium.jpg")
    limiter.wait()
    response = session.get(medium, timeout=(5, 30))
    response.raise_for_status()
    return response.content


def observation_coordinates(obs):
    geo = (obs.get("geojson") or {}).get("coordinates") or []
    if len(geo) == 2 and geo[0] is not None and geo[1] is not None:
        return geo[1], geo[0]
    location = obs.get("location") or ""
    if isinstance(location, str) and "," in location:
        lat_text, lng_text = location.split(",", 1)
        try:
            return float(lat_text), float(lng_text)
        except ValueError:
            return None, None
    return None, None


def load_targets(path=None):
    """Merge a JSON target file over the built-in defaults."""
    target_path = Path(path) if path else DEFAULT_TARGETS_PATH
    merged = json.loads(json.dumps(DEFAULT_TARGETS))
    if target_path.is_file():
        loaded = json.loads(target_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise SystemExit(f"Targets file must be a JSON object: {target_path}")
        for key, value in loaded.items():
            merged[key] = value
    elif path:
        raise SystemExit(f"Targets file not found: {target_path}")
    return merged, str(target_path if target_path.is_file() else "")


def _cell(row, *names):
    for name in names:
        if name in row and row[name] is not None:
            return str(row[name]).strip()
    return ""


def split_list(value, semicolon=False):
    text = (value or "").strip()
    if not text:
        return []
    if semicolon or ";" in text:
        parts = text.split(";")
    else:
        parts = re.split(r"[,|]", text)
    return [part.strip() for part in parts if part.strip()]


def _optional_float(value):
    text = (value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def normalize_item(raw):
    """One manifest row. Missing optional columns stay empty."""
    photo = _cell(raw, "photo_path", "photo", "url", "path")
    expected = _cell(raw, "expected_outcome", "expected").lower().replace(" ", "_")
    confidence = _cell(raw, "key_confidence", "confidence").lower()
    if confidence not in ("verified", "uncertain"):
        confidence = "unspecified"
    return {
        "photo_path": photo,
        "true_taxon": _cell(raw, "true_taxon", "taxon"),
        "true_common": _cell(raw, "true_common", "common"),
        "expected_outcome": expected,
        "lat": _optional_float(_cell(raw, "lat")),
        "lng": _optional_float(_cell(raw, "lng")),
        "region": _cell(raw, "region") or "other",
        "source_url": _cell(raw, "source_url"),
        "license": _cell(raw, "license"),
        "key_confidence": confidence,
        "notes": _cell(raw, "notes"),
        "range_sources": [part.lower() for part in split_list(_cell(raw, "range_sources"), semicolon=True)],
        "nearest_known_record_km": _optional_float(_cell(raw, "nearest_known_record_km")),
        "nearest_known_record_source": _cell(raw, "nearest_known_record_source"),
        "expected_traits": split_list(_cell(raw, "expected_traits")),
    }


def load_manifest(path):
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".csv":
        rows = list(csv.DictReader(text.splitlines()))
    else:
        payload = json.loads(text)
        if isinstance(payload, dict):
            rows = payload.get("items") or payload.get("photos") or payload.get("rows") or []
        else:
            rows = payload
    items = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        item = normalize_item(raw)
        if item["photo_path"] and item["expected_outcome"] in OUTCOMES:
            items.append(item)
    return items, source.parent


def app_range_rule():
    rule = finfindr_server.range_rule()
    return {
        "max_records": rule["max_records"],
        "radius_km": rule["radius_km"],
        "text": rule["text"],
    }


def _norm(value):
    return re.sub(r"\s+", " ", (value or "").strip()).casefold()


def _genus(scientific):
    parts = (scientific or "").split()
    return parts[0].casefold() if parts else ""


def species_names(body):
    names = []

    def add(scientific, front=False):
        scientific = (scientific or "").strip()
        if not scientific:
            return
        if scientific.lower() in {name.lower() for name in names}:
            return
        if front:
            names.insert(0, scientific)
        else:
            names.append(scientific)

    discovery = body.get("discovery") if isinstance(body.get("discovery"), dict) else {}
    add(discovery.get("scientific_name"), front=True)
    for candidate in body.get("candidates") or []:
        add(candidate.get("scientific_name"))
    for relative in discovery.get("closest_relatives") or []:
        add(relative.get("scientific_name"))
    if not names:
        for line in (body.get("result_text") or "").splitlines():
            if line.lower().startswith("scientific name:"):
                add(line.split(":", 1)[1])
                break
    return names


def common_name_of(body):
    discovery = body.get("discovery") if isinstance(body.get("discovery"), dict) else {}
    if discovery.get("common_name"):
        return discovery["common_name"]
    candidates = body.get("candidates") or []
    if candidates and candidates[0].get("common_name"):
        return candidates[0]["common_name"]
    for line in (body.get("result_text") or "").splitlines():
        if line.lower().startswith("common name:"):
            return line.split(":", 1)[1].strip()
    return ""


def app_traits(body):
    discovery = body.get("discovery") if isinstance(body.get("discovery"), dict) else {}
    traits = [str(trait).strip() for trait in (discovery.get("traits") or []) if str(trait).strip()]
    if traits:
        return traits
    for relative in discovery.get("closest_relatives") or []:
        if relative.get("trait"):
            traits.append(str(relative["trait"]).strip())
    for candidate in body.get("candidates") or []:
        if candidate.get("trait"):
            traits.append(str(candidate["trait"]).strip())
    return [trait for trait in traits if trait]


def trait_overlap(expected, traits):
    if not expected:
        return None
    blob = " | ".join(traits).casefold()
    hits = 0
    for item in expected:
        needle = item.casefold().strip()
        if not needle:
            continue
        if needle in blob or any(part.casefold() in needle for part in traits if len(part) >= 4):
            hits += 1
    return hits / len(expected)


def nearest_distance(body):
    discovery = body.get("discovery") if isinstance(body.get("discovery"), dict) else {}
    distances = []
    for record in discovery.get("nearest_records") or []:
        try:
            distances.append(float(record.get("distance_km")))
        except (TypeError, ValueError):
            continue
    return min(distances) if distances else None


def classify_answer(body):
    if not isinstance(body, dict) or body.get("error"):
        return "error"
    quality = (body.get("photo_quality") or "good").lower()
    message = (body.get("message") or "").lower()
    if quality in ("poor", "invalid") or "can't identify" in message or "cant identify" in message:
        return "cant_identify"
    discovery = body.get("discovery") if isinstance(body.get("discovery"), dict) else {}
    if discovery.get("type") in (RANGE_FLAG, "no_match"):
        return "discovery"
    certainty = (body.get("certainty") or "").lower()
    candidates = body.get("candidates") or []
    if certainty == "confident":
        return "confident_name"
    if candidates:
        return "hedged"
    return "cant_identify"


def _is_decoy(item):
    if item.get("expected_outcome") != "match":
        return False
    notes = (item.get("notes") or "").lower()
    return any(token in notes for token in ("look-alike", "lookalike", "look alike", "decoy"))


def answer_from_response(body, item, run, latency_ms, status=200):
    """Classify one app response against one manifest row."""
    body = body if isinstance(body, dict) else {}
    names = species_names(body)
    truth = _norm(item.get("true_taxon"))
    guess = names[0] if names else ""
    species_correct = bool(truth) and _norm(guess) == truth
    top3 = bool(truth) and truth in {_norm(name) for name in names[:3]}
    klass = classify_answer(body)
    discovery = body.get("discovery") if isinstance(body.get("discovery"), dict) else {}
    discovery_type = discovery.get("type") or ""
    traits = app_traits(body)
    overlap = trait_overlap(item.get("expected_traits") or [], traits)
    distance = nearest_distance(body)
    known = item.get("nearest_known_record_km")
    error_km = None if distance is None or known is None else distance - known
    relatives = discovery.get("closest_relatives") or []
    no_match_complete = discovery_type == "no_match" and bool(relatives) and bool(traits)
    range_complete = (
        discovery_type == RANGE_FLAG
        and item.get("expected_outcome") == "range_extension"
        and species_correct
        and distance is not None
        and bool(traits)
        and (overlap is None or overlap > 0)
    )
    if discovery_type == RANGE_FLAG:
        fully_correct = range_complete
    elif discovery_type == "no_match":
        trait_ok = overlap is None or overlap > 0
        fully_correct = item.get("expected_outcome") == "no_match" and no_match_complete and trait_ok
    else:
        fully_correct = False
    decoy = _is_decoy(item)
    reason = ""
    false_accept = False
    asserts_species = klass == "confident_name" or discovery_type == RANGE_FLAG
    if asserts_species and item.get("expected_outcome") == "reject":
        false_accept = True
        reason = "confident identification of a photo that should be rejected"
    elif discovery_type == RANGE_FLAG and item.get("expected_outcome") != "range_extension":
        false_accept = True
        reason = "range-extension flag on a photo that is not a range extension"
    elif asserts_species and not species_correct:
        false_accept = True
        reason = "confident species does not match the labeled taxon"
    if klass == "hedged" and top3 and item.get("expected_outcome") != "reject":
        credit = 0.5
    elif item.get("expected_outcome") == "match" and klass == "confident_name" and species_correct:
        credit = 1.0
    elif item.get("expected_outcome") == "range_extension" and range_complete:
        credit = 1.0
    elif item.get("expected_outcome") == "no_match" and (fully_correct or (klass == "confident_name" and species_correct)):
        credit = 1.0
    elif item.get("expected_outcome") == "reject" and klass == "cant_identify":
        credit = 1.0
    else:
        credit = 0.0
    common = common_name_of(body)
    meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
    return {
        "photo_path": item.get("photo_path") or "",
        "run": run,
        "latency_ms": latency_ms,
        "status": status,
        "true_taxon": item.get("true_taxon") or "",
        "true_common": item.get("true_common") or "",
        "expected_outcome": item.get("expected_outcome") or "",
        "key_confidence": item.get("key_confidence") or "unspecified",
        "notes": item.get("notes") or "",
        "range_sources": list(item.get("range_sources") or []),
        "nearest_known_record_km": known,
        "nearest_known_record_source": item.get("nearest_known_record_source") or "",
        "expected_traits": list(item.get("expected_traits") or []),
        "region": item.get("region") or "",
        "lat": item.get("lat"),
        "lng": item.get("lng"),
        "klass": klass,
        "discovery_type": discovery_type,
        "guess": guess,
        "common": common,
        "genus": _genus(guess),
        "species_correct": species_correct,
        "genus_correct": bool(truth) and _genus(guess) == _genus(item.get("true_taxon")),
        "common_correct": bool(_norm(item.get("true_common"))) and _norm(common) == _norm(item.get("true_common")),
        "top1": species_correct and klass == "confident_name",
        "top3": top3,
        "certainty": (body.get("certainty") or "").lower(),
        "trait_overlap": overlap,
        "nearest_distance_km": distance,
        "nearest_error_km": error_km,
        "flag_fully_correct": fully_correct,
        "decoy": decoy,
        "false_accept": false_accept,
        "false_accept_reason": reason,
        "headline_credit": credit,
        "range_rule": meta.get("range_rule") or "",
        "evidence": meta.get("evidence") or [],
    }


def _rate(numerator, denominator):
    if not denominator:
        return None
    return numerator / denominator


def _round(value, digits=4):
    if value is None:
        return None
    return round(value, digits)


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * fraction
    low = int(math.floor(rank))
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _group_photos(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["photo_path"]].append(row)
    return groups


def _discovery_precision(rows):
    """Worst case: one incorrect flag fails that photo, then take the worse type."""
    groups = _group_photos(rows)
    by_type = {RANGE_FLAG: [], "no_match": []}
    for runs in groups.values():
        present = {row["discovery_type"] for row in runs if row["discovery_type"]}
        for dtype in present:
            flags = [row for row in runs if row["discovery_type"] == dtype]
            other_wrong = any(
                row["discovery_type"] and row["discovery_type"] != dtype and not row["flag_fully_correct"]
                for row in runs
            )
            bad = other_wrong or any(not row["flag_fully_correct"] for row in flags)
            by_type[dtype].append(0 if bad else 1)
    precisions = {}
    for dtype, scores in by_type.items():
        precisions[dtype] = _rate(sum(scores), len(scores))
    available = [value for value in precisions.values() if value is not None]
    combined = min(available) if available else None
    return combined, precisions


def _distance_summary(rows):
    signed = []
    missing = 0
    labeled = 0
    for row in rows:
        if row.get("nearest_known_record_km") is None:
            continue
        labeled += 1
        if row.get("nearest_error_km") is None:
            missing += 1
        else:
            signed.append(row["nearest_error_km"])
    absolute = [abs(value) for value in signed]
    return {
        "labeled": labeled,
        "compared": len(signed),
        "missing_app_distance": missing,
        "mean_abs_km": _round(statistics.mean(absolute), 2) if absolute else None,
        "median_abs_km": _round(statistics.median(absolute), 2) if absolute else None,
        "mean_signed_km": _round(statistics.mean(signed), 2) if signed else None,
    }


def _trait_summary(rows):
    values = [row["trait_overlap"] for row in rows if row.get("trait_overlap") is not None]
    return {
        "n": len(values),
        "mean_overlap": _round(statistics.mean(values), 4) if values else None,
    }


def summarize_answers(rows, include_sources=True):
    """Fred's metrics for one slice of answers. Rates are None when the slice has no cases."""
    groups = _group_photos(rows)
    photos = list(groups.values())
    n = len(rows)
    identifiable = [row for row in rows if row["expected_outcome"] in ("match", "range_extension")]
    should_reject = [row for row in rows if row["expected_outcome"] == "reject"]
    confident = [
        row for row in rows
        if row["klass"] == "confident_name" or row["discovery_type"] == RANGE_FLAG
    ]
    correct_confident = [
        row for row in confident
        if row["species_correct"]
        and row["expected_outcome"] != "reject"
        and (row["discovery_type"] != RANGE_FLAG or row["expected_outcome"] == "range_extension")
    ]
    false_accepts = [row for row in rows if row["false_accept"]]
    decoys = [row for row in rows if row["decoy"]]
    decoy_flags = [row for row in decoys if row["discovery_type"] == RANGE_FLAG]
    hedged = [row for row in rows if row["klass"] == "hedged"]
    refused_identifiable = [
        row for row in identifiable
        if row["klass"] in ("hedged", "cant_identify") or row["discovery_type"] == "no_match"
    ]
    correct_matches = [
        row for row in identifiable
        if row["species_correct"] and (row["klass"] == "confident_name" or row["discovery_type"] == RANGE_FLAG)
    ]
    range_rows = [row for row in rows if row["expected_outcome"] == "range_extension"]
    range_flagged = [
        row for row in range_rows
        if row["discovery_type"] == RANGE_FLAG and row["species_correct"]
    ]
    no_match_rows = [row for row in rows if row["expected_outcome"] == "no_match"]
    no_match_ok = [row for row in no_match_rows if row["flag_fully_correct"] or (row["klass"] == "confident_name" and row["species_correct"])]
    wrong = [row for row in rows if row["true_taxon"] and not row["species_correct"]]
    right = [row for row in rows if row["species_correct"]]
    discovery_precision, precision_by_type = _discovery_precision(rows)
    range_photos = [runs for runs in photos if runs and runs[0]["expected_outcome"] == "range_extension"]
    recalled_photos = [
        runs for runs in range_photos
        if any(row["discovery_type"] == RANGE_FLAG and row["species_correct"] for row in runs)
    ]
    flips = []
    consistent = 0
    for path, runs in groups.items():
        outcomes = [row["discovery_type"] or "none" for row in runs]
        signatures = {(row["klass"], _norm(row["guess"]), row["discovery_type"] or "none") for row in runs}
        if len(signatures) == 1:
            consistent += 1
        if len(set(outcomes)) > 1:
            flips.append({
                "photo_path": path,
                "true_taxon": runs[0]["true_taxon"],
                "expected_outcome": runs[0]["expected_outcome"],
                "key_confidence": runs[0]["key_confidence"],
                "outcomes": outcomes,
            })
    photo_credits = [statistics.mean(row["headline_credit"] for row in runs) for runs in photos]
    latencies = [row["latency_ms"] for row in rows if row.get("latency_ms") is not None]
    metrics = {
        "n_answers": n,
        "n_photos": len(photos),
        "headline_accuracy": _round(statistics.mean(photo_credits)) if photo_credits else None,
        "confident_precision": _round(_rate(len(correct_confident), len(confident))),
        "confident_answers": len(confident),
        "confident_correct": len(correct_confident),
        "confident_wrong": len(false_accepts),
        "false_accepts_per_100": _round(100 * len(false_accepts) / n, 2) if n else None,
        "false_accepts": [
            {
                "photo_path": row["photo_path"],
                "run": row["run"],
                "true_taxon": row["true_taxon"],
                "guess": row["guess"],
                "expected_outcome": row["expected_outcome"],
                "discovery_type": row["discovery_type"],
                "reason": row["false_accept_reason"],
            }
            for row in false_accepts
        ],
        "correct_match_rate": _round(_rate(len(correct_matches), len(identifiable))),
        "false_reject_rate": _round(_rate(len(refused_identifiable), len(identifiable))),
        "correct_reject_rate": _round(_rate(
            sum(1 for row in should_reject if row["klass"] == "cant_identify"),
            len(should_reject),
        )),
        "decoy_false_flag_rate": _round(_rate(len(decoy_flags), len(decoys))),
        "decoy_n": len(decoys),
        "discovery_precision": _round(discovery_precision),
        "discovery_precision_by_type": {key: _round(value) for key, value in precision_by_type.items()},
        "discovery_precision_method": "worst_case_any_wrong_flag_fails_the_photo_then_min_across_types",
        "discovery_recall": _round(_rate(len(recalled_photos), len(range_photos))),
        "discovery_recall_answers": _round(_rate(len(range_flagged), len(range_rows))),
        "no_match_handling": _round(_rate(len(no_match_ok), len(no_match_rows))),
        "coverage": _round(_rate(len(confident), n)),
        "top3_recall": _round(_rate(sum(1 for row in rows if row["top3"]), n)),
        "top3_when_hedged": _round(_rate(sum(1 for row in hedged if row["top3"]), len(hedged))),
        "hedged_when_wrong": _round(_rate(sum(1 for row in wrong if row["klass"] == "hedged"), len(wrong))),
        "hedged_when_right": _round(_rate(sum(1 for row in right if row["klass"] == "hedged"), len(right))),
        "genus_correct": _round(_rate(sum(1 for row in rows if row["genus_correct"]), n)),
        "common_name_correct": _round(_rate(
            sum(1 for row in rows if row["true_common"] and row["common_correct"]),
            sum(1 for row in rows if row["true_common"]),
        )),
        "consistency": _round(_rate(consistent, len(photos))),
        "discovery_flip_rate": _round(_rate(len(flips), len(photos))),
        "discovery_flips": flips,
        "nearest_record_error": _distance_summary(rows),
        "trait_overlap": _trait_summary(rows),
        "latency_ms": {
            "median": int(round(statistics.median(latencies))) if latencies else None,
            "p90": int(round(percentile(latencies, 0.9))) if latencies else None,
            "max": int(round(max(latencies))) if latencies else None,
        },
    }
    if include_sources:
        buckets = defaultdict(list)
        for row in rows:
            for source in row.get("range_sources") or []:
                buckets[source].append(row)
        metrics["by_range_source"] = {
            source: summarize_answers(bucket, include_sources=False)
            for source, bucket in sorted(buckets.items())
        }
    return metrics


def judge(value, spec):
    if not isinstance(spec, dict):
        return "n/a"
    has_bound = spec.get("min") is not None or spec.get("max") is not None
    if not has_bound or value is None:
        return "n/a"
    if spec.get("min") is not None and value < float(spec["min"]) - 1e-9:
        return "fail"
    if spec.get("max") is not None and value > float(spec["max"]) + 1e-9:
        return "fail"
    return "pass"


def score_targets(metrics, targets, rule):
    keys = (
        "confident_precision",
        "false_accepts_per_100",
        "correct_reject_rate",
        "false_reject_rate",
        "decoy_false_flag_rate",
        "discovery_precision",
        "discovery_recall",
    )
    checks = {}
    for key in keys:
        spec = targets.get(key) or {}
        checks[key] = {"value": metrics.get(key), "target": spec, "result": judge(metrics.get(key), spec)}
    rule_spec = targets.get("range_rule") or {}
    if rule_spec:
        matches = (
            rule.get("radius_km") == rule_spec.get("radius_km")
            and rule.get("max_records") == rule_spec.get("max_records")
        )
        checks["range_rule"] = {"value": rule, "target": rule_spec, "result": "pass" if matches else "fail"}
    results = [item["result"] for item in checks.values()]
    if any(result == "fail" for result in results):
        overall = "fail"
    elif any(result == "pass" for result in results):
        overall = "pass"
    else:
        overall = "n/a"
    return {"checks": checks, "overall": overall, "slice": "verified"}


def build_report(rows, label, targets, targets_path, runs):
    rule = app_range_rule()
    verified = [row for row in rows if row["key_confidence"] == "verified"]
    uncertain = [row for row in rows if row["key_confidence"] == "uncertain"]
    return {
        "area": label,
        "runs": runs,
        "targets_path": targets_path,
        "range_rule": rule,
        "targets_result": score_targets(summarize_answers(verified), targets, rule),
        "verified": summarize_answers(verified),
        "uncertain": summarize_answers(uncertain),
        "all": summarize_answers(rows),
        "results": rows,
    }


def _pct(value):
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _target_label(spec):
    if not isinstance(spec, dict):
        return ""
    if spec.get("min") is not None:
        number = spec["min"]
        return f">= {number * 100:.0f}%" if number <= 1 else f">= {number:g}"
    if spec.get("max") is not None:
        number = spec["max"]
        return f"<= {number * 100:.0f}%" if isinstance(number, float) and number <= 1 else f"<= {number:g}"
    if spec.get("radius_km") is not None:
        return f"{spec.get('max_records', 0)} records within {spec.get('radius_km')} km"
    return ""


def _format_check(key, check):
    spec = check.get("target") or {}
    if key == "false_accepts_per_100":
        shown = "n/a" if check.get("value") is None else f"{check['value']:g}"
    elif key == "range_rule":
        value = check.get("value") or {}
        shown = value.get("text") or ""
    else:
        shown = _pct(check.get("value"))
    return f"| {key} | {_target_label(spec)} | {shown} | {check.get('result')} |"


def markdown_report(report):
    verified = report["verified"]
    uncertain = report["uncertain"]
    rule = report["range_rule"]
    lines = [
        f"# FinFindr eval: {report['area']}",
        "",
        f"Range rule: {rule.get('text')} (`FINFINDR_RANGE_KM`={rule.get('radius_km')}, `FINFINDR_RANGE_MAX`={rule.get('max_records')}).",
        "Strict pass/fail uses key_confidence=verified only. Uncertain rows are reported separately.",
        "A discovery flag is fully correct only when a range extension includes a nearest record and supporting traits (and any expected traits overlap), or a no-match result names closest relatives and traits.",
        "Discovery precision is worst-case: any incorrect flag fails that photo, and the score is the lower of the range-extension and no-match precisions.",
        "",
        f"- Targets: {report.get('targets_path') or 'built-in defaults'}",
        f"- Overall: {report['targets_result']['overall']}",
        f"- Photos: {verified['n_photos']} verified, {uncertain['n_photos']} uncertain",
        f"- Runs: {report.get('runs')}",
        "",
        "## Verified targets",
        "",
        "| Metric | Target | Actual | Result |",
        "| --- | --- | --- | --- |",
    ]
    for key, check in report["targets_result"]["checks"].items():
        lines.append(_format_check(key, check))
    lines.extend([
        "",
        "## Verified",
        "",
        f"- Headline accuracy: {_pct(verified['headline_accuracy'])}",
        f"- Confident precision: {_pct(verified['confident_precision'])} ({verified['confident_correct']} / {verified['confident_answers']})",
        f"- False accepts per 100: {verified['false_accepts_per_100']}",
        f"- Correct match rate: {_pct(verified['correct_match_rate'])}",
        f"- False reject rate: {_pct(verified['false_reject_rate'])}",
        f"- Correct reject rate: {_pct(verified['correct_reject_rate'])}",
        f"- Decoy false-flag rate: {_pct(verified['decoy_false_flag_rate'])} (n={verified['decoy_n']})",
        f"- Discovery precision: {_pct(verified['discovery_precision'])}",
        f"- Discovery recall: {_pct(verified['discovery_recall'])}",
        f"- No-match handling: {_pct(verified['no_match_handling'])}",
        f"- Discovery flip rate: {_pct(verified['discovery_flip_rate'])}",
        f"- Top-3 when hedged: {_pct(verified['top3_when_hedged'])}",
        f"- Genus correct: {_pct(verified['genus_correct'])}",
        f"- Common name correct: {_pct(verified['common_name_correct'])}",
        f"- Consistency: {_pct(verified['consistency'])}",
        f"- Latency ms median/p90/max: {verified['latency_ms']['median']} / {verified['latency_ms']['p90']} / {verified['latency_ms']['max']}",
        f"- Nearest-record mean abs error km: {verified['nearest_record_error']['mean_abs_km']}",
        f"- Expected-trait overlap: {verified['trait_overlap']['mean_overlap']}",
        "",
        "### False accepts",
        "",
    ])
    if verified["false_accepts"]:
        lines.extend([
            "| Photo | Run | Truth | Guess | Expected | Reason |",
            "| --- | --- | --- | --- | --- | --- |",
        ])
        for case in verified["false_accepts"]:
            lines.append(
                f"| {case['photo_path']} | {case['run']} | {case['true_taxon']} | {case['guess']} | {case['expected_outcome']} | {case['reason']} |"
            )
    else:
        lines.append("None.")
    lines.extend(["", "### Discovery flips", ""])
    if verified["discovery_flips"]:
        for flip in verified["discovery_flips"]:
            lines.append(f"- {flip['photo_path']}: {', '.join(flip['outcomes'])}")
    else:
        lines.append("None.")
    lines.extend([
        "",
        "## Uncertain",
        "",
        f"- Headline accuracy: {_pct(uncertain['headline_accuracy'])}",
        f"- Confident precision: {_pct(uncertain['confident_precision'])}",
        f"- False accepts per 100: {uncertain['false_accepts_per_100']}",
        f"- Discovery precision: {_pct(uncertain['discovery_precision'])}",
        f"- Discovery recall: {_pct(uncertain['discovery_recall'])}",
        f"- Discovery flip rate: {_pct(uncertain['discovery_flip_rate'])}",
        "",
        "## By range source (verified)",
        "",
    ])
    sources = verified.get("by_range_source") or {}
    if not sources:
        lines.append("No range_sources column on verified rows.")
    for source, bucket in sources.items():
        error = bucket["nearest_record_error"]["mean_abs_km"]
        lines.append(
            f"- {source}: n={bucket['n_answers']}, discovery recall {_pct(bucket['discovery_recall'])}, "
            f"decoy false-flag {_pct(bucket['decoy_false_flag_rate'])}, "
            f"nearest abs error km {error}, trait overlap {bucket['trait_overlap']['mean_overlap']}"
        )
    lines.extend([
        "",
        "## Answers",
        "",
        "| Photo | Run | Expected | Class | Guess | Credit | ms |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ])
    for row in report["results"]:
        if row["key_confidence"] != "verified" and row["key_confidence"] != "uncertain":
            continue
        lines.append(
            f"| {row['photo_path']} | {row['run']} | {row['expected_outcome']} | {row['klass']} | {row['guess']} | {row['headline_credit']} | {row['latency_ms']} |"
        )
    lines.append("")
    return "\n".join(lines)


def load_photo(session, limiter, photo_path, manifest_dir):
    if photo_path.startswith("http://") or photo_path.startswith("https://"):
        if "inaturalist.org" in photo_path or "inaturalist-open-data" in photo_path:
            return photo_bytes(session, limiter, photo_path)
        response = session.get(photo_path, timeout=(5, 30))
        response.raise_for_status()
        return response.content
    candidate = Path(photo_path)
    if not candidate.is_file():
        candidate = manifest_dir / photo_path
    return candidate.read_bytes()


def identify_photo(session, base_url, image, item):
    data = {"region": item.get("region") or "other"}
    if item.get("lat") is not None and item.get("lng") is not None:
        data["lat"] = str(item["lat"])
        data["lng"] = str(item["lng"])
    started = time.perf_counter()
    response = session.post(
        base_url.rstrip("/") + "/identify",
        files={"image": ("eval.jpg", image, "image/jpeg")},
        data=data,
        timeout=130,
    )
    latency_ms = int(round((time.perf_counter() - started) * 1000))
    try:
        body = response.json()
    except ValueError:
        body = {"error": "non-json response"}
    return body, latency_ms, response.status_code


def evaluate_manifest(items, manifest_dir, base_url, runs, session=None):
    session = session or requests.Session()
    session.headers.setdefault("User-Agent", USER_AGENT)
    limiter = RateLimiter(INAT_MIN_INTERVAL)
    rows = []
    for item in items:
        image = load_photo(session, limiter, item["photo_path"], manifest_dir)
        for run in range(1, runs + 1):
            body, latency_ms, status = identify_photo(session, base_url, image, item)
            rows.append(answer_from_response(body, item, run, latency_ms, status))
            print(
                f"{item['photo_path']} run {run}: {rows[-1]['guess'] or '(no name)'} "
                f"{rows[-1]['klass']} credit={rows[-1]['headline_credit']}",
                flush=True,
            )
    return rows


def evaluate_observations(observations, region, base_url, runs, session, limiter):
    rows = []
    for obs in observations:
        taxon = obs.get("taxon") or {}
        photo = (obs.get("photos") or [{}])[0]
        image = photo_bytes(session, limiter, photo.get("url"))
        lat, lng = observation_coordinates(obs)
        item = normalize_item({
            "photo_path": str(obs.get("id") or ""),
            "true_taxon": taxon.get("name") or "",
            "true_common": taxon.get("preferred_common_name") or "",
            "expected_outcome": "match",
            "lat": "" if lat is None else str(lat),
            "lng": "" if lng is None else str(lng),
            "region": region or "other",
            "key_confidence": "verified",
            "notes": "",
        })
        for run in range(1, runs + 1):
            data = {
                "region": region or "other",
                "lat": "" if lat is None else str(lat),
                "lng": "" if lng is None else str(lng),
                "exclude_observation_id": str(obs.get("id") or ""),
            }
            started = time.perf_counter()
            response = session.post(
                base_url.rstrip("/") + "/identify",
                files={"image": ("eval.jpg", image, "image/jpeg")},
                data=data,
                timeout=130,
            )
            latency_ms = int(round((time.perf_counter() - started) * 1000))
            try:
                body = response.json()
            except ValueError:
                body = {"error": "non-json response"}
            rows.append(answer_from_response(body, item, run, latency_ms, response.status_code))
            print(
                f"{item['true_taxon']}: {rows[-1]['guess'] or '(no name)'} {rows[-1]['klass']} {latency_ms}ms",
                flush=True,
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate FinFindr identifications and discovery flags")
    parser.add_argument("--manifest", default="", help="CSV or JSON labeled photo manifest")
    parser.add_argument("--runs", type=int, default=3, help="Identifications per photo (default 3)")
    parser.add_argument("--region", default="", help="Preset region id, used when there is no manifest")
    parser.add_argument("--lat", type=float)
    parser.add_argument("--lng", type=float)
    parser.add_argument("--radius", type=float, default=50, help="Kilometres, used with --lat/--lng")
    parser.add_argument("-n", type=int, default=20, help="Species photos to sample from iNaturalist")
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--targets", default="", help="JSON thresholds. Default scripts/eval_targets.json")
    parser.add_argument("--json", dest="json_path", default="")
    parser.add_argument("--markdown", dest="md_path", default="")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")

    targets, targets_path = load_targets(args.targets or None)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    session.headers["Accept"] = "application/json"
    if args.manifest:
        items, manifest_dir = load_manifest(args.manifest)
        if not items:
            raise SystemExit(f"No usable rows in {args.manifest}")
        label = Path(args.manifest).name
        rows = evaluate_manifest(items, manifest_dir, args.base_url, args.runs, session)
    else:
        params, label = region_params(args)
        limiter = RateLimiter(INAT_MIN_INTERVAL)
        observations = sample_observations(session, limiter, params, args.n, args.seed)
        rows = evaluate_observations(observations, args.region, args.base_url, args.runs, session, limiter)

    report = build_report(rows, label, targets, targets_path, args.runs)
    rendered = markdown_report(report)
    print("\n" + rendered)
    if args.json_path:
        Path(args.json_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.md_path:
        Path(args.md_path).write_text(rendered, encoding="utf-8")
    if report["targets_result"]["overall"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
