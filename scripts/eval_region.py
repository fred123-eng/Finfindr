#!/usr/bin/env python3
"""Score FinFindr against research-grade iNaturalist photos from one area.

Samples observations with photos, posts each photo to /identify, and writes
top-1 accuracy, top-3 accuracy, how often wrong and right answers are marked
unsure, and latency. iNaturalist calls stay at or below 60 per minute.

Example:
  FINFINDR_PREFETCH=0 python3 scripts/eval_region.py --region sea-of-cortez -n 20 \\
    --base-url http://127.0.0.1:5000 --json eval.json --markdown eval.md
"""

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("FINFINDR_PREFETCH", "0")

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import finfindr_server

USER_AGENT = "FinFindr-eval/1.0 (marine ID evaluation; https://github.com/fred123-eng/Finfindr)"
INAT_MIN_INTERVAL = 1.05  # <= 60 requests/minute


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


def score(chosen, truth):
    names = []
    for candidate in chosen.get("candidates") or []:
        scientific = (candidate.get("scientific_name") or "").lower()
        if scientific and scientific not in names:
            names.append(scientific)
    if not names:
        text = chosen.get("result_text") or ""
        for line in text.splitlines():
            if line.lower().startswith("scientific name:"):
                names.append(line.split(":", 1)[1].strip().lower())
                break
    truth_name = truth.lower()
    top1 = bool(names) and names[0] == truth_name
    top3 = truth_name in names[:3]
    certainty = (chosen.get("certainty") or "").lower()
    return top1, top3, certainty


def summarize(rows, label):
    total = len(rows) or 1
    top1 = sum(1 for row in rows if row["top1"]) / total
    top3 = sum(1 for row in rows if row["top3"]) / total
    wrong = [row for row in rows if not row["top1"]]
    right = [row for row in rows if row["top1"]]
    wrong_unsure = (sum(1 for row in wrong if row["certainty"] == "unsure") / len(wrong)) if wrong else None
    right_unsure = (sum(1 for row in right if row["certainty"] == "unsure") / len(right)) if right else None
    latencies = [row["latency_ms"] for row in rows]
    return {
        "area": label,
        "n": len(rows),
        "top1_accuracy": round(top1, 3),
        "top3_accuracy": round(top3, 3),
        "wrong_flagged_unsure": None if wrong_unsure is None else round(wrong_unsure, 3),
        "right_flagged_unsure": None if right_unsure is None else round(right_unsure, 3),
        "latency_ms": {
            "median": int(statistics.median(latencies)) if latencies else 0,
            "mean": int(statistics.mean(latencies)) if latencies else 0,
        },
        "results": rows,
    }


def markdown(report):
    lines = [
        f"# FinFindr eval: {report['area']}",
        "",
        f"- Photos: {report['n']}",
        f"- Top-1 accuracy: {report['top1_accuracy']}",
        f"- Top-3 accuracy: {report['top3_accuracy']}",
        f"- Wrong answers flagged unsure: {report['wrong_flagged_unsure']}",
        f"- Right answers flagged unsure: {report['right_flagged_unsure']}",
        f"- Median latency: {report['latency_ms']['median']} ms",
        "",
        "| Truth | Top guess | Certainty | Top-1 | Top-3 | ms |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["results"]:
        lines.append(
            f"| {row['truth']} | {row['guess']} | {row['certainty']} | {row['top1']} | {row['top3']} | {row['latency_ms']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate FinFindr on iNaturalist photos from one area")
    parser.add_argument("--region", default="", help="Preset region id, such as sea-of-cortez")
    parser.add_argument("--lat", type=float)
    parser.add_argument("--lng", type=float)
    parser.add_argument("--radius", type=float, default=50, help="Kilometres, used with --lat/--lng")
    parser.add_argument("-n", type=int, default=20, help="Number of species photos to score")
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--json", dest="json_path", default="")
    parser.add_argument("--markdown", dest="md_path", default="")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    params, label = region_params(args)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    session.headers["Accept"] = "application/json"
    limiter = RateLimiter(INAT_MIN_INTERVAL)
    observations = sample_observations(session, limiter, params, args.n, args.seed)
    rows = []
    for obs in observations:
        taxon = obs.get("taxon") or {}
        truth = taxon.get("name") or ""
        photo = (obs.get("photos") or [{}])[0]
        image = photo_bytes(session, limiter, photo.get("url"))
        lat, lng = observation_coordinates(obs)
        started = time.perf_counter()
        response = session.post(
            args.base_url.rstrip("/") + "/identify",
            files={"image": ("eval.jpg", image, "image/jpeg")},
            data={
                "region": args.region or "other",
                "lat": "" if lat is None else str(lat),
                "lng": "" if lng is None else str(lng),
                "exclude_observation_id": str(obs.get("id") or ""),
            },
            timeout=130,
        )
        latency_ms = int(round((time.perf_counter() - started) * 1000))
        body = response.json()
        top1, top3, certainty = score(body, truth)
        guess = ""
        if body.get("candidates"):
            guess = body["candidates"][0].get("scientific_name") or ""
        rows.append({
            "truth": truth,
            "guess": guess,
            "certainty": certainty,
            "top1": top1,
            "top3": top3,
            "latency_ms": latency_ms,
            "observation_id": obs.get("id"),
            "status": response.status_code,
        })
        print(f"{truth}: {guess or '(no guess)'} {certainty} top1={top1} {latency_ms}ms", flush=True)

    report = summarize(rows, label)
    rendered = markdown(report)
    print("\n" + rendered)
    if args.json_path:
        Path(args.json_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.md_path:
        Path(args.md_path).write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
