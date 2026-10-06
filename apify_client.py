"""
Thin Apify API client.

The manual workflow is: open the Apify web console, run the TikTok / Instagram
scraper on some hashtags, download the dataset JSON, drop it in data/apify/.
This module does the same run from code, given an API token, so the app can
fetch fresh data without leaving the browser.

Runs the actor asynchronously (POST /runs, then poll /actor-runs/{id} until
it finishes, then GET the dataset items) instead of the "run-sync-get-dataset-
items" convenience endpoint. That endpoint holds one HTTP connection open for
the whole scrape and has a hard ~300s server-side cap - fine for a couple of
hashtags, but a multi-hashtag / high-limit Instagram scrape routinely runs
past that wall and the connection dies before Apify itself is done. Polling
has no such ceiling: `max_wait_s` is purely our own patience, not a server
limit, so a bigger scrape just takes longer instead of failing outright.

Nothing here knows about the Video schema - it returns raw Apify rows.
ingest.load_videos_from_apify_api() maps them.
"""
from __future__ import annotations

import time

import requests

_BASE = "https://api.apify.com/v2"
_POLL_INTERVAL_S = 5
_TERMINAL_STATUSES = {"SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"}
_NETWORK_RETRIES = 3


class ApifyError(RuntimeError):
    pass


def _request_with_retry(method: str, url: str, **kwargs) -> requests.Response:
    """requests.<method> with a few retries on connection-level failures.

    The actor run itself keeps going on Apify's side regardless of whether
    our client is still connected, so a single dropped socket (e.g. a VPN
    that resets long-idle connections) shouldn't sink minutes of already-
    elapsed waiting - only a request that still can't get through after a
    few short backoffs should.
    """
    last_exc: requests.RequestException | None = None
    for attempt in range(_NETWORK_RETRIES):
        try:
            return requests.request(method, url, **kwargs)
        except requests.RequestException as e:
            last_exc = e
            if attempt < _NETWORK_RETRIES - 1:
                time.sleep(2 * (attempt + 1))
    raise ApifyError(f"request to Apify failed after {_NETWORK_RETRIES} attempts: {last_exc}") from last_exc


def run_actor_sync(
    actor_id: str,
    run_input: dict,
    token: str,
    timeout_s: int = 300,
) -> list[dict]:
    """Start `actor_id` with `run_input`, wait for it (polling), return the
    dataset rows. `timeout_s` is the max total wall-clock time to wait for
    the run to finish, not a single request's socket timeout - see module
    docstring for why this doesn't use Apify's run-sync endpoint.

    `actor_id` may be written "owner/name" or "owner~name" - the API path wants
    the tilde form, so we normalise it.

    Raises ApifyError on a missing token, an HTTP error (with Apify's message
    if there is one), a run that fails/aborts/times out on Apify's side, a run
    still going after `timeout_s`, or a non-JSON body.
    """
    if not token:
        raise ApifyError("APIFY_API_TOKEN is not set")

    actor_path = actor_id.replace("/", "~")

    start_resp = _request_with_retry(
        "POST", f"{_BASE}/acts/{actor_path}/runs",
        params={"token": token}, json=run_input, timeout=30,
    )
    _raise_for_apify_error(start_resp, actor_id)

    try:
        run = start_resp.json()["data"]
        run_id = run["id"]
    except (ValueError, KeyError) as e:
        raise ApifyError(f"Apify returned an unexpected run-start body: {e}") from e

    deadline = time.monotonic() + timeout_s
    status = run.get("status", "READY")
    while status not in _TERMINAL_STATUSES:
        if time.monotonic() >= deadline:
            raise ApifyError(
                f"Apify actor {actor_id} run {run_id} did not finish within "
                f"{timeout_s}s (still {status}) - retry with fewer hashtags "
                "or a lower per-hashtag results limit, or check the run in "
                "the Apify console."
            )
        time.sleep(_POLL_INTERVAL_S)
        poll_resp = _request_with_retry(
            "GET", f"{_BASE}/actor-runs/{run_id}", params={"token": token}, timeout=30,
        )
        _raise_for_apify_error(poll_resp, actor_id)
        run = poll_resp.json()["data"]
        status = run.get("status", status)

    if status != "SUCCEEDED":
        detail = run.get("statusMessage") or status
        raise ApifyError(f"Apify actor {actor_id} run {run_id} {status}: {detail}")

    dataset_id = run["defaultDatasetId"]
    items_resp = _request_with_retry(
        "GET", f"{_BASE}/datasets/{dataset_id}/items",
        params={"token": token, "clean": "true"}, timeout=60,
    )
    _raise_for_apify_error(items_resp, actor_id)

    try:
        rows = items_resp.json()
    except ValueError as e:
        raise ApifyError(f"Apify returned a non-JSON body: {e}") from e

    if not isinstance(rows, list):
        raise ApifyError(f"expected a list of rows, got {type(rows).__name__}")
    return rows


def _raise_for_apify_error(resp: requests.Response, actor_id: str) -> None:
    if resp.status_code < 300:
        return
    detail = resp.text.strip()
    try:
        detail = resp.json().get("error", {}).get("message", detail)
    except ValueError:
        pass
    raise ApifyError(f"Apify actor {actor_id} returned HTTP {resp.status_code}: {detail}")
