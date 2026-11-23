"""Task 5: frontend. Static checks, plus behaviour tests that run the real page in headless Chrome with a mocked
fetch (skipped when Chrome/Chromium isn't installed)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import urllib.parse
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"


# ---------------------------------------------------------------- static checks
def test_page_loads_no_third_party_code():
    html = (FRONTEND / "index.html").read_text()
    assert not re.search(r"<script[^>]+src=\"https?://", html)
    assert not re.search(r"<link[^>]+href=\"https?://", html)
    assert 'href="styles.css"' in html and (FRONTEND / "styles.css").exists()


def test_compiled_css_contains_every_class_used():
    css = (FRONTEND / "styles.css").read_text()
    src = (FRONTEND / "index.html").read_text() + (FRONTEND / "app.js").read_text()
    classes: set[str] = set()
    for m in re.finditer(r'class="([^"]+)"', src):
        classes.update(c for c in m.group(1).split() if "${" not in c)
    for m in re.finditer(r'"((?:bg|text|ring)-[a-z]+-\d{2,3}(?: [a-z-]+-\d{2,3})*)"', src):
        classes.update(m.group(1).split())
    missing = [c for c in classes - {"result-card", "group"}
               if "." + re.sub(r"([:\[\]/.%])", r"\\\1", c) not in css]
    assert not missing, f"rebuild frontend/styles.css (README: Frontend CSS); missing {missing}"


def test_no_api_override_and_query_length_matches_backend():
    js = (FRONTEND / "app.js").read_text()
    assert "location.search" not in js and "?api" not in js
    assert 'maxlength="500"' in (FRONTEND / "index.html").read_text()


# ---------------------------------------------------------------- behaviour in a real browser
MOCK_FETCH = r"""
window.__calls = [];
const json = (obj, status = 200) => new Response(JSON.stringify(obj), {status, headers: {"Content-Type": "application/json"}});
const d = new Date();
window.__dob25 = `${d.getFullYear() - 25}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
const patient = (id, name, extra = {}) => ({
  patient: {patient_id: id, mrn: "MRN-000001", name, birth_date: extra.dob || "1958-03-14", gender: "female"},
  summary: null, resources: [],
  bundle: "bundle" in extra ? extra.bundle : {valid: 1, error_count: 0, content_hash: "h"}});
window.fetch = (url, opts = {}) => new Promise((resolve, reject) => {
  url = String(url); window.__calls.push(url);
  let delay = 10, respond;
  if (url.startsWith("/resource-types")) respond = () => json(["DocumentReference", "DiagnosticReport", "AISummary"]);
  else if (url.startsWith("/search")) {
    const q = JSON.parse(opts.body).query;
    if (q === "neterr") respond = () => { throw new TypeError("Failed to fetch"); };
    else if (q === "toolong") respond = () => json({detail: [{msg: "String should have at most 500 characters"}]}, 422);
    else { delay = 500; respond = () => json({query: q, took_ms: 1, filters: {}, results: [{rank: 1, score: 0.9,
      resource_type: "AISummary", resource_id: "x", patient_id: "pat-B", mrn: "MRN-000002", patient_name: "Result " + q,
      date: "2025-01-01", title: "t", subtype: null, snippet: "s", summary_snippet: null}]}); }
  }
  else if (url === "/patients/pat-A") { delay = 1000; respond = () => json(patient("pat-A", "Patient A")); }
  else if (url === "/patients/pat-B") { delay = 50; respond = () => json(patient("pat-B", "Patient B", {dob: window.__dob25})); }
  else if (url === "/patients/pat-NOBUNDLE") respond = () => json(patient("pat-NOBUNDLE", "No Bundle", {bundle: null}));
  else if (url === "/patients/pat-ERR") respond = () => json({detail: "boom"}, 500);
  else respond = () => json({detail: "not found"}, 404);
  const t = setTimeout(() => { try { resolve(respond()); } catch (e) { reject(e); } }, delay);
  opts.signal?.addEventListener("abort", () => { clearTimeout(t); reject(new DOMException("Aborted", "AbortError")); });
});
"""

SCENARIOS = r"""
(async () => {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const $ = (id) => document.getElementById(id);
  const out = {};
  const card = (pid) => { $("results").innerHTML = `<li><article class="result-card" tabindex="0" data-patient="${pid}"></article></li>`;
                          return $("results").querySelector(".result-card"); };
  const key = (k, o = {}) => document.dispatchEvent(new KeyboardEvent("keydown", {key: k, bubbles: true, ...o}));
  const typeQ = (v) => { $("search-input").value = v; $("search-input").dispatchEvent(new Event("input", {bubbles: true})); };
  const change = (el) => el.dispatchEvent(new Event("change", {bubbles: true}));
  try {
    await sleep(50);
    out.options = [...$("filter-type").options].map((o) => o.textContent);
    out.emptyTitle = $("empty-title").textContent;

    card("pat-A").click(); await sleep(10); key("Escape");          // slow A, closed, then fast B
    card("pat-B").click(); await sleep(1500);
    out.titleAfterRace = $("drawer-title").textContent;
    out.subtitle = $("drawer-subtitle").textContent;
    out.bundleHref = $("drawer-body").querySelector("a")?.getAttribute("href");
    key("Escape");

    card("pat-A").click(); await sleep(10); key("Escape"); await sleep(1500);   // response after close
    out.hiddenAfterLateResponse = $("drawer").classList.contains("hidden");
    out.titleAfterLateResponse = $("drawer-title").textContent;

    card("pat-NOBUNDLE").click(); await sleep(100);
    out.noBundleText = $("drawer-body").textContent.includes("No FHIR Bundle is stored");
    out.noBundleLinks = $("drawer-body").querySelectorAll("a").length;
    key("Escape");

    card("pat-ERR").click(); await sleep(100);
    out.focusAfterError = document.activeElement?.id;
    out.drawerError = $("drawer-body").textContent.trim();
    $("drawer").focus(); key("Tab", {shiftKey: true});
    out.focusTrapped = $("drawer").contains(document.activeElement) && document.activeElement !== $("drawer");
    key("Escape");

    typeQ("heart"); await sleep(350); typeQ("h"); await sleep(1200);   // shorter query while "heart" in flight
    out.emptyAfterShorten = !$("empty").classList.contains("hidden") && $("results").classList.contains("hidden");

    typeQ("heart"); await sleep(350);                                  // invalid dates while "heart" in flight
    $("filter-from").value = "2025-12-01"; $("filter-to").value = "2025-01-01"; change($("filter-to"));
    await sleep(1200);
    out.errorAfterBadDates = !$("error").classList.contains("hidden") && $("results").classList.contains("hidden");
    $("filter-from").value = ""; $("filter-to").value = "";

    const before = __calls.filter((u) => u.startsWith("/search")).length;   // typing then a filter change
    typeQ("kidney"); await sleep(50); $("filter-type").value = "AISummary"; change($("filter-type"));
    await sleep(1500);
    out.searchCalls = __calls.filter((u) => u.startsWith("/search")).length - before;
    out.badgeText = $("results").querySelector("span.rounded-full")?.textContent;

    $("filter-type").value = "";
    typeQ("toolong"); await sleep(400); out.err422 = $("error").textContent;
    typeQ("neterr"); await sleep(400); out.errNetwork = $("error").textContent;

    out.allRelative = __calls.every((u) => u.startsWith("/"));
    out.injectedImg = !!document.querySelector("img");
  } catch (e) { out.harnessError = String((e && e.stack) || e); }
  const pre = document.createElement("pre"); pre.id = "test-results"; pre.textContent = JSON.stringify(out);
  document.body.appendChild(pre);
})();
"""


def _chrome() -> str | None:
    for name in ("google-chrome", "chromium", "chromium-browser", "chrome"):
        if path := shutil.which(name):
            return path
    mac = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    return mac if Path(mac).exists() else None


@pytest.fixture(scope="module")
def browser_results(tmp_path_factory) -> dict:
    chrome = _chrome()
    if not chrome:
        pytest.skip("Chrome/Chromium not installed")
    d = tmp_path_factory.mktemp("frontend")
    html = (FRONTEND / "index.html").read_text().replace(
        '<script src="app.js"></script>',
        '<script src="mock.js"></script><script src="app.js"></script><script src="scenarios.js"></script>')
    (d / "index.html").write_text(html)
    (d / "app.js").write_text((FRONTEND / "app.js").read_text())
    (d / "mock.js").write_text(MOCK_FETCH)
    (d / "scenarios.js").write_text(SCENARIOS)
    attack = urllib.parse.quote('"><img src=x onerror=alert(document.cookie)>')
    url = (d / "index.html").as_uri() + f"?api={attack}"
    proc = subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
                           f"--user-data-dir={d / 'profile'}", "--virtual-time-budget=30000", "--dump-dom", url],
                          capture_output=True, text=True, timeout=120)
    m = re.search(r'<pre id="test-results">(.*?)</pre>', proc.stdout, re.DOTALL)
    assert m, f"harness produced no results; stderr: {proc.stderr[-500:]}"
    import html as htmllib

    results = json.loads(htmllib.unescape(m.group(1)))
    assert "harnessError" not in results, results["harnessError"]
    return results


def test_slow_response_for_another_patient_never_replaces_the_drawer(browser_results):
    assert browser_results["titleAfterRace"] == "Patient B"
    assert browser_results["hiddenAfterLateResponse"] is True
    assert browser_results["titleAfterLateResponse"] != "Patient A"


def test_stale_search_never_replaces_empty_or_error_view(browser_results):
    assert browser_results["emptyAfterShorten"] is True
    assert browser_results["errorAfterBadDates"] is True


def test_filter_change_does_not_double_search(browser_results):
    assert browser_results["searchCalls"] == 1


def test_api_query_param_is_ignored_and_cannot_inject(browser_results):
    assert browser_results["allRelative"] is True
    assert browser_results["injectedImg"] is False
    assert browser_results["bundleHref"] == "/patients/pat-B/bundle"


def test_missing_bundle_is_described_without_a_dead_link(browser_results):
    assert browser_results["noBundleText"] is True and browser_results["noBundleLinks"] == 0


def test_age_is_correct_on_the_birthday(browser_results):
    assert "(25 y)" in browser_results["subtitle"]


def test_error_messages_and_focus(browser_results):
    assert browser_results["err422"] == "Search failed (HTTP 422): String should have at most 500 characters."
    assert "could not reach the API" in browser_results["errNetwork"]
    assert "Is it running" not in browser_results["drawerError"] and "HTTP 500" in browser_results["drawerError"]
    assert browser_results["focusAfterError"] == "drawer-close"
    assert browser_results["focusTrapped"] is True


def test_labels_come_from_one_map(browser_results):
    assert browser_results["options"] == ["All types", "DocumentReference (notes)",
                                          "DiagnosticReport (imaging, labs)", "AI summary (patient-level)"]
    assert browser_results["badgeText"] == "AI summary"
    assert browser_results["emptyTitle"] == "Search across all patient media records"
