// EHR Media Intelligence — vanilla JS client for the FastAPI backend.
// The API serves this page, so every request is a same-origin relative URL. There is deliberately no way to
// point the page at another server from the URL: that would send patient data wherever a crafted link says.
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const els = {
    form: $("search-form"), input: $("search-input"), type: $("filter-type"), from: $("filter-from"),
    to: $("filter-to"), clear: $("clear-filters"), status: $("status"), error: $("error"),
    loading: $("loading"), results: $("results"), empty: $("empty"), emptyTitle: $("empty-title"),
    emptyBody: $("empty-body"), suggestions: $("suggestions"), drawer: $("drawer"),
    backdrop: $("drawer-backdrop"), drawerTitle: $("drawer-title"), drawerSub: $("drawer-subtitle"),
    drawerBody: $("drawer-body"), drawerClose: $("drawer-close"),
  };
  // Resource types: dropdown label, badge label and badge colours, in one place.
  const TYPE_META = {
    DocumentReference: { option: "DocumentReference (notes)", badge: "DocumentReference",
                         cls: "bg-sky-100 text-sky-800 ring-sky-200" },
    DiagnosticReport: { option: "DiagnosticReport (imaging, labs)", badge: "DiagnosticReport",
                        cls: "bg-violet-100 text-violet-800 ring-violet-200" },
    AISummary: { option: "AI summary (patient-level)", badge: "AI summary",
                 cls: "bg-teal-100 text-teal-800 ring-teal-200" },
  };
  const LAB_BADGE_CLS = "bg-amber-100 text-amber-800 ring-amber-200";
  const OTHER_BADGE_CLS = "bg-slate-100 text-slate-700 ring-slate-200";
  const EMPTY_TITLE = "Search across all patient media records";
  const EMPTY_BODY = "Ask in plain language. Try one of these:";
  const SUGGESTIONS = [
    "suspicious breast imaging", "heart failure with fluid overload", "low hemoglobin and iron",
    "lung nodule needing follow-up", "stroke with aphasia", "high potassium kidney disease",
  ];

  // ------------------------------------------------------------------ helpers
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const fmtDate = (d) => {
    if (!d) return "Date unknown";
    const dt = new Date(d + "T00:00:00");
    return isNaN(dt) ? d : dt.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
  };

  function badge(r) {
    const meta = TYPE_META[r.resource_type];
    let cls = meta ? meta.cls : OTHER_BADGE_CLS;
    let label = meta ? meta.badge : r.resource_type;
    if (r.resource_type === "DiagnosticReport") {
      const lab = r.subtype === "Laboratory";
      if (lab) cls = LAB_BADGE_CLS;
      label = `${label} · ${lab ? "Lab" : "Imaging"}`;
    }
    return `<span class="inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${cls}">${esc(label)}</span>`;
  }

  function scoreBar(score) {
    const pct = Math.round(score * 100);
    const color = pct >= 50 ? "bg-teal-600" : pct >= 30 ? "bg-teal-400" : "bg-slate-300";
    return `<div class="flex items-center gap-2" title="Cosine similarity ${score.toFixed(3)}">
      <span class="text-xs text-slate-500">Relevance</span>
      <div class="h-1.5 w-20 overflow-hidden rounded-full bg-slate-100" role="meter" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pct}" aria-label="Relevance score">
        <div class="h-full ${color}" style="width:${pct}%"></div></div>
      <span class="w-9 text-right text-xs font-semibold tabular-nums text-slate-700">${pct}%</span></div>`;
  }

  function setView(view) { // "empty" | "loading" | "results" | "error"
    els.loading.classList.toggle("hidden", view !== "loading");
    els.empty.classList.toggle("hidden", view !== "empty");
    els.results.classList.toggle("hidden", view !== "results");
    els.error.classList.toggle("hidden", view !== "error");
    els.results.setAttribute("aria-busy", view === "loading" ? "true" : "false");
  }

  function showEmpty(title, body, withSuggestions) {
    els.emptyTitle.textContent = title;
    els.emptyBody.textContent = body;
    els.suggestions.innerHTML = withSuggestions ? SUGGESTIONS.map((s) =>
      `<button type="button" class="rounded-full border border-slate-300 bg-white px-3 py-1 text-sm text-slate-700 hover:border-teal-600 hover:text-teal-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-teal-600" data-q="${esc(s)}">${esc(s)}</button>`).join("") : "";
    setView("empty");
  }

  // ------------------------------------------------------------------ errors
  class HttpError extends Error {
    constructor(status, body) {
      super(`HTTP ${status}`);
      this.status = status;
      this.detail = body?.detail;
    }
  }

  // Only a network failure suggests the API is down; an HTTP error shows what the server said.
  function errorMessage(err, what) {
    if (err instanceof HttpError) {
      const d = err.detail;
      const text = typeof d === "string" ? d
        : Array.isArray(d) ? d.map((x) => x.msg || String(x)).join("; ") : "";
      return `${what} (HTTP ${err.status})${text ? ": " + text.replace(/\.$/, "") : ""}.`;
    }
    return `${what}: could not reach the API. Is it running? Start it with: uvicorn backend.api.main:app`;
  }

  // ------------------------------------------------------------------ search
  let controller = null;
  let debounce = null;

  async function runSearch() {
    clearTimeout(debounce);   // whatever triggered this search, drop the pending debounced one
    controller?.abort();      // and the in-flight one, so a stale response can't replace this view
    controller = null;
    const q = els.input.value.trim();
    if (els.from.value && els.to.value && els.from.value > els.to.value) {
      els.error.textContent = "The “From” date must be on or before the “To” date.";
      els.status.textContent = "";
      return setView("error");
    }
    if (q.length < 2) {
      els.status.textContent = "";
      return showEmpty(EMPTY_TITLE, EMPTY_BODY, true);
    }
    controller = new AbortController();
    const params = new URLSearchParams();
    if (els.type.value) params.set("resource_type", els.type.value);
    if (els.from.value) params.set("date_from", els.from.value);
    if (els.to.value) params.set("date_to", els.to.value);
    setView("loading");
    els.status.textContent = "Searching…";
    try {
      const res = await fetch(`/search?${params}`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, top_k: 5 }), signal: controller.signal,
      });
      if (!res.ok) throw new HttpError(res.status, await res.json().catch(() => ({})));
      const data = await res.json();
      renderResults(data);
    } catch (err) {
      if (err.name === "AbortError") return;
      els.error.textContent = errorMessage(err, "Search failed");
      els.status.textContent = "";
      setView("error");
    }
  }

  function renderResults(data) {
    if (!data.results.length) {
      els.status.textContent = "No results";
      return showEmpty("No matching records",
        "Try different wording, or widen the resource type and date filters.", false);
    }
    els.status.textContent = `${data.results.length} result${data.results.length > 1 ? "s" : ""} · ${data.took_ms} ms`;
    els.results.innerHTML = data.results.map((r, i) => `
      <li>
        <article tabindex="0" role="button" data-index="${i}" data-patient="${esc(r.patient_id)}"
          aria-label="${esc(`Result ${r.rank}: ${r.title} for ${r.patient_name}, ${fmtDate(r.date)}. Open patient details.`)}"
          class="result-card group cursor-pointer rounded-xl border border-slate-200 bg-white p-4 shadow-sm transition hover:border-teal-500 hover:shadow focus:outline-none focus-visible:border-teal-600 focus-visible:ring-2 focus-visible:ring-teal-600/40">
          <div class="flex flex-wrap items-start justify-between gap-2">
            <div class="min-w-0">
              <p class="font-semibold text-slate-900">${esc(r.patient_name)}
                <span class="ml-1 font-mono text-xs font-normal text-slate-500">${esc(r.mrn)}</span></p>
              <p class="mt-0.5 text-sm text-slate-600"><span class="font-medium text-slate-800">${esc(r.title)}</span>
                <span aria-hidden="true">·</span> <time datetime="${esc(r.date || "")}">${esc(fmtDate(r.date))}</time></p>
            </div>
            <div class="flex flex-col items-end gap-1.5">${badge(r)}${scoreBar(r.score)}</div>
          </div>
          <p class="mt-3 line-clamp-3 text-sm text-slate-700">${esc(r.snippet)}</p>
          ${r.summary_snippet && r.resource_type !== "AISummary" ? `
          <p class="mt-3 flex gap-2 rounded-lg bg-teal-50 p-2.5 text-sm text-teal-900">
            <span class="mt-0.5 shrink-0 rounded bg-teal-600 px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wide text-white" aria-label="AI-generated">AI</span>
            <span class="line-clamp-3">${esc(r.summary_snippet)}</span></p>` : ""}
        </article>
      </li>`).join("");
    setView("results");
  }

  // ------------------------------------------------------------------ drawer
  let lastFocus = null;
  let drawerRequest = 0;          // increments on every open and close; responses for older values are dropped
  let drawerController = null;

  async function openPatient(patientId) {
    drawerController?.abort();
    drawerController = new AbortController();
    const request = ++drawerRequest;
    if (els.drawer.classList.contains("hidden")) lastFocus = document.activeElement;
    els.drawerTitle.textContent = "Loading…";
    els.drawerSub.textContent = "";
    els.drawerBody.innerHTML = `<div class="space-y-3" aria-hidden="true">${'<div class="h-20 animate-pulse rounded-lg bg-slate-100"></div>'.repeat(3)}</div>`;
    els.drawer.classList.remove("hidden");
    els.backdrop.classList.remove("hidden");
    document.body.style.overflow = "hidden";
    els.drawer.focus();
    try {
      const res = await fetch(`/patients/${encodeURIComponent(patientId)}`, { signal: drawerController.signal });
      if (!res.ok) throw new HttpError(res.status, await res.json().catch(() => ({})));
      const data = await res.json();
      if (request !== drawerRequest) return;  // another patient was opened, or the drawer closed, meanwhile
      renderPatient(data);
    } catch (err) {
      if (err.name === "AbortError" || request !== drawerRequest) return;
      els.drawerTitle.textContent = "Error";
      els.drawerBody.innerHTML = `<p role="alert" class="text-sm text-red-700">${esc(errorMessage(err, "Could not load patient"))}</p>`;
    }
    els.drawerClose.focus();
  }

  function closeDrawer() {
    drawerRequest++;
    drawerController?.abort();
    els.drawer.classList.add("hidden");
    els.backdrop.classList.add("hidden");
    document.body.style.overflow = "";
    lastFocus?.focus();
  }

  // Whole years from calendar dates in local time; a 29 Feb birthday counts from 1 Mar in other years.
  function ageInYears(isoDate, today = new Date()) {
    const [y, m, d] = isoDate.split("-").map(Number);
    const month = today.getMonth() + 1;
    const beforeBirthday = month < m || (month === m && today.getDate() < d);
    return today.getFullYear() - y - (beforeBirthday ? 1 : 0);
  }

  const section = (title, inner) =>
    `<section><h3 class="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">${esc(title)}</h3>${inner}</section>`;
  const list = (items, empty = "None documented") => items?.length
    ? `<ul class="list-disc space-y-1 pl-5 text-sm text-slate-800">${items.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>`
    : `<p class="text-sm text-slate-500">${esc(empty)}</p>`;

  function renderPatient({ patient, summary, resources, bundle }) {
    els.drawerTitle.textContent = patient.name;
    const age = patient.birth_date ? ageInYears(patient.birth_date) : null;
    els.drawerSub.textContent = [patient.mrn, patient.birth_date ? `DOB ${fmtDate(patient.birth_date)} (${age} y)` : "DOB unknown",
      patient.gender].join(" · ");

    let html = "";
    if (summary) {
      const conf = { high: "bg-emerald-100 text-emerald-800", medium: "bg-amber-100 text-amber-800", low: "bg-rose-100 text-rose-800" }[summary.confidence];
      html += `
      <div class="rounded-xl border border-teal-200 bg-teal-50/50 p-4">
        <div class="mb-3 flex flex-wrap items-center justify-between gap-2">
          <h3 class="font-semibold text-teal-900">AI clinical summary</h3>
          <div class="flex items-center gap-2 text-xs">
            <span class="rounded-full px-2 py-0.5 font-medium ${conf}">Confidence: ${esc(summary.confidence)}</span>
            <span class="text-slate-500">${esc(summary.model)} · ${summary.word_count} words</span>
          </div>
        </div>
        <div class="space-y-4">
          ${section("Chief concern", `<p class="text-sm text-slate-800">${esc(summary.chief_concern)}</p>`)}
          ${section("Key diagnoses", list(summary.key_diagnoses))}
          ${section("Recent media", summary.recent_media.length ? `<ul class="space-y-1 text-sm">${summary.recent_media.map((m) =>
            `<li class="flex gap-2"><span class="w-24 shrink-0 text-slate-500">${esc(fmtDate(m.date))}</span><span class="font-medium capitalize text-slate-600">${esc(m.kind)}</span><span class="text-slate-800">${esc(m.description)}</span></li>`).join("")}</ul>` : list([]))}
          ${section("Flagged anomalies", summary.flagged_anomalies.length
            ? `<ul class="space-y-1 text-sm">${summary.flagged_anomalies.map((a) => `<li class="flex gap-2 text-rose-800"><span aria-hidden="true">⚠</span><span>${esc(a)}</span></li>`).join("")}</ul>`
            : list([], "No anomalies flagged"))}
          ${section("Narrative", `<p class="text-sm text-slate-800">${esc(summary.narrative)}</p>`)}
          <p class="text-xs text-slate-500"><span class="font-medium">Confidence rationale:</span> ${esc(summary.confidence_rationale)}</p>
          ${summary.quality_warnings?.length ? `<div class="rounded-lg bg-amber-50 p-2 text-xs text-amber-900"><p class="font-medium">Automated quality checks</p>${list(summary.quality_warnings)}</div>` : ""}
        </div>
        <p class="mt-4 rounded-lg border border-amber-300 bg-amber-50 p-3 text-xs font-medium text-amber-900" role="note">⚠ ${esc(summary.disclaimer)}</p>
      </div>`;
    } else {
      html += `<p class="rounded-lg bg-slate-50 p-3 text-sm text-slate-600">No AI summary has been generated for this patient yet.</p>`;
    }

    html += section(`Linked FHIR resources (${resources.length})`, `
      <ul class="divide-y divide-slate-100 rounded-lg border border-slate-200">${resources.map((r) => `
        <li class="p-3">
          <details>
            <summary class="cursor-pointer list-none rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-teal-600">
              <div class="flex flex-wrap items-center justify-between gap-2">
                <div class="min-w-0"><p class="truncate text-sm font-medium text-slate-900">${esc(r.title)}</p>
                  <p class="text-xs text-slate-500">${esc(fmtDate(r.record_date))} · <span class="font-mono">${esc(r.resource_type)}/${esc(r.resource_id)}</span></p></div>
                ${badge({ resource_type: r.resource_type, subtype: r.subtype })}
              </div>
            </summary>
            <pre class="mt-3 whitespace-pre-wrap rounded bg-slate-50 p-3 font-sans text-sm text-slate-800">${esc(r.text)}</pre>
            ${r.audit?.length ? `<details class="mt-2 text-xs text-slate-600"><summary class="cursor-pointer">Data-cleaning audit (${r.audit.length} changes)</summary>
              <ul class="mt-1 space-y-0.5 pl-4">${r.audit.map((a) => `<li><span class="font-mono">${esc(a.field)}</span> ${esc(a.action)}${a.old ? `: “${esc(a.old)}”` : ""}${a.new ? ` → “${esc(a.new)}”` : ""} — ${esc(a.reason)}</li>`).join("")}</ul></details>` : ""}
          </details>
        </li>`).join("")}
      </ul>`);
    html += bundle
      ? `<p class="text-xs text-slate-500">FHIR Bundle ${bundle.valid ? "passed R4 validation" : `failed R4 validation (${esc(bundle.error_count)} errors)`}.
          <a class="text-teal-700 underline hover:text-teal-900" href="${esc(`/patients/${encodeURIComponent(patient.patient_id)}/bundle`)}" target="_blank" rel="noopener">View raw Bundle JSON</a></p>`
      : `<p class="text-xs text-slate-500">No FHIR Bundle is stored for this patient.</p>`;
    els.drawerBody.innerHTML = html;
  }

  // ------------------------------------------------------------------ events
  els.form.addEventListener("submit", (e) => { e.preventDefault(); runSearch(); });
  els.input.addEventListener("input", () => { clearTimeout(debounce); debounce = setTimeout(runSearch, 300); });
  [els.type, els.from, els.to].forEach((el) => el.addEventListener("change", runSearch));
  els.clear.addEventListener("click", () => { els.type.value = els.from.value = els.to.value = ""; runSearch(); });
  els.suggestions.addEventListener("click", (e) => {
    const q = e.target.closest("[data-q]")?.dataset.q;
    if (q) { els.input.value = q; runSearch(); }
  });

  els.results.addEventListener("click", (e) => {
    const card = e.target.closest(".result-card");
    if (card) openPatient(card.dataset.patient);
  });
  els.results.addEventListener("keydown", (e) => {
    const card = e.target.closest(".result-card");
    if (!card) return;
    const cards = [...els.results.querySelectorAll(".result-card")];
    const i = cards.indexOf(card);
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openPatient(card.dataset.patient); }
    else if (e.key === "ArrowDown") { e.preventDefault(); cards[Math.min(i + 1, cards.length - 1)].focus(); }
    else if (e.key === "ArrowUp") { e.preventDefault(); (i === 0 ? els.input : cards[i - 1]).focus(); }
  });
  els.input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { const first = els.results.querySelector(".result-card"); if (first) { e.preventDefault(); first.focus(); } }
  });

  els.drawerClose.addEventListener("click", closeDrawer);
  els.backdrop.addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (e) => {
    const open = !els.drawer.classList.contains("hidden");
    if (open && e.key === "Escape") { e.preventDefault(); closeDrawer(); return; }
    if (open && e.key === "Tab") { // keep focus inside the dialog
      const f = [...els.drawer.querySelectorAll("button, a[href], summary, [tabindex]:not([tabindex='-1'])")].filter((x) => x.offsetParent !== null);
      if (!f.length) { e.preventDefault(); els.drawer.focus(); return; }
      const i = f.indexOf(document.activeElement);
      if (i === -1) { e.preventDefault(); (e.shiftKey ? f[f.length - 1] : f[0]).focus(); }  // e.g. focus on the dialog itself
      else if (e.shiftKey && i === 0) { e.preventDefault(); f[f.length - 1].focus(); }
      else if (!e.shiftKey && i === f.length - 1) { e.preventDefault(); f[0].focus(); }
      return;
    }
    if (!open && e.key === "/" && document.activeElement !== els.input && !/INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName)) {
      e.preventDefault(); els.input.focus();
    }
  });

  // Resource-type dropdown: filled from TYPE_META straight away, then from the backend's list (the source of
  // truth for which types exist); labels always come from TYPE_META.
  function fillTypes(types) {
    const current = els.type.value;
    els.type.innerHTML = `<option value="">All types</option>` +
      types.map((t) => `<option value="${esc(t)}">${esc(TYPE_META[t]?.option || t)}</option>`).join("");
    els.type.value = types.includes(current) ? current : "";
  }
  fillTypes(Object.keys(TYPE_META));
  fetch("/resource-types").then((r) => (r.ok ? r.json() : null)).then((types) => {
    if (Array.isArray(types) && types.length) fillTypes(types);
  }).catch(() => {});

  showEmpty(EMPTY_TITLE, EMPTY_BODY, true);
})();
