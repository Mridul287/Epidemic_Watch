from __future__ import annotations

import csv
import json
import logging
import random
from datetime import date, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "epidemicwatch_data" / "non_iid"
ADAPTER_CANDIDATES = (
    ROOT / "phase3_baseline_results" / "adapter_augmented",
    ROOT / "phase3_baseline_outputs" / "adapter_augmented",
)

CATEGORIES = [
    "Dermatological",
    "Febrile/Systemic",
    "Gastrointestinal",
    "Musculoskeletal/Other",
    "Respiratory",
]
HOSPITALS = ["Hospital A", "Hospital B", "Hospital C"]
TOTAL_DAYS = 90
BASELINE_DAYS = 30
SIMULATION_SEED = 42
OUTBREAK_RELATIVE_INCREASE = 1.0
OUTBREAK_CATEGORY = "Respiratory"
OUTBREAK_START = 50
OUTBREAK_END = 65

MODEL = None
TOKENIZER = None
MODEL_LABELS = None


def _poisson(rng: random.Random, mean: float) -> int:
    if mean <= 0:
        return 0
    limit = pow(2.718281828459045, -mean)
    product = 1.0
    count = 0
    while product > limit:
        count += 1
        product *= rng.random()
    return count - 1


def _read_phase2_counts() -> tuple[list[list[list[int]]], int]:
    """Assign Phase 2 records to seeded synthetic days and inject the Phase 5 surge."""
    counts = [
        [[0 for _ in range(TOTAL_DAYS)] for _ in CATEGORIES]
        for _ in HOSPITALS
    ]
    category_indexes = {name: index for index, name in enumerate(CATEGORIES)}
    category_totals = [[0 for _ in CATEGORIES] for _ in HOSPITALS]

    for hospital_idx in range(len(HOSPITALS)):
        hospital_dir = DATA_ROOT / f"hospital_{hospital_idx}"
        rng = random.Random(SIMULATION_SEED + hospital_idx)
        found_records = False
        for split in ("train", "val", "test"):
            csv_path = hospital_dir / f"{split}.csv"
            if not csv_path.is_file():
                continue
            found_records = True
            with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
                for row in csv.DictReader(file):
                    category = row.get("syndrome_category", "")
                    if category not in category_indexes:
                        continue
                    category_idx = category_indexes[category]
                    category_totals[hospital_idx][category_idx] += 1
                    day_idx = rng.randrange(TOTAL_DAYS)
                    counts[hospital_idx][category_idx][day_idx] += 1
        if not found_records:
            return _fallback_counts()

    respiratory_idx = category_indexes[OUTBREAK_CATEGORY]
    outbreak_hospital = max(
        range(len(HOSPITALS)),
        key=lambda index: category_totals[index][respiratory_idx],
    )
    rng = random.Random(SIMULATION_SEED + 1)
    normal_daily_rate = category_totals[outbreak_hospital][respiratory_idx] / TOTAL_DAYS
    extra_cases_per_day = OUTBREAK_RELATIVE_INCREASE * normal_daily_rate
    for day_idx in range(OUTBREAK_START, OUTBREAK_END):
        counts[outbreak_hospital][respiratory_idx][day_idx] += _poisson(
            rng, extra_cases_per_day
        )
    return counts, outbreak_hospital


def _fallback_counts() -> tuple[list[list[list[int]]], int]:
    counts = [
        [[0 for _ in range(TOTAL_DAYS)] for _ in CATEGORIES]
        for _ in HOSPITALS
    ]
    rates = [2.4, 3.2, 3.5, 2.7, 4.2]
    for hospital_idx in range(len(HOSPITALS)):
        for category_idx, base_rate in enumerate(rates):
            stable_rate = base_rate * (0.85 + hospital_idx * 0.15)
            rng = random.Random(
                SIMULATION_SEED + hospital_idx * len(CATEGORIES) + category_idx
            )
            counts[hospital_idx][category_idx] = [
                _poisson(rng, stable_rate)
                for day in range(TOTAL_DAYS)
            ]
    rng = random.Random(SIMULATION_SEED + 1)
    extra_mean = (
        rates[CATEGORIES.index(OUTBREAK_CATEGORY)]
        * OUTBREAK_RELATIVE_INCREASE
        * (0.85 + 1 * 0.15)
    )
    for day_idx in range(OUTBREAK_START, OUTBREAK_END):
        counts[1][CATEGORIES.index(OUTBREAK_CATEGORY)][day_idx] += _poisson(
            rng, extra_mean
        )
    return counts, 1


def detect_outbreak(
    time_series: list[int], method: str, baseline_days: int = BASELINE_DAYS
) -> tuple[int | None, list[float]]:
    """Phase 5 CUSUM/EWMA detector, returning the first alarm day and its trace."""
    if method not in {"cusum", "ewma"}:
        raise ValueError(f"Unknown method: {method}")
    if not time_series:
        return None, []

    baseline = time_series[:baseline_days]
    mean = sum(baseline) / len(baseline)
    variance = sum((value - mean) ** 2 for value in baseline) / len(baseline)
    sigma = max(variance**0.5, max(mean, 0.5) ** 0.5) + 1e-9
    trace: list[float] = []
    alarm_day = None

    if method == "cusum":
        allowance = 0.5 * sigma
        threshold = 8.0 * sigma
        cumulative = 0.0
        for day_idx, value in enumerate(time_series):
            cumulative = max(0.0, cumulative + value - mean - allowance)
            trace.append(cumulative)
            if alarm_day is None and day_idx >= baseline_days and cumulative > threshold:
                alarm_day = day_idx
        return alarm_day, trace

    lam = 0.2
    ewma_value = mean
    for day_idx, value in enumerate(time_series):
        ewma_value = lam * value + (1 - lam) * ewma_value
        control_limit = mean + 5.0 * sigma * (
            (lam / (2 - lam)) * (1 - (1 - lam) ** (2 * (day_idx + 1)))
        ) ** 0.5
        trace.append(ewma_value)
        if alarm_day is None and day_idx >= baseline_days and ewma_value > control_limit:
            alarm_day = day_idx
    return alarm_day, trace


def _current_status(
    series: list[int], cusum_alarm: int | None, ewma_alarm: int | None
) -> str:
    if cusum_alarm is not None or ewma_alarm is not None:
        return "Alert"
    baseline = series[:BASELINE_DAYS]
    mean = sum(baseline) / len(baseline)
    sigma = max(
        (sum((value - mean) ** 2 for value in baseline) / len(baseline)) ** 0.5,
        max(mean, 0.5) ** 0.5,
    )
    return "Elevated" if series[-1] > mean + 2 * sigma else "Normal"


def _dashboard_data() -> dict:
    counts, outbreak_hospital = _read_phase2_counts()
    start_date = date(2024, 11, 1)
    hospitals = []
    for hospital_idx, name in enumerate(HOSPITALS):
        categories = {
            category: counts[hospital_idx][category_idx]
            for category_idx, category in enumerate(CATEGORIES)
        }
        series = categories[OUTBREAK_CATEGORY]
        cusum_alarm, _ = detect_outbreak(series, "cusum")
        ewma_alarm, _ = detect_outbreak(series, "ewma")
        hospitals.append(
            {
                "name": name,
                "categories": categories,
                "status": _current_status(series, cusum_alarm, ewma_alarm),
                "latest": sum(values[-1] for values in categories.values()),
                "cusumAlarm": cusum_alarm,
                "ewmaAlarm": ewma_alarm,
            }
        )
    return {
        "hospitals": hospitals,
        "categories": CATEGORIES,
        "startDate": start_date.isoformat(),
        "endDate": (start_date + timedelta(days=TOTAL_DAYS - 1)).isoformat(),
        "outbreakHospital": HOSPITALS[outbreak_hospital],
        "outbreakCategory": OUTBREAK_CATEGORY,
        "outbreakStart": OUTBREAK_START,
        "outbreakEnd": OUTBREAK_END - 1,
        "baselineDays": BASELINE_DAYS,
        "synthetic": True,
    }


def _keyword_prediction(text: str) -> str:
    normalized = text.lower()
    keywords = {
        "Respiratory": ("cough", "wheeze", "shortness of breath", "sore throat", "runny nose", "chest tightness"),
        "Febrile/Systemic": ("fever", "chills", "fatigue", "body ache", "weakness", "flu"),
        "Gastrointestinal": ("vomit", "diarrhea", "stomach pain", "nausea", "abdominal", "cramps"),
        "Dermatological": ("rash", "itching", "skin", "blister", "hives", "lesion"),
        "Musculoskeletal/Other": ("joint pain", "back pain", "knee swelling", "headache", "sore muscles"),
    }
    scores = {
        category: sum(phrase in normalized for phrase in phrases)
        for category, phrases in keywords.items()
    }
    return max(scores, key=scores.get)


def _predict(text: str) -> tuple[str, str]:
    global MODEL, TOKENIZER, MODEL_LABELS
    adapter_dir = next(
        (path for path in ADAPTER_CANDIDATES if (path / "adapter_config.json").is_file()),
        None,
    )
    if adapter_dir is not None:
        try:
            if MODEL is None or TOKENIZER is None:
                import torch
                from peft import PeftModel
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                base_name = "emilyalsentzer/Bio_ClinicalBERT"
                label2id = {category: index for index, category in enumerate(CATEGORIES)}
                id2label = {index: category for category, index in label2id.items()}
                tokenizer = AutoTokenizer.from_pretrained(base_name, local_files_only=True)
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    base_name,
                    num_labels=len(CATEGORIES),
                    id2label=id2label,
                    label2id=label2id,
                    local_files_only=True,
                )
                MODEL = PeftModel.from_pretrained(base_model, str(adapter_dir))
                MODEL.eval()
                TOKENIZER = tokenizer
                MODEL_LABELS = id2label

            import torch

            inputs = TOKENIZER(
                text, truncation=True, padding=True, max_length=96, return_tensors="pt"
            )
            with torch.no_grad():
                prediction_id = int(MODEL(**inputs).logits.argmax(dim=-1).item())
            return MODEL_LABELS[prediction_id], "Bio_ClinicalBERT + LoRA"
        except Exception as error:
            logging.exception("Could not load or run the Phase 3 LoRA classifier")
            raise RuntimeError(
                "Phase 3 adapter was found, but loading or inference failed: "
                f"{error}"
            ) from error

    return _keyword_prediction(text), "Keyword demo (Phase 3 adapter unavailable)"


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EpidemicWatch | Syndromic Surveillance</title>
<style>
:root{--navy:#18232d;--ink:#17212b;--muted:#687684;--blue:#5587c9;--paper:#f4f6f9;--card:#fff;--green:#39a767;--red:#cf252e;--orange:#e88919;--line:#e4e8ed}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.4 Inter,Segoe UI,Arial,sans-serif}
.app{min-height:100vh;display:grid;grid-template-columns:260px minmax(0,1fr)}
aside{background:linear-gradient(170deg,var(--navy),#101820);color:white;padding:0 12px 14px;min-height:100vh}
.brand{height:38px;border-bottom:1px solid #39444e;display:flex;align-items:center;gap:7px;font-size:10px;font-weight:700;letter-spacing:.4px;white-space:nowrap}
.brand svg{color:#a9bfce}
.side-content{padding-top:11px}
.side-content h2{font-size:17px;margin:0 0 8px}
.field{margin:9px 0}
.field label{display:block;font-size:12px;margin:0 0 4px;color:#f4f5f6}
select,input,textarea,button{font:inherit}
select,input[type=date],textarea{width:100%;min-width:0;border:1px solid #d4dbe2;border-radius:6px;padding:7px;background:white;color:#202a34}
.date-range{display:grid;grid-template-columns:minmax(0,1fr);gap:5px}
.date-range input[type=date]{display:block;width:100%;min-width:0;padding:6px 8px;font-size:12px}
textarea{height:62px;resize:vertical}
.hint{font-size:10px;line-height:1.35;color:#afbdc8;margin-top:5px}
.predict-result{font-size:12px;margin-top:6px}
.predict-result strong{color:#abdbff}
.main{min-width:0;min-height:100vh;display:flex;flex-direction:column}
.topbar{height:36px;background:#fff;border-bottom:1px solid var(--line);display:flex;align-items:center;padding:0 12px;box-shadow:0 1px 4px #17212b0a}
.menu{border:0;background:none;color:#69747e;font-size:19px;cursor:pointer;padding:0 5px}
.content{flex:1;display:flex;flex-direction:column;width:100%;padding:11px 13px 14px;max-width:1450px;margin:0 auto}
.section-title{font-size:19px;margin:0 0 8px}
.cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-bottom:8px}
.card{background:var(--card);border:1px solid #e8ebef;border-radius:8px;box-shadow:0 2px 7px #17212b10;min-height:78px;padding:9px 12px;display:flex;align-items:center;gap:10px}
.icon{font-size:29px;line-height:1}
.label{font-size:12px;color:#111;margin-bottom:2px}
.value{font-size:24px;font-weight:700;line-height:1.1}
.value.alert{color:var(--red)}.value.elevated{color:var(--orange)}.value.normal{color:var(--green)}
.subvalue{font-size:10px;color:#65717c;margin-top:2px}
.workspace{flex:1;min-height:0;display:grid;grid-template-columns:minmax(0,1fr) 222px;grid-template-rows:minmax(0,1fr);gap:8px;align-items:stretch}
.workspace.overview-mode{grid-template-columns:repeat(2,minmax(0,1fr));grid-auto-rows:min-content;align-content:start}
.overview{display:contents}
.overview-playback{grid-column:1/-1}
.overview-playback h3{margin:0 0 4px}
.workspace.overview-mode .network{align-self:stretch}
.overview-card{display:block;width:100%;text-align:left;color:inherit;cursor:pointer;transition:box-shadow .15s,border-color .15s}
.overview-card:hover,.overview-card:focus-visible{border-color:#7ba3d2;box-shadow:0 3px 12px #17212b20;outline:none}
.overview-heading{display:flex;justify-content:space-between;align-items:center;gap:6px;margin-bottom:3px}
.overview-heading h3{margin:0}
.overview-chart{width:100%;height:180px}
.overview-chart svg{display:block;width:100%;height:100%}
.overview-hint{font-size:10px;color:var(--muted);margin-top:3px}
.view-all-button{margin-left:auto;border:1px solid #d4dbe2;border-radius:5px;background:#f8fafc;color:var(--ink);padding:4px 8px;font-size:11px;cursor:pointer}
.panel{background:white;border:1px solid #e7eaee;border-radius:8px;box-shadow:0 2px 7px #17212b10;padding:9px 10px;min-width:0}
.workspace>#detailPanel{display:flex;flex-direction:column}
.chart-heading{display:flex;align-items:center;gap:8px;margin-bottom:5px}
.chart-heading h3{margin:0}
.panel h3{font-size:12px;font-weight:500;margin:0 0 5px}
.playback{display:grid;grid-template-columns:auto minmax(100px,1fr) auto;align-items:center;gap:8px;margin:0 0 5px}
.playback button,.playback select{border:1px solid #d4dbe2;border-radius:5px;background:#f8fafc;color:var(--ink);padding:4px 8px;font-size:11px}
.playback button{cursor:pointer;min-width:58px}
.playback select{width:auto}
.playback-date{font-size:10px;color:var(--muted);white-space:nowrap}
.timeline-slider{grid-column:1/-1;width:100%;margin:0;accent-color:var(--blue)}
.chart-wrap{position:relative;flex:1;min-height:290px;width:100%;height:auto}
.chart-wrap svg{width:100%;height:100%;display:block;overflow:hidden}
.network{align-self:start;padding:9px}
.network table{width:100%;border-collapse:collapse;font-size:11px;border:1px solid #dce1e7;border-radius:6px;overflow:hidden}
.network th{background:#f0f2f5;text-align:left;padding:6px 5px;font-weight:500}
.network td{padding:6px 5px;border-top:1px solid #e6e9ed}
.status{display:inline-flex;align-items:center;gap:5px}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block}
.dot.Normal{background:var(--green)}.dot.Elevated{background:var(--orange)}.dot.Alert{background:var(--red)}
.legend{display:flex;flex-wrap:wrap;gap:9px;font-size:10px;color:#48535d;margin-top:3px}
.legend span{display:flex;align-items:center;gap:4px}
.legend i{width:9px;height:9px;border-radius:50%;display:inline-block}
.notice{margin:6px 1px 0;color:#6c7781;font-size:10px}
.hidden{display:none!important}
.prediction-form button{margin-top:5px;border:0;border-radius:6px;padding:7px 10px;background:#d8eaf8;color:#133c5a;cursor:pointer;font-weight:600;font-size:12px}
.prediction-source{color:#acbac5;font-size:10px;margin-top:3px}
.period-label{font-size:10px;color:#63717d;margin-left:auto}
@media(max-width:900px){.app{grid-template-columns:230px minmax(0,1fr)}.workspace{grid-template-columns:minmax(0,1fr) minmax(170px,28%)}.chart-wrap{min-height:280px}.overview-chart{height:165px}}
@media(max-width:650px){.app{grid-template-columns:1fr}.main{min-height:100vh}aside{min-height:0}.side-content{display:grid;grid-template-columns:1fr 1fr;gap:0 10px}.side-content h2{grid-column:1/-1}.brand{height:38px}.cards{grid-template-columns:1fr}.card{min-height:68px}.chart-wrap{min-height:245px}.period-label{display:none}.workspace{flex:none}.workspace.overview-mode{grid-template-columns:1fr}.overview-chart{height:210px}}
</style>
</head>
<body>
<div class="app">
<aside id="sidebar">
 <div class="brand"><svg width="21" height="21" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M4 13a3 3 0 1 1 0-6 3 3 0 0 1 0 6Zm0 0v2a4 4 0 0 0 4 4h2V8a3 3 0 1 1 6 0v8a2 2 0 1 0 4 0v-3"/><circle cx="4" cy="10" r="1"/><circle cx="13" cy="8" r="1"/></svg> SYNDROMIC SURVEILLANCE SYSTEM</div>
 <div class="side-content">
  <h2>System Controls</h2>
  <div class="field"><label for="hospitalSelect">Select Hospital</label><select id="hospitalSelect"></select></div>
  <div class="field"><label>Simulation Time Period</label><div class="date-range"><input id="startDate" type="date" aria-label="Start date"><input id="endDate" type="date" aria-label="End date"></div></div>
  <div class="field"><label for="categorySelect">Syndrome Category</label><select id="categorySelect"></select></div>
  <div class="field"><label for="symptomText">Test Symptom Text (Demo)</label><form class="prediction-form" id="predictForm"><textarea id="symptomText" placeholder="Type sample description (e.g., 'cough, fever')."></textarea><button type="submit">Predict category</button></form><div class="predict-result">Predicted Syndrome Category: <strong id="prediction">Enter symptoms above</strong></div><div class="prediction-source" id="predictionSource"></div></div>
  <div class="hint">The timeline replays Phase 5's synthetic 90-day dataset; it is not a live patient feed.</div>
 </div>
</aside>
<main class="main">
 <div class="topbar"><button class="menu" id="menuButton" aria-label="Toggle sidebar">☰</button><span class="period-label" id="periodLabel"></span></div>
 <div class="content">
  <h1 class="section-title">Status Summary</h1>
  <section class="cards">
   <div class="card"><div class="icon" id="alertIcon">🚨</div><div><div class="label">Current Signal:</div><div class="value" id="statusValue">Loading</div></div></div>
   <div class="card"><div class="icon" style="color:#39a767">↗</div><div><div class="label">Selected Day Case Count:</div><div class="value" id="caseValue">—</div><div class="subvalue" id="caseTrend"></div></div></div>
   <div class="card"><div class="icon">⚠️</div><div><div class="label">Status Trend:</div><div class="value elevated" id="trendValue">—</div></div></div>
  </section>
  <section class="workspace" id="workspace">
   <div class="panel overview-playback" id="overviewPlayback"><h3>Play all hospital graphs</h3><div class="playback"><button id="overviewPlayButton" type="button">Play</button><span class="playback-date" id="overviewPlaybackDate">Latest simulated day</span><select id="overviewPlaybackSpeed" aria-label="Playback speed for all graphs"><option value="1">1 day / step</option><option value="4" selected>4 days / step</option></select><input class="timeline-slider" id="overviewTimelineSlider" type="range" min="0" max="89" value="89" aria-label="Simulated day for all graphs"></div></div>
   <div class="overview" id="overview"></div>
   <div class="panel" id="detailPanel"><div class="chart-heading"><h3 id="chartTitle">Daily Case Counts</h3><button class="view-all-button" id="viewAllButton" type="button">All hospital graphs</button></div><div class="playback"><button id="playButton" type="button">Play</button><span class="playback-date" id="playbackDate">Latest simulated day</span><select id="playbackSpeed" aria-label="Playback speed"><option value="1">1 day / step</option><option value="4" selected>4 days / step</option></select><input class="timeline-slider" id="timelineSlider" type="range" min="0" max="89" value="89" aria-label="Simulated day"></div><div class="chart-wrap" id="chart"></div><div class="legend"><span><i style="background:#5688ca"></i>Daily cases</span><span><i style="background:#244d7c"></i>Selected day</span><span><i style="background:#f2bd79"></i>True outbreak window</span><span><i style="background:#c51922"></i>CUSUM/EWMA alarm</span></div></div>
   <div class="panel network" id="networkPanel"><h3>District Hospital Network Status</h3><table><thead><tr><th>Hospital</th><th>Status</th></tr></thead><tbody id="networkRows"></tbody></table><div class="notice" id="networkNote"></div></div>
  </section>
 </div>
</main>
</div>
<script>
let appData;
let dashboardView="overview";
let playbackDay=null;
let playbackTimer=null;
let playbackPlaying=false;
const $=id=>document.getElementById(id);
function statusClass(status){return status.toLowerCase()}
function chooseHospital(){return appData.hospitals.find(h=>h.name===$("hospitalSelect").value)}
function getRange(){const first=new Date(appData.startDate+"T00:00:00"),last=new Date(appData.endDate+"T00:00:00");const start=new Date($("startDate").value+"T00:00:00"),end=new Date($("endDate").value+"T00:00:00");return {first,last,from:Math.max(0,Math.floor((start-first)/86400000)),to:Math.min(89,Math.ceil((end-first)/86400000))}}
function normalizeDateRange(changedId){
 const startInput=$("startDate"),endInput=$("endDate");
 let start=startInput.value||appData.startDate,end=endInput.value||appData.endDate;
 start=start<appData.startDate?appData.startDate:start>appData.endDate?appData.endDate:start;
 end=end<appData.startDate?appData.startDate:end>appData.endDate?appData.endDate:end;
 if(start>end){if(changedId==="startDate")end=start;else start=end}
 startInput.value=start;endInput.value=end;
}
function render(){
 if(!appData)return;
 const h=chooseHospital(),category=$("categorySelect").value,series=h.categories[category],range=getRange();
 const valid=range.to>=range.from,lo=valid?range.from:0,hi=valid?range.to:89;
 const currentDay=Math.max(0,Math.min(playbackDay??hi,hi));
 const observedSeries=series.slice(0,currentDay+1);
 const status=detectorStatus(observedSeries);
 const overviewMode=dashboardView==="overview";
 $("overview").classList.toggle("hidden",!overviewMode);
 $("overviewPlayback").classList.toggle("hidden",!overviewMode);
 $("detailPanel").classList.toggle("hidden",overviewMode);
 $("networkPanel").classList.toggle("hidden",false);
 $("workspace").classList.toggle("overview-mode",overviewMode);
 if(overviewMode)renderOverview(category,lo,hi,currentDay);
 $("chartTitle").textContent=h.name+": Daily "+category+" Case Counts";
 $("statusValue").textContent=status;$("statusValue").className="value "+statusClass(status);
 $("alertIcon").textContent=status==="Alert"?"🚨":status==="Elevated"?"⚠️":"✅";
 $("caseValue").textContent=series[currentDay];$("caseTrend").textContent=category+" cases on the selected simulated day";
 const trend=trendStatus(observedSeries);
 $("trendValue").textContent=trend;$("trendValue").className="value "+statusClass(trend);
 const start=new Date($("startDate").value+"T00:00:00"),end=new Date($("endDate").value+"T00:00:00");
 const selectedDays=valid?hi-lo+1:0;
 $("periodLabel").textContent=start.toLocaleDateString()+" — "+end.toLocaleDateString()+" · "+selectedDays+" simulated "+(selectedDays===1?"day":"days");
 renderPlayback(lo,hi,currentDay,series.length);
 renderTable(currentDay);
 if(!overviewMode)renderChart(h,category,series,lo,hi,currentDay);
}
function detectorStatus(series){
 const cusum=detect(series,"cusum").alarm,ewma=detect(series,"ewma").alarm;
 if(cusum!==null||ewma!==null)return "Alert";
 const base=series.slice(0,30),mean=base.reduce((a,b)=>a+b,0)/base.length;
 const sig=Math.max(Math.sqrt(base.reduce((a,b)=>a+(b-mean)**2,0)/base.length),Math.sqrt(Math.max(mean,.5)));
 return series.at(-1)>mean+2*sig?"Elevated":"Normal";
}
function trendStatus(series){
 if(detect(series,"cusum").alarm!==null||detect(series,"ewma").alarm!==null)return "Elevated";
 return detectorStatus(series);
}
function detect(series,method){
 const base=series.slice(0,30),mean=base.reduce((a,b)=>a+b,0)/base.length;
 const sigma=Math.max(Math.sqrt(base.reduce((a,b)=>a+(b-mean)**2,0)/base.length),Math.sqrt(Math.max(mean,.5)))+1e-9;
 let alarm=null,trace=[],s=0,z=mean;
 series.forEach((x,t)=>{
  if(method==="cusum"){s=Math.max(0,s+x-mean-.5*sigma);trace.push(s);if(alarm===null&&t>=30&&s>8*sigma)alarm=t}
  else{z=.2*x+.8*z;const ucl=mean+5*sigma*Math.sqrt((.2/1.8)*(1-Math.pow(.8,2*(t+1))));trace.push(z);if(alarm===null&&t>=30&&z>ucl)alarm=t}
 });
 return {alarm,trace};
}
function renderTable(currentDay){
 const category=$("categorySelect").value;
 const rows=appData.hospitals.map(h=>{const status=detectorStatus(h.categories[category].slice(0,currentDay+1));return `<tr><td>${h.name}</td><td><span class="status"><i class="dot ${status}"></i>${status}</span></td></tr>`}).join("");
 $("networkRows").innerHTML=rows;$("networkNote").textContent="Status for "+category+" · day "+(currentDay+1);
}
function renderOverview(category,from,to,currentDay){
 $("overview").innerHTML=appData.hospitals.map(h=>{
  const series=h.categories[category],observed=series.slice(0,currentDay+1);
  const status=detectorStatus(observed),cusum=detect(observed,"cusum").alarm,ewma=detect(observed,"ewma").alarm;
  const W=480,H=185,L=30,R=8,T=10,B=24,plotW=W-L-R,plotH=H-T-B;
  const days=Math.max(1,to-from),maxValue=Math.max(1,...series.slice(from,to+1));
  const x=day=>L+((day-from)/days)*plotW,y=value=>T+plotH-(value/maxValue)*plotH;
  const points=series.slice(from,Math.min(to,currentDay)+1).map((value,index)=>`${x(from+index)},${y(value)}`).join(" ");
  const outage=category===appData.outbreakCategory&&h.name===appData.outbreakHospital&&currentDay>appData.outbreakEnd;
  const outbreakStart=Math.max(from,appData.outbreakStart),outbreakEnd=Math.min(to,appData.outbreakEnd);
  const outbreak=outage&&outbreakStart<=outbreakEnd?`<rect x="${x(outbreakStart)}" y="${T}" width="${Math.max(2,x(outbreakEnd)-x(outbreakStart))}" height="${plotH}" fill="#f5b85d" opacity=".32"><title>True outbreak window</title></rect>`:"";
  const alarmMarks=[[cusum,"CUSUM"],[ewma,"EWMA"]].filter(([day])=>day!==null&&day>=from&&day<=Math.min(to,currentDay)).map(([day,label])=>`<circle cx="${x(day)}" cy="${y(series[day])-(label==="EWMA"?7:0)}" r="5" fill="#c51922" stroke="white" stroke-width="1.5"><title>${label} alarm, day ${day+1}</title></circle>`).join("");
  const line=points?`<polyline points="${points}" fill="none" stroke="#5688ca" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>`:"";
  return `<button class="panel overview-card" type="button" data-hospital="${h.name}" aria-label="Open ${h.name} detailed graph"><div class="overview-heading"><h3>${h.name}</h3><span class="status"><i class="dot ${status}"></i>${status}</span></div><div class="overview-chart"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${category} daily cases at ${h.name}"><line x1="${L}" y1="${T+plotH}" x2="${W-R}" y2="${T+plotH}" stroke="#dce2e8"/>${outbreak}${line}${alarmMarks}<circle cx="${x(currentDay)}" cy="${y(series[currentDay])}" r="4" fill="#244d7c" stroke="white" stroke-width="1.5"><title>Selected day ${currentDay+1}: ${series[currentDay]} cases</title></circle><text x="${L}" y="${H-5}" fill="#687684" font-size="10">Day ${from+1}</text><text x="${W-R}" y="${H-5}" text-anchor="end" fill="#687684" font-size="10">Day ${to+1}</text></svg></div><div class="overview-hint">Click to open detailed graph · ${series[currentDay]} cases on day ${currentDay+1}</div></button>`;
 }).join("");
}
function renderPlayback(from,to,currentDay,totalDays){
 for(const id of ["timelineSlider","overviewTimelineSlider"]){
  const slider=$(id);
  slider.min=String(from);slider.max=String(to);slider.value=String(Math.min(currentDay,to));
 }
 $("playButton").textContent=playbackPlaying?"Pause":"Play";
 $("overviewPlayButton").textContent=playbackPlaying?"Pause":"Play";
 $("playbackSpeed").value=$("overviewPlaybackSpeed").value;
 if(currentDay<from)$("playbackDate").textContent="Waiting for selected date range";
 else{
  const currentDate=new Date(appData.startDate+"T00:00:00");
  currentDate.setDate(currentDate.getDate()+currentDay);
  $("playbackDate").textContent=currentDate.toLocaleDateString()+" · day "+(currentDay+1)+"/"+totalDays;
 }
 $("overviewPlaybackDate").textContent=$("playbackDate").textContent;
}
function renderChart(h,category,series,from,to,currentDay){
 if(to<from||currentDay<from){$("chart").innerHTML="<p class='notice'>Waiting for the selected date range.</p>";return}
 const W=720,H=330,L=45,R=12,T=16,B=39,plotW=W-L-R,plotH=H-T-B;
 const visibleEnd=Math.min(to,currentDay),values=series.slice(from,visibleEnd+1),axisLength=to-from+1;
 const highestCount=Math.max(...series.slice(from,to+1)),tickStep=Math.max(1,Math.ceil(highestCount/5));
 const maxValue=tickStep*5,x=i=>L+(i/(Math.max(axisLength-1,1)))*plotW,y=v=>T+plotH-(v/maxValue)*plotH;
 const ticks=5,grid=Array.from({length:ticks+1},(_,i)=>{let val=maxValue*i/ticks,yy=y(val);return `<line x1="${L}" y1="${yy}" x2="${W-R}" y2="${yy}" stroke="#e5e8ec"/><text x="${L-8}" y="${yy+4}" text-anchor="end" fill="#606d78" font-size="10">${Math.round(val)}</text>`}).join("");
 const points=values.map((v,i)=>`${x(i)},${y(v)}`).join(" ");
 const outage=category===appData.outbreakCategory&&h.name===appData.outbreakHospital;
 let shade="";
 if(outage&&currentDay>appData.outbreakEnd){
  const a=Math.max(from,appData.outbreakStart),b=Math.min(to,appData.outbreakEnd);
  if(a<=b){
   const x0=x(a-from),x1=x(b-from),width=Math.max(2,x1-x0);
   const label=width>=100?`<text x="${(x0+x1)/2}" y="${T+25}" text-anchor="middle" font-size="11" fill="#333">True Outbreak Window</text>`:"";
   shade=`<rect x="${x0}" y="${T}" width="${width}" height="${plotH}" fill="#f5b85d" opacity=".32"/>${label}`;
  }
 }
 const alarmDots=[];
 const observedSeries=series.slice(0,currentDay+1);
 const c=detect(observedSeries,"cusum").alarm,e=detect(observedSeries,"ewma").alarm;
 for(const [day,label,offset] of [[c,"CUSUM",0],[e,"EWMA",8]])if(day!==null&&day>=from&&day<=to)alarmDots.push(`<circle cx="${x(day-from)}" cy="${y(series[day])-offset}" r="6.5" fill="#c51922" stroke="white" stroke-width="1.5"><title>${label} alarm, day ${day+1}</title></circle>`);
 const labelStep=Math.max(1,Math.ceil(axisLength/8));
 const labels=Array.from({length:axisLength},(_,i)=>i).filter(i=>i===0||i===axisLength-1||i%labelStep===0).map(i=>{const dt=new Date(appData.startDate+"T00:00:00");dt.setDate(dt.getDate()+from+i);return `<text x="${x(i)}" y="${H-16}" transform="rotate(-35 ${x(i)} ${H-16})" text-anchor="end" fill="#5e6a75" font-size="9">${dt.toLocaleDateString(undefined,{month:"short",day:"numeric"})}</text>`}).join("");
 const line=`<polyline points="${points}" fill="none" stroke="#5688ca" stroke-width="2.6" stroke-linejoin="round" stroke-linecap="round"/>`;
 const dailyPoints=values.map((value,index)=>{
  const day=from+index,dayDate=new Date(appData.startDate+"T00:00:00");
  dayDate.setDate(dayDate.getDate()+day);
  const label=dayDate.toLocaleDateString(undefined,{year:"numeric",month:"short",day:"numeric"});
  return `<circle cx="${x(index)}" cy="${y(value)}" r="2.1" fill="#5688ca" stroke="white" stroke-width=".6"><title>${label}: ${value} cases (day ${day+1})</title></circle>`;
 }).join("");
 const currentMarker=currentDay>=from&&currentDay<=to?`<circle cx="${x(currentDay-from)}" cy="${y(series[currentDay])}" r="5" fill="#244d7c" stroke="white" stroke-width="2"><title>Current simulated day: ${series[currentDay]} cases</title></circle>`:"";
 $("chart").innerHTML=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Daily ${category} cases at ${h.name}">${grid}${shade}<line x1="${L}" y1="${T+plotH}" x2="${W-R}" y2="${T+plotH}" stroke="#adb6bf"/>${line}${dailyPoints}${alarmDots.join("")}${currentMarker}${labels}<text x="12" y="${H/2}" transform="rotate(-90 12 ${H/2})" text-anchor="middle" fill="#586570" font-size="10">Cases per day</text></svg>`;
}
function stopPlayback(){
 if(playbackTimer!==null){clearInterval(playbackTimer);playbackTimer=null}
 playbackPlaying=false;
}
function advancePlayback(){
 const {to}=getRange();
 const next=Math.min(to,playbackDay+Number($("playbackSpeed").value));
 playbackDay=next;
 if(next>=to)stopPlayback();
 render();
}
function startPlayback(){
 const {from,to}=getRange();
 if(to<from)return;
 if(playbackDay===null||playbackDay>=to)playbackDay=from;
 playbackPlaying=true;
 render();
 playbackTimer=setInterval(advancePlayback,750);
}
async function loadData(){
 try{
  const response=await fetch("/api/data");if(!response.ok)throw new Error(await response.text());
  appData=await response.json();
  $("hospitalSelect").innerHTML=appData.hospitals.map(h=>`<option>${h.name}</option>`).join("");
  $("categorySelect").innerHTML=appData.categories.map(c=>`<option>${c}</option>`).join("");
  $("hospitalSelect").value=appData.outbreakHospital;$("categorySelect").value=appData.outbreakCategory;
  $("startDate").value=appData.startDate;$("startDate").min=appData.startDate;$("startDate").max=appData.endDate;
  $("endDate").value=appData.endDate;$("endDate").min=appData.startDate;$("endDate").max=appData.endDate;
  playbackDay=appData.hospitals[0].categories[appData.categories[0]].length-1;
  render();
 }catch(err){$("chart").textContent="Could not load dashboard data: "+err.message}
}
["hospitalSelect","categorySelect"].forEach(id=>$(id).addEventListener("change",render));
$("viewAllButton").addEventListener("click",()=>{dashboardView="overview";render()});
$("overview").addEventListener("click",event=>{
 const card=event.target.closest("[data-hospital]");
 if(!card)return;
 $("hospitalSelect").value=card.dataset.hospital;
 dashboardView="detail";
 render();
});
["startDate","endDate"].forEach(id=>$(id).addEventListener("change",()=>{stopPlayback();normalizeDateRange(id);playbackDay=getRange().from;render()}));
$("menuButton").addEventListener("click",()=>$("sidebar").classList.toggle("hidden"));
for(const id of ["playButton","overviewPlayButton"])$(id).addEventListener("click",()=>{
 if(playbackPlaying){stopPlayback();render()}
 else startPlayback();
});
for(const id of ["timelineSlider","overviewTimelineSlider"])$(id).addEventListener("input",event=>{
 stopPlayback();playbackDay=Number(event.target.value);render();
});
for(const id of ["playbackSpeed","overviewPlaybackSpeed"])$(id).addEventListener("change",event=>{
 const otherId=id==="playbackSpeed"?"overviewPlaybackSpeed":"playbackSpeed";
 $(otherId).value=event.target.value;
 if(playbackPlaying){stopPlayback();startPlayback()}
});
$("predictForm").addEventListener("submit",async event=>{
 event.preventDefault();const text=$("symptomText").value.trim();if(!text){$("prediction").textContent="Enter a symptom description";$("predictionSource").textContent="";return}
 $("prediction").textContent="Predicting…";$("predictionSource").textContent="";
 try{const response=await fetch("/api/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text})});const result=await response.json();if(!response.ok)throw new Error(result.error||"Prediction failed");$("prediction").textContent=result.category;$("predictionSource").textContent=result.source}
 catch(error){$("prediction").textContent="Prediction unavailable";$("predictionSource").textContent=error.message}
});
loadData();
</script>
</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/":
                self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/data":
                data = json.dumps(_dashboard_data()).encode("utf-8")
                self._send(data, "application/json; charset=utf-8")
            else:
                self._send(b"Not found", "text/plain; charset=utf-8", 404)
        except Exception as error:
            logging.exception("Dashboard request failed")
            self._send(
                json.dumps({"error": str(error)}).encode("utf-8"),
                "application/json; charset=utf-8",
                500,
            )

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/predict":
            self._send(b"Not found", "text/plain; charset=utf-8", 404)
            return
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            text = str(payload.get("text", "")).strip()
            if not text:
                raise ValueError("Provide a symptom description.")
            category, source = _predict(text)
            response = json.dumps({"category": category, "source": source}).encode("utf-8")
            self._send(response, "application/json; charset=utf-8")
        except Exception as error:
            logging.exception("Symptom prediction failed")
            self._send(
                json.dumps({"error": str(error)}).encode("utf-8"),
                "application/json; charset=utf-8",
                400,
            )

    def log_message(self, message: str, *args: object) -> None:
        logging.info("%s - %s", self.address_string(), message % args)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the EpidemicWatch browser dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"EpidemicWatch dashboard running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping EpidemicWatch dashboard.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
