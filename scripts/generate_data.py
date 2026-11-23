"""Generate a deterministic, fully synthetic (fake) EHR dataset with deliberate data-quality problems.

No real patient data is used anywhere. Output:
  data/raw/ehr_export.json      simulated Epic-style JSON export (notes + imaging reports)
  data/raw/lab_results.csv      lab results, one row per analyte
  data/raw/notes/*.txt          "scanned" free-text notes with a loose header block

Injected problems (each is handled + audited by the ingestion pipeline):
  * inconsistent date formats (ISO, US, 2-digit years, "14 Mar 1958", "19580314", ...)
  * inconsistent gender codes ("F", "female", "2", "M ", missing)
  * inconsistent MRN formats ("MRN-001001", "1001", "mrn 1001", "001001")
  * inconsistent name formats ("GONZALEZ, MARIA", "Maria Gonzalez")
  * exact and near-exact duplicate records
  * conflicting identifiers (typo'd MRN for the same person; name variant under the same MRN)
  * missing fields (no DOB, no gender, no MRN, no date in a note header)
  * one unrecoverable record (no MRN, no name/DOB) that must be rejected
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

# --------------------------------------------------------------------------------------
# Patient profiles (fictional)
# --------------------------------------------------------------------------------------
# Each lab: (panel, test, loinc, value, unit, ref_range, flag)
PATIENTS = [
    dict(
        mrn=1001, given="Maria", family="Gonzalez", dob=date(1958, 3, 14), sex="female",
        notes=[dict(kind="progress", date=date(2025, 2, 10), title="Endocrinology Progress Note",
            cc="Poorly controlled blood sugars and a non-healing wound on the left foot.",
            body="67-year-old female with type 2 diabetes mellitus for 15 years on metformin and glargine. "
                 "Reports polyuria and fatigue. Exam: 1.5 cm shallow ulcer on plantar surface of left first "
                 "metatarsal head, no purulence, monofilament sensation reduced bilaterally.",
            assessment=["Type 2 diabetes mellitus with hyperglycemia", "Diabetic foot ulcer, left foot",
                        "Diabetic peripheral neuropathy"],
            plan="Increase glargine to 28 units nightly, add empagliflozin, podiatry and wound care referral.")],
        imaging=[dict(date=date(2025, 2, 11), modality="X-ray", title="XR Left Foot 3 Views",
            findings="Soft tissue swelling adjacent to the first metatarsal head. No cortical erosion or periosteal reaction.",
            impression="No radiographic evidence of osteomyelitis.")],
        labs=[(date(2025, 2, 10), "Diabetes Panel", "Hemoglobin A1c", "4548-4", "9.4", "%", "4.0-5.6", "H"),
              (date(2025, 2, 10), "Diabetes Panel", "Glucose", "2345-7", "212", "mg/dL", "70-99", "H"),
              (date(2025, 2, 10), "Diabetes Panel", "Creatinine", "2160-0", "1.1", "mg/dL", "0.6-1.2", "")],
    ),
    dict(
        mrn=1002, given="James", family="Whitfield", dob=date(1949, 11, 2), sex="male",
        notes=[dict(kind="discharge", date=date(2025, 3, 18), title="Discharge Summary",
            cc="Worsening shortness of breath and leg swelling for one week.",
            body="76-year-old male with heart failure with reduced ejection fraction (EF 30%) admitted with acute "
                 "decompensated heart failure. Diuresed with IV furosemide, net negative 4.2 L. Weight down 3.8 kg.",
            assessment=["Acute on chronic systolic heart failure", "Hyponatremia", "Hypertension"],
            plan="Furosemide 40 mg twice daily, continue carvedilol and sacubitril-valsartan. Daily weights. "
                 "Cardiology follow-up in 7 days.")],
        imaging=[dict(date=date(2025, 3, 15), modality="X-ray", title="XR Chest 2 Views",
            findings="Cardiomegaly. Bilateral interstitial opacities and small bilateral pleural effusions.",
            impression="Findings consistent with pulmonary edema and cardiomegaly.")],
        labs=[(date(2025, 3, 15), "Cardiac Panel", "B-type natriuretic peptide", "30934-4", "1250", "pg/mL", "0-100", "H"),
              (date(2025, 3, 15), "Basic Metabolic Panel", "Sodium", "2951-2", "131", "mmol/L", "135-145", "L"),
              (date(2025, 3, 15), "Basic Metabolic Panel", "Potassium", "2823-3", "4.8", "mmol/L", "3.5-5.1", "")],
    ),
    dict(
        mrn=1003, given="Aisha", family="Rahman", dob=date(1990, 7, 22), sex="female",
        notes=[dict(kind="ed", date=date(2025, 1, 6), title="Emergency Department Note",
            cc="Fever, productive cough and right-sided pleuritic chest pain for 3 days.",
            body="34-year-old female, temperature 39.1 C, HR 108, SpO2 93% on room air. Crackles over the right "
                 "lower lung field. No recent travel or hospitalization.",
            assessment=["Community-acquired pneumonia, right lower lobe", "Sepsis ruled out"],
            plan="Amoxicillin-clavulanate plus azithromycin for 5 days, return precautions, PCP follow-up in 1 week.")],
        imaging=[dict(date=date(2025, 1, 6), modality="X-ray", title="XR Chest 2 Views",
            findings="Dense consolidation in the right lower lobe with air bronchograms. No pleural effusion.",
            impression="Right lower lobe pneumonia.")],
        labs=[(date(2025, 1, 6), "CBC", "White blood cell count", "6690-2", "15.2", "10*3/uL", "4.0-11.0", "H"),
              (date(2025, 1, 6), "Inflammatory Markers", "C-reactive protein", "1988-5", "88", "mg/L", "0-10", "H")],
    ),
    dict(
        mrn=1004, given="Robert", family="Chen", dob=date(1962, 1, 30), sex="male",
        notes=[dict(kind="discharge", date=date(2025, 4, 2), title="Discharge Summary",
            cc="Increased dyspnea and sputum production.",
            body="63-year-old male former smoker (40 pack-years) with COPD admitted for acute exacerbation. Treated "
                 "with nebulized bronchodilators, prednisone and doxycycline. CT obtained to exclude pulmonary embolism.",
            assessment=["COPD with acute exacerbation", "Incidental 8 mm pulmonary nodule, right upper lobe"],
            plan="Prednisone taper, tiotropium. Repeat CT chest in 3 months for nodule follow-up per Fleischner guidance.")],
        imaging=[dict(date=date(2025, 3, 30), modality="CT", title="CT Chest Angiogram",
            findings="No pulmonary embolism. Centrilobular emphysema, upper lobe predominant. 8 mm solid nodule in the right upper lobe.",
            impression="No PE. Emphysema. 8 mm indeterminate right upper lobe nodule; follow-up CT recommended.")],
        labs=[(date(2025, 3, 30), "Arterial Blood Gas", "pCO2 arterial", "2019-8", "52", "mm[Hg]", "35-45", "H"),
              (date(2025, 3, 30), "Arterial Blood Gas", "pH arterial", "2744-1", "7.33", "[pH]", "7.35-7.45", "L")],
    ),
    dict(
        mrn=1005, given="Linda", family="Okafor", dob=date(1971, 9, 5), sex="female",
        notes=[dict(kind="consult", date=date(2025, 5, 20), title="Breast Surgery Consultation",
            cc="Palpable lump in the left breast noticed 4 weeks ago.",
            body="53-year-old female, no family history of breast cancer. Exam: 2 cm firm, mobile mass at the left "
                 "breast 2 o'clock position, no axillary lymphadenopathy.",
            assessment=["Left breast mass, BI-RADS 4 on imaging"],
            plan="Ultrasound-guided core needle biopsy scheduled. Discussed possible outcomes."),
               dict(kind="referral", date=date(2025, 5, 12), title="Scanned Referral Letter",
            cc="Referral for evaluation of a left breast mass.",
            body="Dear colleague, I am referring this patient who presented with a self-detected left breast lump. "
                 "Screening mammogram one year ago was normal. Please evaluate.",
            assessment=["Left breast mass"], plan="Surgical evaluation requested.")],
        imaging=[dict(date=date(2025, 5, 15), modality="Mammography", title="Diagnostic Mammogram and Ultrasound, Left",
            findings="Irregular 1.9 cm hypoechoic mass with indistinct margins at 2 o'clock, 6 cm from the nipple.",
            impression="BI-RADS 4C: high suspicion for malignancy. Tissue sampling recommended.")],
        labs=[(date(2025, 5, 20), "CBC", "Hemoglobin", "718-7", "13.1", "g/dL", "12.0-15.5", "")],
    ),
    dict(
        mrn=1006, given="David", family="Miller", dob=date(1985, 12, 11), sex="male",
        notes=[dict(kind="ed", date=date(2025, 6, 3), title="Emergency Department Note",
            cc="Right wrist pain after a fall on an outstretched hand.",
            body="39-year-old male fell from a ladder. Swelling and tenderness of the distal right radius, "
                 "neurovascularly intact.",
            assessment=["Closed fracture of the distal right radius"],
            plan="Closed reduction, sugar-tong splint, orthopedics follow-up in 5 days.")],
        imaging=[dict(date=date(2025, 6, 3), modality="X-ray", title="XR Right Wrist 3 Views",
            findings="Transverse fracture of the distal radial metaphysis with 15 degrees of dorsal angulation.",
            impression="Distal radius fracture (Colles type).")],
        labs=[],
    ),
    dict(
        mrn=1007, given="Patricia", family="Nguyen", dob=date(1953, 4, 18), sex="female",
        notes=[dict(kind="consult", date=date(2025, 7, 8), title="Nephrology Consultation",
            cc="Rising creatinine noted by primary care.",
            body="72-year-old female with long-standing hypertension and type 2 diabetes. On lisinopril and "
                 "spironolactone. No NSAID use.",
            assessment=["Chronic kidney disease stage 3b", "Hyperkalemia", "Hypertension"],
            plan="Stop spironolactone, low-potassium diet, repeat BMP in 1 week.")],
        imaging=[dict(date=date(2025, 7, 10), modality="Ultrasound", title="US Renal Bilateral",
            findings="Right kidney 9.1 cm, left kidney 9.4 cm with increased cortical echogenicity. No hydronephrosis.",
            impression="Findings consistent with medical renal disease. No obstruction.")],
        labs=[(date(2025, 7, 8), "Renal Panel", "Creatinine", "2160-0", "1.6", "mg/dL", "0.5-1.1", "H"),
              (date(2025, 7, 8), "Renal Panel", "eGFR", "62238-1", "38", "mL/min/1.73m2", ">60", "L"),
              (date(2025, 7, 8), "Renal Panel", "Potassium", "2823-3", "5.9", "mmol/L", "3.5-5.1", "H")],
    ),
    dict(
        mrn=1008, given="Kevin", family="O'Brien", dob=date(2001, 6, 27), sex="male",
        notes=[dict(kind="ed", date=date(2025, 8, 14), title="Emergency Department Note",
            cc="Wheezing and chest tightness after exposure to wildfire smoke.",
            body="24-year-old male with mild persistent asthma. Peak flow 55% of predicted, improved to 80% after "
                 "three albuterol-ipratropium nebulizers.",
            assessment=["Asthma exacerbation, moderate"],
            plan="Prednisone 40 mg for 5 days, start inhaled budesonide-formoterol, asthma action plan reviewed.")],
        imaging=[dict(date=date(2025, 8, 14), modality="X-ray", title="XR Chest 1 View",
            findings="Mild hyperinflation. No focal consolidation or pneumothorax.",
            impression="No acute cardiopulmonary process.")],
        labs=[(date(2025, 8, 14), "CBC", "Eosinophils", "26449-9", "0.6", "10*3/uL", "0.0-0.5", "H")],
    ),
    dict(
        mrn=1009, given="Sandra", family="Williams", dob=date(1966, 2, 9), sex="female",
        notes=[dict(kind="progress", date=date(2025, 9, 2), title="Primary Care Progress Note",
            cc="Headaches and elevated home blood pressure readings.",
            body="59-year-old female, BP 168/98 and 164/96 on repeat. Morning occipital headaches for 2 weeks. "
                 "No focal neurologic deficits.",
            assessment=["Uncontrolled essential hypertension", "Hyperlipidemia"],
            plan="Add amlodipine 5 mg, start atorvastatin 40 mg, recheck BP in 2 weeks.")],
        imaging=[dict(date=date(2025, 9, 3), modality="CT", title="CT Head without Contrast",
            findings="No intracranial hemorrhage, mass effect or acute infarct.",
            impression="Negative non-contrast head CT.")],
        labs=[(date(2025, 9, 2), "Lipid Panel", "LDL cholesterol", "13457-7", "182", "mg/dL", "<100", "H"),
              (date(2025, 9, 2), "Lipid Panel", "Total cholesterol", "2093-3", "268", "mg/dL", "<200", "H")],
    ),
    dict(
        mrn=1010, given="Thomas", family="Anderson", dob=date(1944, 8, 15), sex="male",
        notes=[dict(kind="discharge", date=date(2025, 10, 12), title="Discharge Summary",
            cc="Sudden right-sided weakness and difficulty speaking.",
            body="81-year-old male, last known well 90 minutes before arrival, NIHSS 9. Received IV tenecteplase. "
                 "Telemetry showed paroxysmal atrial fibrillation.",
            assessment=["Acute ischemic stroke, left MCA territory", "Paroxysmal atrial fibrillation", "Expressive aphasia"],
            plan="Start apixaban on day 4, atorvastatin 80 mg, inpatient rehabilitation, speech therapy.")],
        imaging=[dict(date=date(2025, 10, 8), modality="MRI", title="MRI Brain without Contrast",
            findings="Restricted diffusion in the left frontal operculum and insula consistent with acute infarct.",
            impression="Acute left MCA territory infarct. No hemorrhagic transformation.")],
        labs=[(date(2025, 10, 8), "Coagulation", "INR", "6301-6", "1.0", "", "0.8-1.2", ""),
              (date(2025, 10, 8), "Chemistry", "Glucose", "2345-7", "148", "mg/dL", "70-99", "H")],
    ),
    dict(
        mrn=1011, given="Grace", family="Kim", dob=date(1995, 10, 3), sex="female",
        notes=[dict(kind="progress", date=date(2025, 4, 22), title="Primary Care Progress Note",
            cc="Fatigue and shortness of breath on exertion for 2 months.",
            body="29-year-old female with heavy menstrual bleeding. Conjunctival pallor. No GI bleeding symptoms.",
            assessment=["Iron deficiency anemia", "Menorrhagia"],
            plan="Oral ferrous sulfate every other day, pelvic ultrasound, gynecology referral.")],
        imaging=[dict(date=date(2025, 4, 29), modality="Ultrasound", title="US Pelvis Transvaginal",
            findings="Enlarged uterus with two intramural fibroids measuring 3.2 cm and 2.1 cm.",
            impression="Uterine fibroids, likely cause of menorrhagia.")],
        labs=[(date(2025, 4, 22), "CBC", "Hemoglobin", "718-7", "8.9", "g/dL", "12.0-15.5", "L"),
              (date(2025, 4, 22), "Iron Studies", "Ferritin", "2276-4", "6", "ng/mL", "15-150", "L")],
    ),
    dict(
        mrn=1012, given="Michael", family="Brown", dob=date(1978, 5, 21), sex="male",
        notes=[dict(kind="discharge", date=date(2025, 11, 4), title="Discharge Summary",
            cc="Periumbilical pain migrating to the right lower quadrant with nausea.",
            body="47-year-old male, McBurney point tenderness, temperature 38.2 C. Underwent uncomplicated "
                 "laparoscopic appendectomy on hospital day 1.",
            assessment=["Acute appendicitis, non-perforated", "Status post laparoscopic appendectomy"],
            plan="Advance diet as tolerated, no heavy lifting for 2 weeks, surgical clinic follow-up.")],
        imaging=[dict(date=date(2025, 11, 2), modality="CT", title="CT Abdomen and Pelvis with Contrast",
            findings="Dilated 11 mm appendix with wall enhancement and periappendiceal fat stranding. No abscess.",
            impression="Acute uncomplicated appendicitis.")],
        labs=[(date(2025, 11, 2), "CBC", "White blood cell count", "6690-2", "13.8", "10*3/uL", "4.0-11.0", "H")],
    ),
    dict(
        mrn=1013, given="Elena", family="Petrova", dob=date(1960, 12, 29), sex="female",
        notes=[dict(kind="consult", date=date(2025, 6, 17), title="Cardiology Consultation",
            cc="Palpitations and irregular heartbeat.",
            body="64-year-old female with new-onset atrial fibrillation, rate 124. CHA2DS2-VASc score 3. "
                 "Weight loss and heat intolerance reported.",
            assessment=["New-onset atrial fibrillation with rapid ventricular response", "Suspected hyperthyroidism"],
            plan="Metoprolol for rate control, start apixaban, endocrinology referral for thyroid workup.")],
        imaging=[dict(date=date(2025, 6, 18), modality="Echocardiogram", title="Transthoracic Echocardiogram",
            findings="Left ventricular ejection fraction 55%. Moderate left atrial enlargement. No significant valve disease.",
            impression="Preserved LV function with moderate left atrial enlargement.")],
        labs=[(date(2025, 6, 17), "Thyroid", "TSH", "3016-3", "0.05", "m[IU]/L", "0.4-4.0", "L"),
              (date(2025, 6, 17), "Thyroid", "Free T4", "3024-7", "3.1", "ng/dL", "0.8-1.8", "H")],
    ),
    dict(
        mrn=1014, given="Samuel", family="Johnson", dob=date(1937, 3, 3), sex="male",
        notes=[dict(kind="progress", date=date(2025, 8, 26), title="Geriatric Psychiatry Follow-up",
            cc="Low mood, poor appetite and memory complaints.",
            body="88-year-old male, PHQ-9 score 17, MoCA 24/30. Lives alone since spouse died last year.",
            assessment=["Major depressive disorder, moderate", "Mild cognitive impairment", "Vitamin B12 deficiency"],
            plan="Start sertraline 25 mg, B12 replacement, home health social work referral.")],
        imaging=[],
        labs=[(date(2025, 8, 26), "Vitamins", "Vitamin B12", "2132-9", "160", "pg/mL", "200-900", "L"),
              (date(2025, 8, 26), "Thyroid", "TSH", "3016-3", "2.1", "m[IU]/L", "0.4-4.0", "")],
    ),
    dict(
        mrn=1015, given="Fatima", family="Hassan", dob=date(1982, 1, 17), sex="female",
        notes=[dict(kind="consult", date=date(2025, 3, 5), title="Rheumatology Consultation",
            cc="Morning stiffness and swelling of both hands for 4 months.",
            body="43-year-old female, morning stiffness lasting 90 minutes. Synovitis of bilateral MCP 2-3 and wrists.",
            assessment=["Seropositive rheumatoid arthritis"],
            plan="Start methotrexate 15 mg weekly with folic acid, short prednisone bridge, baseline hepatitis panel.")],
        imaging=[dict(date=date(2025, 3, 5), modality="X-ray", title="XR Bilateral Hands",
            findings="Periarticular osteopenia and early marginal erosions at the second MCP joints bilaterally.",
            impression="Findings compatible with erosive inflammatory arthritis.")],
        labs=[(date(2025, 3, 5), "Rheumatology", "Rheumatoid factor", "11572-5", "86", "[IU]/mL", "<14", "H"),
              (date(2025, 3, 5), "Inflammatory Markers", "C-reactive protein", "1988-5", "24", "mg/L", "0-10", "H")],
    ),
]

NOTE_KIND_LABEL = {
    "progress": "Progress Note", "discharge": "Discharge Summary", "ed": "Emergency Department Note",
    "consult": "Consultation Note", "referral": "Referral Letter",
}

# --------------------------------------------------------------------------------------
# Messiness helpers (deterministic: chosen by index so the dataset is reproducible)
# --------------------------------------------------------------------------------------
DATE_STYLES = [
    lambda d: d.isoformat(),                  # 2025-02-10
    lambda d: d.strftime("%m/%d/%Y"),         # 02/10/2025
    lambda d: d.strftime("%d %b %Y"),         # 10 Feb 2025
    lambda d: d.strftime("%B %d, %Y"),        # February 10, 2025
    lambda d: d.strftime("%Y%m%d"),           # 20250210
    lambda d: d.strftime("%m/%d/%y"),         # 02/10/25 (2-digit year)
    lambda d: d.strftime("%Y-%m-%dT%H:%M:%SZ").replace("T00:00:00Z", "T09:30:00Z"),
]
GENDER_STYLES = {
    "female": ["F", "female", "2", "Female ", "f", "FEMALE"],
    "male": ["M", "male", "1", "Male", "m ", "MALE"],
}
MRN_STYLES = [
    lambda n: f"MRN-{n:06d}", lambda n: str(n), lambda n: f"{n:06d}",
    lambda n: f"mrn {n}", lambda n: f" {n} ", lambda n: f"MRN{n:08d}",
]


def fmt_date(d: date, i: int) -> str:
    return DATE_STYLES[i % len(DATE_STYLES)](d)


def fmt_gender(sex: str, i: int) -> str:
    opts = GENDER_STYLES[sex]
    return opts[i % len(opts)]


def fmt_mrn(n: int, i: int) -> str:
    return MRN_STYLES[i % len(MRN_STYLES)](n)


def fmt_name(p: dict, i: int) -> str:
    if i % 3 == 0:
        return f"{p['family'].upper()}, {p['given'].upper()}"
    if i % 3 == 1:
        return f"{p['given']} {p['family']}"
    return f"{p['family']}, {p['given']}"


def note_text(n: dict) -> str:
    lines = [f"Chief Complaint: {n['cc']}", "", f"History: {n['body']}", "", "Assessment:"]
    lines += [f"{k}. {a}" for k, a in enumerate(n["assessment"], 1)]
    lines += ["", f"Plan: {n['plan']}"]
    return "\n".join(lines)


def imaging_text(im: dict) -> str:
    return f"Exam: {im['title']}\nFindings: {im['findings']}\nImpression: {im['impression']}"


# --------------------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------------------
def build_json_export() -> list[dict]:
    """Notes for patients 1001-1008 and all imaging reports, in two inconsistent JSON shapes."""
    records: list[dict] = []
    i = 0
    for p in PATIENTS:
        docs = []
        if p["mrn"] <= 1008:
            docs += [("note", n) for n in p["notes"] if n["kind"] != "referral"]
        docs += [("imaging", im) for im in p["imaging"]]
        for kind, d in docs:
            i += 1
            enc = f"ENC-{p['mrn']}-{d['date'].strftime('%Y%m%d')}"
            text = note_text(d) if kind == "note" else imaging_text(d)
            doc_type = NOTE_KIND_LABEL[d["kind"]] if kind == "note" else f"Imaging Report - {d['modality']}"
            if i % 2 == 0:  # shape A: nested patient object
                rec = {
                    "record_id": f"EPIC-{i:04d}",
                    "patient": {"mrn": fmt_mrn(p["mrn"], i), "name": fmt_name(p, i),
                                "dob": fmt_date(p["dob"], i), "sex": fmt_gender(p["sex"], i)},
                    "encounter_id": enc,
                    "document_type": doc_type,
                    "title": d["title"],
                    "service_date": fmt_date(d["date"], i + 1),
                    "content": text,
                }
            else:  # shape B: flat keys with different names
                rec = {
                    "id": f"EPIC-{i:04d}",
                    "patient_id": fmt_mrn(p["mrn"], i),
                    "first_name": p["given"], "last_name": p["family"],
                    "birth_date": fmt_date(p["dob"], i),
                    "gender": fmt_gender(p["sex"], i),
                    "encounter": enc,
                    "type": doc_type,
                    "description": d["title"],
                    "date": fmt_date(d["date"], i + 2),
                    "text": text,
                }
            records.append(rec)

    # --- injected problems -------------------------------------------------------------
    # 1. exact duplicate of a record
    records.append(json.loads(json.dumps(records[0])))
    # 2. near duplicate: same content with different whitespace and a new id
    nd = json.loads(json.dumps(records[3]))
    nd_key = "content" if "content" in nd else "text"
    nd[nd_key] = "  " + nd[nd_key].replace("\n", "\n\n") + "  "
    if "record_id" in nd:
        nd["record_id"] = "EPIC-9001"
    else:
        nd["id"] = "EPIC-9001"
    records.append(nd)
    # 3. name variant under the same MRN (conflicting demographics -> majority wins)
    jw = next(r for r in records if "Whitfield" in json.dumps(r))
    jw_var = json.loads(json.dumps(jw))
    if "patient" in jw_var:
        jw_var["patient"]["name"] = "Jim Whitfield"
        jw_var["record_id"] = "EPIC-9002"
    else:
        jw_var["first_name"] = "Jim"
        jw_var["id"] = "EPIC-9002"
    jw_var["title" if "patient" in jw_var else "description"] = "Cardiology Telephone Note"
    jw_var["document_type" if "patient" in jw_var else "type"] = "Progress Note"
    jw_var.pop("encounter_id", None)
    jw_var.pop("encounter", None)
    body_key = "content" if "content" in jw_var else "text"
    jw_var[body_key] = ("Chief Complaint: Follow-up call after discharge.\n\nHistory: Patient reports weight stable, "
                        "mild ankle swelling, no orthopnea. Adherent to furosemide.\n\nAssessment:\n1. Heart failure "
                        "with reduced ejection fraction, compensated\n\nPlan: Continue current regimen.")
    jw_var["service_date" if "service_date" in jw_var else "date"] = "03/25/2025"
    records.append(jw_var)
    # 4. missing DOB and gender for one Aisha Rahman record
    ar = next(r for r in records if "Rahman" in json.dumps(r))
    if "patient" in ar:
        ar["patient"].pop("dob", None)
        ar["patient"].pop("sex", None)
    else:
        ar.pop("birth_date", None)
        ar.pop("gender", None)
    # 5. unrecoverable: no MRN, no name, no DOB
    records.append({"id": "EPIC-9999", "type": "Progress Note", "date": "2025-05-05",
                    "text": "Chief Complaint: Unknown. Fax page 2 of 3, header missing."})
    return records


def build_lab_rows() -> list[dict]:
    rows: list[dict] = []
    i = 0
    for p in PATIENTS:
        for (d, panel, test, loinc, value, unit, ref, flag) in p["labs"]:
            i += 1
            mrn = p["mrn"]
            if mrn == 1007:
                mrn = 1070  # conflicting identifier: transposed-digit MRN, same name + DOB
            rows.append({
                "MRN": fmt_mrn(mrn, i), "Patient Name": fmt_name(p, i + 1),
                "DOB": fmt_date(p["dob"], i + 3), "Sex": fmt_gender(p["sex"], i + 1),
                "Collected": fmt_date(d, i), "Panel": panel, "Test": test, "LOINC": loinc,
                "Result": value, "Units": unit, "Ref Range": ref, "Flag": flag,
            })
    rows.append(dict(rows[0]))  # exact duplicate row
    rows.append(dict(rows[5]))  # another duplicate
    # missing MRN but recoverable via name + DOB
    rows.append({"MRN": "", "Patient Name": "Grace Kim", "DOB": "10/03/1995", "Sex": "F",
                 "Collected": "2025-04-22", "Panel": "Iron Studies", "Test": "Transferrin saturation",
                 "LOINC": "2502-3", "Result": "7", "Units": "%", "Ref Range": "20-50", "Flag": "L"})
    return rows


def build_txt_notes() -> dict[str, str]:
    files: dict[str, str] = {}
    i = 0
    for p in PATIENTS:
        notes = [n for n in p["notes"] if p["mrn"] > 1008 or n["kind"] == "referral"]
        for n in notes:
            i += 1
            header = [
                f"PATIENT: {fmt_name(p, i)}",
                f"MRN: {fmt_mrn(p['mrn'], i + 2)}",
                f"DOB: {fmt_date(p['dob'], i + 4)}",
                f"SEX: {fmt_gender(p['sex'], i + 2)}",
                f"DATE OF SERVICE: {fmt_date(n['date'], i)}",
                f"DOCUMENT TYPE: {NOTE_KIND_LABEL[n['kind']]}",
                f"TITLE: {n['title']}",
            ]
            if p["mrn"] == 1014:
                # missing service date in header; the date only appears inside the narrative
                header = [h for h in header if not h.startswith("DATE OF SERVICE")]
                body = f"Seen in clinic on {n['date'].strftime('%B %d, %Y')}.\n" + note_text(n)
            else:
                body = note_text(n)
            if p["mrn"] == 1015:
                header = [h for h in header if not h.startswith("SEX")]  # missing gender
            files[f"scan_{p['mrn']}_{n['kind']}.txt"] = "\n".join(header) + "\n----\n" + body + "\n"
    return files


def main(out: Path = RAW) -> None:
    (out / "notes").mkdir(parents=True, exist_ok=True)
    export = build_json_export()
    (out / "ehr_export.json").write_text(json.dumps({"source": "SIMULATED-EPIC-EXPORT", "records": export}, indent=2))
    rows = build_lab_rows()
    with (out / "lab_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for name, text in build_txt_notes().items():
        (out / "notes" / name).write_text(text)
    print(f"Wrote {len(export)} JSON records, {len(rows)} lab rows, "
          f"{len(list((out / 'notes').glob('*.txt')))} text notes to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else RAW)
