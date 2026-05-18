# 🏥 The Diagnostic Odyssey Ender

> **AI-powered rare disease navigator built on Google Gemma 4**  
> Gemma 4 Good Hackathon — Health & Sciences Track

---

## The Problem

A rare disease patient waits an average of **4.8 years** before receiving a correct diagnosis. They see **7 different doctors**, collect multiple wrong diagnoses, and are repeatedly told their symptoms are not real.

There are over **7,000 rare diseases** affecting **300 million people worldwide** — yet most clinicians never encounter them in training. The knowledge exists. Patients just can't access it.

---

## What It Does

The Diagnostic Odyssey Ender lets a patient describe their symptoms in plain language — or upload medical records as a PDF or photo — and receive a structured, evidence-based report with:

- 🔍 **Which rare conditions** match their symptom pattern
- 🧪 **Specific tests** to request at their next appointment
- 👨‍⚕️ **Which specialist** to see
- ❓ **Exact questions** to bring to their doctor
- 📄 **Live PubMed research papers** for the matched conditions

---

## Demo

> Input: *"Female, 24. Heart races every time I stand up, sometimes I pass out. Brain fog so bad I cannot work. Legs turn purple when standing. Doctors said anxiety for 3 years."*

> Output: **POTS (Postural Orthostatic Tachycardia Syndrome) — 45.3% match** + tests + specialist + questions

Real average diagnostic delay for POTS: **5.8 years**. The tool found it in under 30 seconds.

---

## How It Works

```
Patient symptoms (text / PDF / image)
           │
           ▼
┌─────────────────────────────────┐
│   Gemma 4 Agentic Function      │
│   Calling Loop                  │
│   • search_rare_diseases()      │
│   • get_diagnostic_tests()      │
│   • get_specialist_referral()   │
└─────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│   Hybrid RAG Pipeline           │
│   • FAISS dense vector search   │
│   • BM25 sparse keyword search  │
│   • Reciprocal Rank Fusion      │
│   • Cross-encoder reranking     │
└─────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│   650 Disease Database          │
│   • 50 hand-curated diseases    │
│   • 600 from ORPHANET/HPO       │
└─────────────────────────────────┘
           │
           ▼
  Structured 5-section report
```

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| Language Model | Google Gemma 4 4B-IT (4-bit NF4 quantisation) |
| Fine-tuning | PEFT LoRA rank-16 on 1,290 rare disease examples |
| Dense Retrieval | Sentence Transformers (all-MiniLM-L6-v2) + FAISS |
| Sparse Retrieval | BM25 (rank_bm25) |
| Reranking | Cross-encoder (ms-marco-MiniLM-L-6-v2) |
| Disease Database | 650 diseases from ORPHANET/HPO (phenotype.hpoa) |
| Live Research | NCBI PubMed Entrez API |
| Backend | FastAPI + pdfminer.six |
| Public Access | Cloudflare Tunnel (no account needed) |
| Runtime | Google Colab (free T4/A100 GPU) |

---

## Setup & How To Run

### Requirements
- Google account (for Colab)
- No API keys needed
- No local GPU needed
- Free to run

### Step 1 — Open in Google Colab

1. Go to [Google Colab](https://colab.research.google.com)
2. Click **File → Upload notebook**
3. Upload `Diagnostic_Odyssey_Ender.ipynb` from this repo
4. Go to **Runtime → Change runtime type → GPU → T4 or A100**

### Step 2 — Run Cells in Order

Run each cell from top to bottom. Do **not** skip cells.

| Cell | What It Does | Time |
|------|-------------|------|
| Cell 1 | Install all dependencies | ~3 min |
| Cell 2 | Load Gemma 4 with 4-bit quantisation | ~5 min |
| Cell 3 | Load 50-disease base database | instant |
| Cell 3b | Expand to 650 diseases from ORPHANET/HPO | ~1 min |
| Cell 4 | Build FAISS semantic search index | ~1 min |
| Cell 4b | Add BM25 + cross-encoder hybrid RAG | ~2 min |
| Cell 5 | Load diagnostic prompt engine | instant |
| Cell 5b | Enable Gemma 4 agentic function calling | instant |
| Cell 6 | ⭐ Fine-tune with LoRA *(optional, ~45 min on A100)* | optional |
| Cell 7 | Write the UI file automatically | instant |
| Cell 8 | Start FastAPI server + Cloudflare tunnel | ~1 min |

### Step 3 — Open the Live URL

After Cell 8 runs, you will see output like:

```
🌐 Public URL: https://something-random.trycloudflare.com
```

Open that URL in your browser. The tool is live.

### Step 4 — Use the Tool

1. Type your symptoms in plain language in the **Symptoms** field
2. Optionally add previous diagnoses and notes
3. Optionally upload a PDF or photo of medical records
4. Click **Analyse My Symptoms**
5. Wait 15–30 seconds for Gemma 4 to generate the report

---

## Sample Patient Cases to Try

**Case 1 — POTS**
```
Female, 24. Heart races every time I stand up, sometimes I pass out.
Brain fog so bad I cannot work. Legs turn purple when standing.
Doctors said anxiety and depression for 3 years. No treatment helped.
```

**Case 2 — EDS**
```
Male, 19. Joints dislocate constantly — shoulders, knees, fingers.
Skin stretches further than normal and bruises easily. Chronic pain
everywhere. Doctors say it is just hypermobility.
```

**Case 3 — MCAS**
```
Female, 31. Allergic reactions to random foods, smells, temperature.
Hives, flushing, stomach cramps daily. All allergy tests normal.
Antihistamines help slightly.
```

---

## Project Structure

```
diagnostic-odyssey/
├── Diagnostic_Odyssey_Ender.ipynb   # Main Colab notebook (all cells)
├── pubmed_cell.py                   # PubMed live research integration
├── README.md                        # This file
```

---

## Disease Database

650 rare conditions across 12 medical categories:

- **Connective Tissue:** EDS, Marfan, Loeys-Dietz
- **Dysautonomia:** POTS, NMS, Orthostatic Hypotension
- **Autoimmune:** Lupus, APS, Sjogren's, Behcet's, Vasculitis
- **Neurological:** MS, Myasthenia Gravis, Autoimmune Encephalitis, Chiari
- **Hematological:** PNH, HAE, Hemophilia, Thalassemia
- **Metabolic/Genetic:** Fabry, Gaucher, Pompe, Wilson's, Hemochromatosis
- **Immunological:** CVID, Primary Immunodeficiencies
- **Endocrine:** Conn's, Cushing's, MEN1/2
- **Gastrointestinal:** Gastroparesis, EoE, Celiac, MALS, PSC
- **Cardiovascular:** PAH, Cardiac Sarcoidosis, Amyloidosis
- **Mast Cell:** MCAS
- **600+ from ORPHANET/HPO:** Real disease registry data

---

## Results

In testing across 15 real patient vignettes (sourced from rare disease community forums, all de-identified):

- **80% accuracy** — correct diagnosis surfaced in top-3 results
- Hybrid RAG outperforms keyword search on ambiguous presentations
- POTS case: 45.3% relevance vs baseline keyword score of ~12%

---

## Data Sources

- [ORPHANET / Human Phenotype Ontology](https://hpo.jax.org/data/annotations) — phenotype.hpoa v2026-02-16
- [NCBI PubMed Entrez API](https://www.ncbi.nlm.nih.gov/home/develop/api/) — free, no key required
- Synthetic fine-tuning data generated from ORPHANET disease profiles

---

## Important Note

**This tool is not a medical diagnosis.** It is a research and preparation tool designed to help patients have more informed conversations with their doctors. Always consult a qualified medical professional.

---

## License

MIT License — built for the Gemma 4 Good Hackathon, Health & Sciences Track.

---

> *"We built this for everyone who has ever been told their symptoms are not real."*
