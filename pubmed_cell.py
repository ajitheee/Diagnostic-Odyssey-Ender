# ════════════════════════════════════════════════════════════════
# CELL 4c — PubMed Live Research Integration
# YOUR FRIEND'S TASK — Run this AFTER Cell 4b
#
# For each rare disease the AI matches, this fetches the 3 most
# recent real research papers from PubMed (free, no API key needed).
# Papers are included in the API response and shown in the UI.
# ════════════════════════════════════════════════════════════════

import requests, urllib.parse, time

PUBMED_BASE  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_EMAIL = "diagnostic.odyssey@gmail.com"   # NCBI requires an email

def search_pubmed(disease_name, max_results=3):
    """
    Fetch the most recent PubMed papers for a rare disease.
    Uses NCBI Entrez API — completely free, no API key required.
    Stay under 3 requests/second to respect rate limits.
    """
    try:
        # Build search query — disease name in title/abstract
        query     = urllib.parse.quote(f'"{disease_name}"[Title/Abstract] rare disease diagnosis')
        search_url = (
            f"{PUBMED_BASE}/esearch.fcgi"
            f"?db=pubmed&term={query}&retmax={max_results}"
            f"&sort=date&retmode=json&email={PUBMED_EMAIL}"
        )
        search_resp = requests.get(search_url, timeout=8).json()
        pmids = search_resp.get("esearchresult", {}).get("idlist", [])

        if not pmids:
            return []

        # Fetch paper summaries for those IDs
        ids_str     = ",".join(pmids)
        summary_url = (
            f"{PUBMED_BASE}/esummary.fcgi"
            f"?db=pubmed&id={ids_str}&retmode=json&email={PUBMED_EMAIL}"
        )
        summary_resp = requests.get(summary_url, timeout=8).json()

        papers = []
        for pmid in pmids:
            doc = summary_resp.get("result", {}).get(pmid, {})
            if not doc:
                continue
            authors = doc.get("authors", [])
            first_author = authors[0].get("name", "") if authors else "et al."
            papers.append({
                "title":    doc.get("title", ""),
                "author":   first_author,
                "journal":  doc.get("fulljournalname", ""),
                "year":     doc.get("pubdate", "")[:4],
                "pmid":     pmid,
                "url":      f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            })

        return papers

    except Exception as e:
        print(f"  ⚠ PubMed error for '{disease_name}': {e}")
        return []


def enrich_with_pubmed(diseases, top_n=3):
    """
    Add recent PubMed papers to the top N matched diseases.
    Only enriches top 3 to stay within NCBI rate limits.
    """
    enriched = []
    for i, d in enumerate(diseases):
        d_copy = d.copy()
        if i < top_n:
            d_copy["pubmed_papers"] = search_pubmed(d["name"])
            time.sleep(0.4)   # ~2.5 req/s — safely under 3/s NCBI limit
        else:
            d_copy["pubmed_papers"] = []
        enriched.append(d_copy)
    return enriched


# ── Smoke test ────────────────────────────────────────────────────
print("🔬 Testing PubMed integration...")
test_papers = search_pubmed("POTS Postural Orthostatic Tachycardia Syndrome")
print(f"\n✅ PubMed integration ready! ({len(test_papers)} papers found for POTS)")
for p in test_papers:
    print(f"   📄 [{p['year']}] {p['title'][:70]}...")
    print(f"      {p['url']}")


# ════════════════════════════════════════════════════════════════
# HOW TO WIRE THIS INTO CELL 8 (FastAPI /diagnose endpoint)
# ════════════════════════════════════════════════════════════════
#
# In Cell 8, find the return statement inside @app.post("/diagnose")
# and replace it with this:
#
#   result   = run_diagnostic(patient_data)
#   search_t = f"{symptoms} {prev_diagnoses} {notes} {pdf_text}".strip()
#   diseases = retrieve_diseases(search_t or "rare disease chronic", top_k=5)
#   diseases = enrich_with_pubmed(diseases, top_n=3)   # ← ADD THIS LINE
#
#   return {
#       "result":     result,
#       "diseases_checked": [
#           {
#               "name":      d["name"],
#               "relevance": d["relevance_score"],
#               "papers":    d.get("pubmed_papers", [])   # ← ADD THIS
#           }
#           for d in diseases
#       ],
#       "used_vision": image_base64 is not None,
#       "used_pdf":    bool(pdf_text),
#   }
#
# ════════════════════════════════════════════════════════════════
# HOW TO SHOW PAPERS IN THE UI (colab_ui.html)
# ════════════════════════════════════════════════════════════════
#
# In colab_ui.html, find where disease tags are rendered.
# After showing the relevance badge, add this JS snippet:
#
#   if (d.papers && d.papers.length > 0) {
#     const paperList = d.papers.map(p =>
#       `<a href="${p.url}" target="_blank" style="display:block;color:#60a5fa;
#        font-size:11px;margin-top:4px;text-decoration:none;">
#        📄 [${p.year}] ${p.title.substring(0, 60)}...</a>`
#     ).join('');
#     tag.innerHTML += `<div style="margin-top:6px;">${paperList}</div>`;
#   }
#
# ════════════════════════════════════════════════════════════════
