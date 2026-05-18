# ════════════════════════════════════════════════════════════════════════════
# CELL 1 — Semantic RAG with Sentence Transformers + FAISS
# Replace your old retrieve_diseases() with this cell
# Run this AFTER the DISEASE_DB cell
# ════════════════════════════════════════════════════════════════════════════

!pip install sentence-transformers faiss-cpu -q

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

print("⏳ Loading semantic model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ── Build rich text representation of each disease ──────────────────────────
def disease_to_text(d):
    return (
        f"{d['name']} {d['category']} "
        f"symptoms: {' '.join(d['key_symptoms'])} "
        f"misdiagnosed as: {' '.join(d['typical_misdiagnoses'])} "
        f"keywords: {' '.join(d['keywords'])}"
    )

# ── Encode all 50 diseases into vectors ─────────────────────────────────────
disease_texts      = [disease_to_text(d) for d in DISEASE_DB]
disease_embeddings = embedder.encode(disease_texts, normalize_embeddings=True, show_progress_bar=True)

# ── Build FAISS index (cosine similarity via inner product on normalised vecs)
DIM   = disease_embeddings.shape[1]
index = faiss.IndexFlatIP(DIM)
index.add(disease_embeddings.astype(np.float32))

# ── New retrieve_diseases — semantic search ──────────────────────────────────
def retrieve_diseases(patient_text, top_k=5):
    query_vec = embedder.encode([patient_text], normalize_embeddings=True)
    scores, indices = index.search(query_vec.astype(np.float32), top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        d = DISEASE_DB[idx].copy()
        d["relevance_score"] = round(float(score) * 100, 1)   # e.g. 84.3
        results.append(d)
    return results

# ── Quick smoke test ─────────────────────────────────────────────────────────
test = retrieve_diseases(
    "fatigue joint pain heart racing when standing up fainting brain fog dizzy"
)
print(f"\n✅ Semantic RAG ready — {len(DISEASE_DB)} diseases indexed")
print(f"   Embedding model : all-MiniLM-L6-v2  |  Dimension: {DIM}")
print(f"\n🧪 Test — 'fatigue joint pain heart racing standing fainting brain fog'")
for d in test:
    print(f"   {d['relevance_score']:5.1f}%  →  {d['name']}")


# ════════════════════════════════════════════════════════════════════════════
# CELL 2 — Update FastAPI /diagnose to return relevance scores in response
# Replace the return statement in your diagnose() endpoint with this
# ════════════════════════════════════════════════════════════════════════════

# Inside your @app.post("/diagnose") function, change the return to:
#
#   diseases = retrieve_diseases(search_text, top_k=5)
#   return {
#       "result":           result,
#       "diseases_checked": [
#           {"name": d["name"], "relevance": d["relevance_score"]}
#           for d in diseases
#       ],
#       "used_vision": image_base64 is not None,
#       "used_pdf":    bool(pdf_text),
#   }


# ════════════════════════════════════════════════════════════════════════════
# CELL 3 — Unsloth Fine-tuning on Rare Disease Data
# Run this on A100. Takes ~45 mins. Gives you a Gemma 4 model
# specialised in rare disease diagnosis — qualifies for Unsloth prize track
# ════════════════════════════════════════════════════════════════════════════

# Step 3a — Install Unsloth
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" -q
!pip install --no-deps "xformers<0.0.27" trl peft accelerate bitsandbytes -q

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset
import torch, json

# Step 3b — Load Gemma 4 with Unsloth (4-bit, fast)
print("⏳ Loading Gemma 4 with Unsloth...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name    = "unsloth/gemma-4-4b-it-unsloth-bnb-4bit",
    max_seq_length= 2048,
    dtype         = None,
    load_in_4bit  = True,
)

# Step 3c — Add LoRA adapters (fast fine-tuning)
model = FastLanguageModel.get_peft_model(
    model,
    r              = 16,
    target_modules = ["q_proj","k_proj","v_proj","o_proj",
                      "gate_proj","up_proj","down_proj"],
    lora_alpha     = 16,
    lora_dropout   = 0,
    bias           = "none",
    use_gradient_checkpointing = "unsloth",
    random_state   = 42,
)

# Step 3d — Build synthetic training dataset from our 50 diseases
def make_training_examples(disease_db):
    examples = []
    for d in disease_db:
        syms   = ", ".join(d["key_symptoms"][:4])
        missed = ", ".join(d["typical_misdiagnoses"][:2])
        tests  = "\n".join([f"- {t}" for t in d["diagnostic_tests"]])
        prompt = (
            f"Patient history:\n"
            f"Symptoms: {syms}\n"
            f"Previously told: {missed}\n"
            f"Duration: {d['avg_diagnosis_time']} (typical delay)\n\n"
            f"Please analyze and give:\n"
            f"1. Top conditions to investigate\n"
            f"2. Specific tests to request\n"
            f"3. Specialist to see\n"
            f"4. Key questions for doctor\n"
            f"5. Patterns that stand out"
        )
        response = (
            f"1. 🔍 TOP CONDITIONS WORTH INVESTIGATING\n"
            f"{d['name']} ({d['category']}) is a strong match. "
            f"Average diagnosis delay is {d['avg_diagnosis_time']} — this condition is frequently missed.\n\n"
            f"2. 🧪 SPECIFIC TESTS TO REQUEST\n"
            f"{tests}\n\n"
            f"3. 👨‍⚕️ SPECIALIST TO SEE NEXT\n"
            f"{d['specialist']}\n\n"
            f"4. ❓ KEY QUESTIONS TO BRING TO YOUR APPOINTMENT\n"
            f"- Have you ruled out {d['name']}?\n"
            f"- Can we run {d['diagnostic_tests'][0]}?\n"
            f"- Why was {missed} diagnosed instead?\n"
            f"- What would confirm or rule out this condition?\n"
            f"- Should I be referred to a {d['specialist']}?\n\n"
            f"5. 📋 PATTERNS IN YOUR HISTORY THAT STAND OUT\n"
            f"The combination of {', '.join(d['key_symptoms'][:2])} is characteristic of "
            f"{d['name']}, which is commonly misidentified as {d['typical_misdiagnoses'][0]}. "
            f"This pattern warrants urgent specialist evaluation."
        )
        # Alpaca format
        full_text = (
            f"<start_of_turn>user\n{prompt}<end_of_turn>\n"
            f"<start_of_turn>model\n{response}<end_of_turn>"
        )
        examples.append({"text": full_text})

        # Second variation — patient describing in natural language
        natural_prompt = (
            f"I've been suffering for years. I have {syms}. "
            f"Doctors keep telling me it's {missed} but nothing helps. "
            f"Can you help me understand what might really be going on?"
        )
        full_text2 = (
            f"<start_of_turn>user\n{natural_prompt}<end_of_turn>\n"
            f"<start_of_turn>model\n{response}<end_of_turn>"
        )
        examples.append({"text": full_text2})

    return examples

train_data = make_training_examples(DISEASE_DB)
dataset    = Dataset.from_list(train_data)
print(f"✅ Training dataset: {len(train_data)} examples from {len(DISEASE_DB)} diseases")

# Step 3e — Fine-tune
trainer = SFTTrainer(
    model        = model,
    tokenizer    = tokenizer,
    train_dataset= dataset,
    dataset_text_field = "text",
    max_seq_length     = 2048,
    dataset_num_proc   = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps        = 5,
        num_train_epochs    = 3,
        learning_rate       = 2e-4,
        fp16                = not torch.cuda.is_bf16_supported(),
        bf16                = torch.cuda.is_bf16_supported(),
        logging_steps       = 10,
        optim               = "adamw_8bit",
        weight_decay        = 0.01,
        lr_scheduler_type   = "linear",
        output_dir          = "./rare_disease_gemma4",
        report_to           = "none",
    ),
)

print("🚀 Starting fine-tuning on rare disease dataset...")
trainer_stats = trainer.train()
print(f"✅ Fine-tuning complete!")
print(f"   Training loss : {trainer_stats.training_loss:.4f}")
print(f"   Time taken    : {trainer_stats.metrics['train_runtime']:.0f}s")

# Step 3f — Save the fine-tuned model
model.save_pretrained("rare_disease_gemma4_lora")
tokenizer.save_pretrained("rare_disease_gemma4_lora")
print("✅ Fine-tuned model saved to rare_disease_gemma4_lora/")


# ════════════════════════════════════════════════════════════════════════════
# CELL 4 — Use fine-tuned model for inference (replaces original run_diagnostic)
# Run this AFTER Cell 3 completes
# ════════════════════════════════════════════════════════════════════════════

from unsloth import FastLanguageModel

# Load the fine-tuned model for fast inference
ft_model, ft_tokenizer = FastLanguageModel.from_pretrained(
    model_name    = "rare_disease_gemma4_lora",
    max_seq_length= 2048,
    dtype         = None,
    load_in_4bit  = True,
)
FastLanguageModel.for_inference(ft_model)  # 2x faster inference
print("✅ Fine-tuned Gemma 4 loaded for inference (Unsloth optimised)")

def run_diagnostic(patient_data):
    """Run rare disease analysis using fine-tuned Gemma 4 via Unsloth."""

    # Build patient prompt
    symptoms       = patient_data.get("symptoms", "")
    age            = patient_data.get("age", "unknown")
    sex            = patient_data.get("sex", "unknown")
    duration       = patient_data.get("duration", "unknown")
    prev_diagnoses = patient_data.get("prev_diagnoses", "none")
    tests_done     = patient_data.get("tests_done", "none")
    family_history = patient_data.get("family_history", "none")
    notes          = patient_data.get("notes", "")

    # Semantic RAG — retrieve relevant diseases
    search_text = f"{symptoms} {prev_diagnoses} {notes}".strip()
    diseases    = retrieve_diseases(search_text, top_k=5)
    disease_ctx = "\n".join([
        f"• {d['name']} ({d['category']}) — relevance: {d['relevance_score']}%\n"
        f"  Symptoms: {', '.join(d['key_symptoms'])}\n"
        f"  Often missed as: {', '.join(d['typical_misdiagnoses'])}\n"
        f"  Tests: {', '.join(d['diagnostic_tests'])}\n"
        f"  See: {d['specialist']}"
        for d in diseases
    ])

    # Handle vision (image uploaded)
    image_base64 = patient_data.get("image_base64")
    image_mime   = patient_data.get("image_mime")

    prompt = (
        f"Patient: {age}yo {sex} | Symptoms for: {duration}\n"
        f"Symptoms: {symptoms}\n"
        f"Previous diagnoses: {prev_diagnoses}\n"
        f"Tests done: {tests_done}\n"
        f"Family history: {family_history}\n"
        f"Notes: {notes}\n\n"
        f"RARE DISEASE DATABASE MATCHES:\n{disease_ctx}\n\n"
        f"Give a detailed analysis with:\n"
        f"1. Top conditions to investigate\n"
        f"2. Specific tests to request\n"
        f"3. Specialist to see\n"
        f"4. Key questions for doctor\n"
        f"5. Patterns that stand out"
    )

    if image_base64:
        # Vision path — use multimodal input
        import base64
        from PIL import Image
        import io
        img_bytes = base64.b64decode(image_base64)
        image     = Image.open(io.BytesIO(img_bytes))

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": f"Read this medical record carefully, then:\n\n{prompt}"}
        ]}]
    else:
        messages = [{"role": "user", "content": prompt}]

    inputs = ft_tokenizer.apply_chat_template(
        messages,
        return_tensors       = "pt",
        return_dict          = True,
        add_generation_prompt= True,
    ).to(ft_model.device)

    with torch.no_grad():
        output = ft_model.generate(
            **inputs,
            max_new_tokens = 800,
            temperature    = 0.7,
            do_sample      = True,
            repetition_penalty = 1.1,
        )

    input_len = inputs["input_ids"].shape[1]
    response  = ft_tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
    return response.strip()

print("✅ run_diagnostic() ready using fine-tuned Gemma 4 (Unsloth)")
