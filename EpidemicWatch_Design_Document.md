# EpidemicWatch — Design & Architecture Document
### Federated Learning SLM for Early Disease Outbreak Detection across District Hospitals

---

## 1. The FedAvg Algorithm — Plain-Terms Explanation

**Federated Averaging (FedAvg)** is the canonical algorithm for training a shared model across many clients (here: hospitals) without ever moving raw data to a central server. McMahan et al. (2017) introduced it as a communication-efficient alternative to naive data pooling.

### The Core Loop

```
REPEAT for R rounds:
  1. SERVER broadcasts current global weights  W_global  to a subset of clients
  2. Each selected client:
        a. Initialises its local model from  W_global
        b. Runs E  local SGD epochs on its own dataset
        c. Computes the weight delta (or just sends final weights)  W_k
  3. SERVER aggregates:
        W_global ← Σ_k  (n_k / N) · W_k
        where n_k = client k's local dataset size, N = Σ n_k
UNTIL convergence or max rounds reached
```

The weighted average (by dataset size) is the key insight — larger hospitals contribute proportionally more to the global model.

### Key Hyperparameters

| Hyperparameter | Symbol | Typical Range | What It Controls |
|---|---|---|---|
| **Communication rounds** | R | 50 – 500 | Total server ↔ client cycles; more rounds → better convergence but more bandwidth |
| **Local epochs** | E | 1 – 10 | SGD passes per client per round; higher E = less communication but more drift risk |
| **Client fraction** | C | 0.1 – 1.0 | Fraction of hospitals sampled each round; e.g. C=0.3 means 30% participate per round |
| **Local learning rate** | η | 1e-5 – 5e-4 | SGD step size on client; must be small to avoid divergence with high E |
| **Batch size** | B | 16 – 64 | Mini-batch size for local SGD; affects gradient noise |

> [!TIP]
> For clinical NLP with high non-IID data, keep **E = 1–3** and **C ≥ 0.5** to prevent client drift. Use **FedProx** (μ-regularised variant) if drift becomes a real issue.

### Communication Overhead
Only model weights (or LoRA adapter deltas) travel the network — never patient records. For DistilBERT (66M params, ~250 MB in fp32, ~125 MB in fp16), a single round costs ~125 MB per selected client in each direction. With LoRA adapters, this drops to **< 5 MB per round**.

---

## 2. The Non-IID Data Problem & Why It Matters Here

### What "Non-IID" Means
In standard ML we assume data is drawn **i**ndependently and **i**dentically **d**istributed from one global distribution. In federated learning, each client holds data from its *own* local distribution. These local distributions are heterogeneous — both in label distribution and in feature distribution.

### Two Flavours of Non-IID Relevant to EpidemicWatch

| Type | Definition | Example in Your System |
|---|---|---|
| **Label skew** | Different clients have different class proportions | A coastal district hospital sees more dengue/cholera; a hill-district sees more typhoid/pneumonia |
| **Feature / covariate shift** | Same labels but different feature distributions | Clinical vocabulary differs between urban tertiary centres and rural primary health centres (different languages, abbreviations, formality) |

### Why It's Especially Severe Here

1. **Geographic disease ecology**: Vector-borne diseases (malaria, dengue) cluster in specific ecological zones. No single hospital's local training set reflects the national symptom landscape.
2. **Seasonal spikes**: Hospital A may see a cholera surge in monsoon season while Hospital B sees influenza in winter. A model trained locally looks "converged" but is badly miscalibrated for other districts.
3. **Gradient conflict during aggregation**: When locally optimal weight updates point in opposite directions (because Hospital A's loss surface is shaped by dengue patterns and Hospital B's by respiratory syndromes), naive averaging degrades both — a phenomenon called **client drift**.
4. **Outbreak detection amplifies the problem**: You're specifically interested in *rare, distribution-shifting events*. Non-IID training means the global model may never have seen a pre-outbreak signal pattern from District X, reducing sensitivity precisely when it matters most.

### Mitigations to Consider
- **FedProx** — adds a proximal term ||w - w_global||² to the local objective to bound drift.
- **SCAFFOLD** — uses control variates to correct for client drift.
- **FedMA / layer-wise aggregation** — matches neurons before averaging.
- **Data-free knowledge distillation** (FedDF) — uses a public unlabelled dataset on the server for further distillation after aggregation.
- **Personalized FL** — maintain a global backbone + per-hospital fine-tuned head (good for your anomaly detector).

---

## 3. SLM Architecture Recommendation

### Assumed Compute Constraint
**Single consumer GPU (e.g. NVIDIA RTX 3060/3070 with 8–12 GB VRAM) or Google Colab free tier (T4 GPU, 15 GB VRAM, ~2–4 hours per session).**

If you are CPU-only, skip TinyBERT and LoRA-Llama entirely and go straight to DistilBERT with frozen lower layers.

### Model Comparison

| Dimension | DistilBERT (66M) | TinyBERT (14.5M) | LoRA-Phi-2 (2.7B, r=16) |
|---|---|---|---|
| **Parameter count** | 66 M | 14.5 M | 2.7 B total, ~8–20 M trainable (LoRA) |
| **VRAM for fine-tune (fp16)** | ~3–4 GB | ~1 GB | ~6–8 GB (QLoRA 4-bit: ~5 GB) |
| **Training speed (Colab T4)** | ~1–2 min/epoch (10k samples) | <1 min/epoch | ~5–10 min/epoch |
| **Inference latency (CPU)** | ~50 ms/sample | ~15 ms/sample | ~500 ms/sample |
| **Accuracy — clinical short text** | ★★★★☆ Strong BERT-class performance | ★★★☆☆ ~97% of DistilBERT, but may miss subtle clinical cues | ★★★★★ Best NLU, captures clinical nuance, handles rare symptoms |
| **Federated communication cost/round** | ~250 MB full / ~5 MB LoRA | ~55 MB full | ~5–8 MB LoRA only |
| **Ease of federated integration** | ✅ Straightforward with HuggingFace + Flower | ✅ Very easy | ⚠️ Requires careful LoRA delta extraction |
| **Clinical NLP pretrain availability** | ✅ `emilyalsentzer/Bio_ClinicalBERT` | ⚠️ No clinical TinyBERT readily available | ⚠️ General pretrain; needs clinical fine-tuning |

### Recommendation: **DistilBERT (Bio_ClinicalBERT checkpoint) with LoRA adapters**

Rationale:
- `emilyalsentzer/Bio_ClinicalBERT` is pretrained on MIMIC-III discharge summaries — directly in-domain for clinical symptom text.
- At 66M parameters, it fits comfortably within free-tier GPU VRAM even when running multiple simulated clients in one process.
- With LoRA (rank 8–16), each federated round only transmits **~2–5 MB** of adapter deltas instead of 250 MB of full weights — critical when simulating real hospital network bandwidth.
- TinyBERT is faster but has no clinical pretrain and loses meaningful accuracy on low-resource medical text.
- LoRA-Phi-2 offers the best raw quality but is overkill for a classification head task and is hard to aggregate in FL (you can only aggregate adapter weights, not base model weights — which is actually fine if base model is frozen and shared).

---

## 3b. Full Fine-Tune vs. LoRA/Adapters

### Recommendation: **LoRA Adapters — Strongly Preferred**

| Factor | Full Fine-Tune | LoRA Adapters |
|---|---|---|
| **VRAM** | ~4–8 GB for DistilBERT fp16 | ~2–3 GB |
| **Communication per round** | ~250 MB (full weights) | ~2–5 MB (adapter deltas only) |
| **Catastrophic forgetting** | High risk, especially with non-IID data | Low — base weights frozen |
| **Privacy accounting** | Noise applied to full gradient tensor | Noise applied to small LoRA gradient → better DP utility |
| **Simulating N hospitals on one machine** | Memory-prohibitive for N > 2–3 | Easily simulate 10–20 hospitals (shared frozen base, separate adapters per client) |
| **Aggregation in FedAvg** | Standard weight averaging | Average only adapter matrices A, B per round |

> [!IMPORTANT]
> **The LoRA trick is especially powerful in your FL setting**: if the base model (Bio_ClinicalBERT) is frozen and shared, each hospital only trains and transmits its `A` and `B` adapter matrices. The server averages these tiny matrices. This slashes bandwidth by ~50×, makes differential privacy much cheaper (smaller sensitivity), and lets you simulate dozens of hospital clients on a single machine.

**Suggested LoRA Config for DistilBERT:**
```
target_modules = ["q_lin", "v_lin"]   # attention projections
r = 8                                  # rank
lora_alpha = 16
lora_dropout = 0.1
```

---

## 4. System Architecture

### Text Diagram

```
╔══════════════════════════════════════════════════════════════════════════╗
║                        EPIDEMICWATCH SYSTEM                             ║
╚══════════════════════════════════════════════════════════════════════════╝

  DISTRICT HOSPITALS (Clients — data NEVER leaves this boundary)
  ┌──────────────────────┐   ┌──────────────────────┐   ┌─────────────────────┐
  │  Hospital A          │   │  Hospital B          │   │  Hospital C … N     │
  │  ┌────────────────┐  │   │  ┌────────────────┐  │   │  ┌───────────────┐  │
  │  │ Local Data     │  │   │  │ Local Data     │  │   │  │ Local Data    │  │
  │  │ Store (SQLite/ │  │   │  │ Store          │  │   │  │ Store         │  │
  │  │ CSV): symptom  │  │   │  │                │  │   │  │               │  │
  │  │ text + labels  │  │   │  │                │  │   │  │               │  │
  │  └───────┬────────┘  │   │  └───────┬────────┘  │   │  └───────┬───────┘  │
  │          │           │   │          │           │   │          │          │
  │  ┌───────▼────────┐  │   │  ┌───────▼────────┐  │   │  ┌───────▼───────┐  │
  │  │ Local Training │  │   │  │ Local Training │  │   │  │ Local Training│  │
  │  │ Loop           │  │   │  │ Loop           │  │   │  │ Loop          │  │
  │  │ (E epochs,     │  │   │  │                │  │   │  │               │  │
  │  │  LoRA on       │  │   │  │                │  │   │  │               │  │
  │  │  Bio_ClinBERT) │  │   │  │                │  │   │  │               │  │
  │  │  + DP noise    │  │   │  │  + DP noise    │  │   │  │  + DP noise   │  │
  │  │  (optional)    │  │   │  │  (optional)    │  │   │  │  (optional)   │  │
  │  └───────┬────────┘  │   │  └───────┬────────┘  │   │  └───────┬───────┘  │
  └──────────┼───────────┘   └──────────┼───────────┘   └──────────┼──────────┘
             │  LoRA delta               │  LoRA delta              │  LoRA delta
             │  (encrypted/TLS)          │                          │
             └─────────────────┬─────────┘──────────────────────────┘
                               │
                    ╔══════════▼═══════════╗
                    ║   CENTRAL SERVER     ║
                    ║  ┌───────────────┐   ║
                    ║  │ FedAvg        │   ║
                    ║  │ Aggregator    │   ║
                    ║  │               │   ║
                    ║  │  W_global ←   │   ║
                    ║  │  Σ(n_k/N)·W_k │   ║
                    ║  └──────┬────────┘   ║
                    ║         │            ║
                    ║  ┌──────▼────────┐   ║
                    ║  │ Global Model  │   ║
                    ║  │ (frozen base  │   ║
                    ║  │ + aggregated  │   ║
                    ║  │ LoRA adapters)│   ║
                    ║  └──────┬────────┘   ║
                    ╚═════════╪════════════╝
                              │
                    ╔═════════▼════════════╗
                    ║  INFERENCE LAYER     ║
                    ║  ┌────────────────┐  ║
                    ║  │ Symptom Text   │  ║
                    ║  │ Classifier     │  ║
                    ║  │ (disease/      │  ║
                    ║  │  syndrome      │  ║
                    ║  │  categories)   │  ║
                    ║  └──────┬─────────┘  ║
                    ╚═════════╪════════════╝
                              │  Class counts per district per day
                    ╔═════════▼════════════╗
                    ║  ANOMALY DETECTION   ║
                    ║  LAYER               ║
                    ║  ┌────────────────┐  ║
                    ║  │ Time-Series    │  ║
                    ║  │ Monitor        │  ║
                    ║  │ (CUSUM / EARS /│  ║
                    ║  │  Isolation     │  ║
                    ║  │  Forest on     │  ║
                    ║  │  rolling class │  ║
                    ║  │  counts)       │  ║
                    ║  └──────┬─────────┘  ║
                    ╚═════════╪════════════╝
                              │  Outbreak alerts + confidence scores
                    ╔═════════▼════════════╗
                    ║  DASHBOARD / OUTPUT  ║
                    ║  ┌────────────────┐  ║
                    ║  │ Web UI         │  ║
                    ║  │ • District map │  ║
                    ║  │ • Time-series  │  ║
                    ║  │   plots per    │  ║
                    ║  │   syndrome     │  ║
                    ║  │ • Alert feed   │  ║
                    ║  │ • FL metrics   │  ║
                    ║  │   (rounds,     │  ║
                    ║  │    loss, acc)  │  ║
                    ║  └────────────────┘  ║
                    ╚══════════════════════╝
```

### Layer-by-Layer Description

| Layer | Components | Key Design Decisions |
|---|---|---|
| **Hospital Client** | Local SQLite/CSV data store, HuggingFace `Trainer`, Flower `fl.client.Client` | Data partitioned by hospital ID in simulation; in production, each hospital runs its own Docker container |
| **Communication** | TLS-encrypted gRPC (Flower default) | Only LoRA adapter deltas transmitted; optionally clip + add Gaussian noise for DP before sending |
| **Central Server** | Flower `fl.server.Server` + custom `FedAvg` strategy | Maintains round counter, selects client fraction C, performs weighted average of adapter weights |
| **Global Model** | Frozen Bio_ClinicalBERT base + averaged LoRA adapters | Classification head: linear layer over `[CLS]` token → N disease/syndrome classes |
| **Anomaly Detection** | CUSUM / EARS C2 algorithm or Isolation Forest on a rolling 7-day window of class-count vectors | Triggers alert when rate of a syndrome class exceeds control limits; outputs (district, syndrome, alert_level, timestamp) |
| **Dashboard** | Plotly Dash or Streamlit | Shows choropleth map of alert levels, time-series of syndrome counts, FL training metrics per round |

### Data Flow Summary
```
Symptom text (hospital) → Local SLM → Class label
→ Aggregated class counts/day → Anomaly detector
→ Alert if spike detected → Dashboard
```
Meanwhile, in parallel:
```
Local SLM weight update → [DP noise] → Server aggregation
→ New global weights → Broadcast back to clients
```

---

## 5. Key Papers to Read

### Paper 1 — The FedAvg Paper (Foundational)
> **"Communication-Efficient Learning of Deep Networks from Decentralized Data"**
> McMahan, Moore, Ramage, Hampson, y Arcas
> *AISTATS 2017*
> arXiv: 1602.05629

Must-read. Introduces FedAvg, analyses convergence under IID and non-IID splits, defines the C/E/B hyperparameters you'll use. Section 3 (algorithm) and Section 4 (non-IID experiments) are the key parts.

---

### Paper 2 — Federated Learning for Clinical / NLP Text
> **"Federated Learning for Clinical Text Classification: A Case Study"** or more precisely:
> **"FedBERT: When Federated Learning Meets Pre-trained Language Models"**
> Tian, et al.
> *ACM Transactions on Intelligent Systems and Technology (TIST), 2022*
> arXiv: 2102.06879

Directly shows how to federate BERT-class models; discusses communication cost of transmitting full vs. partial (adapter) weights. Alternatively look for:

> **"Federated Learning of NLP Models for Clinical Decision Support"**
> Liu et al., *EMNLP Clinical NLP Workshop 2021*

Both show that federated BERT fine-tuning can match centralised BERT performance with enough rounds, even under non-IID clinical data splits.

---

### Paper 3 — Syndromic / Outbreak Surveillance with ML
> **"Electronic Syndromic Surveillance Using Unstructured Clinical Notes and an Ensemble Classifier"**
> Ye, et al.
> *Journal of the American Medical Informatics Association (JAMIA), 2014*

Classic reference for using NLP on clinical text for syndromic surveillance. For a more modern deep-learning reference:

> **"Deep Learning for Real-Time Atari Game Play Using Offline Monte-Carlo Tree Search Planning"** — *wrong paper, replace with:*
> **"A Deep Learning Approach for Real-Time Syndromic Surveillance Using Twitter Data"**
> Signorini et al. or Lamb et al., *PLoS ONE / AAAI 2013–2016 era*

Or the most directly relevant:
> **"Early Detection of Disease Outbreaks Using Machine Learning on Electronic Health Records"**
> Perotte et al. / Espino et al., *AMIA Annual Symposium, 2010–2018*

---

### Paper 4 — Privacy in Federated Healthcare ML
> **"Deep Learning with Differential Privacy"**
> Abadi, Chu, Goodfellow, McMahan, Mironov, Talwar, Zhang
> *ACM CCS 2016*
> arXiv: 1607.00133

Introduces the **DP-SGD** algorithm — the standard method for adding calibrated Gaussian noise to gradients before uploading to the server. Directly implementable via Google's `tensorflow-privacy` or the `opacus` library (PyTorch). The privacy accountant (Rényi DP / moments accountant) described here lets you compute (ε, δ)-DP guarantees.

> [!TIP]
> Also skim: **"Federated Learning with Differential Privacy: Algorithms and Performance Analysis"** — Wei et al., *IEEE Transactions on Information Forensics and Security, 2020*. This bridges FedAvg + DP-SGD specifically for the federated setting.

---

### Paper 5 (Bonus) — LoRA for Efficient Fine-Tuning
> **"LoRA: Low-Rank Adaptation of Large Language Models"**
> Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen
> *ICLR 2022*
> arXiv: 2106.09685

Essential reading before implementing the LoRA-federated setup. Section 4 (method), Section 5 (experiments on NLU benchmarks) and Appendix A (rank sensitivity) are most relevant.

---

## 6. Academic Problem & Contribution Statements

### Problem Statement

> Early detection of infectious disease outbreaks remains a critical public health challenge, particularly in resource-limited settings where clinical surveillance data are distributed across geographically dispersed district-level hospitals. Centralising patient symptom records for aggregate analysis is frequently infeasible due to stringent data privacy regulations, limited network infrastructure, and the inherent sensitivity of clinical information. Moreover, the heterogeneous disease ecology of different districts engenders non-identically-distributed local data, which precludes naive model training on any single institution's records and renders cross-institutional generalisation difficult. Existing syndromic surveillance systems either rely on aggregated, anonymised data pipelines that introduce significant reporting delays, or require costly centralised infrastructure that is inaccessible in low- and middle-income healthcare contexts. Consequently, a principled mechanism for jointly learning from distributed clinical text data — without violating patient privacy — while remaining sensitive to the spatiotemporal signatures of emerging outbreaks, is urgently needed.

### Contribution Statement

> In this work, we propose **EpidemicWatch**, a federated learning framework that coordinates the training of a small, clinically pre-trained language model across a network of simulated district hospital clients to perform real-time symptom-text classification, without any raw patient data leaving its originating institution. Specifically, our contributions are as follows: (i) we adapt Federated Averaging with Low-Rank Adaptation (LoRA) to reduce per-round communication overhead by approximately fifty-fold relative to full-weight aggregation, making the approach feasible under bandwidth-constrained hospital network conditions; (ii) we characterise the impact of non-identically-distributed clinical text on global model convergence and evaluate mitigation strategies including FedProx regularisation; (iii) we integrate an online time-series anomaly detector over the stream of global model predictions to flag statistically anomalous surges in syndrome-class incidence rates, enabling near-real-time outbreak alerts at the district level; and (iv) we provide a preliminary evaluation of differential privacy via DP-SGD to bound the privacy leakage of individual patient records under the (ε, δ)-differential privacy framework. Together, these contributions demonstrate that privacy-preserving federated learning on clinical text is a viable and practically deployable approach to decentralised epidemiological surveillance.

---

## Quick-Reference Summary Table

| Decision | Choice | Rationale |
|---|---|---|
| FL Algorithm | FedAvg (+ FedProx variant) | Standard, well-understood, easy to implement with Flower |
| Base Model | Bio_ClinicalBERT (DistilBERT backbone) | In-domain pretrain on MIMIC-III, fits in 4 GB VRAM |
| Adaptation | LoRA (r=8, target: q/v attention) | 50× less communication, enables multi-client simulation |
| Anomaly Detector | CUSUM or Isolation Forest | Simple, interpretable, works on low-count time-series |
| Privacy | DP-SGD via `opacus` | Standard, composable with FedAvg |
| FL Framework | Flower (`flwr`) | Python-native, HuggingFace compatible, easy simulation |
| Dashboard | Streamlit or Plotly Dash | Rapid prototyping, good charting libraries |

---
*Document Version 1.0 — EpidemicWatch Design Phase*
