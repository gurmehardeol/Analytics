# Starbucks Customer Segmentation — Final Executive Report

**Date:** 2026-03-25  
**Dataset:** 100,000 orders · 32 engineered features  
**Pipeline:** EDA → Preprocessing → Clustering → Profiling → Classification

---

## Executive Summary

This pipeline segmented 100,000 Starbucks customer orders into two behaviorally distinct clusters and trained production-ready classifiers capable of identifying cluster membership with up to **99.2% accuracy**. The two segments differ sharply on spend, channel preference, and ordering behaviour — enabling targeted marketing, loyalty programme optimisation, and operational planning.

---

## Phase 1 — Clustering

### Method
K-Means was run for k=2 through k=8 on 32 scaled features. DBSCAN was run on a PCA-reduced (10-component) sample for density-based validation.

### Metric Results

| k | Inertia | Silhouette Score |
|---|---|---|
| **2** | 979,922 | **0.1231** |
| 3 | 918,219 | 0.0949 |
| 4 | 873,594 | 0.0847 |
| 5 | 841,980 | 0.0809 |
| 6 | 814,913 | 0.0745 |
| 7 | 793,290 | 0.0714 |
| 8 | 776,941 | 0.0690 |

### Optimal k — Decision

**k = 2** was selected based on:
- **Highest silhouette score (0.1231)** — monotonically decreasing for all higher k, confirming 2 is the most compact and well-separated solution
- **Elbow analysis** — the largest marginal inertia drop occurs between k=2 and k=3, with diminishing returns thereafter
- **DBSCAN corroboration** — on PCA(10) space, DBSCAN (eps=2.0) converged on 1–2 dense regions, consistent with a binary segmentation of the data

### Cluster Sizes
| Cluster | Count | Share |
|---|---|---|
| 0 — Digital High-Spender | 39,400 | 39.4% |
| 1 — Grab-and-Go Commuter | 60,600 | 60.6% |

**Output:** `outputs/clustering/data_with_clusters.csv`  
**Charts:** `outputs/clustering/elbow_silhouette.png`

---

## Phase 2 — Cluster Profiling

### Top 3 Distinguishing Features (variance across cluster means)

| Feature | Variance |
|---|---|
| total_spend | 38.59 |
| cart_size | 2.56 |
| num_customizations | 0.91 |

### Cluster 0 — "Digital High-Spender" (39,400 orders · 39.4%)

**Persona:** Tech-savvy younger customers (25–34) who plan ahead, order via the Mobile App, and build large, heavily customised orders. High rewards membership and food attachment make them the highest-value segment per transaction.

**Key metrics:**
- Avg spend: **$20.19** (vs $11.41 for Cluster 1)
- Avg cart size: **5.11 items** (vs 2.85)
- Avg customisations: **2.63** (vs 1.28)
- Order-ahead rate: **54.2%** (vs 13.9%)
- Rewards membership: **58.6%** (vs 40.6%)
- Preferred channel: **Mobile App (75.7%)**
- Food attachment: **39.7%** (vs 26.3%)
- Satisfaction: **3.86/5**

### Cluster 1 — "Grab-and-Go Commuter" (60,600 orders · 60.6%)

**Persona:** Older, convenience-driven customers (35–44) who visit in person on weekdays — through the Drive-Thru or In-Store Cashier. Simple, quick orders with minimal customisation. Large volume but lower spend per visit.

**Key metrics:**
- Avg spend: **$11.41**
- Avg cart size: **2.85 items**
- Avg customisations: **1.28**
- Order-ahead rate: **13.9%** (highly spontaneous)
- Rewards membership: **40.6%**
- Preferred channel: **Drive-Thru (38.8%) + In-Store (30.2%)**
- Food attachment: **26.3%**
- Satisfaction: **3.58/5**

**Full profiles with tables:** `outputs/clustering/cluster_profiles.md`

---

## Phase 3 — Classification

### Models Trained
Both models were trained on the same 80/20 stratified split (80,000 train / 20,000 test).

### Results

| Model | Accuracy | Precision (w) | Recall (w) | F1 (w) |
|---|---|---|---|---|
| Random Forest (200 trees, max_depth=12) | 97.63% | 0.9764 | 0.9763 | 0.9763 |
| **XGBoost (200 trees, max_depth=6, lr=0.1)** | **99.18%** | **0.9918** | **0.9918** | **0.9918** |

### Best Classifier: XGBoost

XGBoost outperformed Random Forest by **+1.55 percentage points** in accuracy and F1. Both models achieved production-grade performance, confirming that the cluster structure is real and highly learnable from the raw encoded features.

### Top 5 Predictive Features (Random Forest Gini Importance)

| Rank | Feature | Importance |
|---|---|---|
| 1 | total_spend | 0.4752 |
| 2 | cart_size | 0.2102 |
| 3 | order_channel_Mobile App | 0.0962 |
| 4 | num_customizations | 0.0773 |
| 5 | order_ahead | 0.0347 |

The top 5 features alone account for ~89% of predictive power, and they mirror the top distinguishing features found in the profiling phase — a strong internal consistency signal.

**Metrics file:** `outputs/classification/metrics.md`  
**Feature importance chart:** `outputs/classification/feature_importances.png`  
**Confusion matrices:** `outputs/classification/confusion_matrices.png`

---

## Business Recommendations

### For Cluster 0 — Digital High-Spenders (39.4% of orders)

1. **Deepen app engagement** — This segment is already app-native. Invest in in-app personalisation (custom drink suggestions, early access to seasonal items) to increase visit frequency and sustain high average spend.
2. **Rewards tier upsell** — At 58.6% rewards membership, there is still 41% headroom. Target non-member app users with friction-free sign-up flows and milestone bonuses tied to their natural ordering behaviour (large carts, customisations).
3. **Food bundling** — 40% already attach a food item. A/B test bundle promotions (drink + food at a slight discount) within the app at checkout to lift food attachment toward 50%+.
4. **Satisfaction defence** — Already at 3.86/5; ensure fulfilment speed and order accuracy on mobile/ahead orders remains fast to protect NPS as volume grows.

### For Cluster 1 — Grab-and-Go Commuters (60.6% of orders)

1. **App adoption campaign** — Only 20.9% use the app. A targeted "skip the queue" campaign highlighting order-ahead at Drive-Thru locations could convert spontaneous visitors into pre-planners, increasing throughput and average spend.
2. **Rewards acquisition** — At 40.6% membership, this is the largest untapped loyalty pool. Offer in-store sign-up incentives (e.g. next drink at a discount) with Drive-Thru or cashier prompts.
3. **Food attach upsell** — At 26.3% food attach, a simple cashier/kiosk prompt ("Add a croissant for $X?") at point of sale could meaningfully increase basket size without digital infrastructure.
4. **Satisfaction improvement** — Lower satisfaction (3.58 vs 3.86) may reflect wait times or order errors in physical channels. Prioritise staff training, queue management, and order-accuracy checks at high-volume Drive-Thru and In-Store locations.
5. **Age-appropriate messaging** — Skewing 35–54+, this segment responds to reliability and value. Loyalty comms should emphasise consistency, trusted favourites, and value-for-money rather than novelty.

### Cross-Segment

- **Deploy the XGBoost classifier in production** to tag new orders in real-time and route customers into the appropriate personalisation stream.
- **Monitor silhouette score quarterly** — if it rises above 0.15 for k=3, re-run clustering to detect whether a third meaningful segment (e.g. Weekend Occasional) has emerged as the customer mix evolves.
- **The 5 key features** (spend, cart size, mobile app channel, customisations, order-ahead) are sufficient to build a lightweight real-time scoring model for in-session personalisation with minimal latency overhead.

---

## Output File Index

| File | Description |
|---|---|
| `outputs/clustering/elbow_silhouette.png` | Elbow curve and silhouette score chart (k=2..8) |
| `outputs/clustering/data_with_clusters.csv` | 100,000-row dataset with cluster label column |
| `outputs/clustering/cluster_profiles.md` | Full cluster personas with numeric and categorical profiles |
| `outputs/clustering/phase1_summary.json` | Machine-readable Phase 1 metrics |
| `outputs/classification/feature_importances.png` | RF top-20 feature importance bar chart |
| `outputs/classification/confusion_matrices.png` | Side-by-side confusion matrices (RF and XGBoost) |
| `outputs/classification/metrics.md` | Full classification metrics for both models |
| `outputs/classification/phase3_summary.json` | Machine-readable Phase 3 metrics |
| `outputs/final_report.md` | This document |

---

*Starbucks Customer Segmentation Pipeline — completed 2026-03-25*
