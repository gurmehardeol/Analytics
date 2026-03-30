# Customer Segmentation Pipeline — Generic Template

## How to Use This Template

1. Copy this entire `pipeline-template/` folder into your new project directory
2. Drop your CSV into `data/`
3. Edit `pipeline.config.yaml` — set your dataset path, column hints, and domain context
4. Open a terminal and run:

```bash
cd /path/to/your/project
claude
```

Then type: `Run the customer segmentation pipeline per CLAUDE.md`

---

## What the Config Controls

All agent behaviour is driven by `pipeline.config.yaml`. Key settings:

| Section | What it controls |
|---------|-----------------|
| `data.file` | Path to your CSV |
| `data.numeric_columns` | Explicit numeric columns (leave `[]` to auto-detect) |
| `data.categorical_columns` | Explicit categorical columns (leave `[]` to auto-detect) |
| `data.id_column` | Row ID column to drop before modelling |
| `data.ordinal_columns` | Ordered categoricals with their value order |
| `data.drop_columns` | Columns to always exclude |
| `eda.skew_threshold` | Skewness threshold for high-skew flag |
| `eda.imbalance_threshold` | Dominant class % that triggers imbalance flag |
| `preprocessing.missing_value_strategy` | Three-band impute/drop rules |
| `preprocessing.winsorise_percentile` | Clip percentile (default 5th/95th) |
| `preprocessing.auto_engineer_features` | Whether to auto-generate ratio features |
| `clustering.k_min` / `k_max` | Range of k values to sweep |
| `clustering.silhouette_warn/fail_threshold` | Quality gates on clustering |
| `classification.models` | Which models to train (random_forest, xgboost) |
| `classification.accuracy_warning_threshold` | Leakage suspicion threshold |
| `domain.entity` | What a row represents ("customers", "patients", "users") |
| `domain.industry` | Context for persona naming ("retail", "healthcare", etc.) |
| `outputs.dir` | Root output directory |
| `outputs.overwrite` | false = append _v2 instead of overwriting |

---

## Dataset
- File: defined in `pipeline.config.yaml` → `data.file`
- Goal: Cluster records, profile segments, build classifiers to predict segment membership
- Output convention: All outputs go to `{outputs.dir}/`. **Never overwrite existing files** (unless `outputs.overwrite: true`).

---

## Agent Definitions

All subagent definitions live in `.claude/agents/`. Claude Code auto-discovers them.

| Agent | File | Role |
|-------|------|------|
| eda-agent | `.claude/agents/eda-agent.md` | Visual EDA + auto-detect columns + feature recommendations |
| preprocessing-agent | `.claude/agents/preprocessing-agent.md` | Clean, encode, scale using config rules |
| datascience-agent | `.claude/agents/datascience-agent.md` | Clustering + profiling + classification |
| critic-agent | `.claude/agents/critic-agent.md` | Config-aware quality gate between every stage |

---

## Pipeline — Run in Sequence

Each stage must complete and pass the critic gate before the next begins.

### Stage 1 — EDA

1. Invoke `eda-agent`
2. Wait for `EDA COMPLETE` summary
3. Invoke `critic-agent` with `stage=eda`
4. Read verdict:
   - **PASS or PASS WITH CONCERNS** → proceed to Stage 2
   - **FAIL** → surface required actions to user and stop

### Stage 2 — Preprocessing

*Only begin after Stage 1 critic gives PASS or PASS WITH CONCERNS.*

1. Invoke `preprocessing-agent`
2. Wait for `PREPROCESSING COMPLETE` summary
3. Verify `{outputs.dir}/processed/data_scaled.csv` exists
4. Invoke `critic-agent` with `stage=preprocessing`
5. Read verdict:
   - **PASS or PASS WITH CONCERNS** → proceed to Stage 3
   - **FAIL** → surface required actions to user and stop

### Stage 3 — Data Science Pipeline

*Only begin after Stage 2 critic gives PASS or PASS WITH CONCERNS.*

1. Invoke `datascience-agent`
2. Wait for `DATASCIENCE PIPELINE COMPLETE` summary
3. Invoke `critic-agent` with `stage=datascience`
4. Read verdict:
   - **PASS or PASS WITH CONCERNS** → pipeline complete; summarise results to user
   - **FAIL** → surface required actions to user and stop

---

## Pipeline Flow

```
[Read pipeline.config.yaml]
         │
         ▼
    eda-agent
         │
         ▼
critic-agent (stage=eda)
         │
         ├─── FAIL ──► STOP: surface required actions
         │
         ▼ PASS / PASS WITH CONCERNS
         │
preprocessing-agent
         │
         ▼
critic-agent (stage=preprocessing)
         │
         ├─── FAIL ──► STOP: surface required actions
         │
         ▼ PASS / PASS WITH CONCERNS
         │
datascience-agent
         │
         ▼
critic-agent (stage=datascience)
         │
         ├─── FAIL ──► STOP: surface required actions
         │
         ▼ PASS / PASS WITH CONCERNS
         │
 {outputs.dir}/final_report.md ✓
```

---

## Expected Output Structure

```
{outputs.dir}/          ← from pipeline.config.yaml → outputs.dir
├── critic/
│   ├── critic_eda_review.md
│   ├── critic_preprocessing_review.md
│   └── critic_datascience_review.md
├── eda/
│   ├── eda_report.md
│   ├── correlation_heatmap.png
│   ├── pairplot_numeric.png
│   ├── missing_values.png
│   ├── dist_<col>.png          (one per numeric column)
│   └── count_<col>.png         (one per categorical column)
├── processed/
│   ├── preprocessing_report.md
│   ├── data_scaled.csv
│   └── data_encoded.csv
├── clustering/
│   ├── elbow_silhouette.png
│   ├── data_with_clusters.csv
│   ├── cluster_profiles.md
│   └── phase1_summary.json
├── classification/
│   ├── metrics.md
│   ├── feature_importances.png
│   ├── confusion_matrices.png
│   └── phase3_summary.json
└── final_report.md
```

---

## Orchestrator Responsibilities

As the orchestrator you must:

1. **Read `pipeline.config.yaml` first** — confirm it exists and `data.file` points to a real CSV before invoking any agent.
2. **Never skip the critic gate** — it must run after every agent.
3. **Surface FAIL verdicts clearly** — show the user the exact required actions from the critic before stopping.
4. **Report PASS WITH CONCERNS** — note concerns but allow the pipeline to continue.
5. **Provide a final summary** after Stage 3 critic PASS, including:
   - Project name and dataset
   - Number of clusters (k) and silhouette score
   - Persona names for each cluster (with sizes)
   - Best classifier and its F1 score
   - Whether a high-accuracy warning was triggered
   - Link to `{outputs.dir}/final_report.md`

---

## Adapting for a New Dataset — Quick Checklist

- [ ] Copy `pipeline-template/` into your project folder
- [ ] Place your CSV in `data/`
- [ ] Set `data.file` in config
- [ ] Set `domain.entity` (e.g. "patients", "subscribers", "students")
- [ ] Set `domain.industry` (e.g. "healthcare", "SaaS", "education")
- [ ] Optionally list `data.id_column` and `data.drop_columns`
- [ ] Optionally specify `data.ordinal_columns` if you have ordered categories
- [ ] Run: `claude` → `Run the customer segmentation pipeline per CLAUDE.md`
