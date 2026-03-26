# Starbucks Customer Ordering Patterns — EDA Report

**Date:** 2026-03-25
**Data file:** `data/starbucks_customer_ordering_patterns.csv`
**Analyst:** EDA Agent

---

## 1. Dataset Shape & Column Overview

| Property | Value |
|---|---|
| Rows | 100,000 |
| Columns | 20 |
| Missing values | **0** (dataset is complete) |

### Column inventory

| Column | Type | Role |
|---|---|---|
| customer_id | object | Identifier |
| order_id | object | Identifier |
| order_date | object | Temporal |
| order_time | object | Temporal |
| day_of_week | object | Categorical |
| order_channel | object | Categorical |
| store_id | object | Identifier |
| store_location_type | object | Categorical |
| region | object | Categorical |
| customer_age_group | object | Categorical |
| customer_gender | object | Categorical |
| is_rewards_member | bool | Binary flag |
| cart_size | int64 | Numeric |
| num_customizations | int64 | Numeric |
| total_spend | float64 | Numeric |
| fulfillment_time_min | float64 | Numeric |
| drink_category | object | Categorical |
| has_food_item | bool | Binary flag |
| order_ahead | bool | Binary flag |
| customer_satisfaction | int64 | Numeric / target |

---

## 2. Numeric Columns — Key Observations

### cart_size (items per order)
- **Range:** 1–10 | **Mean:** 3.74 | **Median:** 4
- **Skewness:** +0.63 (moderate right skew — most orders are small, a tail of large orders)
- **IQR outliers:** 900 rows (0.90%) above the upper fence
- Distribution is roughly unimodal, peaking around 3–4 items. The box plot shows isolated high-end outliers (7–10 items).

### num_customizations (add-ons / modifications)
- **Range:** 0–8 | **Mean:** 2.33 | **Std:** 1.90
- **Skewness:** +0.89 (strongest right skew among numerics)
- **IQR outliers:** 609 rows (0.61%)
- Many orders have 0–2 customizations; orders with 6+ are uncommon and flag as outliers. This column shows the most non-normality.

### total_spend (USD)
- **Range:** ~$2–$45 | **Mean:** ~$15.60
- **Skewness:** +0.67 (right-skewed, consistent with cart_size skew)
- **IQR outliers:** 1,415 rows (1.42%) — highest absolute outlier count
- Strong positive correlation expected with cart_size. Log-transformation recommended before clustering.

### fulfillment_time_min
- **Range:** ~1–30 min | **Mean:** ~8.5 min
- **Skewness:** +0.36 (mildest skew — closest to symmetric)
- **IQR outliers:** 811 rows (0.81%)
- Near-normal distribution with a slight right tail. Likely driven by channel (Drive-Thru vs. Mobile App).

### customer_satisfaction (1–5 Likert scale)
- **Range:** 1–5 | **Mean:** 3.69 | **Median:** 4
- **Skewness:** -0.71 (left-skewed — ratings lean towards 4 and 5)
- **IQR outliers:** 0 (bounded integer scale, no statistical outliers)
- A discrete ordinal variable. Scores of 4 and 5 dominate; low ratings (1–2) are rare.

---

## 3. Categorical Columns — Key Observations

### customer_gender
- Near equal split: Female 45.3%, Male 44.8%, Non-binary 5.1%, Prefer not to say 4.9%.
- No dominant imbalance; all groups adequately represented.

### order_channel
- **Mobile App** dominates (42.5%), followed by Drive-Thru (28.0%), In-Store Cashier (22.1%), Kiosk (7.4%).
- Large imbalance between Mobile App and Kiosk — consider binary encoding or grouping for ML.

### store_location_type
- Relatively balanced: Suburban 35.7%, Urban 32.6%, Rural 31.7%.
- Slight suburban lean but no extreme skew.

### region
- West (22.6%), Southeast (20.2%), Southwest (19.7%), Midwest (19.5%), Northeast (18.0%).
- Very even distribution across five regions.

### customer_age_group
- Youngest-heavy: 25–34 (29.8%), 35–44 (24.5%), 18–24 (20.3%), 45–54 (15.4%), 55+ (10.0%).
- Expected demographic pyramid for a mobile-forward brand.

### day_of_week
- Almost perfectly uniform (~14,100–14,400 per day). No meaningful day-of-week effect in volume.

### drink_category
- Six categories, all nearly equal (~16,500–16,800 each): Refresher, Tea, Espresso, Frappuccino, Other, Brewed Coffee.
- No dominant drink type; uniform distribution.

### is_rewards_member
- Non-members: 52.3%, Members: 47.7% — near 50/50 split. Useful binary feature.

### has_food_item
- No food: 68.4%, Has food: 31.6% — meaningful imbalance; food orders are the minority.

### order_ahead
- No ahead: 70.2%, Order ahead: 29.8% — roughly 30% pre-order rate.

---

## 4. Correlation Analysis (Numeric Features)

From the lower-triangle heatmap:

| Pair | Pearson r | Interpretation |
|---|---|---|
| cart_size ↔ total_spend | ~+0.85 | Strong positive — larger carts cost more (expected) |
| num_customizations ↔ total_spend | ~+0.45 | Moderate — customizations add cost |
| num_customizations ↔ cart_size | ~+0.30 | Mild positive |
| fulfillment_time_min ↔ cart_size | ~+0.20 | Slight — bigger orders take longer |
| customer_satisfaction ↔ fulfillment_time_min | ~-0.15 | Very mild negative — longer wait, slightly lower satisfaction |
| customer_satisfaction ↔ total_spend / cart_size | ~0.00–0.05 | No meaningful correlation |

**Key takeaway:** `cart_size` and `total_spend` are highly collinear (~r = 0.85). Only one should be used as a clustering feature to avoid redundancy; `total_spend` is preferred as it captures both size and price variation.

---

## 5. Skewness & Outlier Flags

| Column | Skewness | Outlier % | Action |
|---|---|---|---|
| num_customizations | +0.89 | 0.61% | Log-transform or robust scaler |
| total_spend | +0.67 | 1.42% | Log-transform; watch for price outliers |
| cart_size | +0.63 | 0.90% | Log or cap at 95th percentile |
| customer_satisfaction | -0.71 | 0% | Ordinal — encode as-is or discretize |
| fulfillment_time_min | +0.36 | 0.81% | Mild skew; standard scaling acceptable |

Outlier rows are present but modest (<1.5% per column). Recommend **robust scaling** (median/IQR) during preprocessing rather than removal, to retain all 100,000 rows.

---

## 6. Missing Value Summary

**Zero missing values detected across all 20 columns.** No imputation required.

---

## 7. Recommended Features for Clustering

Based on variance, interpretability, and low collinearity:

### Primary numeric features (after scaling)
1. `total_spend` — best single proxy for spend behavior
2. `num_customizations` — signals beverage preferences / engagement
3. `fulfillment_time_min` — proxy for channel and order complexity
4. `customer_satisfaction` — outcome signal

### Engineered / binary features
5. `is_rewards_member` (0/1) — strong behavioural differentiator
6. `has_food_item` (0/1) — spend pattern indicator
7. `order_ahead` (0/1) — channel behaviour

### Encoded categoricals (label or one-hot)
8. `order_channel` — Mobile App vs. other channels is a strong segmentation axis
9. `customer_age_group` — ordinal, encode as 1–5
10. `store_location_type` — Urban / Suburban / Rural context

### Exclude from clustering
- `cart_size` — collinear with `total_spend` (r ≈ 0.85)
- `customer_id`, `order_id`, `store_id` — identifiers only
- `order_date`, `order_time` — not yet feature-engineered
- `region`, `day_of_week`, `drink_category`, `customer_gender` — near-uniform; low clustering signal

---

## 8. Generated Plot Files

| File | Description |
|---|---|
| `dist_cart_size.png` | Histogram + boxplot for cart_size |
| `dist_num_customizations.png` | Histogram + boxplot for num_customizations |
| `dist_total_spend.png` | Histogram + boxplot for total_spend |
| `dist_fulfillment_time_min.png` | Histogram + boxplot for fulfillment_time_min |
| `dist_customer_satisfaction.png` | Histogram + boxplot for customer_satisfaction |
| `count_customer_gender.png` | Count plot for customer_gender |
| `count_order_channel.png` | Count plot for order_channel |
| `count_store_location_type.png` | Count plot for store_location_type |
| `count_region.png` | Count plot for region |
| `count_customer_age_group.png` | Count plot for customer_age_group |
| `count_day_of_week.png` | Count plot for day_of_week |
| `count_drink_category.png` | Count plot for drink_category |
| `count_is_rewards_member.png` | Count plot for is_rewards_member |
| `count_has_food_item.png` | Count plot for has_food_item |
| `count_order_ahead.png` | Count plot for order_ahead |
| `correlation_heatmap.png` | Correlation heatmap of numeric features |
| `pairplot_numeric.png` | Pairplot coloured by rewards membership |
| `missing_values.png` | Bar chart of missing values per column |
