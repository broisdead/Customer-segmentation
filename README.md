# 🛍️ Customer Segmentation & Marketing Intelligence

> **An industry-grade customer segmentation system that transforms raw CRM data into actionable marketing personas — with measurable business impact.**

---

## 📋 Project Overview

A retail company was spending its entire marketing budget on undifferentiated campaigns, achieving only ~6% average acceptance rates and wasting thousands of dollars per campaign on non-responders. This project applies end-to-end machine learning to segment 2,240 customers into four distinct behavioural groups, each with a tailored marketing strategy.

**Result:** Four clearly defined customer segments with estimated campaign ROI uplift of 20–30% through precision targeting.

---

## 🎯 Business Problem

> *"How can we group our customers into distinct behavioural segments so that each marketing campaign targets the right audience — maximising ROI and minimising churn?"*

### Stakeholders & Decisions Supported

| Stakeholder | Decision Supported |
|---|---|
| Marketing Team | Which channels and offers to deploy per segment |
| Product Team | Which categories to promote to which customers |
| CRM / Retention | Which customers to prioritise for win-back |
| Finance | Revenue forecasting and CLV estimation by segment |

---

## 📊 Dataset

**Source:** IBM Marketing Campaign Dataset  
**Size:** 2,240 customers × 29 features  
**Coverage:** 2-year purchase window, 6 marketing campaigns

**Raw features include:**
- Demographic: `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome`
- Spend: `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds`
- Purchases: `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumDealsPurchases`
- Engagement: `AcceptedCmp1–5`, `Response`, `Recency`, `NumWebVisitsMonth`

---

## 🔬 Methodology

```
Raw Data (29 features)
    │
    ▼
EDA — 4 business questions explored:
  • Who are our customers demographically?
  • How do spending patterns differ?
  • Which channels do customers prefer?
  • Why are our campaigns underperforming?
    │
    ▼
Feature Engineering (14 derived features)
  • RFM-style: Total_Spend, Spend_Per_Month, Recency (R)
  • Behavioural: Premium_Spend_Ratio, Deal_Sensitivity
  • Channel: Online_Purchase_Ratio, Total_Purchases
  • Engagement: Total_Campaigns_Accepted, Spend_Per_Campaign
    │
    ▼
Preprocessing
  • Missing income (1.1%) → dropped
  • Outlier removal (IQR × 1.5) on Age and Income
  • MinMax scaling (distance-based algorithm requirement)
  • PCA → n components retaining ≥85% variance
    │
    ▼
K Selection (Elbow + Silhouette + Davies-Bouldin) → K=4
    │
    ▼
Model Comparison (4 algorithms)
  K-Means · Agglomerative · GMM · DBSCAN
    │
    ▼
Final Model: K-Means (K=4)
  Best Silhouette score + highest business interpretability
    │
    ▼
Business Layer
  Named personas · Radar fingerprints · Revenue analysis
  Campaign ROI simulation · Marketing playbook
```

---

## 👥 Key Results — Customer Segments

| Segment | Size | Avg Annual Spend | Revenue Share | Strategic Priority |
|---------|------|-----------------|---------------|--------------------|
| 💎 Premium Loyalists | ~25% | Highest | ~50% | Retain at all costs |
| 📈 Mid-tier Engagers | ~25% | Moderate | ~25% | Convert to Premium |
| 👨‍👩‍👧‍👦 Family First | ~25% | Moderate | ~15% | Grow basket size |
| 🔍 Budget Browsers | ~25% | Lowest | ~10% | Selective activation |

### Segment Characteristics

**💎 Premium Loyalists**
- High income, high spend, low deal sensitivity
- Catalog and in-store buyers
- Purchased across multiple campaigns — brand loyal
- **Action:** VIP programs, early access, exclusive bundles. Never discount.

**📈 Mid-tier Engagers**
- Moderate income, balanced channel use
- Highest campaign acceptance rate of any group
- Aspiring buyers who can be moved upmarket
- **Action:** Loyalty program enrolment, upgrade nudges, omnichannel campaigns

**👨‍👩‍👧‍👦 Family First**
- Multiple children, larger households
- Practical, staple-category buyers; in-store preferred
- Moderate spend per person but high purchase volume
- **Action:** Family packs, subscriptions, BOGO — weekend/holiday timing

**🔍 Budget Browsers**
- Lower income, high web visits but low conversion
- Deal-sensitive; respond only to clear value offers
- High browse-without-buying behaviour
- **Action:** Flash sales, email retargeting, time-limited vouchers only

---

## 💡 Business Insights & Recommendations

1. **The 80/20 Rule Holds:** ~25% of customers (Premium Loyalists) generate ~50% of revenue. Every $ spent retaining them delivers outsized return.

2. **Campaigns Are Wasted on the Wrong People:** Budget Browsers have the lowest campaign ROI but likely receive the same campaigns as Premium Loyalists. Segment-first targeting can reduce wasted contact cost by 20–30%.

3. **Mid-tier Engagers Are the Growth Engine:** They are the most campaign-responsive segment. A loyalty programme nudging them toward Premium behaviour is the highest-upside marketing investment.

4. **Family First Needs Volume Offers:** This segment doesn't buy premium — they buy practical. Subscription boxes and family packs increase basket size without requiring income they don't have.

5. **Web Experience Needs Work:** Budget Browsers have the highest web visit rate but lowest conversion — an untapped retargeting opportunity.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `Python 3.10+` | Core language |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Clustering, PCA, metrics |
| `scipy` | Statistics, dendrogram |
| `matplotlib` / `seaborn` | Static visualisations |
| `plotly` | Interactive charts |
| `streamlit` | Dashboard |

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn plotly streamlit
```

### 2. Get the data

Download the [IBM Marketing Campaign Dataset](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis) and save as `marketing_campaign.csv` in the project root.

### 3. Run the notebook

```bash
jupyter notebook customer_segmentation_enhanced.ipynb
```

Or on Google Colab — update the file path in Cell 2 to your Drive location.

### 4. Launch the dashboard

After running the notebook (which exports `customers_segmented.csv`):

```bash
streamlit run dashboard.py
```

---

## 📁 Project Structure

```
customer-segmentation/
├── customer_segmentation_enhanced.ipynb  # Main analysis notebook
├── dashboard.py                           # Streamlit interactive dashboard
├── marketing_campaign.csv                 # Raw input data (download separately)
├── customers_segmented.csv                # Exported: customers with cluster labels
├── segment_revenue_summary.csv            # Exported: revenue table by segment
└── README.md                              # This file
```

---

## 📈 Model Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Silhouette Score | ~0.35 | Good cluster separation (> 0.30 threshold) |
| Davies-Bouldin Index | < 1.0 | Compact clusters relative to their separation |
| Calinski-Harabasz | High | Cluster variance clearly exceeds within-cluster variance |
| Cluster balance | ~25% each | No single cluster dominates — all are actionable at scale |

---

## 🔄 Reproducibility

- Random seed fixed at `42` across all stochastic operations
- All preprocessing steps logged with before/after counts
- Persona assignment is dynamic (rank-based) — stable across re-runs even if cluster IDs shift

---

## 🧠 What Makes This Industry-Grade

✅ **Business framing first** — problem statement, stakeholders, success criteria defined before any code  
✅ **EDA structured as business questions** — not just charts, but insights + implications  
✅ **Feature engineering with rationale** — every feature maps to a marketing decision  
✅ **4-algorithm comparison** — model selection justified with 3 metrics, not just silhouette  
✅ **Business layer** — segments named, profiled in non-technical language, with marketing playbooks  
✅ **Revenue quantification** — CLV proxies, campaign ROI simulation per segment  
✅ **Interactive dashboard** — stakeholders can explore without touching the notebook  
✅ **Reproducible pipeline** — fixed seeds, documented preprocessing, exportable outputs  


