"""
Customer Segmentation Dashboard — Streamlit App
================================================
Run with: streamlit run dashboard.py

Requires:  customers_segmented.csv  (exported from the main notebook)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Colour palette (matches notebook) ──────────────────────────────────────
PALETTE = {
    "Premium Loyalists" : "#2E86AB",
    "Mid-tier Engagers" : "#A23B72",
    "Family First"      : "#F18F01",
    "Budget Browsers"   : "#C73E1D",
}

# ─── Load data ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str = "customers_segmented.csv") -> pd.DataFrame:
    return pd.read_csv(path)

try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️  `customers_segmented.csv` not found. Run the notebook first to generate it.")
    st.stop()

PERSONAS = df["Persona"].unique().tolist()

# ─── Sidebar filters ─────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/bar-chart.png", width=60)
st.sidebar.title("🔎 Filters")

selected_personas = st.sidebar.multiselect(
    "Customer Segments", PERSONAS, default=PERSONAS
)

age_range = st.sidebar.slider(
    "Age Range", int(df.Age.min()), int(df.Age.max()),
    (int(df.Age.min()), int(df.Age.max()))
)

income_range = st.sidebar.slider(
    "Income Range ($k)",
    int(df.Income.min() / 1000), int(df.Income.max() / 1000),
    (int(df.Income.min() / 1000), int(df.Income.max() / 1000))
)

# Apply filters
mask = (
    df["Persona"].isin(selected_personas) &
    df["Age"].between(*age_range) &
    df["Income"].between(income_range[0] * 1000, income_range[1] * 1000)
)
dff = df[mask]

# ─── Header ──────────────────────────────────────────────────────────────────
st.title("🛍️  Customer Segmentation Intelligence Dashboard")
st.caption("IBM Marketing Campaign Dataset · 2,240 Customers · K-Means Clustering (K=4)")
st.markdown("---")

# ─── KPI Row ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("Total Customers",    f"{len(dff):,}")
k2.metric("Avg Annual Spend",   f"${dff['Spend_Per_Month'].mean()*12:,.0f}")
k3.metric("Avg Income",         f"${dff['Income'].mean():,.0f}")
k4.metric("Avg Recency (days)", f"{dff['Recency'].mean():.0f}")
k5.metric("Segments Shown",     f"{dff['Persona'].nunique()}")

st.markdown("---")

# ─── Row 1: Segment overview ──────────────────────────────────────────────────
st.subheader("📊 Segment Overview")

col1, col2 = st.columns([1, 2])

with col1:
    seg_counts = dff["Persona"].value_counts().reset_index()
    seg_counts.columns = ["Persona", "Count"]
    fig_pie = px.pie(
        seg_counts, names="Persona", values="Count",
        color="Persona", color_discrete_map=PALETTE,
        title="Customer Distribution by Segment",
        hole=0.45
    )
    fig_pie.update_layout(margin=dict(t=40, b=10, l=10, r=10), height=350)
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    seg_spend = dff.groupby("Persona")["Total_Spend"].agg(
        Total_Revenue="sum", Avg_Spend="mean"
    ).reset_index().sort_values("Total_Revenue", ascending=False)

    fig_rev = px.bar(
        seg_spend, x="Persona", y="Total_Revenue",
        color="Persona", color_discrete_map=PALETTE,
        title="Total Revenue Contribution by Segment (2yr)",
        text_auto=".2s"
    )
    fig_rev.update_traces(textposition="outside")
    fig_rev.update_layout(showlegend=False, height=350, margin=dict(t=40,b=10))
    st.plotly_chart(fig_rev, use_container_width=True)

# ─── Row 2: Scatter + Boxplot ────────────────────────────────────────────────
st.subheader("📍 Customer Map — Income vs Spend")

col3, col4 = st.columns([2, 1])

with col3:
    fig_scatter = px.scatter(
        dff, x="Income", y="Total_Spend",
        color="Persona", color_discrete_map=PALETTE,
        hover_data=["Age", "Family_Size", "Total_Purchases", "Recency"],
        title="Income vs Total 2yr Spend",
        labels={"Income": "Annual Income ($)", "Total_Spend": "Total Spend ($)"},
        template="plotly_white", opacity=0.65
    )
    fig_scatter.update_traces(marker=dict(size=6))
    fig_scatter.update_layout(height=420)
    st.plotly_chart(fig_scatter, use_container_width=True)

with col4:
    feature = st.selectbox(
        "Feature to compare by segment:",
        ["Total_Spend", "Income", "Recency", "Deal_Sensitivity",
         "Online_Purchase_Ratio", "Total_Purchases", "Premium_Spend_Ratio"]
    )
    fig_box = px.box(
        dff, x="Persona", y=feature, color="Persona",
        color_discrete_map=PALETTE,
        title=f"{feature} by Segment",
        template="plotly_white"
    )
    fig_box.update_layout(showlegend=False, height=420,
                           xaxis=dict(tickangle=20))
    st.plotly_chart(fig_box, use_container_width=True)

# ─── Row 3: Radar fingerprints ────────────────────────────────────────────────
st.subheader("🕸️  Segment Fingerprints")

radar_features = ["Income", "Total_Spend", "Total_Purchases",
                   "Online_Purchase_Ratio", "Deal_Sensitivity",
                   "Total_Campaigns_Accepted", "Premium_Spend_Ratio"]
radar_labels   = ["Income", "Spend", "Frequency",
                   "Online", "Deals", "Campaigns", "Premium"]

seg_means  = dff.groupby("Persona")[radar_features].mean()
seg_norm   = (seg_means - seg_means.min()) / (seg_means.max() - seg_means.min() + 1e-9)

fig_radar  = go.Figure()
for persona in selected_personas:
    if persona not in seg_norm.index:
        continue
    vals = seg_norm.loc[persona].tolist()
    vals += vals[:1]
    fig_radar.add_trace(go.Scatterpolar(
        r=vals,
        theta=radar_labels + [radar_labels[0]],
        fill="toself",
        name=persona,
        line_color=PALETTE.get(persona, "#999"),
        opacity=0.6
    ))

fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    title="Normalised Feature Profile per Segment",
    height=500,
    template="plotly_white"
)
st.plotly_chart(fig_radar, use_container_width=True)

# ─── Row 4: Channel preference ────────────────────────────────────────────────
st.subheader("📡 Channel & Category Analysis")

col5, col6 = st.columns(2)

with col5:
    ch_cols  = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
    ch_names = ["Web", "Catalog", "In-Store"]
    ch_means = dff.groupby("Persona")[ch_cols].mean().rename(columns=dict(zip(ch_cols, ch_names)))
    fig_ch   = px.bar(
        ch_means.reset_index().melt(id_vars="Persona"),
        x="Persona", y="value", color="variable",
        barmode="group",
        title="Avg Purchases by Channel",
        labels={"value": "Avg # Purchases", "variable": "Channel"},
        template="plotly_white"
    )
    fig_ch.update_layout(xaxis=dict(tickangle=15), height=380)
    st.plotly_chart(fig_ch, use_container_width=True)

with col6:
    cat_cols   = ["MntWines", "MntMeatProducts", "MntFishProducts",
                   "MntFruits", "MntSweetProducts", "MntGoldProds"]
    cat_labels = ["Wines", "Meat", "Fish", "Fruits", "Sweets", "Gold"]
    # Only use cats if they exist in df
    available  = [c for c in cat_cols if c in dff.columns]
    avail_lbls = [l for c, l in zip(cat_cols, cat_labels) if c in dff.columns]

    if available:
        cat_means = dff.groupby("Persona")[available].mean().rename(
            columns=dict(zip(available, avail_lbls)))
        fig_cat = px.bar(
            cat_means.reset_index().melt(id_vars="Persona"),
            x="Persona", y="value", color="variable",
            barmode="stack",
            title="Avg Spend by Product Category",
            labels={"value": "Avg Spend ($)", "variable": "Category"},
            template="plotly_white"
        )
        fig_cat.update_layout(xaxis=dict(tickangle=15), height=380)
        st.plotly_chart(fig_cat, use_container_width=True)

# ─── Row 5: Segment profile cards ────────────────────────────────────────────
st.subheader("🃏 Segment Profile Cards")

playbook = {
    "Premium Loyalists": {
        "emoji"      : "💎",
        "description": "High-earning brand loyalists. Low deal sensitivity. Catalog & in-store buyers.",
        "channel"    : "📮 Catalog + 🏪 In-Store",
        "offer"      : "VIP early access, exclusive bundles, loyalty rewards",
        "avoid"      : "Heavy discounting",
        "roi"        : "⭐⭐⭐⭐⭐"
    },
    "Mid-tier Engagers": {
        "emoji"      : "📈",
        "description": "Moderate spenders, highly campaign-responsive. Aspiring to Premium.",
        "channel"    : "🌐 Omnichannel (web + store)",
        "offer"      : "Loyalty enrolment, upgrade nudges, new product trials",
        "avoid"      : "Over-discounting",
        "roi"        : "⭐⭐⭐⭐"
    },
    "Family First": {
        "emoji"      : "👨‍👩‍👧‍👦",
        "description": "Families with children. Practical, high-volume, staple category buyers.",
        "channel"    : "🏪 In-Store + 🌐 Web",
        "offer"      : "Family packs, subscriptions, BOGO on staples",
        "avoid"      : "Luxury single-serve products",
        "roi"        : "⭐⭐⭐"
    },
    "Budget Browsers": {
        "emoji"      : "🔍",
        "description": "Price-sensitive deal-seekers. High web visits, low conversion.",
        "channel"    : "🌐 Web + 📧 Email retargeting",
        "offer"      : "Flash sales, limited-time vouchers, bundle deals",
        "avoid"      : "Catalog (low ROI), premium positioning",
        "roi"        : "⭐⭐"
    },
}

cards = st.columns(len(selected_personas)) if selected_personas else []

for col, persona in zip(cards, selected_personas):
    info = playbook.get(persona, {})
    with col:
        st.markdown(
            f"""
            <div style="background:#f9f9f9;border-left:5px solid {PALETTE.get(persona,'#999')};
                        border-radius:8px;padding:16px;margin-bottom:8px;">
              <h4 style='margin:0'>{info.get('emoji','')} {persona}</h4>
              <p style='font-size:0.85em;color:#444'>{info.get('description','')}</p>
              <p><b>Channel:</b> {info.get('channel','')}</p>
              <p><b>Offer:</b> {info.get('offer','')}</p>
              <p><b>Avoid:</b> ❌ {info.get('avoid','')}</p>
              <p><b>ROI Potential:</b> {info.get('roi','')}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built with Python · scikit-learn · Plotly · Streamlit | IBM Marketing Campaign Dataset")
