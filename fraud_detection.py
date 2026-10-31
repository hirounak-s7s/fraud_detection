import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────── Page Config ────────────────────────────
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────── Custom CSS ─────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root variables ── */
:root {
    --bg-primary: #0a0e17;
    --bg-card: rgba(15, 23, 42, 0.65);
    --border-glow: rgba(99, 102, 241, 0.25);
    --accent: #6366f1;
    --accent-light: #818cf8;
    --success: #10b981;
    --danger: #ef4444;
    --danger-light: #fca5a5;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
}

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header {visibility: hidden;}

/* ── Glass cards ── */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border-glow);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.15);
}

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 50%, #0c4a6e 100%);
    border-radius: 20px;
    padding: 48px 40px;
    margin-bottom: 32px;
    text-align: center;
    border: 1px solid rgba(99, 102, 241, 0.2);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 40%, rgba(99,102,241,0.08) 0%, transparent 50%),
                radial-gradient(circle at 70% 60%, rgba(14,165,233,0.06) 0%, transparent 50%);
    animation: drift 12s ease-in-out infinite alternate;
}
@keyframes drift {
    0%   { transform: translate(0, 0); }
    100% { transform: translate(-30px, 20px); }
}
.hero h1 {
    position: relative;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #c7d2fe 0%, #818cf8 40%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}
.hero p {
    position: relative;
    color: var(--text-secondary);
    font-size: 1.05rem;
    max-width: 640px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Section titles ── */
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-subtitle {
    font-size: 0.85rem;
    color: var(--text-secondary);
    margin-bottom: 18px;
}

/* ── Result cards ── */
.result-safe {
    background: linear-gradient(135deg, rgba(16,185,129,0.12) 0%, rgba(15,23,42,0.65) 100%);
    border: 1px solid rgba(16,185,129,0.35);
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    animation: fadeSlideUp 0.5s ease-out;
}
.result-fraud {
    background: linear-gradient(135deg, rgba(239,68,68,0.14) 0%, rgba(15,23,42,0.65) 100%);
    border: 1px solid rgba(239,68,68,0.35);
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    animation: fadeSlideUp 0.5s ease-out;
}
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-icon { font-size: 3rem; margin-bottom: 8px; }
.result-label {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 4px;
}
.result-prob {
    font-size: 0.95rem;
    color: var(--text-secondary);
}

/* ── Probability bar ── */
.prob-bar-bg {
    width: 100%;
    height: 10px;
    background: rgba(255,255,255,0.07);
    border-radius: 999px;
    margin-top: 16px;
    overflow: hidden;
}
.prob-bar-fill-safe {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #10b981, #34d399);
    transition: width 0.8s cubic-bezier(0.22, 1, 0.36, 1);
}
.prob-bar-fill-fraud {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #ef4444, #f87171);
    transition: width 0.8s cubic-bezier(0.22, 1, 0.36, 1);
}

/* ── Metric pills ── */
.metric-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 16px;
    justify-content: center;
}
.metric-pill {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 10px 18px;
    text-align: center;
    min-width: 120px;
}
.metric-pill .label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-secondary);
    margin-bottom: 4px;
}
.metric-pill .value {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
}

/* ── Sidebar polish ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%) !important;
    border-right: 1px solid rgba(99,102,241,0.15) !important;
}

/* ── Streamlit input overrides ── */
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    border-radius: 10px !important;
}

/* ── Button overrides ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    padding: 14px 0 !important;
    border-radius: 12px !important;
    border: none !important;
    cursor: pointer;
    transition: all 0.3s ease !important;
    letter-spacing: 0.3px;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(99,102,241,0.35) !important;
}
.stButton > button:active {
    transform: scale(0.98) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    font-weight: 600 !important;
    color: var(--accent-light) !important;
}

/* ── Tooltip info cards in sidebar ── */
.info-chip {
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 0.82rem;
    line-height: 1.5;
    color: var(--text-secondary);
}
.info-chip strong { color: var(--accent-light); }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────── Load Model ─────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("Fraud_Detection_Pipeline.pkl")


model = load_model()

# ──────────────────────────── Hero ───────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🛡️ Fraud Detection System</h1>
    <p>
        AI-powered transaction screening built on logistic regression.
        Enter transaction details below to get an instant risk verdict
        with probability scores.
    </p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────── Sidebar ────────────────────────────────
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    st.markdown("---")
    st.markdown("""
    <div class="info-chip">
        <strong>Model:</strong> Logistic Regression<br>
        <strong>Preprocessing:</strong> StandardScaler + OneHotEncoder<br>
        <strong>Class Weight:</strong> Balanced
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📖 Feature Guide")
    st.markdown("""
    <div class="info-chip">
        <strong>Transaction Type</strong> – Category of the mobile-money transfer
        (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER).
    </div>
    <div class="info-chip">
        <strong>Amount</strong> – The monetary value of the transaction.
    </div>
    <div class="info-chip">
        <strong>Old Balance (Origin)</strong> – Sender's balance before the transaction.
    </div>
    <div class="info-chip">
        <strong>New Balance (Origin)</strong> – Sender's balance after the transaction.
    </div>
    <div class="info-chip">
        <strong>Old Balance (Dest)</strong> – Recipient's balance before the transaction.
    </div>
    <div class="info-chip">
        <strong>New Balance (Dest)</strong> – Recipient's balance after the transaction.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;color:#64748b;font-size:0.75rem;'>"
        "Built with Streamlit &bull; scikit-learn</p>",
        unsafe_allow_html=True,
    )

# ──────────────────────────── Input Form ─────────────────────────────
st.markdown("""
<div class="section-title">📝 Transaction Details</div>
<div class="section-subtitle">Fill in the fields below to analyse a transaction.</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    transaction_type = st.selectbox(
        "💳 Transaction Type",
        options=["CASH_OUT", "TRANSFER", "CASH_IN", "DEBIT", "PAYMENT"],
        help="Select the type of mobile-money transaction.",
    )
    amount = st.number_input(
        "💰 Transaction Amount",
        min_value=0.0,
        value=10000.0,
        step=100.0,
        format="%.2f",
        help="The monetary value of the transaction.",
    )
    old_balance_org = st.number_input(
        "🏦 Old Balance (Origin)",
        min_value=0.0,
        value=50000.0,
        step=100.0,
        format="%.2f",
        help="Sender's account balance before the transaction.",
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    new_balance_org = st.number_input(
        "📤 New Balance (Origin)",
        min_value=0.0,
        value=40000.0,
        step=100.0,
        format="%.2f",
        help="Sender's account balance after the transaction.",
    )
    old_balance_dest = st.number_input(
        "📥 Old Balance (Destination)",
        min_value=0.0,
        value=20000.0,
        step=100.0,
        format="%.2f",
        help="Recipient's account balance before the transaction.",
    )
    new_balance_dest = st.number_input(
        "🏦 New Balance (Destination)",
        min_value=0.0,
        value=30000.0,
        step=100.0,
        format="%.2f",
        help="Recipient's account balance after the transaction.",
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ── Derived features (auto-computed) ──
balance_diff_orig = new_balance_org - old_balance_org
balance_diff_dest = new_balance_dest - old_balance_dest

with st.expander("🔍 Auto-Computed Derived Features", expanded=False):
    dc1, dc2 = st.columns(2)
    dc1.metric("Balance Diff (Origin)", f"${balance_diff_orig:,.2f}")
    dc2.metric("Balance Diff (Destination)", f"${balance_diff_dest:,.2f}")

# ──────────────────────────── Predict ────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

predict_clicked = st.button("🔍  Analyse Transaction", use_container_width=True)

if predict_clicked:
    # Build the DataFrame the pipeline expects
    input_df = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": old_balance_org,
        "newbalanceOrig": new_balance_org,
        "oldbalanceDest": old_balance_dest,
        "newbalanceDest": new_balance_dest,
        "balanceDiffOrig": balance_diff_orig,
        "balanceDiffDest": balance_diff_dest,
    }])

    # Prediction
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    prob_legit = probabilities[0] * 100
    prob_fraud = probabilities[1] * 100

    st.markdown("---")
    st.markdown("""
    <div class="section-title">📊 Analysis Result</div>
    """, unsafe_allow_html=True)

    res_col1, res_col2 = st.columns([2, 1], gap="large")

    with res_col1:
        if prediction == 1:
            st.markdown(f"""
            <div class="result-fraud">
                <div class="result-icon">🚨</div>
                <div class="result-label" style="color:#fca5a5;">FRAUDULENT TRANSACTION</div>
                <div class="result-prob">
                    Fraud probability: <strong style="color:#f87171;">{prob_fraud:.1f}%</strong>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill-fraud" style="width:{prob_fraud}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-safe">
                <div class="result-icon">✅</div>
                <div class="result-label" style="color:#6ee7b7;">LEGITIMATE TRANSACTION</div>
                <div class="result-prob">
                    Legitimacy confidence: <strong style="color:#34d399;">{prob_legit:.1f}%</strong>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill-safe" style="width:{prob_legit}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with res_col2:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;">
            <div class="section-title" style="justify-content:center;">Probability Breakdown</div>
            <div class="metric-row" style="flex-direction:column;align-items:center;">
                <div class="metric-pill" style="width:100%;">
                    <div class="label">Legitimate</div>
                    <div class="value" style="color:#10b981;">{prob_legit:.2f}%</div>
                </div>
                <div class="metric-pill" style="width:100%;">
                    <div class="label">Fraudulent</div>
                    <div class="value" style="color:#ef4444;">{prob_fraud:.2f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Transaction summary pills ──
    st.markdown(f"""
    <div class="metric-row" style="justify-content:center;margin-top:24px;">
        <div class="metric-pill">
            <div class="label">Type</div>
            <div class="value">{transaction_type}</div>
        </div>
        <div class="metric-pill">
            <div class="label">Amount</div>
            <div class="value">${amount:,.2f}</div>
        </div>
        <div class="metric-pill">
            <div class="label">Origin Δ</div>
            <div class="value">${balance_diff_orig:,.2f}</div>
        </div>
        <div class="metric-pill">
            <div class="label">Dest Δ</div>
            <div class="value">${balance_diff_dest:,.2f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Raw data expander ──
    with st.expander("🗂️ Raw Input Sent to Model"):
        st.dataframe(input_df, use_container_width=True)
