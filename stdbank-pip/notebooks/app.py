"""
Fraud detection dashboard for the PaySim mobile money dataset.

Run with:
    streamlit run app.py

Expects the artifacts folder produced by train_and_save.py to sit next to this file.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st

matplotlib.use('Agg')

HERE = os.path.dirname(os.path.abspath(__file__))
REQUIRED = ('fraud_model.joblib', 'model_metadata.json', 'eda_summary.json')


def candidate_dirs():
    """Every sensible place the artifacts might be, in priority order."""
    seen, out = set(), []
    roots = [os.environ.get('FRAUD_ARTIFACTS'), HERE, os.getcwd(),
             os.path.dirname(HERE), os.path.dirname(os.getcwd())]
    for root in roots:
        if not root:
            continue
        for path in (root, os.path.join(root, 'artifacts'),
                     os.path.join(root, 'fraud_dashboard', 'artifacts'),
                     os.path.join(root, 'notebooks', 'artifacts'),
                     os.path.join(root, 'models'), os.path.join(root, 'output')):
            full = os.path.abspath(path)
            if full not in seen:
                seen.add(full)
                out.append(full)
    return out


def find_artifacts():
    """Return the first directory holding all three artifacts, plus the search trail."""
    tried = []
    for path in candidate_dirs():
        missing = [f for f in REQUIRED if not os.path.isfile(os.path.join(path, f))]
        tried.append((path, missing))
        if not missing:
            return path, tried
    return None, tried


ARTIFACTS, SEARCH_TRAIL = find_artifacts()

# ----------------------------------------------------------------- palette
PAPER = '#F4F1EA'
INK = '#1B1D1F'
CLAY = '#A8482A'
SAGE = '#4F6B52'
SAND = '#C9A227'
MUTED = '#8B857A'
LINE = '#D8D2C6'

st.set_page_config(page_title='Fraud Detection', layout='wide',
                   initial_sidebar_state='expanded')

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@300;400;500;600&display=swap');

.stApp {{ background: {PAPER}; }}
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; color: {INK}; }}

/* Force readable text even when Streamlit is running its dark theme */
.stApp, .stApp p, .stApp li, .stApp span, .stApp label, .stApp div,
.stApp td, .stApp th, .stApp strong, .stApp b, .stApp em,
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] * {{
    color: {INK};
}}
.stApp a {{ color: {CLAY}; }}
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] * {{ color: {INK} !important; }}
div[data-baseweb="select"] *, div[data-baseweb="input"] * {{ color: {INK} !important; }}
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {{
    background: #FBFAF7 !important; border-color: {LINE} !important; }}
.stApp input, .stApp textarea {{ color: {INK} !important; background: #FBFAF7 !important; }}
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"],
[data-testid="stThumbValue"] {{ color: {MUTED} !important; }}
[data-testid="stDataFrame"] * {{ color: {INK} !important; }}

h1, h2, h3 {{ font-family: 'Fraunces', Georgia, serif !important;
              font-weight: 600 !important; letter-spacing: -0.4px; color: {INK} !important; }}
h1 {{ font-size: 2.6rem !important; line-height: 1.1; margin-bottom: 0.2rem !important; }}
h2 {{ font-size: 1.55rem !important; margin-top: 2.2rem !important; }}
h3 {{ font-size: 1.15rem !important; }}

.eyebrow {{ font-family:'Inter',sans-serif; font-size:0.72rem; font-weight:600;
            letter-spacing:0.18em; text-transform:uppercase; color:{CLAY}; margin-bottom:0.5rem; }}
.lede {{ font-size:1.02rem; line-height:1.65; color:#3A3D40; max-width:60rem; }}
.rule {{ height:1px; background:{LINE}; margin:1.6rem 0 1.4rem 0; }}

.statgrid {{ display:flex; gap:1rem; flex-wrap:wrap; margin:0.6rem 0 0.4rem 0; }}
.stat {{ flex:1; min-width:150px; background:#FBFAF7; border:1px solid {LINE};
         border-left:3px solid {CLAY}; padding:0.9rem 1.05rem; }}
.stat .k {{ font-family:'Fraunces',serif; font-size:1.85rem; font-weight:600; line-height:1.1; }}
.stat .l {{ font-size:0.74rem; letter-spacing:0.07em; text-transform:uppercase;
            color:{MUTED}; margin-top:0.3rem; }}
.stat.sage {{ border-left-color:{SAGE}; }}
.stat.sand {{ border-left-color:{SAND}; }}

.note {{ background:#FBFAF7; border:1px solid {LINE}; border-left:3px solid {SAGE};
         padding:1rem 1.15rem; margin:1.1rem 0; font-size:0.93rem; line-height:1.6; }}
.warn {{ background:#FBF6F3; border:1px solid #E4D3CA; border-left:3px solid {CLAY};
         padding:1rem 1.15rem; margin:1.1rem 0; font-size:0.93rem; line-height:1.6; }}

.verdict {{ padding:1.4rem 1.6rem; border:1px solid {LINE}; background:#FBFAF7; }}
.verdict .headline {{ font-family:'Fraunces',serif; font-size:1.7rem; font-weight:600; }}
.verdict .sub {{ color:{MUTED}; font-size:0.85rem; letter-spacing:0.06em;
                 text-transform:uppercase; margin-top:0.25rem; }}

section[data-testid="stSidebar"] {{ background:#EDE9E0; border-right:1px solid {LINE}; }}
section[data-testid="stSidebar"] * {{ color:{INK} !important; }}
section[data-testid="stSidebar"] .stRadio label {{ font-size:0.95rem; }}
.stat .l, .verdict .sub {{ color:{MUTED} !important; }}

div[data-testid="stDataFrame"] {{ border:1px solid {LINE}; }}
.stButton button {{ background:{INK}; color:{PAPER}; border:none; border-radius:0;
                    font-weight:500; letter-spacing:0.03em; padding:0.55rem 1.6rem; }}
.stButton button:hover {{ background:{CLAY}; color:{PAPER}; }}
#MainMenu, footer {{ visibility:hidden; }}
</style>
""", unsafe_allow_html=True)


def style_axes(ax, ylabel='', xlabel=''):
    ax.set_facecolor(PAPER)
    ax.figure.patch.set_facecolor(PAPER)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.grid(axis='y', color=LINE, linewidth=0.7)
    ax.set_axisbelow(True)
    return ax


@st.cache_resource
def load_model():
    return joblib.load(os.path.join(ARTIFACTS, 'fraud_model.joblib'))


@st.cache_data
def load_json(name):
    with open(os.path.join(ARTIFACTS, name)) as fh:
        return json.load(fh)


if ARTIFACTS is None:
    st.markdown("<div class='eyebrow'>Setup needed</div>", unsafe_allow_html=True)
    st.markdown('# The model files were not found')
    st.markdown(
        "<p class='lede'>The dashboard needs three files: <code>fraud_model.joblib</code>, "
        "<code>model_metadata.json</code> and <code>eda_summary.json</code>. "
        "They should sit together in a folder named <code>artifacts</code>.</p>",
        unsafe_allow_html=True)

    st.markdown('### Fastest fix')
    st.markdown('Point the app straight at the folder holding those three files:')
    st.code('# Windows PowerShell\n'
            '$env:FRAUD_ARTIFACTS = "C:\\\\path\\\\to\\\\artifacts"\n'
            'streamlit run app.py', language='powershell')
    st.markdown('Or generate them from the raw data:')
    st.code('python train_and_save.py', language='bash')
    st.markdown('Or run the final section of the notebook, which writes the same three files.')

    manual = st.text_input('Alternatively, paste the folder path here')
    if manual:
        clean = os.path.abspath(manual.strip().strip('"').strip("'"))
        missing = [f for f in REQUIRED if not os.path.isfile(os.path.join(clean, f))]
        if missing:
            st.warning(f'That folder is missing: {", ".join(missing)}')
        else:
            st.success('Found them. Set FRAUD_ARTIFACTS to this path and restart, '
                       'or move the folder next to app.py.')

    st.markdown('### Where the app looked')
    st.dataframe(pd.DataFrame(
        [{'folder searched': p,
          'result': 'all three present' if not m else f'missing {len(m)} of 3'}
         for p, m in SEARCH_TRAIL]),
        hide_index=True, use_container_width=True)

    st.markdown(f"<div class='note'>Current working directory: "
                f"<code>{os.getcwd()}</code><br>Location of app.py: "
                f"<code>{HERE}</code></div>", unsafe_allow_html=True)
    st.stop()

def normalise_meta(meta, eda):
    """Fill in anything a leaner metadata file is missing.

    The notebook writes a compact metadata file and train_and_save.py writes a
    fuller one. Rather than demand a particular version, we derive what we can
    and mark the rest as unavailable so pages can degrade gracefully.
    """
    meta = dict(meta)
    costs = meta.setdefault('costs', {})
    cost_fn = costs.setdefault('missed_fraud', 5000.0)
    cost_fp = costs.setdefault('false_alarm', 50.0)

    # Number of frauds in the test set is recoverable from the do nothing cost
    if 'test_fraud_count' not in meta:
        if costs.get('baseline_no_model'):
            meta['test_fraud_count'] = int(round(costs['baseline_no_model'] / cost_fn))
        else:
            meta['test_fraud_count'] = 0
    n_fraud = meta['test_fraud_count']
    costs.setdefault('baseline_no_model', float(n_fraud * cost_fn))

    meta.setdefault('test_fraud_rate', eda.get('fraud_rate', 0.0))
    if 'test_rows' not in meta:
        rate = meta['test_fraud_rate'] or eda.get('fraud_rate', 0.0)
        meta['test_rows'] = int(round(n_fraud / rate)) if rate else 0

    # Each threshold entry needs its confusion counts and total cost
    for entry in meta.get('thresholds', {}).values():
        precision = entry.get('precision', 0.0)
        recall = entry.get('recall', 0.0)
        tp = entry.get('true_positives', int(round(recall * n_fraud)))
        fn = entry.get('false_negatives', max(n_fraud - tp, 0))
        alerts = entry.get('alerts', int(round(tp / precision)) if precision > 0 else tp)
        fp = entry.get('false_positives', max(alerts - tp, 0))
        entry.update({'true_positives': tp, 'false_negatives': fn,
                      'false_positives': fp, 'alerts': alerts})
        entry.setdefault('total_cost', float(fn * cost_fn + fp * cost_fp))

    if 'accuracy_trap' not in meta:
        rate = meta['test_fraud_rate'] or eda.get('fraud_rate', 0.0)
        meta['accuracy_trap'] = {'always_normal_accuracy': 1.0 - rate}

    # Results carried over from the notebook, used only for the comparison chart
    meta.setdefault('model_comparison', [
        {'model': 'Logistic Regression (baseline)', 'basic': 0.055, 'balances': 0.091, 'errors': 0.466},
        {'model': 'Decision Tree', 'basic': 0.236, 'balances': 0.198, 'errors': 0.272},
        {'model': 'Random Forest', 'basic': 0.204, 'balances': 0.716, 'errors': 0.845},
        {'model': 'XGBoost', 'basic': 0.283, 'balances': 0.836, 'errors': 0.990},
    ])
    meta['has_pr_curve'] = 'pr_curve' in meta
    return meta


model = load_model()
meta = normalise_meta(load_json('model_metadata.json'), load_json('eda_summary.json'))
eda = load_json('eda_summary.json')

FEATURES = list(meta['features'])

# The model knows the exact column order it was trained on. Trust that over the
# metadata file: the notebook and train_and_save.py list the features in different
# orders, and feeding columns in the wrong sequence gives silently wrong answers.
try:
    trained_order = model.get_booster().feature_names
except Exception:
    trained_order = None

if trained_order and set(trained_order) == set(FEATURES):
    FEATURE_ORDER_FIXED = trained_order != FEATURES
    FEATURES = list(trained_order)
else:
    FEATURE_ORDER_FIXED = False

TYPE_COLS = [c for c in FEATURES if c.startswith('type_')]
ALL_TYPES = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']


LOW_ACTIVITY = meta.get('low_activity_positions',
                        eda.get('low_activity_positions', []))


def build_row(txn_type, amount, old_org, old_dest, cycle_position):
    """Turn raw transaction inputs into the model feature vector."""
    row = {
        'log_amount': np.log1p(max(amount, 0)),
        'cycle_position': cycle_position,
        'is_low_activity': int(cycle_position in LOW_ACTIVITY),
        'log_old_org': np.log1p(max(old_org, 0)),
        'log_old_dest': np.log1p(max(old_dest, 0)),
    }
    for col in TYPE_COLS:
        row[col] = int(col == f'type_{txn_type}')
    return pd.DataFrame([row])[FEATURES].astype(float)


# ------------------------------------------------------------------ sidebar
st.sidebar.markdown(
    f"<div style='font-family:Fraunces,serif;font-size:1.3rem;font-weight:600;"
    f"padding:0.4rem 0 0.1rem 0;'>Fraud Detection</div>"
    f"<div style='font-size:0.76rem;letter-spacing:0.13em;text-transform:uppercase;"
    f"color:{MUTED};margin-bottom:1.2rem;'>PaySim mobile money</div>",
    unsafe_allow_html=True)

page = st.sidebar.radio('Section', [
    'Overview',
    'Explore the data',
    'Model performance',
    'Score a transaction',
    'Recommendations',
], label_visibility='collapsed')

st.sidebar.markdown(f"<div class='rule'></div>", unsafe_allow_html=True)
st.sidebar.markdown(
    f"<div style='font-size:0.8rem;color:{MUTED};line-height:1.6;'>"
    f"Model: XGBoost<br>Features: {len(FEATURES)}<br>"
    f"PR-AUC: {meta['pr_auc']:.3f}<br>ROC-AUC: {meta['roc_auc']:.3f}</div>",
    unsafe_allow_html=True)

if FEATURE_ORDER_FIXED:
    st.sidebar.markdown(
        f"<div style='font-size:0.76rem;color:{CLAY};line-height:1.5;margin-top:0.8rem;'>"
        f"Note: the metadata listed features in a different order to the trained model. "
        f"The model's own order is being used.</div>", unsafe_allow_html=True)


# ===================================================================== pages
def page_overview():
    st.markdown("<div class='eyebrow'>Standard Bank challenge</div>", unsafe_allow_html=True)
    st.markdown('# Catching fraud in mobile money')
    st.markdown(
        "<p class='lede'>Fraud is roughly one transaction in every seven hundred and fifty. "
        "That rarity is the whole problem: a model that simply approves everything is right "
        "99.87 percent of the time and catches nothing. This dashboard covers what the data "
        "shows, how the model performs, how it scores a single transaction, and what we "
        "recommend the bank actually does.</p>", unsafe_allow_html=True)
    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)

    cost = meta['costs']
    co = meta['thresholds']['cost_optimal']
    st.markdown(f"""
    <div class='statgrid'>
      <div class='stat'><div class='k'>{eda['fraud_rate']*100:.2f}%</div>
        <div class='l'>Fraud rate</div></div>
      <div class='stat sage'><div class='k'>{co['recall']*100:.0f}%</div>
        <div class='l'>Fraud caught</div></div>
      <div class='stat sand'><div class='k'>{meta['pr_auc']:.3f}</div>
        <div class='l'>PR-AUC</div></div>
      <div class='stat sage'><div class='k'>R{(cost['baseline_no_model']-co['total_cost'])/1000:,.0f}k</div>
        <div class='l'>Loss avoided</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('## What we found')
    left, right = st.columns(2)
    with left:
        st.markdown(f"""
**Fraud hides in two places only.** Every fraudulent transaction in the data is either a
TRANSFER or a CASH_OUT. Payments, debits and cash deposits are clean. That alone removes more
than half the book from suspicion.

**The quiet part of the cycle is where it shows.** Transaction volume follows a repeating
24 step cycle. In the quietest positions the fraud rate is
{eda['activity_split']['low_rate']*100:.2f} percent against
{eda['activity_split']['busy_rate']*100:.2f} percent in the busy ones, because genuine
activity drops away while fraud continues at a steady pace.
""")
    with right:
        st.markdown(f"""
**Amounts are far larger.** The median fraudulent transaction is
R{eda['amount_stats']['fraud_median']:,.0f} against R{eda['amount_stats']['normal_median']:,.0f}
for a normal one, roughly six times bigger.

**Accuracy is a trap.** Approving every transaction scores
{meta['accuracy_trap']['always_normal_accuracy']*100:.2f} percent accuracy and catches zero
fraud. Every number on the performance page is built to avoid that illusion.
""")

    st.markdown("<div class='warn'><b>An honest caveat.</b> PaySim is simulated. Its fraud "
                "almost always empties the victim account to the last cent, so the amount and "
                "the balance together give the answer away more reliably than they ever would "
                "on real bank data. We expect lower numbers in production and have said so "
                "throughout rather than presenting these figures as a forecast.</div>",
                unsafe_allow_html=True)


def page_eda():
    st.markdown("<div class='eyebrow'>Deliverable one</div>", unsafe_allow_html=True)
    st.markdown('# Explore the data')
    st.markdown("<p class='lede'>Four questions: how rare is fraud, where does it hide, "
                "when does it happen, and how large is it.</p>", unsafe_allow_html=True)
    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='statgrid'>
      <div class='stat'><div class='k'>{eda['total_rows']:,}</div><div class='l'>Transactions</div></div>
      <div class='stat sand'><div class='k'>{eda['fraud_count']:,}</div><div class='l'>Fraud cases</div></div>
      <div class='stat'><div class='k'>{eda['fraud_rate']*100:.3f}%</div><div class='l'>Fraud rate</div></div>
      <div class='stat sage'><div class='k'>1 in {int(1/eda['fraud_rate'])}</div><div class='l'>Odds of fraud</div></div>
    </div>""", unsafe_allow_html=True)

    # ---------------------------------------------------------- by type
    st.markdown('## Where fraud hides')
    by_type = pd.DataFrame(eda['fraud_by_type'])
    c1, c2 = st.columns([1.15, 1])
    with c1:
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        colors = [CLAY if r > 0 else MUTED for r in by_type['fraud']]
        ax.bar(by_type['type'], by_type['transactions'], color=colors, width=0.62)
        style_axes(ax, 'Transactions')
        ax.set_title('Volume by type, types containing fraud in clay',
                     fontsize=10, color=INK, loc='left', pad=12)
        plt.xticks(rotation=20)
        st.pyplot(fig, use_container_width=True)
    with c2:
        show = by_type.copy()
        show['rate'] = (show['rate'] * 100).round(3).astype(str) + '%'
        show.columns = ['Type', 'Transactions', 'Fraud', 'Fraud rate']
        st.dataframe(show, hide_index=True, use_container_width=True)
        st.markdown("<div class='note'>TRANSFER and CASH_OUT carry every single fraud case. "
                    "A first line of defence could ignore the other three types entirely and "
                    "lose nothing.</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------- by hour
    st.markdown('## Where fraud sits in the activity cycle')
    st.markdown("<div class='note'>A note on what we can and cannot claim. The data card "
                "tells us one step is one hour and the simulation runs for 30 days. It does "
                "<b>not</b> tell us which clock time step 1 corresponds to, so we do not label "
                "these positions with real hours. We only use what the data supports: a "
                "repeating 24 step cycle, and the fact that some positions carry far less "
                "traffic than others.</div>", unsafe_allow_html=True)

    cycle = pd.DataFrame(eda['cycle'])
    fig, ax = plt.subplots(figsize=(11, 3.4))
    bar_colours = ['#B9AE97' if p in LOW_ACTIVITY else '#CFC8B8'
                   for p in cycle['cycle_position']]
    ax.bar(cycle['cycle_position'], cycle['transactions'], color=bar_colours, width=0.7,
           label='transaction volume')
    style_axes(ax, 'Transactions', 'Position in the 24 step cycle')
    ax2 = ax.twinx()
    ax2.plot(cycle['cycle_position'], cycle['rate'] * 100, color=CLAY, linewidth=2,
             marker='o', markersize=4, label='fraud rate')
    ax2.set_ylabel('Fraud rate, percent', color=CLAY, fontsize=9)
    ax2.tick_params(colors=CLAY, labelsize=8.5)
    for side in ('top', 'left'):
        ax2.spines[side].set_visible(False)
    ax2.spines['right'].set_color(LINE)
    ax.set_title('Volume swings across the cycle, fraud stays flat',
                 fontsize=10, color=INK, loc='left', pad=12)
    st.pyplot(fig, use_container_width=True)

    st.markdown(f"<div class='note'>Genuine volume collapses in part of the cycle while the "
                f"<i>count</i> of frauds barely moves, so the fraud <i>share</i> climbs sharply "
                f"there. Quiet positions {LOW_ACTIVITY} carry a "
                f"{eda['activity_split']['low_rate']*100:.2f} percent fraud rate against "
                f"{eda['activity_split']['busy_rate']*100:.2f} percent elsewhere, while holding "
                f"only {eda['activity_split']['low_volume_share']*100:.1f} percent of all "
                f"traffic. Those positions were picked by volume alone, not by assuming a clock "
                f"time, which is why both cycle position and a low activity flag are model "
                f"inputs.</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------- amounts
    st.markdown('## How large fraud is')
    c1, c2 = st.columns(2)
    with c1:
        hist = eda['amount_hist']
        edges = np.array(hist['edges'])
        centres = (edges[:-1] + edges[1:]) / 2
        normal = np.array(hist['normal'], dtype=float)
        fraud = np.array(hist['fraud'], dtype=float)
        fig, ax = plt.subplots(figsize=(5.6, 3.3))
        ax.fill_between(centres, normal / normal.max(), color=MUTED, alpha=0.42, label='normal')
        ax.fill_between(centres, fraud / fraud.max(), color=CLAY, alpha=0.62, label='fraud')
        style_axes(ax, 'Relative frequency', 'log of amount')
        ax.legend(frameon=False, fontsize=8.5, labelcolor=MUTED)
        ax.set_title('Fraud sits to the right, the amounts are bigger',
                     fontsize=10, color=INK, loc='left', pad=12)
        st.pyplot(fig, use_container_width=True)
    with c2:
        bands = pd.DataFrame(eda['amount_bands'])
        fig, ax = plt.subplots(figsize=(5.6, 3.3))
        ax.barh(bands['band'], bands['rate'] * 100, color=SAGE, height=0.62)
        style_axes(ax, '', 'Fraud rate, percent')
        ax.grid(axis='x', color=LINE, linewidth=0.7)
        ax.grid(axis='y', visible=False)
        ax.set_title('Fraud rate climbs with transaction size',
                     fontsize=10, color=INK, loc='left', pad=12)
        st.pyplot(fig, use_container_width=True)

    st.markdown(f"""
    <div class='statgrid'>
      <div class='stat'><div class='k'>R{eda['amount_stats']['normal_median']:,.0f}</div>
        <div class='l'>Median normal</div></div>
      <div class='stat sand'><div class='k'>R{eda['amount_stats']['fraud_median']:,.0f}</div>
        <div class='l'>Median fraud</div></div>
      <div class='stat'><div class='k'>R{eda['amount_stats']['normal_mean']:,.0f}</div>
        <div class='l'>Mean normal</div></div>
      <div class='stat sand'><div class='k'>R{eda['amount_stats']['fraud_mean']:,.0f}</div>
        <div class='l'>Mean fraud</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('## The three findings that shaped the model')
    st.markdown("""
1. **Type is the strongest filter.** Fraud lives only in TRANSFER and CASH_OUT, so transaction
   type is encoded directly as a model input.
2. **Position in the cycle carries real signal.** The fraud rate in the quiet part of the
   cycle is many times higher than in the busy part, so cycle position and a low activity
   flag both go in. We deliberately avoid calling these positions clock hours, because the
   data never tells us when the simulation started.
3. **Size matters but is skewed.** Fraudulent amounts are far larger, but the raw column is so
   skewed that we use the logarithm instead, which the models handle far better.
""")


def page_performance():
    st.markdown("<div class='eyebrow'>Deliverable two</div>", unsafe_allow_html=True)
    st.markdown('# Detect fraud')
    st.markdown("<p class='lede'>Four models were trained on three feature sets. This page "
                "covers which won, why accuracy is the wrong yardstick, and how the decision "
                "threshold was chosen from business cost rather than convention.</p>",
                unsafe_allow_html=True)
    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)

    st.markdown('## Why accuracy misleads')
    trap = meta['accuracy_trap']
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class='verdict'>
          <div class='headline'>{trap['always_normal_accuracy']*100:.2f}%</div>
          <div class='sub'>Accuracy, approve everything</div>
          <div style='margin-top:0.9rem;font-size:0.9rem;color:{MUTED};'>
          Catches zero fraud. Costs the bank every rand lost.</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        co = meta['thresholds']['cost_optimal']
        st.markdown(f"""
        <div class='verdict'>
          <div class='headline'>{co['recall']*100:.1f}%</div>
          <div class='sub'>Fraud caught by our model</div>
          <div style='margin-top:0.9rem;font-size:0.9rem;color:{MUTED};'>
          {co['true_positives']} of {co['true_positives']+co['false_negatives']} frauds stopped,
          from {co['alerts']:,} alerts.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='note'>A model that predicts <i>normal</i> every time scores "
                "almost 99.9 percent accuracy while being completely useless. We therefore "
                "judge everything on <b>PR-AUC</b>, which measures how well the model ranks "
                "rare positives, together with precision and recall at a chosen threshold.</div>",
                unsafe_allow_html=True)

    st.markdown('## Model comparison')
    comp = pd.DataFrame(meta['model_comparison'])
    fig, ax = plt.subplots(figsize=(10, 3.4))
    y = np.arange(len(comp))
    ax.barh(y - 0.26, comp['basic'], height=0.25, color=MUTED, label='basic features')
    ax.barh(y, comp['balances'], height=0.25, color=CLAY, label='with balances, used')
    ax.barh(y + 0.26, comp['errors'], height=0.25, color='#D9C7A0',
            label='with balance errors, rejected')
    ax.set_yticks(y)
    ax.set_yticklabels(comp['model'], fontsize=9)
    style_axes(ax, '', 'Test PR-AUC')
    ax.grid(axis='x', color=LINE, linewidth=0.7)
    ax.grid(axis='y', visible=False)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=MUTED, loc='lower right')
    st.pyplot(fig, use_container_width=True)

    st.markdown("<div class='warn'><b>Why the third bar was rejected.</b> Adding the balance "
                "error columns pushes PR-AUC to 0.99, but those columns are computed after the "
                "transaction has settled and they encode the simulator's fraud pattern almost "
                "perfectly. The model stops learning and starts memorising. We kept the middle "
                "option, which uses only information available at the moment of decision.</div>",
                unsafe_allow_html=True)

    st.markdown('## Where the model draws the line')
    f1t, cot = meta['thresholds']['f1_optimal'], meta['thresholds']['cost_optimal']
    if not meta['has_pr_curve']:
        st.markdown("<div class='note'>The precision recall curve is not stored in this "
                    "metadata file. Regenerate the artifacts with train_and_save.py to see "
                    "it here. The two candidate operating points are still shown below.</div>",
                    unsafe_allow_html=True)
    pr = meta.get('pr_curve', {'recall': [cot['recall'], f1t['recall']],
                               'precision': [cot['precision'], f1t['precision']]})
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(pr['recall'], pr['precision'], color=INK, linewidth=1.8)
    ax.axhline(meta['test_fraud_rate'], color=MUTED, linestyle=':', linewidth=1.2)
    ax.scatter([f1t['recall']], [f1t['precision']], s=95, color=SAND, zorder=5,
               edgecolor=INK, linewidth=0.8, label='best F1')
    ax.scatter([cot['recall']], [cot['precision']], s=95, color=CLAY, zorder=5,
               edgecolor=INK, linewidth=0.8, label='lowest cost, chosen')
    style_axes(ax, 'Precision, share of alerts that are real', 'Recall, share of fraud caught')
    ax.legend(frameon=False, fontsize=8.5, labelcolor=MUTED)
    ax.set_title('Every possible trade off, and the two candidates',
                 fontsize=10, color=INK, loc='left', pad=12)
    st.pyplot(fig, use_container_width=True)

    st.markdown('## Justifying the threshold with money')
    st.markdown(f"""
The usual default of 0.5 has no business meaning. We compared two defensible choices using
stated costs: **R{meta['costs']['missed_fraud']:,.0f}** lost on average when a fraud goes
through, and **R{meta['costs']['false_alarm']:,.0f}** of analyst time to review one alert.
Missing a fraud is therefore {meta['costs']['missed_fraud']/meta['costs']['false_alarm']:.0f}
times more expensive than a false alarm, which pushes the threshold far below 0.5.
""")

    rows = pd.DataFrame([
        {'Strategy': 'No model, approve all', 'Threshold': 'n/a', 'Precision': '0.000',
         'Recall': '0.000', 'Alerts': 0,
         'Total cost': f"R{meta['costs']['baseline_no_model']:,.0f}"},
        {'Strategy': 'Best F1 balance', 'Threshold': f"{f1t['threshold']:.4f}",
         'Precision': f"{f1t['precision']:.3f}", 'Recall': f"{f1t['recall']:.3f}",
         'Alerts': f"{f1t['alerts']:,}", 'Total cost': f"R{f1t['total_cost']:,.0f}"},
        {'Strategy': 'Lowest business cost', 'Threshold': f"{cot['threshold']:.4f}",
         'Precision': f"{cot['precision']:.3f}", 'Recall': f"{cot['recall']:.3f}",
         'Alerts': f"{cot['alerts']:,}", 'Total cost': f"R{cot['total_cost']:,.0f}"},
    ])
    st.dataframe(rows, hide_index=True, use_container_width=True)

    saving = meta['costs']['baseline_no_model'] - cot['total_cost']
    st.markdown(f"""
    <div class='note'>
    <b>The counterintuitive result.</b> The lowest cost option has precision of only
    {cot['precision']:.2f}, meaning roughly four in five alerts are false. That looks poor until
    you price it: catching {cot['recall']*100:.0f} percent of fraud saves
    <b>R{saving:,.0f}</b> against doing nothing, and the extra review work costs a fraction of
    the fraud it prevents. Optimising F1 would look tidier and cost the bank
    R{f1t['total_cost']-cot['total_cost']:,.0f} more.
    </div>""", unsafe_allow_html=True)

    st.markdown('## What the model relies on')
    imp = pd.Series(meta['importance']).reindex(FEATURES).dropna().sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.barh(imp.index, imp.values, color=SAGE, height=0.6)
    style_axes(ax, '', 'Importance')
    ax.grid(axis='x', color=LINE, linewidth=0.7)
    ax.grid(axis='y', visible=False)
    st.pyplot(fig, use_container_width=True)


def page_predict():
    st.markdown("<div class='eyebrow'>Live scoring</div>", unsafe_allow_html=True)
    st.markdown('# Score a transaction')
    st.markdown("<p class='lede'>Enter the details of a transaction as they would be known at "
                "the moment of decision. The model returns a fraud probability, a recommended "
                "action, and a breakdown of what drove the score.</p>", unsafe_allow_html=True)
    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)

    quiet_example = LOW_ACTIVITY[len(LOW_ACTIVITY) // 2] if LOW_ACTIVITY else 4

    preset = st.selectbox('Start from an example, or enter your own below', [
        'Custom entry',
        'Account emptied by transfer, quiet part of the cycle',
        'Ordinary payment, busy part of the cycle',
        'Large but legitimate cash out',
    ])

    defaults = {
        'Custom entry': ('TRANSFER', 181000.0, 181000.0, 0.0, quiet_example),
        'Account emptied by transfer, quiet part of the cycle':
            ('TRANSFER', 482000.0, 482000.0, 0.0, quiet_example),
        'Ordinary payment, busy part of the cycle': ('PAYMENT', 4800.0, 62000.0, 0.0, 14),
        'Large but legitimate cash out': ('CASH_OUT', 210000.0, 940000.0, 380000.0, 11),
    }
    d_type, d_amt, d_org, d_dest, d_pos = defaults[preset]

    c1, c2, c3 = st.columns(3)
    with c1:
        txn_type = st.selectbox('Transaction type', ALL_TYPES, index=ALL_TYPES.index(d_type))
        cycle_position = st.slider('Position in the 24 step cycle', 0, 23, d_pos,
                                   help='Which step of the repeating daily cycle this '
                                        'transaction falls in. Not a clock hour: the data '
                                        'does not tell us when the simulation began.')
        if cycle_position in LOW_ACTIVITY:
            st.caption('This is one of the low activity positions.')
    with c2:
        amount = st.number_input('Amount, rand', min_value=0.0, value=float(d_amt), step=1000.0)
        old_org = st.number_input('Sender balance before', min_value=0.0,
                                  value=float(d_org), step=1000.0)
    with c3:
        old_dest = st.number_input('Receiver balance before', min_value=0.0,
                                   value=float(d_dest), step=1000.0)
        threshold = st.slider(
            'Decision threshold', min_value=0.01, max_value=0.99,
            value=float(np.clip(meta['thresholds']['cost_optimal']['threshold'],
                                0.01, 0.99)),
            step=0.01, format='%.2f')

    row = build_row(txn_type, amount, old_org, old_dest, cycle_position)
    proba = float(model.predict_proba(row)[0, 1])
    flagged = proba >= threshold

    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
    left, right = st.columns([1, 1.5])
    with left:
        colour = CLAY if flagged else SAGE
        label = 'Refer for review' if flagged else 'Allow'
        st.markdown(f"""
        <div class='verdict' style='border-left:4px solid {colour};'>
          <div class='sub'>Recommended action</div>
          <div class='headline' style='color:{colour};margin-top:0.35rem;'>{label}</div>
          <div style='margin-top:1.1rem;font-family:Fraunces,serif;font-size:2.4rem;
                      font-weight:600;'>{proba*100:.1f}%</div>
          <div class='sub'>Fraud probability</div>
          <div style='margin-top:0.9rem;font-size:0.86rem;color:{MUTED};'>
            Threshold in use: {threshold:.4f}</div>
        </div>""", unsafe_allow_html=True)

    with right:
        fig, ax = plt.subplots(figsize=(6.2, 1.5))
        ax.barh([0], [1], color='#E4DFD4', height=0.42)
        ax.barh([0], [proba], color=CLAY if flagged else SAGE, height=0.42)
        ax.axvline(threshold, color=INK, linewidth=1.6, linestyle='--')
        ax.text(threshold, 0.42, ' threshold', fontsize=8, color=INK, va='bottom')
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel('Fraud probability', color=MUTED, fontsize=9)
        ax.set_facecolor(PAPER)
        fig.patch.set_facecolor(PAPER)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(LINE)
        ax.tick_params(colors=MUTED, labelsize=8.5)
        st.pyplot(fig, use_container_width=True)

        if flagged:
            st.markdown("<div class='warn'>This transaction scores above the threshold. Under "
                        "the current settings it would be held and sent to an analyst rather "
                        "than declined outright, since roughly four in five alerts turn out to "
                        "be legitimate.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='note'>This transaction scores below the threshold and "
                        "would pass through without review.</div>", unsafe_allow_html=True)

    # ------------------------------------------------------- explanation
    st.markdown('## Why the model gave this score')
    booster = model.get_booster()
    dmatrix = __import__('xgboost').DMatrix(row, feature_names=FEATURES)
    contribs = booster.predict(dmatrix, pred_contribs=True)[0]
    base_value, feature_contribs = contribs[-1], contribs[:-1]

    order = np.argsort(np.abs(feature_contribs))[::-1]
    labels = [FEATURES[i] for i in order]
    values = [feature_contribs[i] for i in order]

    readable = {
        'log_amount': f'Amount, R{amount:,.0f}',
        'cycle_position': f'Cycle position {cycle_position}',
        'is_low_activity': ('Falls in a low activity position'
                            if cycle_position in LOW_ACTIVITY
                            else 'Falls in a busy position'),
        'log_old_org': f'Sender balance, R{old_org:,.0f}',
        'log_old_dest': f'Receiver balance, R{old_dest:,.0f}',
    }
    for col in TYPE_COLS:
        readable[col] = f'Type is {col.replace("type_", "")}'

    c1, c2 = st.columns([1.3, 1])
    with c1:
        fig, ax = plt.subplots(figsize=(6.6, 3.8))
        pretty = [readable.get(l, l) for l in labels][::-1]
        vals = values[::-1]
        ax.barh(pretty, vals, color=[CLAY if v > 0 else SAGE for v in vals], height=0.6)
        ax.axvline(0, color=INK, linewidth=1)
        style_axes(ax, '', 'Push towards fraud, clay, or towards normal, green')
        ax.grid(axis='x', color=LINE, linewidth=0.7)
        ax.grid(axis='y', visible=False)
        ax.tick_params(axis='y', labelsize=8.5)
        st.pyplot(fig, use_container_width=True)
    with c2:
        st.markdown(f"""
The model starts from a baseline score for an average transaction, then each feature pushes
the score up or down. Clay bars argue for fraud, green bars argue against it.

The three strongest influences here:
""")
        for i in range(min(3, len(labels))):
            direction = 'towards fraud' if values[i] > 0 else 'towards normal'
            st.markdown(f"- **{readable.get(labels[i], labels[i])}** pushes "
                        f"{direction} by {abs(values[i]):.2f}")
        st.markdown(f"<div class='note' style='margin-top:0.9rem;'>These contributions come "
                    f"from the model itself, not from a guess about what it might be doing. "
                    f"They add up exactly to the final score, which means an analyst can always "
                    f"be given a reason for a decision.</div>", unsafe_allow_html=True)


def page_recommend():
    st.markdown("<div class='eyebrow'>Deliverable three</div>", unsafe_allow_html=True)
    st.markdown('# Recommendations')
    st.markdown("<p class='lede'>What the bank should do with these findings, which "
                "transaction types deserve attention, and what the two kinds of mistake "
                "actually cost.</p>", unsafe_allow_html=True)
    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)

    # ------------------------------------------------ cost calculator
    st.markdown('## The cost of missing fraud against false alarms')
    st.markdown("Both numbers below are assumptions, so they are left adjustable. Change them "
                "to match the bank's real figures and the recommended threshold moves with them.")

    c1, c2 = st.columns(2)
    with c1:
        cost_fn = st.number_input('Cost of one missed fraud, rand', min_value=100.0,
                                  value=float(meta['costs']['missed_fraud']), step=500.0)
    with c2:
        cost_fp = st.number_input('Cost of reviewing one false alarm, rand', min_value=1.0,
                                  value=float(meta['costs']['false_alarm']), step=10.0)

    ratio = cost_fn / max(cost_fp, 1e-9)
    n_fraud = meta['test_fraud_count']
    n_rows = meta['test_rows']

    if not meta['has_pr_curve']:
        st.markdown("<div class='note'>This metadata file does not include the full precision "
                    "recall curve, so the cost curve below is drawn from the two stored "
                    "operating points. Regenerate with train_and_save.py for the full "
                    "version.</div>", unsafe_allow_html=True)
    _f1t, _cot = meta['thresholds']['f1_optimal'], meta['thresholds']['cost_optimal']
    pr = meta.get('pr_curve', {'recall': [_cot['recall'], _f1t['recall']],
                               'precision': [_cot['precision'], _f1t['precision']]})
    recalls = np.array(pr['recall'])
    precisions = np.array(pr['precision'])
    with np.errstate(divide='ignore', invalid='ignore'):
        tp = recalls * n_fraud
        alerts = np.where(precisions > 0, tp / precisions, 0)
        fp = np.clip(alerts - tp, 0, None)
        fn = n_fraud - tp
        total = fn * cost_fn + fp * cost_fp
    ok = np.isfinite(total) & (alerts > 0)
    best = int(np.argmin(np.where(ok, total, np.inf)))

    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(recalls[ok], total[ok] / 1000, color=INK, linewidth=1.8)
    ax.scatter([recalls[best]], [total[best] / 1000], s=110, color=CLAY, zorder=5,
               edgecolor=INK, linewidth=0.8)

    # Keep the y-axis scaled to the cost curve itself. The "doing nothing"
    # baseline can be an order of magnitude higher than anything on the curve
    # (especially with only two fallback points), and letting it into the
    # autoscale squashes the real data into a sliver at the bottom of an
    # otherwise empty chart. So only draw it as a line if it actually falls
    # in range; otherwise call out its value in the corner instead.
    data_max = float(np.nanmax(total[ok])) / 1000 if ok.any() else 0.0
    ylim_top = max(data_max * 1.25, data_max + 20, 1.0)
    baseline = n_fraud * cost_fn / 1000
    if baseline <= ylim_top:
        ax.axhline(baseline, color=MUTED, linestyle=':', linewidth=1.3)
        ax.text(0.02, baseline, ' doing nothing', fontsize=8.5,
                color=MUTED, va='bottom')
    else:
        ax.text(0.02, ylim_top, f' doing nothing costs R{baseline:,.0f}k, off this scale',
                fontsize=8.5, color=MUTED, va='top')
    ax.set_ylim(0, ylim_top)

    style_axes(ax, 'Total cost, thousands of rand', 'Share of fraud caught')
    ax.set_title('Total cost against how much fraud we chase',
                 fontsize=10, color=INK, loc='left', pad=12)
    st.pyplot(fig, use_container_width=True)

    st.markdown(f"""
    <div class='statgrid'>
      <div class='stat'><div class='k'>{ratio:,.0f}x</div>
        <div class='l'>Missed fraud vs false alarm</div></div>
      <div class='stat sage'><div class='k'>{recalls[best]*100:.0f}%</div>
        <div class='l'>Recommended fraud caught</div></div>
      <div class='stat sand'><div class='k'>{int(alerts[best]):,}</div>
        <div class='l'>Alerts per {n_rows:,} txns</div></div>
      <div class='stat sage'><div class='k'>R{(n_fraud*cost_fn - total[best])/1000:,.0f}k</div>
        <div class='l'>Saved against doing nothing</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='note'>
    Because a missed fraud costs about {ratio:,.0f} times more than a wasted review, the bank
    should deliberately accept a <b>high false alarm rate</b>. Chasing
    {recalls[best]*100:.0f} percent of fraud means analysts review roughly
    {int(alerts[best]):,} transactions out of {n_rows:,}, and that work is worth it. Tuning for
    a tidy looking precision score would quietly cost more money.
    </div>""", unsafe_allow_html=True)

    # ------------------------------------------------ type priority
    st.markdown('## Which transaction types need most attention')
    by_type = pd.DataFrame(eda['fraud_by_type']).sort_values('fraud', ascending=False)
    c1, c2 = st.columns([1, 1.1])
    with c1:
        show = by_type.copy()
        show['share_of_fraud'] = (show['fraud'] / show['fraud'].sum() * 100).round(1)
        show['rate'] = (show['rate'] * 100).round(3)
        show = show[['type', 'transactions', 'fraud', 'rate', 'share_of_fraud']]
        show.columns = ['Type', 'Volume', 'Frauds', 'Fraud rate %', 'Share of all fraud %']
        st.dataframe(show, hide_index=True, use_container_width=True)
    with c2:
        st.markdown("""
**Priority one: TRANSFER.** Highest fraud rate of any type. This is the first leg of the
attack, where the criminal moves money out of the compromised account. Stopping it here
prevents the loss entirely, so real time scoring should be applied to every transfer.

**Priority two: CASH_OUT.** This is the second leg, where the money leaves the system. Once it
completes, recovery is unlikely. Worth scoring in real time, and worth linking back to any
recent transfer of the same amount.

**Deprioritise the rest.** PAYMENT, CASH_IN and DEBIT contain no fraud at all in this data.
They can be routed around the model, which frees capacity for the two types that matter.
""")

    st.markdown('## What the bank should do')
    st.markdown("""
**1. Score transfers and cash outs in real time, and leave the rest alone.**
Fraud is confined to two transaction types that together make up a minority of volume.
Restricting the model to those two cuts the scoring load substantially with no loss of cover.

**2. Hold suspicious transactions for review rather than declining them.**
At the recommended operating point roughly four in five alerts are legitimate customers.
Automatic declines at that rate would do real damage to customer trust. A short hold with a
quick analyst check, or a step up verification such as a one time password, gets the benefit
without the harm.

**3. Concentrate review capacity on the quiet part of the cycle.**
Transaction volume follows a repeating 24 step cycle, and in its quietest stretch the fraud
rate is many times higher while total volume is at its lowest. A small team covering that
window therefore addresses a disproportionate share of the risk at low cost. Mapping those
cycle positions onto real clock times is a one line exercise once the bank knows when the
data collection started, which the dataset itself never states.

**4. Set the threshold from cost, and revisit it quarterly.**
The threshold is a business decision, not a technical one. It should be recalculated whenever
the average fraud loss or the cost of review changes materially.

**5. Treat the balance columns with suspicion when moving to real data.**
In this simulation, fraud empties the account to the last cent, which makes the balance
features far more predictive than they will be in production. The model should be retrained
and revalidated on real transactions before anyone relies on these numbers.

**6. Monitor for drift.**
Fraud adapts. Track the alert rate, the share of alerts confirmed as fraud, and the score
distribution week by week. A sudden move in any of them is a signal to retrain.
""")

    st.markdown("<div class='warn'><b>The honest limitation.</b> Every figure here comes from "
                "simulated data. The ranking of transaction types, the night time pattern and "
                "the cost logic should all carry over to real transactions. The precise "
                "performance numbers will not, and should be re-established on the bank's own "
                "data before deployment.</div>", unsafe_allow_html=True)


PAGES = {
    'Overview': page_overview,
    'Explore the data': page_eda,
    'Model performance': page_performance,
    'Score a transaction': page_predict,
    'Recommendations': page_recommend,
}
PAGES[page]()