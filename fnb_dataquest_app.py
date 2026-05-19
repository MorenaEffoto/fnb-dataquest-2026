"""
FNB DataQuest 2026 — "From Roots to Rise"
Interpretable Credit Modelling App
Run with: python fnb_dataquest_app.py
Then open http://127.0.0.1:8050 in your browser

Requirements:
    pip install dash dash-bootstrap-components plotly scikit-learn pandas openpyxl scipy
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. DATA LOADING & CLEANING
# ─────────────────────────────────────────────

def load_and_clean(path="loan_book.csv"):
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    # Standardise messy categoricals
    def normalise(val, mapping):
        s = str(val).strip().lower()
        for keys, canonical in mapping.items():
            if s in keys:
                return canonical
        return "other"

    own_map = {
        ("mortgage",): "MORTGAGE",
        ("rent", "renting"): "RENT",
        ("own", "owner"): "OWN",
        ("other",): "OTHER",
    }
    df["home_ownership"] = df["home_ownership"].apply(lambda x: normalise(x, own_map))

    purpose_map = {
        ("debt_consolidation", "debt consolidation"): "debt_consolidation",
        ("home_improvement", "home improvement"): "home_improvement",
        ("major_purchase", "major purchase"): "major_purchase",
        ("small_business", "small business"): "small_business",
        ("medical",): "medical",
        ("education",): "education",
        ("other",): "other",
    }
    df["loan_purpose"] = df["loan_purpose"].apply(lambda x: normalise(x, purpose_map))

    # Informative missingness flag (key feature engineering insight)
    df["ever_delinquent"] = df["months_since_last_delinquency"].notna().astype(int)

    # Impute
    df["months_since_last_delinquency"] = df["months_since_last_delinquency"].fillna(999)
    df["annual_income"] = df["annual_income"].fillna(df["annual_income"].median())
    df["employment_length_years"] = df["employment_length_years"].fillna(df["employment_length_years"].median())
    df["num_open_accounts"] = df["num_open_accounts"].fillna(df["num_open_accounts"].median())

    return df


# ─────────────────────────────────────────────
# 2. WOE / IV ENGINE
# ─────────────────────────────────────────────

def compute_woe_iv(df, feature, target="default_flag", bins=10):
    """
    Returns a DataFrame with WoE and IV per bin,
    plus the total IV for the feature.
    """
    tmp = df[[feature, target]].copy().dropna()
    total_events = tmp[target].sum()
    total_non_events = len(tmp) - total_events

    is_numeric = pd.api.types.is_numeric_dtype(tmp[feature]) and tmp[feature].nunique() > 10

    if not is_numeric:
        grouped = tmp.groupby(feature.astype(str) if hasattr(feature, 'astype') else feature)[target].agg(["sum", "count"])
        tmp[feature] = tmp[feature].astype(str)
        grouped = tmp.groupby(feature)[target].agg(["sum", "count"])
        grouped.columns = ["events", "total"]
    else:
        tmp[feature] = pd.to_numeric(tmp[feature], errors="coerce")
        tmp = tmp.dropna(subset=[feature])
        try:
            tmp["bin"] = pd.qcut(tmp[feature], bins, duplicates="drop")
        except Exception:
            tmp["bin"] = pd.cut(tmp[feature], bins, duplicates="drop")
        grouped = tmp.groupby("bin", observed=True)[target].agg(["sum", "count"])
        grouped.columns = ["events", "total"]

    grouped["non_events"] = grouped["total"] - grouped["events"]
    grouped["pct_events"] = grouped["events"] / total_events
    grouped["pct_non_events"] = grouped["non_events"] / total_non_events
    grouped["pct_events"] = grouped["pct_events"].replace(0, 0.0001)
    grouped["pct_non_events"] = grouped["pct_non_events"].replace(0, 0.0001)
    grouped["woe"] = np.log(grouped["pct_non_events"] / grouped["pct_events"])
    grouped["iv"] = (grouped["pct_non_events"] - grouped["pct_events"]) * grouped["woe"]
    grouped["default_rate"] = grouped["events"] / grouped["total"]
    total_iv = grouped["iv"].sum()
    grouped = grouped.reset_index()
    grouped.columns = [str(c) for c in grouped.columns]
    return grouped, total_iv


def iv_strength(iv):
    if iv < 0.02:
        return "Useless"
    elif iv < 0.1:
        return "Weak"
    elif iv < 0.3:
        return "Medium"
    else:
        return "Strong"


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING & MODELLING
# ─────────────────────────────────────────────

def engineer_features(df):
    fe = df.copy()

    # Ratio features
    fe["loan_to_income"] = fe["loan_amount"] / (fe["annual_income"] + 1)
    fe["income_per_year_employed"] = fe["annual_income"] / (fe["employment_length_years"] + 1)

    # Log transform skewed numerics
    for col in ["annual_income", "total_revolving_balance", "loan_amount"]:
        fe[f"log_{col}"] = np.log1p(fe[col])

    # Bin age into credit-risk buckets
    fe["age_band"] = pd.cut(
        fe["age"],
        bins=[0, 25, 35, 45, 55, 100],
        labels=["18-25", "26-35", "36-45", "46-55", "55+"],
    )

    # Binary flags
    fe["high_utilisation"] = (fe["credit_utilisation_pct"] > 0.75).astype(int)
    fe["has_delinquency_2yr"] = (fe["num_delinquencies_2yr"] > 0).astype(int)
    fe["recent_inquiries"] = (fe["num_hard_inquiries_6mo"] >= 3).astype(int)

    # Informative missingness (already created in clean)
    return fe


NUMERIC_FEATURES = [
    "age", "log_annual_income", "employment_length_years",
    "num_open_accounts", "num_delinquencies_2yr", "credit_utilisation_pct",
    "months_since_oldest_account", "num_hard_inquiries_6mo",
    "log_loan_amount", "interest_rate", "dti_ratio",
    "months_since_last_delinquency", "pct_accounts_current",
    "loan_to_income", "income_per_year_employed", "ever_delinquent",
    "high_utilisation", "has_delinquency_2yr", "recent_inquiries",
]

CATEGORICAL_FEATURES = ["home_ownership", "loan_purpose", "email_domain_type"]


def fit_model(df):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline

    fe = engineer_features(df)
    train = fe[fe["set"] == "train"].copy()
    test = fe[fe["set"] == "test"].copy()

    # WoE encode categoricals
    woe_maps = {}
    for cat in CATEGORICAL_FEATURES:
        woe_df, _ = compute_woe_iv(train, cat)
        col_name = woe_df.columns[0]
        woe_maps[cat] = dict(zip(woe_df[col_name].astype(str), woe_df["woe"]))

    def apply_woe(df_in):
        out = df_in.copy()
        for cat, mapping in woe_maps.items():
            out[f"{cat}_woe"] = out[cat].astype(str).map(mapping).fillna(0)
        return out

    train = apply_woe(train)
    test = apply_woe(test)

    woe_feature_cols = [f"{c}_woe" for c in CATEGORICAL_FEATURES]
    feature_cols = NUMERIC_FEATURES + woe_feature_cols

    X_train = train[feature_cols].fillna(0)
    y_train = train["default_flag"]
    X_test = test[feature_cols].fillna(0)
    y_test = test["default_flag"]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced")),
    ])
    pipe.fit(X_train, y_train)

    train_proba = pipe.predict_proba(X_train)[:, 1]
    test_proba = pipe.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_train, train_proba)
    test_auc = roc_auc_score(y_test, test_proba)

    # Coefficients
    scaler = pipe.named_steps["scaler"]
    lr = pipe.named_steps["lr"]
    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": lr.coef_[0],
        "std_coef": lr.coef_[0] * scaler.scale_,
    }).sort_values("coefficient", key=abs, ascending=False)

    return pipe, woe_maps, feature_cols, train_auc, test_auc, coef_df, test_proba, y_test, test


# ─────────────────────────────────────────────
# 4. DASH APP
# ─────────────────────────────────────────────

def build_app(df, pipe, woe_maps, feature_cols, train_auc, test_auc,
              coef_df, test_proba, y_test, test_df):
    import dash
    from dash import dcc, html, Input, Output, dash_table
    import dash_bootstrap_components as dbc
    import plotly.graph_objects as go
    import plotly.express as px
    from sklearn.metrics import roc_curve, precision_recall_curve

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY],
                    suppress_callback_exceptions=True)

    # ── colour palette (FNB teal + gold) ──
    TEAL = "#00a3ad"
    GOLD = "#f5a623"
    DARK = "#1a2d3b"
    LIGHT_BG = "#f4f6f9"

    # ── reusable card ──
    def card(children, **kwargs):
        return dbc.Card(dbc.CardBody(children), className="mb-3 shadow-sm", **kwargs)

    # ── numeric and categorical feature lists for dropdowns ──
    raw_numerics = [c for c in df.select_dtypes(include=np.number).columns
                    if c not in ["default_flag", "branch_code_id"]]
    raw_cats = ["home_ownership", "loan_purpose", "email_domain_type",
                "application_dow", "region", "phone_verified"]

    # ─────────────── LAYOUT ───────────────
    app.layout = dbc.Container(fluid=True, style={"backgroundColor": LIGHT_BG}, children=[

        # Header
        dbc.Row(dbc.Col(html.Div([
            html.H2("FNB DataQuest 2026", style={"color": "white", "margin": 0}),
            html.P("From Roots to Rise — Interpretable Credit Modelling",
                   style={"color": "#cce9eb", "margin": 0}),
        ], style={
            "backgroundColor": DARK, "padding": "20px 30px",
            "borderBottom": f"4px solid {TEAL}", "marginBottom": "20px"
        }))),

        dbc.Tabs(id="tabs", active_tab="tab-eda-uni", children=[

            # ══════════════════════════════════════
            # TAB 1 — UNIVARIATE EDA
            # ══════════════════════════════════════
            dbc.Tab(label="📊 Univariate EDA", tab_id="tab-eda-uni", children=[
                dbc.Row([
                    dbc.Col([
                        card([
                            html.Label("Select Feature", style={"fontWeight": "bold"}),
                            dcc.Dropdown(
                                id="uni-feature",
                                options=[{"label": c, "value": c} for c in raw_numerics + raw_cats],
                                value="credit_utilisation_pct",
                                clearable=False,
                            ),
                            html.Br(),
                            html.Label("Number of Bins (numeric only)"),
                            dcc.Slider(id="uni-bins", min=5, max=20, step=1, value=10,
                                       marks={i: str(i) for i in range(5, 21, 5)}),
                        ])
                    ], width=3),
                    dbc.Col([
                        card([dcc.Graph(id="uni-dist-plot")]),
                        card([dcc.Graph(id="uni-woe-plot")]),
                    ], width=9),
                ]),
                dbc.Row(dbc.Col(card([
                    html.H5("IV Summary — All Features", style={"color": TEAL}),
                    dcc.Graph(id="iv-summary-bar"),
                ]))),
            ]),

            # ══════════════════════════════════════
            # TAB 2 — BIVARIATE EDA
            # ══════════════════════════════════════
            dbc.Tab(label="🔗 Bivariate EDA", tab_id="tab-eda-bi", children=[
                dbc.Row([
                    dbc.Col(card([
                        html.Label("X axis", style={"fontWeight": "bold"}),
                        dcc.Dropdown(id="bi-x", options=[{"label": c, "value": c}
                                                          for c in raw_numerics],
                                     value="credit_utilisation_pct", clearable=False),
                        html.Br(),
                        html.Label("Y axis"),
                        dcc.Dropdown(id="bi-y", options=[{"label": c, "value": c}
                                                          for c in raw_numerics],
                                     value="dti_ratio", clearable=False),
                        html.Br(),
                        html.Label("Colour by"),
                        dcc.Dropdown(id="bi-color",
                                     options=[{"label": c, "value": c} for c in raw_cats],
                                     value="home_ownership", clearable=False),
                    ]), width=3),
                    dbc.Col([
                        card([dcc.Graph(id="bi-scatter")]),
                        card([dcc.Graph(id="bi-heatmap")]),
                    ], width=9),
                ]),
            ]),

            # ══════════════════════════════════════
            # TAB 3 — DATA QUALITY
            # ══════════════════════════════════════
            dbc.Tab(label="🔍 Data Quality", tab_id="tab-dq", children=[
                dbc.Row([
                    dbc.Col(card([
                        html.H5("Missing Values", style={"color": TEAL}),
                        dcc.Graph(id="dq-missing"),
                    ]), width=6),
                    dbc.Col(card([
                        html.H5("Data Types & Cardinality", style={"color": TEAL}),
                        dash_table.DataTable(
                            id="dq-table",
                            style_header={"backgroundColor": TEAL, "color": "white",
                                          "fontWeight": "bold"},
                            style_cell={"textAlign": "left", "padding": "8px"},
                            style_data_conditional=[{
                                "if": {"row_index": "odd"},
                                "backgroundColor": "#f0f8f9",
                            }],
                        ),
                    ]), width=6),
                ]),
                dbc.Row(dbc.Col(card([
                    html.H5("Dirty Category Values Detected", style={"color": GOLD}),
                    html.P("The following columns had inconsistent casing/formatting before cleaning:"),
                    html.Ul([
                        html.Li("home_ownership — 14 raw variants → 4 canonical (MORTGAGE, RENT, OWN, OTHER)"),
                        html.Li("loan_purpose — 21 raw variants → 7 canonical categories"),
                    ]),
                    html.H6("Informative Missingness:"),
                    html.Ul([
                        html.Li("months_since_last_delinquency: 49.9% missing — "
                                "but missing means 'never delinquent' → engineered as ever_delinquent flag"),
                        html.Li("annual_income: 7.2% missing → imputed with median"),
                        html.Li("employment_length_years: 3.1% missing → imputed with median"),
                    ]),
                ]))),
            ]),

            # ══════════════════════════════════════
            # TAB 4 — MODEL
            # ══════════════════════════════════════
            dbc.Tab(label="🤖 Model", tab_id="tab-model", children=[
                dbc.Row([
                    dbc.Col(card([
                        html.H5("Model Performance", style={"color": TEAL}),
                        html.Table([
                            html.Tr([html.Th("Metric"), html.Th("Value")]),
                            html.Tr([html.Td("Baseline AUC (old model)"), html.Td("0.680")]),
                            html.Tr([html.Td("Our Train AUC"), html.Td(f"{train_auc:.3f}",
                                     style={"color": TEAL, "fontWeight": "bold"})]),
                            html.Tr([html.Td("Our Test AUC"), html.Td(f"{test_auc:.3f}",
                                     style={"color": TEAL, "fontWeight": "bold"})]),
                            html.Tr([html.Td("LightGBM ceiling AUC"), html.Td("0.820")]),
                        ], style={"width": "100%", "borderCollapse": "collapse"}),
                        html.Br(),
                        html.P("Improvement over baseline: "
                               f"+{(test_auc - 0.68):.3f} AUC points",
                               style={"color": GOLD, "fontWeight": "bold"}),
                    ]), width=4),
                    dbc.Col(card([dcc.Graph(id="model-roc")]), width=8),
                ]),
                dbc.Row([
                    dbc.Col(card([dcc.Graph(id="model-coef")]), width=8),
                    dbc.Col(card([
                        html.H5("Key Feature Engineering", style={"color": TEAL}),
                        html.Ul([
                            html.Li("loan_to_income ratio"),
                            html.Li("Log transforms: income, loan_amount, revolving_balance"),
                            html.Li("ever_delinquent flag (informative missingness)"),
                            html.Li("high_utilisation flag (>75%)"),
                            html.Li("WoE encoding for categoricals"),
                            html.Li("recent_inquiries flag (≥3 in 6mo)"),
                        ]),
                    ]), width=4),
                ]),
            ]),

            # ══════════════════════════════════════
            # TAB 5 — BUSINESS DASHBOARD
            # ══════════════════════════════════════
            dbc.Tab(label="💼 Business Dashboard", tab_id="tab-biz", children=[
                dbc.Row([
                    dbc.Col(card([
                        html.H6("Approval Threshold", style={"fontWeight": "bold"}),
                        dcc.Slider(id="biz-threshold", min=0.05, max=0.95, step=0.01,
                                   value=0.50,
                                   marks={v: f"{v:.0%}" for v in [0.1, 0.25, 0.5, 0.75, 0.9]},
                                   tooltip={"placement": "bottom"}),
                        html.Br(),
                        html.Div(id="biz-kpis"),
                    ]), width=4),
                    dbc.Col(card([dcc.Graph(id="biz-volume-risk")]), width=8),
                ]),
                dbc.Row([
                    dbc.Col(card([dcc.Graph(id="biz-pr-curve")]), width=6),
                    dbc.Col(card([dcc.Graph(id="biz-score-dist")]), width=6),
                ]),
            ]),

            # ══════════════════════════════════════
            # TAB 6 — RESEARCH
            # ══════════════════════════════════════
            dbc.Tab(label="📚 Research Notes", tab_id="tab-research", children=[
                dbc.Row(dbc.Col(card([
                    html.H4("Credit Modelling Concepts", style={"color": TEAL}),

                    html.H5("GLMs vs Non-Linear Models"),
                    html.P("Logistic regression (a Generalised Linear Model) models the log-odds of default "
                           "as a linear combination of features: logit(p) = β₀ + β₁x₁ + … + βₙxₙ. "
                           "Each coefficient has a direct interpretation — a unit increase in x increases "
                           "log-odds by β. Non-linear models (e.g. Random Forest, LightGBM) can capture "
                           "interactions and non-linearities automatically, typically achieving higher AUC, "
                           "but their decisions cannot be easily explained to regulators or risk committees."),

                    html.H5("Weight of Evidence (WoE) & Information Value (IV)"),
                    html.P("WoE transforms a feature bin into: ln(% Non-Events / % Events). "
                           "A high positive WoE bin is low-risk; a negative WoE bin is high-risk. "
                           "IV = Σ (% Non-Events − % Events) × WoE. "
                           "Rule of thumb: IV < 0.02 useless, 0.02–0.1 weak, 0.1–0.3 medium, >0.3 strong. "
                           "WoE encoding linearises the relationship between a feature and log-odds, "
                           "making it ideal for logistic regression."),

                    html.H5("Key Metrics in Credit Context"),
                    html.Ul([
                        html.Li("AUC: Probability model ranks a random defaulter above a random non-defaulter. "
                                "Baseline 0.68 → our model improves this."),
                        html.Li("Gini = 2×AUC − 1. Standard in credit scoring."),
                        html.Li("Precision (approved loans): of those approved, how many actually repay? "
                                "Low precision = high bad debt."),
                        html.Li("Recall: of all good customers, how many did we approve? "
                                "Low recall = missed revenue."),
                        html.Li("F1: harmonic mean of precision and recall."),
                    ]),

                    html.H5("Regulatory Considerations", style={"color": GOLD}),
                    html.P("Even on simulated data, features like email_domain_type and phone_verified "
                           "may act as proxies for socioeconomic status. Regulators (e.g. FSCA in SA, "
                           "FCA in UK) prohibit protected characteristics (race, gender, age in some contexts) "
                           "and proxy variables that produce disparate impact. "
                           "application_dow (day of week) and region could similarly encode demographic "
                           "patterns. A compliant model must demonstrate fairness across protected groups."),
                ])))
            ]),
        ]),
    ])

    # ─────────────────────────────────────────────
    # CALLBACKS
    # ─────────────────────────────────────────────

    # ── IV Summary bar (computed once) ──
    @app.callback(Output("iv-summary-bar", "figure"), Input("tabs", "active_tab"))
    def update_iv_bar(tab):
        iv_rows = []
        for col in raw_numerics + raw_cats:
            try:
                _, iv = compute_woe_iv(df, col)
                iv_rows.append({"feature": col, "iv": iv, "strength": iv_strength(iv)})
            except Exception:
                pass
        iv_df = pd.DataFrame(iv_rows).sort_values("iv", ascending=True)
        color_map = {"Useless": "#ccc", "Weak": "#f5a623",
                     "Medium": "#00a3ad", "Strong": "#1a2d3b"}
        fig = go.Figure()
        for strength, grp in iv_df.groupby("strength"):
            fig.add_bar(y=grp["feature"], x=grp["iv"], orientation="h",
                        name=strength, marker_color=color_map.get(strength, "#999"))
        fig.update_layout(barmode="stack", title="Information Value by Feature",
                          xaxis_title="IV", height=500,
                          plot_bgcolor="white", paper_bgcolor="white")
        return fig

    # ── Univariate distribution + WoE ──
    @app.callback(
        Output("uni-dist-plot", "figure"),
        Output("uni-woe-plot", "figure"),
        Input("uni-feature", "value"),
        Input("uni-bins", "value"),
    )
    def update_uni(feature, bins):
        is_numeric = feature in raw_numerics

        # Distribution split by default
        fig_dist = go.Figure()
        for flag, label, color in [(0, "Non-Default", TEAL), (1, "Default", GOLD)]:
            subset = df[df["default_flag"] == flag][feature].dropna()
            if is_numeric:
                fig_dist.add_trace(go.Histogram(
                    x=subset, name=label, opacity=0.7,
                    marker_color=color, nbinsx=bins, histnorm="percent"))
            else:
                vc = subset.value_counts(normalize=True).reset_index()
                fig_dist.add_trace(go.Bar(x=vc[feature], y=vc["proportion"] * 100,
                                          name=label, marker_color=color, opacity=0.8))
        fig_dist.update_layout(
            title=f"Distribution of {feature} by Default Status",
            barmode="overlay" if is_numeric else "group",
            xaxis_title=feature, yaxis_title="% of Group",
            plot_bgcolor="white", paper_bgcolor="white",
        )

        # WoE chart
        try:
            woe_df, total_iv = compute_woe_iv(df, feature, bins=bins)
            bin_col = woe_df.columns[0]
            fig_woe = go.Figure()
            fig_woe.add_bar(x=woe_df[bin_col].astype(str), y=woe_df["woe"],
                            marker_color=[GOLD if w < 0 else TEAL for w in woe_df["woe"]],
                            name="WoE")
            fig_woe.add_scatter(x=woe_df[bin_col].astype(str),
                                y=woe_df["default_rate"] * 100,
                                mode="lines+markers", name="Default Rate %",
                                yaxis="y2", line=dict(color=DARK, width=2))
            fig_woe.update_layout(
                title=f"WoE by Bin — {feature} | IV = {total_iv:.3f} ({iv_strength(total_iv)})",
                yaxis=dict(title="WoE"),
                yaxis2=dict(title="Default Rate %", overlaying="y", side="right"),
                plot_bgcolor="white", paper_bgcolor="white",
            )
        except Exception as e:
            fig_woe = go.Figure()
            fig_woe.add_annotation(text=f"Could not compute WoE: {e}",
                                   x=0.5, y=0.5, showarrow=False)

        return fig_dist, fig_woe

    # ── Bivariate scatter + heatmap ──
    @app.callback(
        Output("bi-scatter", "figure"),
        Output("bi-heatmap", "figure"),
        Input("bi-x", "value"),
        Input("bi-y", "value"),
        Input("bi-color", "value"),
    )
    def update_bi(x_col, y_col, color_col):
        sample = df.sample(min(3000, len(df)), random_state=42)
        fig_scatter = go.Figure()
        for cat_val in sample[color_col].unique():
            sub = sample[sample[color_col] == cat_val]
            for flag, symbol in [(0, "circle"), (1, "x")]:
                s2 = sub[sub["default_flag"] == flag]
                fig_scatter.add_trace(go.Scatter(
                    x=s2[x_col], y=s2[y_col], mode="markers",
                    marker=dict(symbol=symbol, opacity=0.5, size=5),
                    name=f"{cat_val} | {'Default' if flag else 'OK'}",
                ))
        fig_scatter.update_layout(
            title=f"{x_col} vs {y_col} (coloured by {color_col})",
            xaxis_title=x_col, yaxis_title=y_col,
            plot_bgcolor="white", paper_bgcolor="white",
        )

        num_cols = [c for c in raw_numerics if c in df.columns][:12]
        corr = df[num_cols].corr()
        fig_heatmap = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale="RdBu", zmid=0,
            text=corr.round(2).values, texttemplate="%{text}",
        ))
        fig_heatmap.update_layout(title="Correlation Heatmap (numeric features)",
                                   plot_bgcolor="white", paper_bgcolor="white")
        return fig_scatter, fig_heatmap

    # ── Data quality ──
    @app.callback(
        Output("dq-missing", "figure"),
        Output("dq-table", "data"),
        Output("dq-table", "columns"),
        Input("tabs", "active_tab"),
    )
    def update_dq(tab):
        miss = df.isnull().sum().reset_index()
        miss.columns = ["feature", "missing"]
        miss["pct"] = (miss["missing"] / len(df) * 100).round(2)
        miss = miss[miss["missing"] > 0].sort_values("pct", ascending=True)
        fig = go.Figure(go.Bar(
            x=miss["pct"], y=miss["feature"], orientation="h",
            marker_color=TEAL,
            text=miss["pct"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
        ))
        fig.update_layout(title="Missing Values (%)", xaxis_title="% Missing",
                          plot_bgcolor="white", paper_bgcolor="white")

        dq_rows = []
        for col in df.columns:
            dq_rows.append({
                "Column": col,
                "Dtype": str(df[col].dtype),
                "Unique": df[col].nunique(),
                "Missing %": f"{df[col].isnull().mean() * 100:.1f}%",
            })
        cols = [{"name": c, "id": c} for c in ["Column", "Dtype", "Unique", "Missing %"]]
        return fig, dq_rows, cols

    # ── Model ROC + Coefficients ──
    @app.callback(
        Output("model-roc", "figure"),
        Output("model-coef", "figure"),
        Input("tabs", "active_tab"),
    )
    def update_model(tab):
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, test_proba)
        fig_roc = go.Figure()
        fig_roc.add_scatter(x=fpr, y=tpr, mode="lines", name=f"Our Model (AUC={test_auc:.3f})",
                            line=dict(color=TEAL, width=2))
        fig_roc.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Baseline AUC=0.68",
                            line=dict(color=GOLD, dash="dash"))
        fig_roc.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR",
                               plot_bgcolor="white", paper_bgcolor="white")

        top = coef_df.head(15)
        colors = [GOLD if c < 0 else TEAL for c in top["coefficient"]]
        fig_coef = go.Figure(go.Bar(
            x=top["coefficient"], y=top["feature"], orientation="h",
            marker_color=colors,
        ))
        fig_coef.update_layout(
            title="Top 15 Logistic Regression Coefficients",
            xaxis_title="Coefficient (standardised scale)",
            plot_bgcolor="white", paper_bgcolor="white",
        )
        return fig_roc, fig_coef

    # ── Business dashboard ──
    @app.callback(
        Output("biz-kpis", "children"),
        Output("biz-volume-risk", "figure"),
        Output("biz-pr-curve", "figure"),
        Output("biz-score-dist", "figure"),
        Input("biz-threshold", "value"),
    )
    def update_biz(threshold):
        from sklearn.metrics import precision_recall_curve

        proba = test_proba
        y_true = y_test.values

        # KPIs at threshold
        preds = (proba >= threshold).astype(int)
        approved = preds == 0  # approve if predicted non-default
        approval_rate = approved.mean()
        bad_rate_in_approved = y_true[approved].mean() if approved.sum() > 0 else 0
        missed_good = ((preds == 1) & (y_true == 0)).sum()

        kpis = html.Div([
            html.H6("At current threshold:", style={"color": DARK}),
            dbc.Row([
                dbc.Col(html.Div([
                    html.H3(f"{approval_rate:.1%}", style={"color": TEAL}),
                    html.Small("Approval Rate"),
                ])),
                dbc.Col(html.Div([
                    html.H3(f"{bad_rate_in_approved:.1%}", style={"color": GOLD}),
                    html.Small("Bad Rate in Approved"),
                ])),
                dbc.Col(html.Div([
                    html.H3(f"{missed_good:,}", style={"color": DARK}),
                    html.Small("Good Customers Declined"),
                ])),
            ]),
        ])

        # Volume vs risk curve
        thresholds = np.linspace(0.05, 0.95, 50)
        vol_risk = []
        for t in thresholds:
            p = (proba >= t).astype(int)
            app_mask = p == 0
            vol_risk.append({
                "threshold": t,
                "approval_rate": app_mask.mean(),
                "bad_rate": y_true[app_mask].mean() if app_mask.sum() > 0 else 0,
            })
        vr = pd.DataFrame(vol_risk)
        fig_vr = go.Figure()
        fig_vr.add_scatter(x=vr["approval_rate"], y=vr["bad_rate"],
                           mode="lines+markers", line=dict(color=TEAL, width=2),
                           text=vr["threshold"].round(2),
                           hovertemplate="Approval: %{x:.1%}<br>Bad Rate: %{y:.1%}<br>Threshold: %{text}")
        fig_vr.add_vline(x=approval_rate, line_dash="dash", line_color=GOLD,
                         annotation_text=f"Current: {threshold:.0%}")
        fig_vr.update_layout(title="Volume vs Risk Trade-off",
                              xaxis_title="Approval Rate", yaxis_title="Bad Rate in Approved",
                              plot_bgcolor="white", paper_bgcolor="white")

        # PR curve
        prec, rec, thr = precision_recall_curve(y_true, proba)
        fig_pr = go.Figure()
        fig_pr.add_scatter(x=rec, y=prec, mode="lines", line=dict(color=TEAL, width=2))
        fig_pr.update_layout(title="Precision–Recall (business: precision = portfolio quality)",
                              xaxis_title="Recall (good customers captured)",
                              yaxis_title="Precision (quality of approvals)",
                              plot_bgcolor="white", paper_bgcolor="white")

        # Score distribution
        fig_sd = go.Figure()
        for flag, label, color in [(0, "Non-Default", TEAL), (1, "Default", GOLD)]:
            mask = y_true == flag
            fig_sd.add_trace(go.Histogram(
                x=proba[mask], name=label, opacity=0.7, nbinsx=40,
                marker_color=color, histnorm="percent",
            ))
        fig_sd.add_vline(x=threshold, line_dash="dash", line_color=DARK,
                         annotation_text=f"Threshold {threshold:.2f}")
        fig_sd.update_layout(barmode="overlay",
                              title="Model Score Distribution by True Outcome",
                              xaxis_title="Predicted Default Probability",
                              yaxis_title="% of Group",
                              plot_bgcolor="white", paper_bgcolor="white")

        return kpis, fig_vr, fig_pr, fig_sd

    return app


# ─────────────────────────────────────────────
# MAIN
#─────────────────────────────────────────────

if __name__ == "__main__":
    print("⏳ Loading and cleaning data...")
    df = load_and_clean("loan_book.csv")

    print("⏳ Fitting logistic regression model...")
    pipe, woe_maps, feature_cols, train_auc, test_auc, coef_df, test_proba, y_test, test_df = fit_model(df)
    print(f"✅ Model fitted — Train AUC: {train_auc:.3f} | Test AUC: {test_auc:.3f}")

    print("⏳ Building Dash app...")
    app = build_app(df, pipe, woe_maps, feature_cols, train_auc, test_auc,
                    coef_df, test_proba, y_test, test_df)

    print("\n🚀 App running at http://127.0.0.1:8050\n")
    app.run(debug=False, port=8050)
