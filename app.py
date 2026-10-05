"""
2026 eThekwini Metro Election Analytics & Forecasting - Streamlit dashboard

Run:   streamlit run app.py
Needs: National.csv and Voting.pkl in the same folder as this file
       (or upload them from the sidebar).

Note: Voting.pkl contains ONLY the Random Forest (rf_model). The one-hot
preprocessor and label encoder were not saved in the notebook, so this app
rebuilds them from National.csv using the exact same steps and the same
train/test split (test_size=0.20, random_state=42, stratified). If the rebuilt
feature space does not match the pickle, the app retrains an equivalent
Random Forest and tells you so.
"""

import io
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Config (mirrors the notebook)
# ----------------------------------------------------------------------------
CSV_PATH = "National.csv"
MODEL_PATH = "Voting.pkl"
DEFAULT_MUNI = "ETH - eThekwini"
ELECTION_DATE = "2026-11-04"

FEATURES = [
    "Province", "Municipality", "Ward", "VotingDistrict", "VotingStationName",
    "RegisteredVoters", "BallotType", "SpoiltVotes", "DateGenerated",
]
CATEGORICAL = [
    "Province", "Municipality", "Ward", "VotingDistrict",
    "VotingStationName", "BallotType", "DateGenerated",
]
NUMERICAL = ["RegisteredVoters", "SpoiltVotes"]
WARD_KEYS = ["Province", "Municipality", "Ward"]

st.set_page_config(
    page_title="eThekwini 2026 Election Forecast",
    page_icon="🗳️",
    layout="wide",
)


# ----------------------------------------------------------------------------
# Data + model loading
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading election data…")
def load_csv(source) -> pd.DataFrame:
    """source is a file path (str) or raw bytes from an upload."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    # DateGenerated is left as text - the notebook's final training frame
    # (re-read from CSV before Stage 4) treated it as a categorical string.
    return pd.read_csv(source)


def add_winning_party(df: pd.DataFrame) -> pd.DataFrame:
    """Winning party per ward (sum votes per party; 'TIE' if tied) - as in notebook."""
    ward_party = (
        df.groupby(WARD_KEYS + ["PartyName"], as_index=False)["TotalValidVotes"].sum()
    )
    ward_party["MaxVotes"] = ward_party.groupby(WARD_KEYS)["TotalValidVotes"].transform("max")
    winners = ward_party[ward_party["TotalValidVotes"] == ward_party["MaxVotes"]].copy()
    n_winners = winners.groupby(WARD_KEYS)["PartyName"].transform("count")
    winners["WinningParty"] = winners["PartyName"]
    winners.loc[n_winners > 1, "WinningParty"] = "TIE"
    winning = winners.groupby(WARD_KEYS, as_index=False)["WinningParty"].first()
    return df.drop(columns=["WinningParty"], errors="ignore").merge(
        winning, on=WARD_KEYS, how="left"
    )


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("numerical", "passthrough", NUMERICAL),
        ]
    )


@st.cache_resource(show_spinner="Rebuilding preprocessor and checking the model…")
def build_artifacts(_df: pd.DataFrame, _rf, data_key: str):
    """Recreate encoder + preprocessor exactly as in the notebook and validate the pickle."""
    labelled = add_winning_party(_df)
    X = labelled[FEATURES]
    y = labelled["WinningParty"].astype(str)

    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
    )

    preprocessor = make_preprocessor()
    Xt_train = preprocessor.fit_transform(X_train)
    n_features = Xt_train.shape[1]

    model = _rf
    retrained = False
    if model is None or getattr(model, "n_features_in_", None) != n_features:
        retrained = True
        model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        model.fit(Xt_train, y_train)

    test_acc = accuracy_score(y_test, model.predict(preprocessor.transform(X_test)))
    train_acc = accuracy_score(y_train, model.predict(Xt_train))

    return {
        "model": model,
        "preprocessor": preprocessor,
        "label_encoder": label_encoder,
        "retrained": retrained,
        "n_features": n_features,
        "pickle_features": getattr(_rf, "n_features_in_", None),
        "train_acc": train_acc,
        "test_acc": test_acc,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def predict_proba_named(art, X: pd.DataFrame) -> pd.DataFrame:
    """Probabilities with party names as columns."""
    proba = art["model"].predict_proba(art["preprocessor"].transform(X[FEATURES]))
    names = art["label_encoder"].inverse_transform(art["model"].classes_.astype(int))
    return pd.DataFrame(proba, columns=names, index=X.index)


# ----------------------------------------------------------------------------
# Helper calculations
# ----------------------------------------------------------------------------
def station_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per station/ballot: registered voters, valid and spoilt votes.
    Registered/spoilt are repeated on each party row, so take max (not sum)."""
    return (
        df.groupby(WARD_KEYS + ["VotingDistrict", "BallotType"], as_index=False)
        .agg(
            RegisteredVoters=("RegisteredVoters", "max"),
            SpoiltVotes=("SpoiltVotes", "max"),
            ValidVotes=("TotalValidVotes", "sum"),
        )
    )


def turnout_pct(st_tbl: pd.DataFrame) -> float:
    reg = st_tbl["RegisteredVoters"].sum()
    return 100 * (st_tbl["ValidVotes"].sum() + st_tbl["SpoiltVotes"].sum()) / reg if reg else np.nan


def top_n_with_others(series: pd.Series, n: int = 10) -> pd.Series:
    series = series.sort_values(ascending=False)
    top = series.head(n)
    rest = series.iloc[n:].sum()
    return pd.concat([top, pd.Series({"OTHERS": rest})]) if rest > 0 else top


# ----------------------------------------------------------------------------
# Sidebar: data sources
# ----------------------------------------------------------------------------
st.sidebar.title("🗳️ eThekwini 2026")

csv_upload = st.sidebar.file_uploader("National.csv", type="csv")
pkl_upload = st.sidebar.file_uploader("Voting.pkl", type="pkl")

if csv_upload is not None:
    df = load_csv(csv_upload.getvalue())
    data_key = f"upload-{csv_upload.name}-{csv_upload.size}"
elif os.path.exists(CSV_PATH):
    df = load_csv(CSV_PATH)
    data_key = f"disk-{os.path.getmtime(CSV_PATH)}"
else:
    st.title("2026 eThekwini Metro Election Forecast")
    st.info(
        "Upload **National.csv** in the sidebar (or place it next to `app.py`).\n\n"
        "Source: IEC Data Downloads - https://results.elections.org.za/home/Downloads/ME-Results/"
    )
    st.stop()

missing = [c for c in FEATURES + ["PartyName", "TotalValidVotes"] if c not in df.columns]
if missing:
    st.error(f"National.csv is missing required columns: {missing}")
    st.stop()

rf_model = None
try:
    if pkl_upload is not None:
        rf_model = joblib.load(io.BytesIO(pkl_upload.getvalue()))
        data_key += f"-pkl-{pkl_upload.size}"
    elif os.path.exists(MODEL_PATH):
        rf_model = joblib.load(MODEL_PATH)
        data_key += f"-pkl-{os.path.getmtime(MODEL_PATH)}"
except Exception as e:  # noqa: BLE001
    st.sidebar.warning(f"Could not load model file ({e}). A new model will be trained.")

art = build_artifacts(df, rf_model, data_key)

munis = sorted(df["Municipality"].dropna().unique())
default_idx = munis.index(DEFAULT_MUNI) if DEFAULT_MUNI in munis else 0
muni = st.sidebar.selectbox("Municipality", munis, index=default_idx)
mdf = df[df["Municipality"] == muni]

page = st.sidebar.radio(
    "Page", ["Overview", "Ward explorer", "2026 forecast", "Single-station what-if", "Model"]
)
st.sidebar.caption(
    "Forecasts are data-driven estimates based on historical patterns, "
    "not confirmed election results."
)

st.title("2026 eThekwini Metro Election Analytics")
st.caption(f"Local Government Elections - {ELECTION_DATE} · Municipality: {muni}")

# ----------------------------------------------------------------------------
# Page: Overview
# ----------------------------------------------------------------------------
if page == "Overview":
    stn = station_table(mdf)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Wards", f"{mdf['Ward'].nunique():,}")
    c2.metric("Voting districts", f"{mdf['VotingDistrict'].nunique():,}")
    c3.metric("Parties", f"{mdf['PartyName'].nunique():,}")
    c4.metric("Historical turnout", f"{turnout_pct(stn):.1f}%")

    left, right = st.columns(2)

    with left:
        st.subheader("Top 10 parties by valid votes")
        votes = mdf.groupby("PartyName")["TotalValidVotes"].sum()
        plot_data = top_n_with_others(votes, 10).reset_index()
        plot_data.columns = ["PartyName", "TotalValidVotes"]
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(data=plot_data, x="TotalValidVotes", y="PartyName", palette="viridis", ax=ax)
        ax.set_xlabel("Total valid votes")
        ax.set_ylabel("")
        st.pyplot(fig)

    with right:
        st.subheader("Vote share (top 5 + others)")
        pie = top_n_with_others(votes, 5)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.pie(
            pie, labels=pie.index, autopct="%1.1f%%", startangle=140,
            colors=sns.color_palette("Set2", len(pie)),
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        st.pyplot(fig)

    st.subheader("Historical ward winners")
    winners = add_winning_party(mdf).drop_duplicates(WARD_KEYS)["WinningParty"].value_counts()
    st.bar_chart(winners)

    st.subheader("Registered voters vs valid votes per voting district")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(stn["RegisteredVoters"], stn["ValidVotes"], alpha=0.4, color="#2b5c8f")
    lim = max(stn["RegisteredVoters"].max(), 1)
    ax.plot([0, lim], [0, lim], "r--", label="100% turnout line")
    ax.set_xlabel("Registered voters")
    ax.set_ylabel("Valid votes")
    ax.legend()
    st.pyplot(fig)

    with st.expander("Raw data preview"):
        st.dataframe(mdf.head(200))

# ----------------------------------------------------------------------------
# Page: Ward explorer
# ----------------------------------------------------------------------------
elif page == "Ward explorer":
    wards = sorted(mdf["Ward"].dropna().unique(), key=lambda w: (len(str(w)), str(w)))
    ward = st.selectbox("Ward", wards)
    wdf = mdf[mdf["Ward"] == ward]

    ballots = sorted(wdf["BallotType"].dropna().unique())
    ballot_choice = st.radio("Ballot type", ["All"] + ballots, horizontal=True)
    if ballot_choice != "All":
        wdf = wdf[wdf["BallotType"] == ballot_choice]

    stn = station_table(wdf)
    hist_winner = add_winning_party(mdf[mdf["Ward"] == ward])["WinningParty"].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registered voters", f"{stn['RegisteredVoters'].sum():,.0f}")
    c2.metric("Valid votes", f"{stn['ValidVotes'].sum():,.0f}")
    c3.metric("Turnout", f"{turnout_pct(stn):.1f}%")
    c4.metric("Historical winner", hist_winner)

    share = wdf.groupby("PartyName")["TotalValidVotes"].sum()
    share = (100 * share / share.sum()).sort_values(ascending=False)
    st.subheader("Vote share by party (%)")
    st.bar_chart(share.head(12))

    st.subheader("Station detail")
    st.dataframe(
        wdf.groupby(["VotingDistrict", "VotingStationName", "BallotType"], as_index=False)
        .agg(RegisteredVoters=("RegisteredVoters", "max"),
             SpoiltVotes=("SpoiltVotes", "max"),
             ValidVotes=("TotalValidVotes", "sum")),
    )

# ----------------------------------------------------------------------------
# Page: 2026 forecast
# ----------------------------------------------------------------------------
elif page == "2026 forecast":
    st.subheader("Predicted ward winners for 4 November 2026")
    st.write(
        "The model scores every voting station in the selected wards using its latest "
        "known registered-voter and spoilt-vote figures, then averages the party "
        "probabilities per ward."
    )

    all_wards = sorted(mdf["Ward"].dropna().unique(), key=lambda w: (len(str(w)), str(w)))
    selected = st.multiselect("Target wards (leave empty for all)", all_wards)
    target = mdf[mdf["Ward"].isin(selected)] if selected else mdf

    base = target[FEATURES].drop_duplicates().copy()
    base["DateGenerated"] = ELECTION_DATE  # future date -> unseen category, ignored by encoder

    if st.button("Run forecast", type="primary"):
        proba = predict_proba_named(art, base)
        ward_proba = proba.groupby([base["Ward"].astype(str)]).mean()
        result = pd.DataFrame(
            {
                "Predicted winner": ward_proba.idxmax(axis=1),
                "Confidence": ward_proba.max(axis=1),
                "Runner-up": ward_proba.apply(lambda r: r.nlargest(2).index[-1], axis=1),
            }
        )
        result["Confidence"] = (100 * result["Confidence"]).round(1)
        result = result.sort_index(key=lambda s: s.map(lambda w: (len(w), w)))
        st.session_state["forecast"] = (result, ward_proba)

    if "forecast" in st.session_state:
        result, ward_proba = st.session_state["forecast"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Wards forecast", len(result))
        c2.metric("Leading party", result["Predicted winner"].value_counts().idxmax())
        c3.metric("Avg confidence", f"{result['Confidence'].mean():.1f}%")

        st.subheader("Wards predicted per party")
        st.bar_chart(result["Predicted winner"].value_counts())

        st.subheader("Ward-by-ward forecast")
        st.dataframe(result.rename_axis("Ward"))

        st.download_button(
            "Download forecast (CSV)",
            result.rename_axis("Ward").to_csv().encode(),
            "ethekwini_2026_forecast.csv",
            "text/csv",
        )

        with st.expander("Average party probabilities per ward"):
            st.dataframe((100 * ward_proba).round(1))

        st.warning(
            "These are model estimates built from historical patterns - not confirmed "
            "results. Treat low-confidence wards as too close to call."
        )

# ----------------------------------------------------------------------------
# Page: Single-station what-if
# ----------------------------------------------------------------------------
elif page == "Single-station what-if":
    st.subheader("Predict the winning party for one voting station")

    wards = sorted(mdf["Ward"].dropna().unique(), key=lambda w: (len(str(w)), str(w)))
    ward = st.selectbox("Ward", wards)
    wdf = mdf[mdf["Ward"] == ward]

    district = st.selectbox("Voting district", sorted(wdf["VotingDistrict"].unique()))
    ddf = wdf[wdf["VotingDistrict"] == district]
    station = st.selectbox("Voting station", sorted(ddf["VotingStationName"].dropna().unique()))
    sdf = ddf[ddf["VotingStationName"] == station]
    ballot = st.selectbox("Ballot type", sorted(sdf["BallotType"].unique()))
    row0 = sdf[sdf["BallotType"] == ballot].iloc[0]

    c1, c2 = st.columns(2)
    registered = c1.number_input(
        "Registered voters", min_value=0, value=int(row0["RegisteredVoters"]), step=10
    )
    spoilt = c2.number_input(
        "Spoilt votes", min_value=0, value=int(row0["SpoiltVotes"]), step=1
    )

    if st.button("Predict", type="primary"):
        X_new = pd.DataFrame(
            [{
                "Province": row0["Province"],
                "Municipality": row0["Municipality"],
                "Ward": row0["Ward"],
                "VotingDistrict": district,
                "VotingStationName": station,
                "RegisteredVoters": registered,
                "BallotType": ballot,
                "SpoiltVotes": spoilt,
                "DateGenerated": ELECTION_DATE,
            }]
        )
        proba = predict_proba_named(art, X_new).iloc[0].sort_values(ascending=False)
        st.success(f"Predicted winner: **{proba.index[0]}** ({100 * proba.iloc[0]:.1f}%)")
        st.bar_chart((100 * proba).rename("Probability (%)"))

# ----------------------------------------------------------------------------
# Page: Model
# ----------------------------------------------------------------------------
else:
    st.subheader("Model details")
    if art["retrained"] and art["pickle_features"] is None:
        st.warning(
            "No Voting.pkl was loaded, so the app trained an equivalent Random Forest "
            "(100 trees, max_depth=10, random_state=42)."
        )
    elif art["retrained"]:
        st.warning(
            f"Voting.pkl expects {art['pickle_features']} features but rebuilding the "
            f"preprocessor from this National.csv gives {art['n_features']}. The app "
            "trained an equivalent Random Forest (100 trees, max_depth=10, "
            "random_state=42) instead. Use the same National.csv as the notebook to use "
            "the saved model."
        )
    else:
        st.success(
            f"Voting.pkl loaded and verified against the rebuilt preprocessor "
            f"({art['n_features']:,} one-hot features)."
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train accuracy", f"{100 * art['train_acc']:.2f}%")
    c2.metric("Test accuracy", f"{100 * art['test_acc']:.2f}%")
    c3.metric("Train rows", f"{art['n_train']:,}")
    c4.metric("Test rows", f"{art['n_test']:,}")

    st.write("**Target classes (winning party per ward):**")
    st.write(list(art["label_encoder"].classes_))

    st.write("**Input features:**")
    st.code(", ".join(FEATURES))

    st.info(
        "Caveats: the target (ward winner) is the same for every station in a ward, and "
        "the features include ward/station identifiers, so the model largely learns which "
        "ward a station belongs to. A large train-vs-test gap indicates overfitting. "
        "`DateGenerated` is a file-generation date, so for 2026 it is an unseen value and "
        "contributes nothing."
    )
