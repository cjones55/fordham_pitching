#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import joblib
import tempfile

# ============================
# STREAMLIT PAGE CONFIG
# ============================

HEADER_MAROON = "#A00000"
BACKGROUND_DARK = "#1e1e1e"

st.set_page_config(
    page_title="Fordham Pitching Postgame",
    page_icon="⚾",
    layout="wide",
)

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {BACKGROUND_DARK};
        color: white;
    }}
    h1, h2, h3 {{
        color: {HEADER_MAROON};
        font-weight: 800;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1>Fordham Pitching Postgame – Stuff+</h1>", unsafe_allow_html=True)
st.write("Upload a TrackMan CSV to generate Stuff+ reports, movement charts, and downloadable PNG summaries.")

# ============================
# FILE UPLOADER
# ============================

uploaded_file = st.file_uploader("Upload Fordham TrackMan CSV", type=["csv"])

if uploaded_file is None:
    st.stop()

# Save uploaded file to a temp path
tmp_dir = Path(tempfile.mkdtemp())
csv_path = tmp_dir / "uploaded.csv"
with open(csv_path, "wb") as f:
    f.write(uploaded_file.getbuffer())

# ============================
# AXIS STYLE FIX
# ============================

def style_axes(ax):
    ax.tick_params(colors="white", which="both")
    for spine in ax.spines.values():
        spine.set_color("white")

# ============================
# LOAD CSV
# ============================

df_original = pd.read_csv(csv_path, encoding="latin1", sep=None, engine="python")
df = df_original.copy()

home_team = df_original["HomeTeam"].iloc[0]
away_team = df_original["AwayTeam"].iloc[0]

# Detect Fordham team string
if "Fordham" in str(home_team):
    fordham_team = home_team
elif "Fordham" in str(away_team):
    fordham_team = away_team
else:
    fordham_team = home_team  # fallback

df = df[df["PitcherTeam"] == fordham_team].copy()
df["Pitcher"] = df["Pitcher"].astype(str).str.strip()

game_date = pd.to_datetime(df["Date"].iloc[0]).strftime("%B %d, %Y")

# ============================
# SAFE COLUMN RENAMING
# ============================

rename_map = {
    "RelSpeed": "Velo",
    "InducedVertBreak": "IVB",
    "HorzBreak": "HB",
    "SpinRate": "Spin",
    "RelHeight": "RelH",
    "RelSide": "RelS",
    "Extension": "Ext",
    "VertApprAngle": "VAA",
    "HorzApprAngle": "HAA",
    "ZoneSpeed": "ZONE%",
}
df = df.rename(columns=rename_map)

# ============================
# LOAD LIGHTGBM STUFF+ MODEL
# ============================

model_dir = Path(__file__).parent

stuff_model = joblib.load(model_dir / "stuff_lgbm_model.pkl")
league_stats = joblib.load(model_dir / "stuff_lgbm_league.pkl")

league_mu = league_stats["mean"]
league_sigma = league_stats["std"] if league_stats["std"] > 0 else 1.0

# ============================
# PITCH TYPE NORMALIZATION
# ============================

pitch_map = {
    "Fastball": "FB",
    "FourSeamFastBall": "FB",
    "4-Seam": "FB",
    "FF": "FB",
    "Sinker": "SI",
    "Cutter": "FC",
    "Slider": "SL",
    "Sweeper": "SW",
    "Curveball": "CU",
    "ChangeUp": "CH",
    "Changeup": "CH"
}

df["pitch_abbr"] = df["TaggedPitchType"].map(pitch_map)
df["pitch_abbr"] = df["pitch_abbr"].fillna(
    df["TaggedPitchType"].astype(str).str[:2].str.upper()
)

# ============================
# RUN / ER ENGINE
# ============================

def compute_pitcher_runs(df_in: pd.DataFrame):
    df_local = df_in.copy()

    sort_cols = []
    if "Inning" in df_local.columns:
        sort_cols.append("Inning")
    if "PitchNo" in df_local.columns:
        sort_cols.append("PitchNo")
    elif "PitchOfPA" in df_local.columns:
        sort_cols.append("PitchOfPA")
    elif "PitchNumber" in df_local.columns:
        sort_cols.append("PitchNumber")

    if sort_cols:
        df_local = df_local.sort_values(sort_cols)
    else:
        df_local = df_local.sort_index()

    if "Inning" not in df_local.columns:
        df_local["Inning"] = 1
    if "RunsScored" not in df_local.columns:
        df_local["RunsScored"] = 0
    if "OutsOnPlay" not in df_local.columns:
        df_local["OutsOnPlay"] = 0
    if "PlayResult" not in df_local.columns:
        df_local["PlayResult"] = ""

    base_owner = {1: None, 2: None, 3: None}
    current_inning = df_local["Inning"].iloc[0]
    outs = 0
    error_flag = False

    pitcher_R = {}
    pitcher_ER = {}

    for idx, row in df_local.iterrows():
        p = row["Pitcher"]
        if p not in pitcher_R:
            pitcher_R[p] = 0
            pitcher_ER[p] = 0

        if row["Inning"] != current_inning:
            current_inning = row["Inning"]
            outs = 0
            base_owner = {1: None, 2: None, 3: None}
            error_flag = False

        if "Error" in str(row["PlayResult"]):
            error_flag = True

        runs_scored = int(row["RunsScored"])

        if runs_scored > 0:
            scorers = [
                (base_owner[3], "3B"),
                (base_owner[2], "2B"),
                (base_owner[1], "1B"),
                (p, "Batter")
            ]
            for _ in range(runs_scored):
                owner, _label = scorers.pop(0)
                if owner is None:
                    owner = p
                pitcher_R[owner] += 1
                if not error_flag:
                    pitcher_ER[owner] += 1

        pr = str(row["PlayResult"])
        if pr in ["Single", "Double", "Triple", "HomeRun"]:
            if pr == "HomeRun":
                base_owner = {1: None, 2: None, 3: None}
            elif pr == "Triple":
                base_owner = {1: None, 2: None, 3: p}
            elif pr == "Double":
                base_owner[3] = base_owner[1]
                base_owner[2] = p
                base_owner[1] = None
            elif pr == "Single":
                base_owner[3] = base_owner[2]
                base_owner[2] = base_owner[1]
                base_owner[1] = p

        outs += int(row["OutsOnPlay"])

        if outs >= 3:
            outs = 0
            base_owner = {1: None, 2: None, 3: None}
            error_flag = False

    return pitcher_R, pitcher_ER

pitcher_R_map, pitcher_ER_map = compute_pitcher_runs(df)

# ============================
# OUTPUT DIR (TEMP)
# ============================

output_dir = tmp_dir / "fordham_postgame_pitching"
output_dir.mkdir(exist_ok=True)

# ============================
# LOOP THROUGH PITCHERS
# ============================

pitchers = df["Pitcher"].unique()

progress = st.progress(0.0)
total_pitchers = len(pitchers)

pitch_colors = {
    "FB": "#1f77b4",
    "SI": "#17becf",
    "FC": "#ff7f0e",
    "SL": "#d62728",
    "CU": "#9467bd",
    "CH": "#2ca02c",
    "SW": "#8c564b"
}

# ============================
# OPPONENT + HOME/AWAY AUTO-DETECTION
# ============================

if home_team == fordham_team:
    opponent = away_team
    matchup_title = f"Fordham vs {opponent}"
else:
    opponent = home_team
    matchup_title = f"{opponent} vs Fordham"

# ============================
# PROCESS EACH PITCHER
# ============================

for i, pitcher in enumerate(pitchers, start=1):

    pdf = df[df["Pitcher"] == pitcher].copy()

    # GAME LINE
    total_pitches = len(pdf)
    strike_calls = ["StrikeCalled", "StrikeSwinging", "FoulBall", "FoulBallNotFieldable"]

    strikes = pdf["PitchCall"].isin(strike_calls).sum()
    strike_pct = round(strikes / total_pitches * 100, 1) if total_pitches else 0

    whiffs = pdf["PitchCall"].eq("StrikeSwinging").sum()
    walks = pdf["KorBB"].eq("Walk").sum()
    strikeouts = pdf["KorBB"].eq("Strikeout").sum()
    hbp = pdf["PitchCall"].eq("HitByPitch").sum()

    hits = pdf["PlayResult"].isin(["Single", "Double", "Triple", "HomeRun"]).sum()
    hr = pdf["PlayResult"].eq("HomeRun").sum()

    runs = pitcher_R_map.get(pitcher, 0)
    er = pitcher_ER_map.get(pitcher, 0)

    outs_on_play = pdf["OutsOnPlay"].sum()
    total_outs = outs_on_play + strikeouts
    ip = total_outs // 3 + (total_outs % 3) / 10 if total_outs else 0.0

    # STUFF+ MODEL
    stuff_features = ["Velo","IVB","HB","Spin","RelH","RelS","Ext","VAA","HAA"]
    Xs = pdf[stuff_features].fillna(0)
    pdf["stuff_prob"] = stuff_model.predict_proba(Xs)[:, 1]
    pdf["Stuff+"] = 100 + 50 * ((pdf["stuff_prob"] - league_mu) / league_sigma)

    # FLAGS
    pdf["is_csw"] = pdf["PitchCall"].isin(["StrikeCalled","StrikeSwinging"])
    pdf["is_whiff"] = pdf["PitchCall"].eq("StrikeSwinging")
    pdf["is_swing"] = pdf["PitchCall"].isin([
        "StrikeSwinging","FoulBall","FoulBallNotFieldable",
        "InPlay","InPlayNoOut","InPlayOut"
    ])
    pdf["is_strike"] = pdf["PitchCall"].isin(strike_calls)

    pdf["in_zone"] = (
        pdf["PlateLocSide"].between(-0.83, 0.83) &
        pdf["PlateLocHeight"].between(1.5, 3.5)
    )

    # AGGREGATE
    agg = pdf.groupby("pitch_abbr").agg(
        N=("PitchCall","count"),
        Velo=("Velo","mean"),
        IVB=("IVB","mean"),
        HB=("HB","mean"),
        Spin=("Spin","mean"),
        RelH=("RelH","mean"),
        RelS=("RelS","mean"),
        Ext=("Ext","mean"),
        VAA=("VAA","mean"),
        HAA=("HAA","mean"),
        Stuff_plus=("Stuff+","mean"),
        CSW=("is_csw","sum"),
        Whiffs=("is_whiff","sum"),
        Swings=("is_swing","sum"),
        Strikes=("is_strike","sum"),
        InZone=("in_zone","sum")
    ).reset_index()

    agg = agg.rename(columns={"Stuff_plus": "Stuff+", "pitch_abbr": "Pitch"})

    total_N = agg["N"].sum()
    agg["Usage%"] = (agg["N"] / total_N * 100).round(1)
    agg["CSW%"] = (agg["CSW"] / agg["N"] * 100).round(1)
    agg["Whiff%"] = np.where(
        agg["Swings"] > 0,
        (agg["Whiffs"] / agg["Swings"] * 100).round(1),
        0.0
    )
    agg["Strike%"] = (agg["Strikes"] / agg["N"] * 100).round(1)
    agg["Zone%"] = (agg["InZone"] / agg["N"] * 100).round(1)

    # FIGURE LAYOUT
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(BACKGROUND_DARK)

    # LOGO
    logo_path = model_dir / "rams.png"
    if logo_path.exists():
        logo_img = mpimg.imread(logo_path)
        fig.figimage(logo_img, xo=40, yo=fig.bbox.ymax - 180, zorder=50, alpha=1.0)

    # TITLE
    title = f"{pitcher} – {matchup_title}"
    summary = (
        f"IP: {ip:.1f}  H: {hits}  R: {runs}  ER: {er}  "
        f"BB: {walks}  K: {strikeouts}  HR: {hr}  HBP: {hbp}  "
        f"Whiffs: {whiffs}  Strike%: {strike_pct}%"
    )

    fig.suptitle(title, fontsize=26, fontweight="bold", color=HEADER_MAROON, y=0.97)
    plt.text(0.5, 0.93, summary, ha="center", va="center", color="white", fontsize=14)

    # MOVEMENT (SCATTER + CENTROIDS)
    ax1 = plt.subplot2grid((5, 4), (0, 0), rowspan=2)
    style_axes(ax1)
    ax1.set_facecolor(BACKGROUND_DARK)
    ax1.set_xlim(-25, 25)
    ax1.set_ylim(-25, 25)

    throws = pdf["PitcherThrows"].iloc[0] if "PitcherThrows" in pdf.columns else "Right"

    if throws.upper().startswith("R"):
        arm_color = (0.10, 0.30, 0.60, 0.10)
        glove_color = (0.60, 0.10, 0.10, 0.10)
        arm_xmin, arm_xmax = 0, 25
        glove_xmin, glove_xmax = -25, 0
    else:
        arm_color = (0.10, 0.30, 0.60, 0.10)
        glove_color = (0.60, 0.10, 0.10, 0.10)
        arm_xmin, arm_xmax = -25, 0
        glove_xmin, glove_xmax = 0, 25

    ax1.axvspan(arm_xmin, arm_xmax, facecolor=arm_color, zorder=0)
    ax1.axvspan(glove_xmin, glove_xmax, facecolor=glove_color, zorder=0)

    ax1.axhline(0, color="white", linestyle=":", linewidth=1.2)
    ax1.axvline(0, color="white", linestyle=":", linewidth=1.2)

    # scatter
    for _, row in pdf.iterrows():
        c = pitch_colors.get(row["pitch_abbr"], "white")
        ax1.scatter(
            row["HB"], row["IVB"],
            s=40, color=c, edgecolor="white", linewidth=0.5, alpha=0.9
        )

    # centroids
    centroids = pdf.groupby("pitch_abbr")[["HB", "IVB"]].mean().reset_index()
    for _, row in centroids.iterrows():
        pitch = row["pitch_abbr"]
        c = pitch_colors.get(pitch, "white")
        ax1.scatter(
            row["HB"], row["IVB"],
            s=250, color=c, edgecolor="white", linewidth=1.5, zorder=5
        )
        ax1.text(
            row["HB"], row["IVB"], pitch,
            color="white", fontsize=9, weight="bold",
            ha="center", va="center", zorder=6
        )

    ax1.set_title("Movement", color="white")

    # LOCATION PLOTS
    def draw_mlb_zone(ax):
        ax.set_facecolor(BACKGROUND_DARK)
        style_axes(ax)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(0, 5)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)

        zone_x = [-0.83, 0.83, 0.83, -0.83, -0.83]
        zone_y = [1.5, 1.5, 3.5, 3.5, 1.5]

        ax.plot(zone_x, zone_y, color="white", linewidth=2.5)
        ax.fill_between([-0.83, 0.83], 1.5, 3.5, color="white", alpha=0.06)

    def draw_home_plate(ax):
        plate_x = [-0.83, 0.83, 0.83, 0, -0.83, -0.83]
        plate_y = [0, 0, 0.17, 0.34, 0.17, 0]
        ax.plot(plate_x, plate_y, color="white", linewidth=2)
        ax.fill(plate_x, plate_y, color="white", alpha=0.10)

    # LHH
    axL = plt.subplot2grid((5, 4), (0, 1), rowspan=2)
    draw_mlb_zone(axL)
    draw_home_plate(axL)

    LHH = pdf[pdf["BatterSide"] == "Left"]
    for _, row in LHH.iterrows():
        c = pitch_colors.get(row["pitch_abbr"], "white")
        axL.scatter(
            row["PlateLocSide"], row["PlateLocHeight"],
            s=85, color=c, edgecolor="white"
        )

    axL.set_title("LHH", color="white")

    # RHH
    axR = plt.subplot2grid((5, 4), (0, 2), rowspan=2)
    draw_mlb_zone(axR)
    draw_home_plate(axR)

    RHH = pdf[pdf["BatterSide"] == "Right"]
    for _, row in RHH.iterrows():
        c = pitch_colors.get(row["pitch_abbr"], "white")
        axR.scatter(
            row["PlateLocSide"], row["PlateLocHeight"],
            s=85, color=c, edgecolor="white"
        )

    axR.set_title("RHH", color="white")

    # RELEASE
    axRel = plt.subplot2grid((5, 4), (0, 3), rowspan=2)
    style_axes(axRel)
    axRel.set_facecolor(BACKGROUND_DARK)
    axRel.set_xlim(-4, 4)
    axRel.set_ylim(3, 7)
    axRel.set_aspect("equal", adjustable="box")

    for _, row in pdf.iterrows():
        c = pitch_colors.get(row["pitch_abbr"], "white")
        axRel.scatter(
            row["RelS"], row["RelH"],
            s=25, color=c, edgecolor="white"
        )

    axRel.set_title("Release", color="white")

    # TABLE
    axT = plt.subplot2grid((5, 4), (2, 0), colspan=4, rowspan=2)
    axT.axis("off")

    table_df = agg[[
        "Pitch","N","Usage%","Velo","IVB","HB",
        "Spin","Stuff+","CSW%","Whiff%","Strike%","Zone%"
    ]].round(2)

    tbl = axT.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
        bbox=[0, 0, 1, 1]
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(HEADER_MAROON)
            cell.set_text_props(color="white", weight="bold")
        else:
            pitch = table_df.iloc[r - 1]["Pitch"]
            bg = pitch_colors.get(pitch, BACKGROUND_DARK)
            cell.set_facecolor(bg)
            cell.set_text_props(color="white")

    # FOOTER
    axFooter = plt.subplot2grid((5, 4), (4, 0), colspan=4)
    axFooter.axis("off")

    axFooter.text(
        0.5, 0.55, summary,
        ha="center", va="center",
        fontsize=14, color="white", weight="bold"
    )

    axFooter.text(
        0.98, 0.15, f"Game Date: {game_date}",
        ha="right", va="center",
        fontsize=12, color="white"
    )

    # SAVE IMAGE
    out = output_dir / f"{pitcher.replace(',','')}_Summary.png"
    plt.savefig(out, dpi=300, facecolor=fig.get_facecolor())
    plt.close()

    progress.progress(i / total_pitchers)

# ============================
# DISPLAY & DOWNLOAD SECTION
# ============================

st.subheader("📁 Generated Pitching Reports")

png_files = sorted(output_dir.glob("*.png"))

if not png_files:
    st.error("No images were generated. Check your CSV formatting.")
else:
    for img_path in png_files:
        st.markdown(f"### {img_path.stem.replace('_', ' ')}")
        st.image(str(img_path))

        with open(img_path, "rb") as f:
            st.download_button(
                label="Download PNG",
                data=f,
                file_name=img_path.name,
                mime="image/png",
                key=str(img_path)
            )
