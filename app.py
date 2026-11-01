"""Python & XGBoost Fuel Strategy Simulator: lap-by-lap pit windows and fuel-mileage finishes.

Run:  streamlit run app.py

Reads the small tables in data/ (written by build_data.py), the XGBoost models in models/
and their backtest in reports/ (written by train_models.py), and simulates races with
race_sim.py. Tabs: Race forecast · Replay a race · Fuel windows · The models · How it works
"""
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import app_style as s
import models as ml
import race_sim as rs

TRACK_LABELS = {"michigan": "Michigan", "pocono": "Pocono", "nashville": "Nashville", "kansas": "Kansas",
                "las-vegas": "Las Vegas", "texas": "Texas", "homestead": "Homestead-Miami",
                "martinsville": "Martinsville"}
SHORT_TRACKS = {"martinsville"}
DEFAULT_TRACK = "homestead"     # where the app opens
N_SIMS = 2000
XGB, SIMPLE = "XGBoost models", "Simple averages"
MODEL_LABELS = {"caution": "Caution risk", "green": "Green-flag stops", "yellow": "Yellow-flag stops"}
MODEL_QUESTIONS = {"caution": "Will a caution come out on the next lap?",
                   "green": "Does this car pit under green on this lap?",
                   "yellow": "Does this car pit under this caution?"}

st.set_page_config(page_title=s.TITLE, page_icon="⛽", layout="wide")
theme = st.context.theme.type if st.context.theme.type in ("light", "dark") else "light"
st.html(s.style_css(theme))
st.html(s.HERO)


# ---------------------------------------------------------------- data and simulations

@st.cache_data
def get_data():
    return rs.load()


@st.cache_data
def get_stats():
    """Headline counts for the hook and the tech panel."""
    d = get_data()
    return {"races": len(d["races"]), "tracks": d["races"]["track"].nunique(),
            "stops": int(d["stints"]["end_how"].str.startswith("pit").sum()), "stints": len(d["stints"]),
            "car_laps": int(d["stints"]["laps"].sum()), "run_outs": len(d["run_outs"])}


@st.cache_data(show_spinner=False)
def get_params(track, race_id, yellow_burn, save):
    return rs.track_params(get_data(), track, race_id, yellow_burn=yellow_burn, save=save)


@st.cache_resource
def get_models():
    """The trained XGBoost models and their notes, or (None, None) before train_models.py has run."""
    return ml.load()


@st.cache_data
def get_reports():
    names = ("backtest_metrics", "backtest_races", "shap_importance", "shap_dependence")
    if not all((ml.REPORTS / f"{n}.csv").exists() for n in names):
        return None
    return {n: pd.read_csv(ml.REPORTS / f"{n}.csv") for n in names}


@st.cache_resource(show_spinner="Turning the models into lap-by-lap tables...")
def get_tables(track, race_id):
    """The models' answers for every lap of this race, as lookup tables for the simulation."""
    return ml.tables(get_models()[0], rs.race_format(get_data(), track, race_id))


@st.cache_data(show_spinner="Simulating the race 2,000 times...")
def get_forecast(track, yellow_burn, save, engine):
    t = get_params(track, None, yellow_burn, save)
    tables = get_tables(track, None) if engine == XGB else None
    return rs.summary(t, rs.simulate(t, n_sims=N_SIMS, tables=tables))


@st.cache_data(show_spinner="Checking against every real race at this track...")
def get_check(track, yellow_burn, save, engine):
    models = get_models()[0] if engine == XGB else None
    return rs.check_track(get_data(), track, n_sims=300, models=models, yellow_burn=yellow_burn, save=save)


@st.cache_data(show_spinner="Simulating the rest of the race...")
def get_replay(track, race_id, lap, yellow_burn, save, engine):
    data = get_data()
    t = get_params(track, race_id, yellow_burn, save)
    state = rs.replay_state(data, race_id, lap, yellow_burn)
    tables = get_tables(track, race_id) if engine == XGB else None
    sim = rs.simulate(t, n_sims=N_SIMS, start_lap=lap, state=state, seed=lap, tables=tables)
    table = rs.replay_table(t, state, sim, lap)
    return table.join(rs.actual_next_stop(data, race_id, table, lap)), state["caution_left"]


def panel(key):
    """A bordered card with the three-colour stripe across the top (styled in app_style)."""
    return st.container(border=True, key=f"panel_{key}")


def pct(label, top=1.0, help=None):
    return st.column_config.ProgressColumn(label, format="percent", min_value=0, max_value=top, help=help)


def tech_story(k):
    """How the app is built: the pipeline from raw feeds to the page, with real counts."""
    return f"""
**1 · Data pipeline.** Python and Requests pull public race timing feeds (every lap time,
every pit stop and race control's notes) for {k['races']} races at {k['tracks']} ovals since
2022. Each feed is downloaded once and cached. `build_data.py` boils hundreds of files down
to five small CSV tables (about 1 MB), so the app never touches the raw feeds.

**2 · Inference with pandas.** {k['car_laps']:,} car-laps are cut into {k['stints']:,}
stints between {k['stops']:,} pit stops, and every lap is marked green or yellow. A lap-time
check (any green lap 25% off a car's own pace) catches stops the pit data missed, so only
clean stints count as evidence. The tank estimate is then tested against race control's
{k['run_outs']} notes about cars running out of fuel.

**3 · XGBoost models.** Three classifiers answer the questions the simulation asks every
lap: will a caution come out, does this car pit under green, does it pit under yellow? Each is
backtested walk-forward (every race from 2024 on predicted only from earlier races) against the
simple averages it replaces, and explained with SHAP values. The scores are in "The models"
tab, wins and losses alike.

**4 · Monte Carlo in NumPy.** The models are turned into lap-by-lap lookup tables, then every
car, in {N_SIMS:,} races at once, is played forward lap by lap in one vectorized loop: one to
two seconds per track.

**5 · The app.** Streamlit with cached simulations and light and dark themes. Plotly charts
use a palette checked for colour-blind readers. It's tested with Streamlit's AppTest, plus
screenshots driven by Playwright.
"""


def model_scores(reports):
    """Model vs baseline per model: log loss, and how far off the per-race counts were."""
    met = reports["backtest_metrics"]
    races = reports["backtest_races"]
    mae = races.assign(m=(races["model"] - races["actual"]).abs(), b=(races["baseline"] - races["actual"]).abs())
    mae = mae.groupby("model_kind")[["m", "b"]].mean()
    out = {}
    for kind in MODEL_LABELS:
        mm = met[met["model"] == kind]
        xgb_row = mm[mm["who"].str.startswith("XGBoost")].iloc[0]
        base_row = mm[mm["who"].str.startswith("Simple")].iloc[0]
        out[kind] = {"skill": float(xgb_row["skill_vs_baseline"]), "auc_model": float(xgb_row["auc"]),
                     "auc_base": float(base_row["auc"]),
                     "mae_model": float(mae.loc[kind, "m"]), "mae_base": float(mae.loc[kind, "b"])}
    return out


# ---------------------------------------------------------------- charts

def pit_chart(t, fc):
    """Share of the field pitting on each lap, under green and under yellow."""
    p = fc["pit_by_lap"]
    p = p.loc[1:p.index[(p.sum(axis=1) > 0)].max()] * 100
    colours = s.SERIES[theme]
    fig = go.Figure()
    for col, name in (("green", "Green-flag stops"), ("yellow", "Yellow-flag stops")):
        fig.add_bar(x=p.index, y=p[col], name=name, marker=dict(color=colours[col], line_width=0),
                    hovertemplate=f"Lap %{{x}}: %{{y:.1f}}% of cars pit under {col}<extra></extra>")
    # Stage breaks send most of the field down pit road at once; those spikes would flatten
    # everything else, so the axis is capped at the ordinary laps and the spikes are labelled.
    total = p.sum(axis=1)
    cap = max(float(total.quantile(0.97)) * 1.6, 2.0)
    for lap, value in total[total > cap].items():
        fig.add_annotation(x=lap, y=cap, text=f"↑ {value:.0f}%", showarrow=False, yshift=10,
                           xanchor="right", xshift=-4, font=dict(size=11, color=s.INK[theme]))
    for i, lap in enumerate(t.stage_breaks, 1):
        fig.add_vline(x=lap + 0.5, line=dict(color=s.MUTED, width=1, dash="dot"),
                      annotation=dict(text=f"Stage {i} ends", font=dict(color=s.MUTED, size=11)),
                      annotation_position="top right", annotation_yshift=-18)
    s.style_fig(fig, theme, height=380, x_title="Lap", y_title="Cars pitting on this lap (%)")
    fig.update_layout(barmode="stack", bargap=0.15, legend_traceorder="normal")
    fig.update_xaxes(range=[0, p.index.max() + 1])
    fig.update_yaxes(range=[0, cap * 1.12])
    return fig


def next_stop_chart(table, lap, laps):
    """Each car's likely next stop (middle 80% of simulations) and where it really stopped."""
    d = table[table["p_stop"] >= 0.05].sort_values("next_stop_mid", ascending=False)
    names = "#" + d["car"] + " " + d["driver"]
    blue = s.SERIES[theme]["green"]
    fig = go.Figure()
    fig.add_bar(y=names, x=d["next_stop_p90"] - d["next_stop_p10"] + 1, base=d["next_stop_p10"] - 0.5,
                orientation="h", name="Likely next stop (80% of simulations)",
                marker=dict(color=blue, opacity=0.35, cornerradius=4),
                customdata=d[["next_stop_p10", "next_stop_p90"]],
                hovertemplate="%{y}: laps %{customdata[0]:.0f}–%{customdata[1]:.0f}<extra></extra>")
    fig.add_scatter(y=names, x=d["next_stop_mid"], mode="markers", name="Most likely",
                    marker=dict(color=blue, size=9, line=dict(color=s.SURFACE[theme], width=2)),
                    hovertemplate="%{y}: around lap %{x:.0f}<extra></extra>")
    real = d["actual_next_stop"].notna()
    fig.add_scatter(y=names[real], x=d.loc[real, "actual_next_stop"], mode="markers", name="Real next stop",
                    marker=dict(color=s.INK[theme], size=9, symbol="diamond"),
                    customdata=d.loc[real, "actual_flag"],
                    hovertemplate="%{y}: really pitted on lap %{x} under %{customdata}<extra></extra>")
    s.style_fig(fig, theme, height=max(320, 24 * len(d) + 80), x_title="Lap")
    fig.update_layout(bargap=0.4)
    fig.update_xaxes(range=[lap, laps + 2], showgrid=True, gridcolor=s.GRID[theme])
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=11))
    return fig


def window_chart(t):
    """Where real green-flag stops came, against the size of the tank."""
    blue = s.SERIES[theme]["green"]
    fig = go.Figure()
    fig.add_histogram(x=t.green_targets, xbins=dict(size=2), name="Green-flag stops",
                      marker=dict(color=blue, line=dict(color=s.SURFACE[theme], width=1)),
                      hovertemplate="%{x} laps of fuel: %{y} stops<extra></extra>")
    for x, text in ((t.full_window, "Tank at full pace"), (t.window, "Tank with saving")):
        fig.add_vline(x=x, line=dict(color=s.INK[theme], width=1.5, dash="dash"),
                      annotation=dict(text=f"{text}: {x:.0f}", font=dict(size=11)),
                      annotation_position="top left" if x == t.full_window else "top right")
    s.style_fig(fig, theme, height=340, x_title="Laps of fuel used when the car pitted under green",
                y_title="Stops")
    fig.update_layout(showlegend=False)
    return fig


def importance_chart(imp, kind):
    """Average size of each feature's SHAP value: how much it moves the model's answer."""
    d = imp[imp["model"] == kind].sort_values("mean_abs_shap")
    fig = go.Figure(go.Bar(x=d["mean_abs_shap"], y=d["feature"].map(ml.NAMES), orientation="h",
                           marker=dict(color=s.SERIES[theme]["green"], cornerradius=4),
                           hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
    s.style_fig(fig, theme, height=60 + 34 * len(d), x_title="Average push on the answer (SHAP, log-odds)")
    fig.update_layout(showlegend=False)
    fig.update_xaxes(showgrid=True, gridcolor=s.GRID[theme])
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    return fig


def dependence_chart(dep, kind, feature):
    """How one feature's SHAP value changes across its range: above zero raises the chance."""
    d = dep[(dep["model"] == kind) & (dep["feature"] == feature)]
    blue = s.SERIES[theme]["green"]
    if feature == "track":
        d = d.assign(x=d["x"].map(TRACK_LABELS)).sort_values("shap")
        fig = go.Figure(go.Bar(x=d["shap"], y=d["x"], orientation="h", marker=dict(color=blue, cornerradius=4),
                               hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
        s.style_fig(fig, theme, height=340, x_title="Push on the answer (SHAP, log-odds)")
        fig.update_xaxes(showgrid=True, gridcolor=s.GRID[theme], zeroline=True, zerolinecolor=s.MUTED)
        fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    else:
        d = d.astype({"x": float}).sort_values("x")
        fig = go.Figure(go.Scatter(x=d["x"], y=d["shap"], mode="lines+markers",
                                   line=dict(color=blue, width=2), marker=dict(size=8, color=blue),
                                   hovertemplate=f"{ml.NAMES[feature]} %{{x:.2f}}: %{{y:+.2f}}<extra></extra>"))
        fig.add_hline(y=0, line=dict(color=s.MUTED, width=1))
        s.style_fig(fig, theme, height=340, x_title=ml.NAMES[feature], y_title="Push on the answer (log-odds)")
    fig.update_layout(showlegend=False)
    return fig


def caution_curve_chart(t, tables):
    """The caution model's risk lap by lap (if no random caution comes) next to the flat average."""
    restarts = [0] + [b + int(np.median(t.stage_len)) for b in t.stage_breaks]
    laps = np.arange(1, t.laps)
    since = np.array([lap - max(r for r in restarts if r < lap) for lap in laps])
    green = ~np.isin(laps, [b + k for b in t.stage_breaks for k in range(1, int(np.median(t.stage_len)) + 1)])
    risk = np.where(green, tables["caution"][laps, np.minimum(since, ml.MAX_SINCE)] * 100, np.nan)
    fig = go.Figure()
    fig.add_scatter(x=laps, y=risk, mode="lines", name="XGBoost model", line=dict(color=s.SERIES[theme]["green"], width=2),
                    hovertemplate="Lap %{x}: %{y:.1f}% chance of a caution next lap<extra></extra>")
    fig.add_scatter(x=laps, y=np.where(green, t.hazard * 100, np.nan), mode="lines", name="Simple average",
                    line=dict(color=s.INK[theme], width=1.5, dash="dash"),
                    hovertemplate="Lap %{x}: %{y:.1f}% (the same every lap)<extra></extra>")
    for i, lap in enumerate(t.stage_breaks, 1):
        fig.add_vline(x=lap + 0.5, line=dict(color=s.MUTED, width=1, dash="dot"),
                      annotation=dict(text=f"Stage {i} ends", font=dict(color=s.MUTED, size=11)),
                      annotation_position="top right")
    s.style_fig(fig, theme, height=340, x_title="Lap", y_title="Chance of a caution next lap (%)")
    fig.update_layout(legend_traceorder="normal")
    fig.update_yaxes(rangemode="tozero")
    return fig


# ---------------------------------------------------------------- controls

data = get_data()
stats = get_stats()

# The hook for fans, and a button for the tech side
h1, h2 = st.columns([4, 1], vertical_alignment="center")
h1.html(f'<div class="hook"><b>Will they make it on fuel?</b> Every pit stop from {stats["races"]} Cup races '
        f'since 2022, rebuilt lap by lap, then raced {N_SIMS:,} more times to see who has to stop late, who can '
        f'stretch it, and who runs dry.</div>')
with h2.popover("Under the hood", icon=":material/settings:", width="stretch", key="tech"):
    st.html(s.eyebrow("Under the hood"))
    st.html('<div class="hook"><b>How it\'s built</b></div>')
    st.html(s.chips([f"{stats['races']} races", f"{stats['stops']:,} pit stops", f"{stats['car_laps']:,} car-laps",
                     (f"{N_SIMS:,} simulations per track", "hot")]))
    st.html(s.tech_badges())
    st.markdown(tech_story(stats))

c1, c2, c3 = st.columns([2, 2, 3])
track = c1.selectbox("Track", list(TRACK_LABELS), format_func=TRACK_LABELS.get,
                     index=list(TRACK_LABELS).index(DEFAULT_TRACK))
have_models = get_models()[0] is not None
engine = c2.segmented_control(
    "Simulation engine", [XGB, SIMPLE], default=XGB if have_models else SIMPLE, key="engine",
    disabled=not have_models,
    help="XGBoost models decide cautions and pit stops lap by lap. Simple averages are the per-track "
         "averages they replaced. Compare the two in \"The models\" tab.") or (XGB if have_models else SIMPLE)
with c3.expander("Assumptions (nobody publishes fuel burn)"):
    yellow_burn = st.slider("Fuel a yellow lap uses, as a share of a green lap", 0.1, 0.8, rs.YELLOW_BURN, 0.05,
                            help="Not measurable from public data yet. Pocono 2026 suggests about a third or less.")
    save = st.slider("How much further a tank goes with hard fuel saving", 0.0, 0.2, rs.SAVE, 0.01,
                     format="%.2f", help="Michigan 2025 suggests about 8%, Pocono about 16%.")

t = get_params(track, None, yellow_burn, save)
races = data["races"][data["races"]["track"] == track]
latest = races.iloc[-1]
limited = "Fuel-limited" if t.fuel_limited else "Tire-limited"
st.html(s.chips([
    latest["track_name"], f"{latest['track_miles']:.3g}-mile oval", f"{t.laps} laps",
    "Stages end on laps " + ", ".join(map(str, t.stage_breaks)), (limited, "hot"),
    (f"{len(races)} races since 2022", "blue"),
]))
if t.fuel_limited:
    st.html(s.callout(f"<b>Fuel sets the stops here.</b> Green-flag stops come about "
                      f"{np.median(t.green_targets):.0f} laps into a run, right up against a tank of about "
                      f"{t.full_window:.0f} laps at full pace ({t.window:.0f} with hard saving)."))
else:
    note = (f"<b>Tires set the stops here, not fuel.</b> Cars pit about {np.median(t.green_targets):.0f} laps "
            f"into a run, long before the tank is empty. Nobody had to save fuel to run the longest stint "
            f"seen ({t.longest:.0f} laps of fuel), so that's treated as a full-pace tank, and it's only a lower "
            f"bound. Fuel decides a race here only when a late caution leaves a very long green run to the end.")
    if track in SHORT_TRACKS:
        note += (" <b>On a short track like this one, it almost never does:</b> stage breaks come long "
                 "before anyone runs low.")
    st.html(s.callout(note))

forecast_tab, replay_tab, window_tab, models_tab, about_tab = st.tabs(
    ["Race forecast", "Replay a race", "Fuel windows", "The models", "How it works"])

# ---------------------------------------------------------------- race forecast

with forecast_tab:
    fc = get_forecast(track, yellow_burn, save, engine)
    st.html(s.eyebrow(f"{N_SIMS:,} simulated races · {engine}"))
    st.subheader(f"The next race at {TRACK_LABELS[track]}")
    m = st.columns(4)
    m[0].metric("Fuel-mileage finish", f"{fc['p_fuel_finish']:.0%}",
                help="Share of simulated races where at least a third of the field has to save fuel, make a "
                     "late green-flag stop for fuel, or runs dry.")
    m[1].metric("Cautions", f"{fc['cautions']:.1f}", help="Including the stage breaks.")
    m[2].metric("Stops per car", f"{fc['stops_per_car']:.1f}",
                help=f"{fc['green_stops_per_car']:.1f} of them under green.")
    m[3].metric("Car runs dry", f"{fc['p_run_dry']:.1%}", help="Chance any one car runs out of fuel.")

    with panel("pits"):
        st.markdown("**When do cars pit?** The share of the field pitting on each lap. Stage breaks bring a "
                    "caution, and with it a wave of yellow-flag stops.")
        st.plotly_chart(pit_chart(t, fc), width="stretch", config={"displayModeBar": False})
        with st.expander("Show as a table"):
            p = fc["pit_by_lap"]
            p = p[p.sum(axis=1) > 0].reset_index()
            st.dataframe(p, hide_index=True, column_config={
                "lap": st.column_config.NumberColumn("Lap"),
                "green": pct("Pit under green", float(p["green"].max())),
                "yellow": pct("Pit under yellow", float(p["yellow"].max()))})

    with panel("check"):
        st.markdown("**How did the real races compare?** The same test run on each real race's final stints. "
                    "The parameters come from every race here, including the one being checked, so this is a "
                    "consistency check rather than a true forecast test.")
        chk = get_check(track, yellow_burn, save, engine)
        show = chk.assign(fuel_finish=chk["fuel_finish"].map({True: "Yes", False: "No"}),
                          p_fuel_finish=chk["p_fuel_finish"].round(3), share_caught=chk["share_caught"].round(2))
        st.dataframe(show.drop(columns=["race_id"]), hide_index=True, column_config={
            "season": st.column_config.NumberColumn("Season", format="%d"),
            "date": "Date",
            "p_fuel_finish": pct("Simulated chance of a fuel finish", 0.5),
            "fuel_finish": "Was it one?",
            "share_caught": pct("Field caught out", 1.0, help="Had to save, stopped late for fuel or ran dry."),
            "sim_cautions": st.column_config.NumberColumn("Cautions (sim)", format="%.1f"),
            "cautions": st.column_config.NumberColumn("Cautions (real)"),
            "sim_stops": st.column_config.NumberColumn("Stops per car (sim)", format="%.1f"),
            "stops": st.column_config.NumberColumn("Stops per car (real)", format="%.1f",
                                                   help="Cars that finished on the lead lap."),
        })

# ---------------------------------------------------------------- replay a race

with replay_tab:
    st.html(s.eyebrow("Pick a lap and look ahead"))
    st.subheader("Who has to stop, and who can stretch it?")
    # Labelled by date, not the official race name: those are sponsor names.
    labels = {r["race_id"]: f"{r['season']} · {pd.Timestamp(r['race_date']):%B %d} · {r['actual_laps']} laps"
              for _, r in races.iloc[::-1].iterrows()}
    r1, r2 = st.columns([2, 3])
    race_id = r1.selectbox("Race", list(labels), format_func=labels.get)
    race = races[races["race_id"] == race_id].iloc[0]
    lap = r2.slider("Replay from lap", 1, int(race["actual_laps"]) - 1, int(race["scheduled_laps"] * 0.7))
    rt = get_params(track, race_id, yellow_burn, save)
    table, caution = get_replay(track, race_id, lap, yellow_burn, save, engine)
    flag = rs.race_flags(data, race_id).get(lap)
    st.html(s.chips([f"Lap {lap} of {rt.laps}", f"{rt.laps - lap} to go",
                     ("Under caution", "hot") if flag == 2 else ("Green flag", "blue"),
                     f"Tank: {rt.full_window:.0f} laps full pace, {rt.window:.0f} saving"]))
    v = table["verdict"].value_counts()
    m = st.columns(3)
    m[0].metric("Make it at full pace", int(v.get("Makes it at full pace", 0)),
                help="Cars that can reach the finish on the fuel they have, without saving or more cautions.")
    m[1].metric("Make it if they save", int(v.get("Makes it if it saves", 0)))
    m[2].metric("Need another stop", int(v.get("Needs another stop", 0)),
                help="Even with saving, unless cautions bring the fuel need down.")

    with panel("replay"):
        st.markdown("**Every car on track at this lap.** Sorted by where it finished. Chances come from "
                    f"{N_SIMS:,} simulations of the rest of the race, with random cautions.")
        show = table.sort_values("finish_pos").copy()
        show["car"] = "#" + show["car"]
        chances = ["p_stop", "p_green_stop", "p_save", "p_dry"]
        show[chances] = show[chances].round(3)
        show["next_stop"] = np.where(show["p_stop"] >= 0.05,
                                     show["next_stop_p10"].round().astype("Int64").astype(str) + "–"
                                     + show["next_stop_p90"].round().astype("Int64").astype(str), "")
        show["real"] = np.where(show["actual_next_stop"].notna(),
                                "Lap " + show["actual_next_stop"].astype("Int64").astype(str) + " ("
                                + show["actual_flag"].fillna("") + ")", "No more stops")
        cols = {"car": "Car", "driver": "Driver", "stopped_on": "Last stop", "used": "Fuel used",
                "verdict": "Without a caution", "laps_short": "Laps short", "p_stop": "Stops again",
                "p_green_stop": "Stops under green", "p_save": "Has to save", "p_dry": "Runs dry",
                "next_stop": "Likely next stop", "real": "Really did", "finish_pos": "Finished"}
        st.dataframe(show[list(cols)].rename(columns=cols), hide_index=True, height=min(36 * len(show) + 40, 760),
                     column_config={
                         "Last stop": st.column_config.NumberColumn(format="Lap %d"),
                         "Fuel used": st.column_config.NumberColumn(format="%.1f laps",
                                                                    help="Yellow laps count as part of a lap."),
                         "Laps short": st.column_config.NumberColumn(format="%.1f",
                                                                     help="Laps of fuel missing even with saving."),
                         "Stops again": pct("Stops again"), "Stops under green": pct("Stops under green"),
                         "Has to save": pct("Has to save"), "Runs dry": pct("Runs dry", 0.2),
                         "Finished": st.column_config.NumberColumn(format="P%d"),
                     })

    if (table["p_stop"] >= 0.05).any():
        with panel("nextstop"):
            st.markdown("**When will they stop next?** The bar is the middle 80% of simulations; the diamond is "
                        "when the car really stopped.")
            st.plotly_chart(next_stop_chart(table, lap, rt.laps), width="stretch",
                            config={"displayModeBar": False})

# ---------------------------------------------------------------- fuel windows

with window_tab:
    st.html(s.eyebrow("The evidence"))
    st.subheader(f"How far a tank goes at {TRACK_LABELS[track]}")
    m = st.columns(4)
    m[0].metric("Tank at full pace", f"{t.full_window:.0f} laps")
    m[1].metric("Tank with saving", f"{t.window:.0f} laps")
    m[2].metric("Green-flag stops at", f"{np.median(t.green_targets):.0f} laps", help="Median.")
    m[3].metric("Stops set by", "Fuel" if t.fuel_limited else "Tires")
    with panel("window"):
        st.markdown("**Where real green-flag stops came.** Laps of fuel used, counting each yellow lap as "
                    f"{yellow_burn:.2f} of a green one. Stops bunched against the tank line mean fuel sets them; "
                    "stops well short of it mean tires do.")
        st.plotly_chart(window_chart(t), width="stretch", config={"displayModeBar": False})

    st_all = data["stints"].merge(data["cars"][["race_id", "car", "driver"]], on=["race_id", "car"])
    st_all = st_all[st_all["race_id"].isin(races["race_id"])]
    clean = st_all[st_all["end_how"].isin(["pit_green", "pit_yellow", "finish"])
                   & (st_all["laps_behind"] == 0) & (st_all["slow_green"] == 0)]
    clean = clean.assign(fuel=clean["green"] + yellow_burn * clean["yellow"]).nlargest(8, "fuel")
    with panel("longest"):
        st.markdown("**The longest clean stints.** On the lead lap, with no sign of a missed stop in the lap "
                    "times. A tank lasts at least this long.")
        how = {"pit_green": "Green-flag stop", "pit_yellow": "Yellow-flag stop", "finish": "Checkered flag"}
        st.dataframe(clean.assign(car="#" + clean["car"], end_how=clean["end_how"].map(how))[
            ["season", "car", "driver", "start", "end", "green", "yellow", "fuel", "end_how"]],
            hide_index=True, column_config={
                "season": st.column_config.NumberColumn("Season", format="%d"), "car": "Car", "driver": "Driver",
                "start": "From lap", "end": "To lap", "green": "Green laps", "yellow": "Yellow laps",
                "fuel": st.column_config.NumberColumn("Laps of fuel", format="%.1f"), "end_how": "Ended with"})

    notes = data["run_outs"][data["run_outs"]["race_id"].isin(races["race_id"])]
    with panel("dry"):
        st.markdown("**Cars that ran out.** From race control's notes. These set the upper end of the window.")
        if notes.empty:
            st.markdown("Nobody ran out of fuel here since 2022, so the tank figure is only a lower bound.")
        else:
            st.dataframe(notes.merge(races[["race_id", "season"]])[["season", "lap", "car", "note"]],
                         hide_index=True, column_config={
                             "season": st.column_config.NumberColumn("Season", format="%d"), "lap": "Lap",
                             "car": "Car", "note": st.column_config.TextColumn("Race control note", width="large")})

    with panel("alltracks"):
        st.markdown("**Every track side by side.**")
        rows = []
        for key, label in TRACK_LABELS.items():
            p = get_params(key, None, yellow_burn, save)
            rows.append({"Track": label, "Stops set by": "Fuel" if p.fuel_limited else "Tires",
                         "Green stops at": np.median(p.green_targets), "Tank, full pace": p.full_window,
                         "Tank, saving": p.window, "Green laps per caution": 1 / p.hazard})
        st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={
            c: st.column_config.NumberColumn(format="%.0f") for c in
            ("Green stops at", "Tank, full pace", "Tank, saving", "Green laps per caution")})

# ---------------------------------------------------------------- the models

with models_tab:
    st.html(s.eyebrow("XGBoost · walk-forward backtest · SHAP"))
    st.subheader("The models, and are they any good?")
    reports = get_reports()
    if reports is None or not have_models:
        st.info("The models haven't been trained on this computer yet. Run `python train_models.py`.")
    else:
        meta = get_models()[1]
        st.markdown(
            f"Three XGBoost classifiers answer the questions the simulation asks every lap. Each was tested "
            f"**walk-forward**: every race from {pd.Timestamp(meta['backtest_from']):%Y} on was predicted by a "
            f"model trained only on the races before it, then scored against the **simple average** the "
            "simulation used before (each track's caution rate, its usual green-stop lap, and its yellow-flag "
            "pit table). Scores are shown as they came out, wins and losses alike.")
        sc = model_scores(reports)
        cols = st.columns(3)
        for col, kind in zip(cols, MODEL_LABELS):
            k = sc[kind]
            verdict = "better" if k["skill"] > 0 else "worse"
            col.metric(MODEL_LABELS[kind], f"{abs(k['skill']):.0%} {verdict}",
                       delta=f"ranks events {k['auc_model']:.2f} vs {k['auc_base']:.2f}",
                       delta_color="normal" if k["auc_model"] >= k["auc_base"] else "inverse",
                       help=f"{MODEL_QUESTIONS[kind]} The headline is how much lower the model's log loss is "
                            "than the simple average's (log loss punishes confident wrong answers; lower is "
                            "better). The second line is the AUC: how often a real event is ranked above a "
                            "non-event (0.5 is a coin flip, 1.0 is perfect).")

        with panel("backtest"):
            st.markdown("**The scorecard.** Lower log loss and Brier score are better; higher AUC is better. "
                        "\"Off by per race\" is how far the expected count of events in a race was from the "
                        "real count, on average.")
            met = reports["backtest_metrics"].copy()
            met["model"] = met["model"].map(MODEL_LABELS)
            off = {(kind, w): v for kind in MODEL_LABELS
                   for w, v in (("XGBoost model", sc[kind]["mae_model"]), ("Simple average (step 2)", sc[kind]["mae_base"]))}
            met["off_per_race"] = [off[(k, w)] for k, w in zip(reports["backtest_metrics"]["model"], met["who"])]
            st.dataframe(met.drop(columns=["skill_vs_baseline"]), hide_index=True, column_config={
                "model": "Question", "who": "Method",
                "rows": st.column_config.NumberColumn("Predictions", format="localized"),
                "events": st.column_config.NumberColumn("Real events", format="localized"),
                "log_loss": st.column_config.NumberColumn("Log loss", format="%.4f"),
                "brier": st.column_config.NumberColumn("Brier score", format="%.4f"),
                "auc": st.column_config.NumberColumn("AUC", format="%.3f"),
                "off_per_race": st.column_config.NumberColumn("Off by per race", format="%.2f")})

        with panel("caution_curve"):
            st.markdown(f"**Caution risk lap by lap at {TRACK_LABELS[track]}.** The model's chance of a caution "
                        "on the next lap, if no random caution comes, next to the flat average it replaces.")
            st.plotly_chart(caution_curve_chart(t, get_tables(track, None)), width="stretch",
                            config={"displayModeBar": False})

        with panel("shap"):
            st.markdown("**What drives each model (SHAP).** SHAP splits every prediction into a push from each "
                        "feature. Bigger bars move the answer more.")
            kind = st.segmented_control("Model", list(MODEL_LABELS), format_func=MODEL_LABELS.get,
                                        default="caution", key="shap_model") or "caution"
            a, b = st.columns(2)
            a.plotly_chart(importance_chart(reports["shap_importance"], kind), width="stretch",
                           config={"displayModeBar": False})
            imp = reports["shap_importance"]
            order = imp[imp["model"] == kind].sort_values("mean_abs_shap", ascending=False)["feature"].tolist()
            feat = b.selectbox("How one feature pushes the answer", order, format_func=ml.NAMES.get,
                               key=f"shap_feat_{kind}")
            b.plotly_chart(dependence_chart(reports["shap_dependence"], kind, feat), width="stretch",
                           config={"displayModeBar": False})
            b.caption("Above zero raises the chance; below zero lowers it. Each point is an average over "
                      "real situations from the races.")

# ---------------------------------------------------------------- how it works

with about_tab:
    st.html(s.eyebrow("The honest version"))
    st.subheader("How it works")
    st.markdown(f"""
**Nobody publishes fuel burn.** Teams keep it secret. The fuel cell size is in the rulebook
(about 20 gallons), but how much goes in at each stop and how fast it burns are not. So this
app never talks about gallons. It works in **laps of fuel**, inferred from what cars actually did.

**1. Measure the tank.** Every car's race is cut into stints between pit stops, and every lap is
marked green or yellow. A stint a car finished without stopping proves the tank lasts at least
that long. Only clean stints count: on the lead lap, with no sign in the lap times of a stop the
pit data missed. Cars that ran out of fuel, found in race control's notes, show where the tank
really ends. At Michigan and Pocono, the two match.

**2. Measure the race.** From every race here since 2022: how often a caution comes out per green
lap, how long cautions last, how many laps into a run cars make green-flag stops, and how likely
cars on the lead lap were to pit under yellow, given how old their stint was and whether they
could make the finish without stopping.

**3. Simulate it, lap by lap, {N_SIMS:,} times.** Stage breaks bring a caution at their known
laps. Other cautions come at random. Green laps burn a lap of fuel; yellow laps burn
{yellow_burn:.2f} of one (an assumption you can change above). Cars pit under green when their
stint reaches the length real cars ran, unless they can reach the next stage break or the finish
on what's left. Each car's real range differs a little from what its team planned on, so cars
that stretch it sometimes run dry.

**A fuel-mileage finish** here means at least a third of the field has to save fuel, make a late
green-flag stop for fuel, or runs dry.

**4. Let XGBoost make the calls.** With the "XGBoost models" engine (the default), three
models replace those averages: one for the chance of a caution on each lap, one for the chance
a car pits under green, and one for the chance it pits under a caution. They see the lap, laps
to go, the stage, laps since the last restart and fuel used, so they can learn things a single
average can't, such as cautions bunching up after restarts. Each was backtested race by race
against the average it replaced; see "The models" tab.

**What it doesn't know yet:** how much less fuel a yellow lap burns, team-by-team differences,
track position, and tire wear beyond when cars usually stop.
""")
    st.html(s.eyebrow("Under the hood"))
    st.subheader("The tech stack")
    st.html(s.tech_badges())
    st.markdown(tech_story(stats))

st.html(s.FOOTER)
