"""The app's look: colours, fonts, banner and footer.

Black, white and bold condensed capitals with yellow, red and blue racing stripes.
Theme colours for Streamlit's own widgets live in .streamlit/config.toml. The chart
colours below are checked for colour-blind safety by check_colours.py.
"""
import html

# Chart series, per theme (validated in both; see docs/app_style.md).
SERIES = {
    "light": {"green": "#2a78d6", "yellow": "#eda100"},
    "dark": {"green": "#3987e5", "yellow": "#c98500"},
}
SURFACE = {"light": "#ffffff", "dark": "#121212"}
INK = {"light": "#0b0b0b", "dark": "#f0efec"}
MUTED = "#898781"
GRID = {"light": "#e1e0d9", "dark": "#2c2c2a"}
EYEBROW = {"light": "#0062b1", "dark": "#5aa9f0"}   # small blue capitals above headings

LINKEDIN_URL = "https://www.linkedin.com/in/tony-honesto-4195023"
GITHUB_URL = "https://github.com/atonyhonesto"
TITLE = "Python & XGBoost Fuel Strategy Simulator"   # the keep-awake check looks for this exact text

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:ital,wght@0,600;0,700;0,800;1,800&display=swap');
:root { --ink: #0b0b0b; --yellow: #ffd400; --red: #e4002b; --blue: #0071ce; --eyebrow: __EYEBROW__;
        --race: 'Barlow Condensed', 'Arial Narrow', sans-serif;
        --stripe: linear-gradient(90deg, var(--yellow) 0 33.3%, var(--red) 33.3% 66.6%, var(--blue) 66.6%); }

/* Banner: black bar, condensed capitals, a three-colour stripe along the bottom */
.hero { position: relative; overflow: hidden; border-radius: 4px; padding: 18px 26px 26px; background: var(--ink); }
.hero::after { content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 6px; background: var(--stripe); }
.hero .eyebrow { color: var(--yellow); font-family: var(--race); font-weight: 700; font-size: 1rem;
                 letter-spacing: 1.5px; text-transform: uppercase; }
/* The title keeps its own case: the keep-awake check looks for the exact title text. */
.hero .name { font-family: var(--race); font-style: italic; font-weight: 800;
              line-height: .95; letter-spacing: .5px; font-size: clamp(2.1rem, 5vw, 3.6rem); color: #fff; }
.hero .name span { color: var(--yellow); }
.hero .by { margin-top: 10px; font-size: .95rem; color: #cfcfcf; }
.hero .by a { color: var(--yellow); font-weight: 600; text-decoration: none; }
.hero .by a:hover { text-decoration: underline; }

/* Headings: heavy condensed capitals, with a small blue label above where used */
[data-testid="stMain"] h2, [data-testid="stMain"] h3 { font-family: var(--race); font-weight: 800;
    text-transform: uppercase; letter-spacing: .4px; }
[data-testid="stMain"] h3 { font-size: 1.7rem; }
.eyebrow { color: var(--eyebrow); font-family: var(--race); font-weight: 700; font-size: .95rem;
           letter-spacing: 1.2px; text-transform: uppercase; margin-bottom: -10px; }

/* Tabs */
[data-testid="stTab"] p { font-family: var(--race); font-weight: 700; font-size: 1.15rem;
    text-transform: uppercase; letter-spacing: .8px; }
@media (max-width: 640px) { [data-testid="stTab"] p { font-size: .95rem; letter-spacing: .2px; } }

/* Panels around charts and tables: the three-colour stripe across the top */
[class*="st-key-panel_"] { position: relative; overflow: hidden; }
[class*="st-key-panel_"]::before { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 4px;
    background: var(--stripe); }

/* Stat tiles */
[data-testid="stMetric"] { border-left: 5px solid var(--red) !important; padding-left: 12px; }
[data-testid="stMetricValue"] { font-family: var(--race); font-weight: 800; }
[data-testid="stMetricLabel"] p { text-transform: uppercase; letter-spacing: .6px; font-size: .78rem; font-weight: 600; }

/* Info chips */
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 2px 0 8px; }
.chip { font-size: .85rem; padding: 3px 11px; border-radius: 999px; white-space: nowrap;
        border: 1px solid rgba(137,135,129,.55); }
.chip.hot { background: var(--ink); color: var(--yellow); border-color: var(--ink); font-weight: 600; }
.chip.blue { background: rgba(0,113,206,.12); border-color: rgba(0,113,206,.55); }

/* The hook under the banner, and the "Under the hood" button beside it */
.hook { font-size: 1.08rem; line-height: 1.45; padding: 4px 0 2px; }
.hook b { font-family: var(--race); font-weight: 800; font-size: 1.35rem; text-transform: uppercase;
          letter-spacing: .4px; margin-right: 4px; }
/* Black with yellow type in every state (Streamlit greys it out while the panel is open) */
button[data-testid="stPopoverButton"], button[data-testid="stPopoverButton"]:focus,
button[data-testid="stPopoverButton"]:active, button[data-testid="stPopoverButton"][aria-expanded="true"] {
    background: var(--ink) !important; color: var(--yellow) !important; border: 2px solid var(--ink) !important; }
button[data-testid="stPopoverButton"] p { font-family: var(--race); font-weight: 700; font-size: 1.05rem;
                                          text-transform: uppercase; letter-spacing: .8px; }
button[data-testid="stPopoverButton"]:hover, button[data-testid="stPopoverButton"][aria-expanded="true"] {
    border-color: var(--yellow) !important; }

/* Tech badges */
.tech { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 4px; }
.tech span { font-family: var(--race); font-weight: 700; letter-spacing: .5px; text-transform: uppercase;
             font-size: .95rem; padding: 2px 10px; border-radius: 3px; border: 1px solid rgba(137,135,129,.55); }
.tech span.next { border-style: dashed; opacity: .75; }

/* Track notes */
.callout { border-left: 5px solid var(--yellow); background: rgba(255,212,0,.12); padding: 10px 14px;
           border-radius: 3px; margin: 4px 0 12px; }

/* Footer */
.foot { margin-top: 36px; padding-top: 16px; font-size: .85rem; position: relative;
        display: flex; flex-wrap: wrap; gap: 10px 24px; justify-content: space-between; align-items: center; }
.foot::before { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 4px; background: var(--stripe); }
.foot a { color: var(--eyebrow); font-weight: 600; text-decoration: none; }
.foot .stack span { font-family: var(--race); font-weight: 700; letter-spacing: .5px; text-transform: uppercase;
                    padding: 2px 9px; margin-right: 6px; border-radius: 3px; border: 1px solid rgba(137,135,129,.45); }
.foot .note { opacity: .65; flex-basis: 100%; }
</style>
"""

HERO = f"""
<div class="hero">
  <div class="eyebrow">Cup Series &middot; lap by lap</div>
  <div class="name"><span>Python &amp; XGBoost</span> Fuel Strategy Simulator</div>
  <div class="by">Built by <b>Tony Honesto</b> &middot;
    <a href="{LINKEDIN_URL}" target="_blank">LinkedIn</a> &middot;
    <a href="{GITHUB_URL}" target="_blank">GitHub</a></div>
</div>
"""

FOOTER = f"""
<div class="foot">
  <div>Built by <b>Tony Honesto</b> &middot; <a href="{LINKEDIN_URL}" target="_blank">LinkedIn</a> &middot;
       <a href="{GITHUB_URL}" target="_blank">GitHub</a></div>
  <div class="stack"><span>Python</span><span>XGBoost</span><span>SHAP</span><span>NumPy</span><span>Monte Carlo</span><span>Streamlit</span></div>
  <div class="note">An independent fan project built on public race timing data. Not affiliated with any
       racing series, team or sponsor. Simulations are for fun, not betting advice.</div>
</div>
"""


def style_css(theme):
    return STYLE.replace("__EYEBROW__", EYEBROW[theme])


def eyebrow(text):
    return f'<div class="eyebrow">{html.escape(text)}</div>'


def chips(items):
    """Pill labels; each item is text or (text, kind) with kind 'hot' or 'blue'."""
    out = []
    for item in items:
        text, kind = item if isinstance(item, tuple) else (item, "")
        out.append(f'<span class="chip {kind}">{html.escape(text)}</span>')
    return '<div class="chips">' + "".join(out) + "</div>"


def callout(text):
    return f'<div class="callout">{text}</div>'


# The tech stack, in badge order; anything in TECH_NEXT shows as "next" with a dashed outline.
TECH = ["Python", "XGBoost", "SHAP", "Requests", "pandas", "NumPy", "Monte Carlo", "Plotly", "Streamlit",
        "Playwright"]
TECH_NEXT = []


def tech_badges():
    now = "".join(f"<span>{html.escape(t)}</span>" for t in TECH)
    soon = "".join(f'<span class="next" title="Coming next">{html.escape(t)} · next</span>' for t in TECH_NEXT)
    return f'<div class="tech">{now}{soon}</div>'


def style_fig(fig, theme, height=380, x_title=None, y_title=None):
    """Shared chart chrome: transparent background, recessive grid, legend across the top."""
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=36, b=10), hovermode="closest",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=INK[theme]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif')),
    )
    fig.update_xaxes(title=x_title, showgrid=False, linecolor=GRID[theme], color=MUTED)
    fig.update_yaxes(title=y_title, gridcolor=GRID[theme], zeroline=False, color=MUTED)
    return fig
