# The app's look (`app_style.py`)

Everything about how the app looks lives here, so `app.py` can stick to what it shows.

## The style

- **Black, white and bold condensed capitals,** with a yellow, red and blue racing
  stripe.
- **The banner** is a black bar. Its title is "Python & XGBoost Fuel Strategy
  Simulator" in heavy slanted type, with "Python & XGBoost" in yellow and a small
  yellow label above it. The browser tab shows the same title (`TITLE`). The
  three-colour stripe runs along the bottom. The title is deliberately *not*
  uppercased with CSS: the keep-awake check looks for its exact text.
- **The hook and the "Under the hood" button.** The hook opens with large condensed
  capitals. The button is black with yellow type, and stays that way while its panel
  is open (Streamlit would otherwise grey it out).
- **Tech badges** are outlined condensed capitals. Ones still to come (XGBoost, SHAP)
  have a dashed outline and say "next".
- **Headings** are heavy condensed capitals (the Barlow Condensed font, loaded from
  Google Fonts). Small blue capital labels sit above them.
- **Panels** around charts and tables carry the three-colour stripe across the top.
- **Stat tiles** have a red bar down the left side.
- **Chips** are the pill labels under the track picker. The black one with yellow text
  marks whether fuel or tires set the stops.
- **Callouts** are the yellow-edged notes about a track.
- **The footer** has the same stripe, the builder's links, and a note that this is an
  independent fan project. It doesn't name or show any series, team, sponsor or logo.

Streamlit's own buttons, sliders and tabs take their colours from
`.streamlit/config.toml`: blue on light, yellow on dark.

## Light and dark

The app follows the viewer's theme. `app.py` asks Streamlit which theme is showing and
passes it in. The charts then use the right colours for that background, and the small
blue labels use a lighter blue on dark so they stay readable.

## Chart colours

| Use | Light | Dark |
|---|---|---|
| Green-flag stops, and other single-colour charts | `#2a78d6` blue | `#3987e5` blue |
| Yellow-flag stops | `#eda100` gold | `#c98500` gold |

These come from a validated colour-blind-safe palette and are re-checked with
`check_colours.py` (see `docs/check_colours.md`). The brighter stripe colours (yellow
`#ffd400`, red `#e4002b`, blue `#0071ce`) are only decoration. They never carry
meaning in a chart.

## The helpers

- `style_css(theme)`: the page CSS for the theme.
- `eyebrow(text)`: a small blue capital label.
- `chips(items)`: a row of pill labels.
- `callout(text)`: a yellow-edged note.
- `style_fig(fig, theme)`: the shared chart look. It has a see-through background,
  faint grid lines and the legend along the top.
