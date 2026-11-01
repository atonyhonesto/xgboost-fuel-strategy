# Checking chart colours (`check_colours.py`)

About one man in twelve has some colour blindness. Two colours that look clearly
different to most people can look the same to them. This script measures, rather
than guesses, whether the chart colours stay distinguishable.

```powershell
.venv\Scripts\python check_colours.py
```
With no arguments, it checks the app's own chart colours (from `app_style.py`) on the
app's light and dark backgrounds. To check any other set:
```powershell
.venv\Scripts\python check_colours.py "#2a78d6,#eda100" light "#ffffff"
```

## The checks

It's a Python copy of a standard data-viz validator (written in JavaScript; this PC
has no Node.js to run it). It gives the same numbers: for example, the documented
13.7 for yellow next to orange.

- **Lightness band:** each colour is neither too pale nor too dark for the background.
- **Chroma floor:** each colour is vivid enough not to read as grey.
- **Colour-blind separation:** how far apart neighbouring colours stay when red-green
  colour blindness is simulated. 8 or more passes; 6 to 8 needs labels as well;
  under 6 fails.
- **Normal vision:** neighbouring colours must also be clearly different to everyone
  (15 or more).
- **Contrast:** each colour against the background. Under 3:1 is allowed only if
  the chart has labels or a table view.

## The app's result

Blue for green-flag stops and gold for yellow-flag stops pass in both modes, by a wide
margin (31.5 light, 27.4 dark). Gold is under 3:1 on the white background, so the
pit chart has a legend, hover labels and a "Show as a table" view.
