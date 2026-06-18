# Optocoupler Thermal-Cycling Reliability V5

This repository contains the data, analysis code, tables, and figures for the V5 optocoupler thermal-cycling reliability analysis.

The only executable analysis script is:

`code/optocoupler_thermal_reliability_study_pubrev_v5.py`

Run it from the package root with:

```bash
python code/optocoupler_thermal_reliability_study_pubrev_v5.py
```

The script reads `data/optocoupler_ttf_unit_level.csv`, performs the censored reliability analysis, generates all tables and figures, writes the figure manifest/contact sheet/QA report, and creates a clean GitHub package.

Included outputs:

- `data/`: raw and canonical unit-level data.
- `tables/`: generated result tables.
- `figures/all/`: source figure PNGs using the analysis figure names.
- `figures/paper/`: paper-ready figure PNGs using the figure-slot names.
- `figures/figure_manifest.csv` and `figures/contact_sheet.png`: figure index and visual QA sheet.
- `qa/`: figure QA report.

Interpretation note: 15 deg C and 200 deg C results are model-conditional extrapolations, not independently validated test conditions.
