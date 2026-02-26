# Global Pay Auditor

Streamlit-based compensation diagnostics dashboard with validation, peer benchmarking, pay-gap analysis, outlier detection, and export support.

## Files
- `app_streamlit.py` — main Streamlit app
- `validation.py` — validation and cleaning
- `peer_groups.py` — peer grouping logic
- `modeling_gap.py` — adjusted pay gap model
- `modeling_outliers.py` — outlier detection
- `reporting.py` — reporting helpers
- `exports.py` — export helpers
- `requirements.txt` — Python dependencies
- `sample_template.csv` — sample input format

## Deploy locally
```bash
pip install -r requirements.txt
streamlit run app_streamlit.py
```

## Deploy on Streamlit Cloud
- Main file path: `app_streamlit.py`
- Python version: use default supported version
