# CLL microCT clustering and statistical analysis

## Installation

Python 3.12 is recommended. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1` instead.

## Run the analysis

The default command reads `Supplementary Data 2.xlsx` from the repository root and writes generated files under `results/`:

```bash
python run_analysis.py
```

The full explicit command is:

```bash
python run_analysis.py \
  --input "Supplementary Data 2.xlsx" \
  --output-dir results \
  --seed 42 \
  --bootstrap-iterations 1000
```
