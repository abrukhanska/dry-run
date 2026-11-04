# Dry Run: Which Grape Is in the Glass?

A Machine Learning mini-project at the National University of Kyiv-Mohyla Academy.

**Team:** Alina Brukhanska, Milena Mashchenko and Viktoriia Boriak.

We classify English-language wine reviews into five grape varieties and examine how performance changes when explicit names and geographical references are masked. The project compares TF-IDF + logistic regression, TF-IDF + ComplementNB, and TF-IDF + TruncatedSVD + histogram gradient boosting on shared winery-grouped folds.

The main notebook is [dry_run.ipynb](dry_run.ipynb). It includes the analysis, saved outputs, conclusions and the team's AI use disclosure.

**Repository:** [abrukhanska/dry-run](https://github.com/abrukhanska/dry-run) · **[Open dry_run.ipynb in Colab](https://colab.research.google.com/github/abruhanska/dry-run/blob/main/dry_run.ipynb)**

## Team contributions

Sections **1–3 and 9–11 were prepared jointly**. These cover the research questions and hypotheses, setup and initial data inspection, the joint model comparison, shared-error and confidence analyses, and the final conclusions and limitations.

The primary responsibilities for the remaining sections were:

| Team member | Participant | Notebook sections | Main contribution |
|---|---|---|---|
| **Milena Mashchenko** | 1 | **4 and 8.1** | Data cleaning, duplicate and label-conflict checks, the frozen balanced sample and descriptive analysis; the TF-IDF + logistic regression experiment, DEV selection, grouped and ordinary-fold evaluation, coefficient interpretation and model-specific diagnostics. |
| **Viktoriia Boriak** | 2 | **6 and 8.2** | The L0–L3 masking rules and dictionaries, their implementation and DEV checks; the TF-IDF + ComplementNB experiment, DEV selection, MAIN evaluation, metadata-only reference experiments and model-specific diagnostics. |
| **Alina Brukhanska** | 3 | **5, 7 and 8.3** | The shared winery-grouped validation design, DEV/MAIN split and fold checks; common metrics, evaluation helpers and dummy baselines; the TF-IDF + SVD + gradient boosting experiment, component selection, frozen configuration records, verified reuse of saved results, and per-variety, error and computational-cost analyses. |

The shared validation assignments and evaluation procedure connect the three individual experiments. The final cross-model interpretation in Sections 9–11 is a joint team contribution. AI assistance is described separately at the end of the notebook.

## 1. Choose a way to run the project

| Goal | What to run | Original Kaggle CSV needed? | What is fitted? |
|---|---|---|---|
| Read the completed report | Open the notebook without executing it | No | Nothing |
| Reproduce the full data inspection, cleaning and model workflow | Run from the beginning with `DATA_STAGE = "raw"` | Yes | The first two models and reference experiments are fitted again. SVD results are reused after verification when the matching saved files are present. |
| Use the frozen sample without rebuilding it | Run from the beginning with `DATA_STAGE = "sample"` | No | The model sections behave as above; selecting sample mode does **not** disable training. |
| Rebuild the joint comparison and error analysis from saved predictions | In a fresh kernel, run the code cells in Sections 9–10 in order; read Section 11 | No | Nothing |
| Repeat SVD training deliberately | Follow Section 10 of this README in a separate project copy | Only if using raw mode | Selected MAIN fits, or both DEV search and MAIN fits, depending on the chosen option |

**Recommended for the full report:** raw mode, which is the supplied notebook's default. It displays the original-data inspection and cleaning results during execution. Sample mode is an alternative for working without the original CSV.

**Recommended for quickly checking the reported comparison:** the saved-results route in Section 8. It avoids all model fitting.

The project files must be available **before** the first notebook code cell is run. In particular, the environment check reads `results/svd_hgb/decision_before_main.json`.

## 2. Research question and experiment

How well can one of five grape varieties be identified from a tasting note, and what remains after masking the obvious clues?

The target labels, in the shared class order, are:

1. Chardonnay;
2. Riesling;
3. Pinot Noir;
4. Cabernet Sauvignon;
5. Syrah.

Only these exact dataset labels are retained. Alternative labels and separately labelled blends are excluded from the classification task. A retained dataset label does not establish that the wine contains only one grape variety.

### Text versions

| Version | Processing |
|---|---|
| L0 | Basic text normalization; no name masking |
| L1 | L0 with target-variety names, documented variants and broad short forms removed |
| L2 | L1 with names of other grape varieties removed |
| L3 | L2 with listed geographical names, origin adjectives and the corresponding record's full winery name removed |

L2 adds **other grape names**, not geographical masking. Geography is introduced at L3. Broad entries such as `Pinot` and `Cabernet` mean that L1 and L2 do not perfectly isolate target and non-target grape information.

The fixed dictionaries contain 20 target-variety terms, 263 other-grape terms and 1,623 geographical terms. The notebook documents their construction and DEV checks. The classifier receives processed descriptions, but L3 preprocessing also needs the record's winery metadata.

### Sample and validation

The original file contains 129,971 records. Selecting the five exact labels gives 43,828 records; description deduplication leaves 40,581. A fixed seed selects 3,000 reviews per variety, producing a balanced sample of **15,000 reviews** and **4,695 normalized winery groups**.

The frozen DEV/MAIN split contains:

| Partition | Reviews | Distinct winery groups | Use |
|---|---:|---:|---|
| DEV | 4,021 | 1,174 | Dictionary inspection and model selection |
| MAIN | 10,979 | 3,521 | Final grouped evaluation |

DEV and MAIN have disjoint normalized winery groups. DEV uses five grouped folds once. MAIN uses five grouped folds repeated three times, giving **15 fold scores per model and text version**. Each model uses the same assignments. TF-IDF and SVD are fitted within each training fold.

The primary metric is macro-F1. Accuracy, balanced accuracy and per-variety precision, recall and F1 are also reported. Fold-level sample SD describes score variation; it is not a confidence interval. First-repeat OOF files contain one prediction per MAIN review. Their pooled metrics need not equal the means over all 15 folds.

## 3. Files supplied with the project

Keep the relative paths below. The notebook defines its project root as `Path.cwd()`, so the **current working directory** must be this directory, even when the notebook is opened through Colab or an editor.

| Path | Contents |
|---|---|
| `dry_run.ipynb` | Main notebook, including AI use disclosure |
| `requirements.txt` | Recorded numerical/plotting package versions |
| `.gitignore` | Excludes local raw downloads, environments, backups and temporary archives |
| `data/wine_sample.csv` | Frozen balanced sample, including descriptions, IDs and retained metadata |
| `data/wine_sample.sha256` | SHA-256 of the sample CSV |
| `masking/target_varieties.txt` | Target-grape masking terms |
| `masking/other_grapes.txt` | Other-grape masking terms |
| `masking/geography.txt` | Geographical masking terms |
| `results/setup/` | Recorded versions and original-data inspection tables |
| `results/data_preparation/` | Cleaning flow, class counts and sample statistics |
| `results/validation/` | DEV/MAIN assignments, shared folds and their manifest |
| `results/masking_manifest.json` | Dictionary version and file hashes |
| `results/scores/` | Fold-level metrics, separated by partition and validation scheme |
| `results/oof/` | Saved out-of-fold predictions |
| `results/svd_hgb/` | SVD selection record, cache-verification receipts and diagnostics |
| `results/joint_comparison/` | Combined tables used in Sections 9–10 |
| `results/logreg_top_words_L0.csv`, `results/logreg_top_words_L3.csv` | Saved coefficient-ranked terms |
| `figures/` | Exported plots |
| `winemag-data-130k-v2.csv` | Optional local original CSV, downloaded separately; required for `raw` + `local` |

The supplied runtime export contains 143 files across `data/`, `masking/`, `results/` and `figures/`. The prepared project package adds this README, `requirements.txt`, `.gitignore` and the main notebook. It contains no serialized fitted classifiers and no original Kaggle CSV.

Do not flatten the directory structure or upload only the notebook. The sample, result tables, manifests and dictionaries are required to reproduce the supplied workflow. Original raw-data charts are also available as saved PNGs for reading outside execution.

## 4. Software environment

The recorded SVD experiment used **Python 3.13.18** and these package versions:

| Package | Version |
|---|---|
| NumPy | 2.1.3 |
| pandas | 2.2.3 |
| SciPy | 1.16.3 |
| scikit-learn | 1.6.1 |
| joblib | 1.6.0 |
| threadpoolctl | 3.7.0 |
| Matplotlib | 3.10.0 |

JupyterLab and ipykernel are additional tools for local notebook execution. `kagglehub` is optional and is needed only for `RAW_SOURCE = "kaggle"`.

### Exact Python version and the supplied SVD cache

The current notebook reconstructs the complete SVD decision record in Section 8.3.4 and compares it with the saved JSON. That comparison includes `platform.python_version()`. Consequently, **even a different Python patch version can stop the unchanged full notebook at this check**, despite matching numerical packages.

Installing packages with pip does not change the Python interpreter. For unchanged full-notebook replay with the supplied decision record, use the recorded Python version. If the available Colab runtime or local interpreter differs, the saved-results route in Section 8 avoids this training-environment check. Alternatively, a deliberate fresh SVD DEV+MAIN run can be recorded in a separate copy as described in Section 10.

Do not edit the recorded Python string or replace hashes merely to make verification pass. That would misrepresent the provenance of the saved experiment.

### What the first environment cell does

Before the main imports, Section 2.1 reads the package versions from the existing SVD decision file. It tests numerical-library imports in a separate Python process and reinstalls the recorded packages if versions differ or imports fail. If incompatible libraries are already loaded, it terminates the current Python process so that a clean kernel can be used.

Prepare dependencies and perform any necessary kernel restart **before** the main execution. A restart during `Run all` interrupts that run; it is not successful uninterrupted execution. After a restart, restore the working directory and run from the beginning again.

No GPU is needed. The SVD code uses one worker and limits numerical threads to one. Fresh SVD fitting is substantially slower than loading its saved results.

## 5. Google Colab: get the project files

Choose either GitHub cloning or ZIP upload. You do not need both.

### Option A: clone the repository from GitHub

1. Open [abruhanska/dry-run](https://github.com/abruhanska/dry-run). The clone URL is already filled in below.
2. Open the main notebook in Colab through **File → Open notebook → GitHub**, or upload the `.ipynb` through **File → Upload notebook**.
3. Connect to a CPU runtime.
4. Add a temporary setup cell **before Section 2.1** and run:

```python
from pathlib import Path
import os
import subprocess

repository_url = "https://github.com/abruhanska/dry-run.git"
project_dir = Path("/content/dry-run")

if not project_dir.exists():
    subprocess.run(
        ["git", "clone", repository_url, str(project_dir)],
        check=True,
    )

required_manifest = (
    project_dir / "results" / "svd_hgb" / "decision_before_main.json"
)
if not required_manifest.is_file():
    raise FileNotFoundError(
        "The selected repository folder does not contain the project files."
    )

os.chdir(project_dir)
print("Project root:", Path.cwd())
```

Opening an `.ipynb` from GitHub opens the notebook document; the associated data and results must also be made available to the runtime. The clone step supplies them.

The snippet reuses an existing clone; it does not pull updates or overwrite its files. For a private repository, use an authenticated download method or the ZIP route instead of embedding credentials in the notebook.

### Option B: upload a project ZIP

1. Download the repository using **Code → Download ZIP**, or use the supplied project package.
2. Open/upload the main notebook in Colab.
3. Run the following temporary setup cell before Section 2.1. Select the **project ZIP**, not the original dataset ZIP:

```python
from pathlib import Path
from zipfile import ZipFile
from google.colab import files
import os

uploaded = files.upload()
zip_names = [name for name in uploaded if name.lower().endswith(".zip")]
if len(zip_names) != 1:
    raise ValueError("Upload exactly one project ZIP.")

extract_dir = Path("/content/dry-run-files")
extract_dir.mkdir(parents=True, exist_ok=True)

with ZipFile(zip_names[0]) as archive:
    for name in archive.namelist():
        target = (extract_dir / name).resolve()
        if not target.is_relative_to(extract_dir.resolve()):
            raise ValueError("The archive contains an invalid path.")
    archive.extractall(extract_dir)

markers = list(
    extract_dir.rglob("results/svd_hgb/decision_before_main.json")
)
if len(markers) != 1:
    raise ValueError("Expected exactly one project root in the extracted ZIP.")

project_dir = markers[0].parents[2]
os.chdir(project_dir)
print("Project root:", Path.cwd())
```

Use this extraction route once in a fresh runtime. Extracting the same archive again can replace generated files with the archived versions. For subsequent work in the same session, simply return to the printed project directory.

### Install dependencies before running the study

After changing to the project root:

```python
%pip install -r requirements.txt
```

If a restart is required, restart the Python session while retaining the runtime's files. Afterwards, change back to the same project directory. For the GitHub route:

```python
%cd /content/dry-run
```

For the prepared project ZIP route:

```python
%cd /content/dry-run-files/dry-run
```

If a GitHub ZIP has a different containing directory, use the project-root path printed by the extraction cell. Check:

```python
from pathlib import Path
import platform

print("Python:", platform.python_version())
print("Working directory:", Path.cwd())
assert Path("results/svd_hgb/decision_before_main.json").is_file()
assert Path("data/wine_sample.csv").is_file()
```

Installing the recorded packages does not guarantee that the Colab Python version matches the frozen SVD record; see Section 4.

Colab runtime files are temporary. Download the updated notebook and any newly generated results before deleting the runtime. Saving the notebook to Drive alone does not preserve every file in its runtime.

## 6. Original dataset: download and configure it

### Source and attribution

The source is [Wine Reviews on Kaggle](https://www.kaggle.com/datasets/zynicide/wine-reviews), shared by Zack Thoutt (`zynicide`), containing reviews originally published by Wine Enthusiast. The notebook records the dataset license as **CC BY-NC-SA 4.0** and documents the cleaning and sampling changes. Retain that attribution and the dataset-license information when redistributing the derived sample; this does not assign a license to the project's code.

The notebook requests Kaggle dataset version **4** and uses **`winemag-data-130k-v2.csv`**. The `v2` in the filename and Kaggle's dataset version number refer to different things.

### Option A: manual download (default raw workflow)

1. Open the dataset page and select the recorded dataset version where available. Sign in if Kaggle requests it.
2. Download and extract the dataset archive.
3. Locate `winemag-data-130k-v2.csv`. Other files in the download are not used by this project.
4. Place this CSV directly in the project root, beside `README.md` and the notebook. Do not put it inside `data/` unless you also intentionally change the loader configuration.
5. Use these settings in Section 2.3:

```python
DATA_STAGE = "raw"
RAW_SOURCE = "local"
```

In Colab, upload the extracted CSV with the Files panel into the current project directory. If it appears in `/content` instead, move it to the printed project root. For the GitHub setup above:

```python
from pathlib import Path
import shutil

source = Path("/content/winemag-data-130k-v2.csv")
destination = Path.cwd() / source.name

if source.resolve() != destination.resolve():
    if destination.exists():
        raise FileExistsError("The destination CSV already exists; check it first.")
    shutil.move(str(source), str(destination))

print(destination)
```

### Option B: download through kagglehub

Install the optional downloader in the active environment:

```python
%pip install kagglehub
```

Then set:

```python
DATA_STAGE = "raw"
RAW_SOURCE = "kaggle"
KAGGLE_DATASET = "zynicide/wine-reviews"
KAGGLE_VERSION = 4
```

The notebook uses `kagglehub.dataset_download` with the versioned handle `zynicide/wine-reviews/versions/4`. It locates the CSV in the downloader's returned cache directory. The file does not need to be copied into the project root for this route.

If access requires authentication, follow Kaggle's current instructions. Do not commit credentials or token files. A download/authentication problem is not a reason to replace the recorded input checksum.

### Verify the exact raw file

The expected SHA-256 of the **extracted CSV**, not of the downloaded ZIP, is:

```text
52af2643c8ac29f010f0cc629dfbdda1c74aa0f332d11762af9ef3de4e567ac9
```

Section 3.1 checks this automatically. For a manually placed file, it can also be checked before execution:

```python
from pathlib import Path
import hashlib

raw_path = Path("winemag-data-130k-v2.csv")
digest = hashlib.sha256()
with raw_path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)

print(digest.hexdigest())
```

Use the unchanged downloaded CSV. Opening and resaving it in a spreadsheet application can alter its bytes and invalidate the checksum.

### What raw mode does to the frozen sample

Raw mode repeats exact-label filtering, required-field checks, description deduplication and seeded sampling. If the frozen sample CSV and checksum already exist, Section 4.8 compares the newly generated CSV bytes with the existing file. It **does not silently overwrite a different frozen sample**.

A mismatch can indicate different input data, settings, software behavior or preparation code. Investigate the cause; do not delete the reference hash just to continue. No mismatch is expected merely because raw mode is used with the supplied matching files.

## 7. Local execution on a laptop

Use a dedicated virtual environment and a CPU. Internet access is needed for initial downloads and package installation; local-data execution does not need to download the raw CSV repeatedly.

### Get the repository

Either download and extract the project ZIP, or copy the HTTPS clone URL from GitHub and run:

```bash
git clone https://github.com/abruhanska/dry-run.git
cd dry-run
```

If using the prepared ZIP, open a terminal in its `dry-run` directory instead. Check that `data/`, `masking/` and `results/` are directly inside the current directory.

### Windows / PowerShell

For unchanged full-notebook replay, first confirm that the selected interpreter is the recorded Python version:

```powershell
py -3.13 --version
```

The command must report **3.13.18** for the exact frozen SVD decision comparison. Selecting `-3.13` alone does not guarantee that patch version. Once the intended interpreter is installed:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install jupyterlab ipykernel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m ipykernel install --user --name dry-run --display-name "Dry Run"
.\.venv\Scripts\python.exe -m jupyterlab
```

These commands do not require PowerShell activation scripts. In JupyterLab, open `dry_run.ipynb` and select the **Dry Run** kernel.

### macOS / Linux

Select a Python installation matching the recorded version for unchanged full replay. Confirm its patch version, then run:

```bash
python3.13 --version
python3.13 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install jupyterlab ipykernel
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -m ipykernel install --user --name dry-run --display-name "Dry Run"
.venv/bin/python -m jupyterlab
```

For `RAW_SOURCE = "kaggle"`, also install `kagglehub` using the virtual environment's Python. Downloading the raw CSV manually does not require that package.

### Run the notebook

1. Start JupyterLab from the project root and open the main notebook.
2. Select the project's kernel.
3. Choose the desired settings in Section 2.3:
   - full original-data workflow: `DATA_STAGE = "raw"`, normally `RAW_SOURCE = "local"`;
   - frozen-sample workflow: `DATA_STAGE = "sample"`.
4. For raw/local mode, confirm the CSV is in the project root.
5. Run the environment cell. If it restarts/stops the kernel, reconnect and start again after setup is complete.
6. Execute the notebook from top to bottom using **Run All**. Do not recover a failed run by executing unrelated later cells out of order.
7. Save the executed notebook if you want its new outputs retained.

When using VS Code or another Jupyter front end, also check the actual kernel working directory. A correct notebook location does not by itself guarantee a correct `Path.cwd()`.

## 8. Sample mode and saved-results-only analysis

### Run the notebook with the frozen sample

Set in Section 2.3:

```python
DATA_STAGE = "sample"
```

`RAW_SOURCE` is ignored when selecting the input in this mode. The original Kaggle CSV is not required.

The frozen sample's expected SHA-256 is:

```text
2925cef15298f314f52449d2d743c7a8075ffd8af59abaa1c5c4583f4ed05633
```

The notebook verifies the sample, its IDs, five balanced classes, normalized winery groups and description uniqueness. It skips sample creation rather than sampling the already sampled data again.

**Expected differences from raw mode:**

- Original-file inspection in Section 3 is skipped, including raw-data charts and related-label tables. Those calculations require records outside the frozen sample.
- Preparation cells 4.2–4.9 skip rebuilding the sample and cleaning reports.
- The non-target-label overlap diagnostic in 4.11b is skipped because the non-target records are absent.
- Sample diagnostics 4.12a–4.12c still run. The cleaning-flow chart in 4.12a reads the bundled `results/data_preparation/class_summary.csv` and `cleaning_flow.csv`.
- Validation, masking, baselines and the model sections still execute normally. **Sample mode is a data-source choice, not a no-training switch.**

The original-data findings remain documented in the Markdown narrative and exported files. To recompute all of them, use raw mode. Viewing a saved raw-data PNG is different from recomputing it on the original records.

### Rebuild Sections 9–10 without any training

This is the shortest route to checking the combined results:

1. Make the complete project files available and change to the project root.
2. Use an environment with NumPy, pandas, Matplotlib, scikit-learn and IPython. The supplied requirements are the recorded numerical/plotting versions.
3. Start a fresh notebook kernel and, if necessary, run only a working-directory setup cell.
4. **Do not run Sections 1–8.** Begin with the first code cell under **9.1 Checking the saved results**.
5. Run every code cell in Sections 9 and 10 in order, then read the conclusions in Section 11.

Section 9 imports its own dependencies, loads the sample and saved experiment files, and reconstructs MAIN fold membership from the stored assignments. It does not need `main`, `dev` or model objects created by earlier sections. In a fresh kernel, `Path.cwd()` determines the project root.

This route verifies all 14 primary/reference experiment tables used in the joint comparison: 12 text experiments and two dummy baselines. It checks file alignment, fold identities and sizes, valid probabilities and first-repeat metrics reconstructed from predictions. It then regenerates comparison tables, confusion-change plots, shared-error counts and confidence diagnostics.

It does **not** rerun cleaning, DEV tuning, model fitting, the metadata-only experiments or coefficient extraction. It does not prove fresh-training reproducibility and does not independently reconstruct later-repeat predictions, which were not saved.

For this route, the exact Python-version equality check in Section 8.3.4 is not executed. Do not run Section 2.1 merely as a prerequisite for Section 9: it is not needed for the saved-results analysis once dependencies and paths are prepared.

## 9. Why the third model can finish without training

The third pipeline is TF-IDF → TruncatedSVD → HistGradientBoostingClassifier. It has two separate reuse mechanisms.

### DEV search

Section 8.3.3 checks `results/svd_hgb/decision_before_main.json` and the candidate score file. When the recorded setup, inputs and versions match, it loads `dev_candidate_fold_scores.csv`, reconstructs the summary and confirms the same selected components:

| Text version | Selected SVD components |
|---|---:|
| L0 | 100 |
| L1 | 200 |
| L2 | 200 |
| L3 | 200 |

A fresh search would require 4 text versions × 2 component counts × 5 DEV folds = **40 complete pipeline fits**. The archived candidate fits sum to approximately 15.2 minutes of fitting time in the recorded run, excluding additional overhead.

### MAIN evaluation

Section 8.3.5 processes L0–L3 separately. Each level has:

- `results/scores/MAIN/grouped/SVD_HGB__Lx.csv`;
- `results/oof/MAIN/grouped/SVD_HGB__Lx.csv`;
- `results/svd_hgb/main_receipt_Lx.json`.

Here, `Lx` means one of L0, L1, L2 or L3. A receipt records hashes of the decision, MAIN row order, processed text and two result files. The loader verifies these and checks the result contents before reusing them.

Expected messages include:

```text
Saved DEV results verified and loaded. No models were trained.
[L0] Saved results verified and loaded ...
L0: 15 fold scores; 10,979 first-repeat OOF predictions; verified saved results.
```

These messages mean that **previously executed experiments are being checked and reused**. They do not mean that the experiment originally involved no fitting. The archive stores scores and predictions, not trained estimators that can classify arbitrary new reviews.

If a level has no completion receipt, the MAIN function evaluates that level afresh and writes its results and receipt. An existing receipt with missing, changed or inconsistent files is an error, not permission to reuse them or to silently start a different experiment.

Fresh MAIN evaluation requires 4 × 15 = **60 complete fits**. The original recorded MAIN fits sum to about **41.7 minutes**, while the full original MAIN cell took approximately **45.12 minutes**, including prediction, checks and saving. Loading results can take seconds, but does not replace these historical timings. Hardware and numerical-thread differences affect runtime.

Some preprocessing audits still fit TF-IDF vectorizers even when complete SVD+boosting fits are reused. Thus, reuse does not mean that every earlier operation in Section 8.3 is skipped.

### What happens to the other models?

In the supplied notebook, LogisticRegression and ComplementNB do **not** have the same completed-run reuse mechanism. Their DEV searches and MAIN fits run again when their sections execute. The baselines, ordinary-fold comparison and metadata-only experiments also execute in a full top-to-bottom run. Logistic-regression interpretation fits are performed again for coefficient inspection.

Later comparison and diagnostic sections read the saved scores and predictions; they do not fit a new ensemble or recalibrate probabilities. Fresh fits can update recorded timings even when predictive scores remain unchanged. Keep an untouched project copy if preserving the exact supplied execution records matters.

## 10. Optional: deliberately repeat SVD fitting

Use a **separate extracted or cloned project copy**. Keep the supplied archive unchanged. No reset is necessary for ordinary viewing or verified reuse.

Choose one option:

- **MAIN only:** keep the recorded DEV choice and repeat the 60 MAIN fits. This still needs the environment to match the frozen decision, including its current Python-version check.
- **DEV and MAIN:** repeat the 40 DEV fits, record a new decision with the current environment, and then repeat the 60 MAIN fits. This is a new execution of the study, not reuse of the original SVD decision. Input files and the planned candidate/configuration choices remain unchanged.

In the separate copy:

1. Complete environment setup while the supplied decision file is still present; Section 2.1 needs it to read package versions.
2. Run the notebook in order through **8.3.2 Checking the SVD grid on DEV**.
3. Before running 8.3.3, execute the following temporary reset cell. Set `RESET_DEV_SEARCH = True` only for a fresh DEV+MAIN execution.

```python
from pathlib import Path
from datetime import datetime, timezone
import shutil

RESET_DEV_SEARCH = False  # False: repeat MAIN; True: repeat DEV and MAIN
root = Path(PROJECT_ROOT).resolve()
levels = ["L0", "L1", "L2", "L3"]

paths = [
    root / "results" / "svd_hgb" / f"main_receipt_{level}.json"
    for level in levels
]

partitions = ["MAIN"]
if RESET_DEV_SEARCH:
    partitions.append("DEV")
    paths += [
        root / "results" / "svd_hgb" / filename
        for filename in [
            "decision_before_main.json",
            "dev_candidate_fold_scores.csv",
            "dev_search_summary.csv",
        ]
    ]

for partition in partitions:
    for kind in ["scores", "oof"]:
        paths += [
            root / "results" / kind / partition / "grouped"
            / f"SVD_HGB__{level}.csv"
            for level in levels
        ]

stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
backup = root / "local_backups" / f"svd_before_refit_{stamp}"

for path in paths:
    if path.is_file():
        destination = backup / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(destination))

print("Original SVD records moved to:", backup)
print("Continue with Section 8.3.3, then run the remaining cells in order.")
```

4. Continue with 8.3.3, 8.3.4 and 8.3.5 in order, followed by the diagnostics and joint comparison.
5. Keep the new decision, score files, OOF files and receipts together. Regenerate downstream summaries after training.

The reset cell moves only the selected SVD records into a timestamped local backup. It does not alter the sample, validation assignments, masking files or the original download. Do not run it in your sole reference copy, commit `local_backups/`, or clear all of `results/` indiscriminately.

When repeating DEV+MAIN with another Python interpreter, the required numerical package versions must still be installed and supported by that interpreter. This option does not promise identical floating-point results across arbitrary platforms.

## 11. Main results in the supplied run

Mean MAIN macro-F1 over the 15 shared grouped folds:

| Pipeline | L0 | L1 | L2 | L3 |
|---|---:|---:|---:|---:|
| Logistic regression | 0.8549 | 0.7924 | 0.7900 | 0.7847 |
| ComplementNB | 0.8222 | 0.7552 | 0.7515 | 0.7450 |
| SVD + boosting | 0.8187 | 0.7481 | 0.7470 | 0.7409 |

The stratified dummy reaches approximately **0.1986** macro-F1; the most-frequent-class dummy reaches **0.0606**. Logistic regression has the highest mean macro-F1, accuracy and balanced accuracy at all four levels, and the highest mean per-variety F1. It does not lead in every individual precision or recall comparison.

The largest adjacent decrease for all pipelines is L0→L1. Total paired L3−L0 mean differences are −0.0702 for logistic regression, −0.0771 for ComplementNB and −0.0779 for SVD + boosting. SVD uses 100 components at L0 and 200 at the masked levels, so its comparisons involving L0 also change representation size.

After masking, Syrah and Cabernet Sauvignon remain difficult and are frequently confused. Among first-repeat L3 errors on true red varieties, another red variety is predicted in approximately 94.0% of cases for logistic regression, 85.6% for ComplementNB and 92.5% for SVD + boosting.

All three pipelines remain well above the text-free references at L3. This establishes surviving predictive information under the specified rules, not that predictions use sensory descriptions alone. Unlisted names, writing style and other indirect associations can remain.

## 12. Outputs, limitations and interpretation

The notebook covers the research questions, data inspection, cleaning, validation, masking, metrics/baselines, individual models, joint comparison, error analysis and conclusions. Its AI use disclosure appears after Section 11.

The main combined tables are:

- `results/joint_comparison/overall_summary.csv`;
- `results/joint_comparison/paired_model_differences.csv`;
- `results/joint_comparison/paired_masking_differences.csv`;
- `results/joint_comparison/per_variety_summary.csv`;
- `results/joint_comparison/fit_and_cost.csv`;
- `results/joint_comparison/directed_error_changes.csv`;
- `results/joint_comparison/number_of_wrong_pipelines.csv`;
- `results/joint_comparison/L3_confidence_summary.csv`;
- `results/joint_comparison/L3_selective_accuracy.csv`.

Results concern five exact dataset labels in a deliberately balanced sample. They do not estimate accuracy for all varieties or natural source class frequencies. Winery-name normalization can miss aliases, and winery separation does not separate authors, origins, vintages or writing styles. There is no independent future dataset.

The pipelines are compared as complete DEV-selected systems, including different n-gram representations and SVD sizes. The masking dictionaries can miss cues or remove ordinary words inside names. L3 changes geography and winery-name masking together, so their contributions are not separately identified.

Error and confidence diagnostics use first-repeat OOF predictions. High confidence is not a guarantee of correctness; confidence-selected subsets can have different class compositions. Shared successes do not establish the performance of an ensemble, which was not evaluated. Coefficients and selected mistakes do not prove a causal sensory mechanism. Recorded fit times are not a controlled hardware benchmark.

## 13. Troubleshooting

| Symptom | Explanation and next step |
|---|---|
| Missing `results/svd_hgb/decision_before_main.json` in the first cell | Only the notebook was opened, or the working directory is wrong. Clone/extract the complete project and change to its root before the environment cell. |
| Raw CSV not found | For raw/local mode, put the extracted `winemag-data-130k-v2.csv` in the project root. Alternatively use the configured Kaggle downloader or choose sample mode. |
| The CSV has the right name but a wrong checksum | Check the dataset version and file contents. Do not use another Wine Reviews CSV, hash the ZIP, resave the CSV or replace the reference hash. |
| Missing `data/wine_sample.sha256` or a sample mismatch | Restore the matching CSV/checksum pair from the supplied project. |
| Missing preparation CSVs in 4.12a | Restore `results/data_preparation/`; sample mode does not recreate the cleaning-history reports. |
| Raw charts or non-target-label checks are skipped | Expected in sample mode. Use raw mode to recompute them, or inspect the exported original-run figures separately. |
| `numpy: use the recorded version ...` | The active environment differs from the frozen setup. Install the recorded versions in the correct kernel and restart before continuing. |
| NumPy/scikit-learn import errors after package installation | Previously loaded modules may belong to the old installation. Restart the Python kernel; do not keep executing later cells in an inconsistent session. |
| Python disconnects during Section 2.1 | The environment cell deliberately clears an inconsistent kernel. Retain the files, reconnect, restore the working directory and begin again after setup. |
| SVD decision mismatch despite matching package versions | Check the exact Python version, inputs, settings and pipeline parameters. Section 8.3.4 compares the Python string as well. Use exact-environment replay, saved-results analysis, or a deliberate fresh SVD run in a separate copy. |
| SVD loads results quickly and prints “No models were trained” | Expected when the recorded cache verifies. See Section 9 of this README. |
| SVD has a receipt but a result file is missing/changed | Restore matching archived files. A receipt mismatch is an integrity failure; do not replace hashes to bypass it. |
| `NameError` after starting midway | Earlier cells define shared state. Run in order, or use the explicitly self-contained Section 9 saved-results entry point. |
| Kaggle authentication or download fails | Use the manual download route or follow Kaggle's current authentication instructions. Raw data is unnecessary for the sample/results-only routes. |
| Timings differ after rerunning models | Fit time depends on the run environment. New timings are expected; compare predictive metrics and input integrity separately. |

## 14. Saving work and sharing the repository

Keep the main notebook, this README, `requirements.txt`, the frozen sample and checksum, all three dictionaries, result files and figures together. Downloading only the `.ipynb` from Colab does not download its runtime directories.

The original Kaggle CSV is obtained separately. Keep local raw downloads, credentials, virtual environments, Python caches, `.ipynb_checkpoints/`, local backups and temporary export ZIPs out of the repository. Retain the dataset attribution and license information for the derived sample.

A Colab runtime's disk files and installed packages are not preserved merely by saving the notebook. Before deleting a runtime, save the current `.ipynb` and export any result files you want to retain. Copying the prepared project to another machine preserves the recorded results; it does not copy an active Python session.

The supplied hashes and receipts provide consistency checks. The presence of saved outputs does not by itself prove a fresh, uninterrupted end-to-end execution. Claims about reproduction should specify whether they refer to saved-result analysis, replay with reused SVD results, or complete refitting.

## 15. Reference documentation

- [Wine Reviews dataset](https://www.kaggle.com/datasets/zynicide/wine-reviews)


