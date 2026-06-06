# DLPulsar

A CLI toolkit for training and evaluating a neural-network pulsar denoising model. It supports data preprocessing, synthetic dataset generation, model training and inference, pulsar timing, SNR analysis, and exporting comparative metrics to CSV.

## Directory Structure

```
Pulsar Denoising Code Replication/
├── cli.py                  # CLI entry point
├── setup.py                # Package install (registers `dlpulsar` command)
├── requirements.txt
├── Data/
│   ├── template_data/      # Profile templates used to generate training data (.fit, .std)
│   ├── training_data/      # Generated training datasets (.npy)
│   ├── testing_data/       # Real pulsar observation archives for evaluation
│   │   ├── input_data/     # Raw / baseline input copies
│   │   └── output_data/    # Denoised copies (one subfolder per method)
│   │       ├── output_data_fft/
│   │       ├── scrunch_data/
│   │       └── wavelet_data/
│   └── testing_toas/       # Timing outputs (.txt) and SNR results
│       ├── input_data/
│       └── output_data/
├── ModelTraining/
│   └── model.py            # Neural network architecture and training loop
└── ModelTesting/
    ├── run_model.py        # Inference, timing, and SNR utilities
    └── metrics.py          # Comparative metrics and CSV export
```

All data should live under `Data/`. The template, training, testing, and timing directories serve different stages of the pipeline (see **Data Layout** below).

## Prerequisites

**psrchive** is required and must be installed before running this project. It is not included in `requirements.txt`. Conda is the recommended install path:

```bash
conda install -c conda-forge psrchive
```

You will also need the psrchive command-line tools on your `PATH` (`pam`, `pat`, `psrstat`, `psrsmooth`, etc.).

## Installation

From the project root:

```bash
pip install -r requirements.txt
pip install -e .
```

This registers the `dlpulsar` CLI command.

## CLI Commands

| Command | Description |
|---|---|
| `preprocessindividual --path <file>` | Preprocess a single pulsar archive |
| `preprocessall --path <dir>` | Preprocess all archives under a directory (default: `Data/testing_data`) |
| `createdataset` | Generate a synthetic training dataset from templates |
| `trainmodel` | Train the denoising model |
| `runmodel` | Run the trained model on an input archive |
| `timeindividual <path>` | Time a single pulsar file |
| `timeall` | Time all files in the testing directories |
| `getresults` | Compute comparative metrics and write a CSV |

### Common Options

**`createdataset`**
- `-n, --num-examples` — number of examples (default: `10000`)
- `--template-path` — template directory (default: `Data/template_data`)
- `--output-path` — output `.npy` path (default: `Data/training_data/sim_data.npy`)

**`trainmodel`**
- `--dataset-path` — training data path (default: `Data/training_data/sim_data.npy`)
- `--epochs` — training epochs (default: `100`)
- `--batch-size` — batch size (default: `10`)
- `--lr` — learning rate (default: `1e-4`)
- `--save-path` — model output path (default: `wideband_denoise.pth`)

**`runmodel`**
- `--model-path` — trained model path (default: `wideband_denoise.pth`)
- `--input-path` — input archive (required)
- `--output-path` — output archive (required)

**`getresults`**
- `--output-csv` — results CSV path (default: `results.csv`)

## Workflow

### 1. Prepare data

Upload your testing archives and templates into `Data/`. **You must manually copy testing data into separate directories** for each denoising method (input, neural network, scrunch, wavelet thresholding). The CLI does not copy files for you.

Then preprocess both the testing data and templates:

```bash
dlpulsar preprocessall --path Data/testing_data
dlpulsar preprocessall --path Data/template_data
```

Preprocessing standardizes each archive's frequency and time dimensions before training or inference, keeping channel counts consistent across the pipeline.

### 2. Generate training data

```bash
dlpulsar createdataset
```

Output is written to `Data/training_data/`.

### 3. Train the model

```bash
dlpulsar trainmodel
```

A `.pth` checkpoint is saved (default: `wideband_denoise.pth`).

### 4. Run inference

```bash
dlpulsar runmodel --input-path <input> --output-path <output>
```

Denoised archives are written in place to the specified output path.

### 5. Time the data

Time all input and output archives:

```bash
dlpulsar timeall
```

**Important:** For timing to work, the **first 10 characters** of each pulsar observation filename must match the corresponding template name. For example, an archive named `J2129-5721_uwl_190412.ar` will be matched to a template named `J2129-5721.fit`, because both share the prefix `J2129-5721`. This prefix-based lookup is how `timeall` and `timeindividual` select the correct template for each file.

Timing results are written as `.txt` files under `Data/testing_toas/`. SNRs are computed automatically as part of this step.

### 6. Export results

```bash
dlpulsar getresults
```

Produces a comparative CSV (default: `results.csv`) across denoising methods.

## Data Layout

| Directory | Contents |
|---|---|
| `Data/template_data/` | Profile templates (`.fit`, `.std`) used by `createdataset` |
| `Data/training_data/` | Synthetic training arrays (`.npy`) |
| `Data/testing_data/` | Real pulsar observation archives, split across `input_data/` and `output_data/` subfolders |
| `Data/testing_toas/` | Timing outputs (`.txt`) and SNR results produced by `timeall` |
