import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, skew, kurtosis


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_path_names(input_dir=None, output_dir=None):
    input_files = []
    if input_dir is None:
        input_dir = os.path.join(BASE_DIR, "Data", "testing_data", "input_data")
    for dirname, _, filenames in os.walk(input_dir):
        for name in filenames:
            if name.endswith('.txt'): continue
            input_files.append(os.path.join(dirname, name))

    output_files = {}
    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, "Data", "testing_data", "output_data")
    for dirname, _, filenames in os.walk(output_dir):
        if dirname == output_dir: continue
        folder_name = os.path.basename(dirname)
        if folder_name not in output_files:
            output_files[folder_name] = []
        for name in filenames:
            if name.endswith('.txt'): continue
            output_files[folder_name].append(os.path.join(dirname, name))
    return input_files, output_files


def wrapped_phase_diff(phi1, phi2):
    return ((phi1 - phi2 + 0.5) % 1.0) - 0.5


def comparative_results(input_dir=None, output_dir=None, toas_dir=None):
    if toas_dir is None:
        toas_dir = os.path.join(BASE_DIR, "Data", "testing_toas")
        
    _, output_files = get_path_names(input_dir, output_dir)
    variants = list(output_files.keys())

    toas = {variant: [[], [], []] for variant in variants}

    input_data_txt = os.path.join(toas_dir, "input_data", "input_data.txt")
    if not os.path.exists(input_data_txt):
        print(f"File not found: {input_data_txt}")
        return {}
    with open(input_data_txt, "r") as f:
        x_lines = f.readlines()     

    snr_results_txt = os.path.join(toas_dir, "snr_results.txt")
    if os.path.exists(snr_results_txt):
        with open(snr_results_txt, "r") as f:
            lines = f.readlines()
            for i in range(0, len(lines), len(variants) + 1):
                if i + len(variants) >= len(lines): break
                base = lines[i].split()
                if len(base) < 2: continue
                base_snr = float(base[1])

                for j in range(1, len(variants) + 1):
                    method = lines[i + j].split()
                    if len(method) < 2: continue
                    snr = float(method[1])
                    toas[variants[j - 1]][2].append(float(np.log(snr / (base_snr + 1e-8))))

    results = {variant: [] for variant in variants}

    for key in toas.keys():
        variant_txt = os.path.join(toas_dir, "output_data", f"{key}.txt")
        if not os.path.exists(variant_txt):
            continue
        with open(variant_txt, "r") as f:
            var_lines = f.readlines()
        
        for i in range(min(len(x_lines), len(var_lines))):
            x_parts = x_lines[i].split()
            var_parts = var_lines[i].split()
            if len(x_parts) < 5 or len(var_parts) < 5: continue
            
            x_toa = np.float64(x_parts[3])
            x_unc = np.float64(x_parts[4])

            var_toa = np.float64(var_parts[3])
            var_unc = np.float64(var_parts[4])

            pct_change = wrapped_phase_diff(x_toa, var_toa)

            toas[key][0].append(pct_change)
            toas[key][1].append(np.log((var_unc / (x_unc)) + 1e-8))

        if not toas[key][0]: continue

        results[key].append(np.mean(toas[key][0]))
        results[key].append(np.median(toas[key][0]))
        results[key].append(np.std(toas[key][0]))

        results[key].append(np.mean(toas[key][1]))
        results[key].append(np.median(toas[key][1]))
        results[key].append(np.std(toas[key][1]))

        results[key].append(np.mean(np.abs(toas[key][0])))

        results[key].append(np.percentile(np.abs(toas[key][0]), 90))
        results[key].append(np.percentile(np.abs(toas[key][0]), 95))
        results[key].append(np.percentile(np.abs(toas[key][0]), 99))

        results[key].append(wilcoxon(toas[key][0])[1] if len(toas[key][0]) > 0 else np.nan)
        results[key].append(wilcoxon(toas[key][1])[1] if len(toas[key][1]) > 0 else np.nan)

        results[key].append(skew(toas[key][0]))
        results[key].append(kurtosis(toas[key][0]))

        if toas[key][2]:
            results[key].append(np.mean(toas[key][2]))
        else:
            results[key].append(np.nan)
    
    return results


def save_results_to_csv(results, output_csv):
    columns = [
        "mean_phase_diff", "median_phase_diff", "std_phase_diff", 
        "mean_unc_log_ratio", "median_unc_log_ratio", "std_unc_log_ratio", 
        "mean_abs_phase_diff", "p90_abs_phase_diff", "p95_abs_phase_diff", "p99_abs_phase_diff",
        "wilcoxon_phase_diff", "wilcoxon_unc", "skew_phase_diff", "kurtosis_phase_diff", "mean_snr_log_ratio"
    ]
    df = pd.DataFrame.from_dict(results, orient='index', columns=columns)
    df.to_csv(output_csv)
    print(f"Results saved to {output_csv}")

