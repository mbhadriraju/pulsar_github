import psrchive
import numpy as np
import torch.nn as nn
import torch
from Data.preprocess import load_data, replace_data
import subprocess
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class ModelRunner:
    def __init__(self, model_path=None):
        self.model_path = model_path

    def run_model(self, input_path, output_path):
        input_data, _ = load_data(input_path)

        model = torch.load(self.model_path)
        model.eval()
        with torch.no_grad():
            output_data = model(torch.tensor(input_data).float().unsqueeze(0).unsqueeze(0)).squeeze().numpy()

        replace_data(output_path, output_data)


    def run_wt(self, input_path, output_path):
        subprocess.run([
            'psrsmooth',
            f"{output_path}"
        ])


    def run_scrunch(self, input_path, output_path):
        input_data, _ = load_data(input_path)
        scrunched = input_data.mean(axis=0)
        
        replace_data(output_path, np.tile(scrunched, (input_data.shape[0], 1)))

    def get_timing_paths(self):
        input_files = []
        in_dir = os.path.join(BASE_DIR, "Data", "testing_data", "input_data")
        for dirname, _, filenames in os.walk(in_dir):
            for name in filenames:
                if name.endswith('.txt'): continue
                input_files.append(os.path.join(dirname, name))

        output_files = {}
        out_dir = os.path.join(BASE_DIR, "Data", "testing_data", "output_data")
        for dirname, _, filenames in os.walk(out_dir):
            if dirname == out_dir: continue
            folder_name = os.path.basename(dirname)
            if folder_name not in output_files:
                output_files[folder_name] = []
            for name in filenames:
                if name.endswith('.txt'): continue
                output_files[folder_name].append(os.path.join(dirname, name))

        return input_files, output_files
    
    def get_template_paths(self):
        template_paths = {}
        templ_dir = os.path.join(BASE_DIR, "Data", "template_data")
        for dirname, _, filenames in os.walk(templ_dir):
            for name in filenames:
                if name.endswith('.fit') or name.endswith('.std'):
                    template_paths[name.replace(".fit", "").replace(".std", "")] = os.path.join(dirname, name)

        return template_paths

    def time_all(self):
        template_paths = self.get_template_paths()
        input_files, output_files = self.get_timing_paths()

        input_data_txt = os.path.join(BASE_DIR, 'Data', 'testing_toas', 'input_data', 'input_data.txt')
        for file in input_files:
            fname = os.path.basename(file)[:10]
            if fname in template_paths:
                subprocess.run([
                    "pat", "-TP", "-A", "FDM", "-s", template_paths[fname], "-R", file
                ], stdout=open(input_data_txt, "w"), stderr=subprocess.PIPE)

        for dir_name, values in output_files.items():
            out_txt = os.path.join(BASE_DIR, 'Data', 'testing_toas', 'output_data', f'{dir_name}.txt')
            for file in values:
                fname = os.path.basename(file)[:10]
                if fname in template_paths:
                    subprocess.run([
                        "pat", "-TP", "-A", "FDM", "-s", template_paths[fname], "-R", file
                    ], stdout=open(out_txt, "w"), stderr=subprocess.PIPE)

    def time_individual(self, path):
        template_paths = self.get_template_paths()
        fname = os.path.basename(path)[:10]
        if fname in template_paths:
            subprocess.run([
                "pat", "-TP", "-A", "FDM", "-s", template_paths[fname], "-R", path
            ])
        else:
            print(f"Template not found for {fname}")

    def calculate_snrs(self):
        input_paths, output_paths = self.get_timing_paths()
        variants = list(output_paths.keys())
        snr_file = os.path.join(BASE_DIR, "Data", "testing_toas", "snr_results.txt")

        with open(snr_file, "w") as f:
            for file in input_paths:
                input_result = subprocess.run(
                    f"psrstat -c snr -c weff {file}",
                    capture_output=True, text=True, shell=True
                )
                input_snr = None
                if input_result.stdout:
                    out = input_result.stdout.splitlines()[0].split()
                    if len(out) > 1 and "snr=" in out[1]:
                        input_snr = out[1].replace("snr=", "")

                f.write(f"{file} {input_snr}\n")

                for v in variants:
                    repl_file = file.replace(
                        "input_data",
                        f"output_data/{v}"
                    )

                    if os.path.exists(repl_file):
                        result = subprocess.run(
                            f"psrstat -c snr -c weff {repl_file}",
                            capture_output=True, text=True, shell=True
                        )

                        if result.stdout:
                            output = result.stdout.splitlines()[0].split()
                            if len(output) > 1 and "snr=" in output[1]:
                                snr_val = output[1].replace("snr=", "")

                    f.write(f"{file} {snr_val}\n")
