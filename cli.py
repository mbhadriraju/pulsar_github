import argparse
import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from Data.data_generator import generate_individual
from ModelTraining.model import train
from ModelTesting.run_model import ModelRunner
from ModelTesting.metrics import comparative_results, save_results_to_csv
from Data.preprocess import preprocess_individual, preprocess_all

def main():
    parser = argparse.ArgumentParser(description="Pulsar Denoising CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # preprocessindividual
    parser_preprocess_ind = subparsers.add_parser("preprocessindividual", help="Preprocess a single pulsar file")
    parser_preprocess_ind.add_argument("--path", type=str, help="Path to the file to preprocess")

    # preprocessall
    parser_preprocess_all = subparsers.add_parser("preprocessall", help="Preprocess all pulsar files in a directory")
    parser_preprocess_all.add_argument("--path", type=str, default=os.path.join(BASE_DIR, "Data", "testing_data"), help="Path to the root directory to preprocess")

    # createdataset
    parser_create = subparsers.add_parser("createdataset", help="Generate a dataset of simulated pulsars")
    parser_create.add_argument("-n", "--num-examples", type=int, default=10000, help="Number of examples to generate")
    parser_create.add_argument("--template-path", type=str, default=os.path.join(BASE_DIR, "Data", "template_data"), help="Path to template files")
    parser_create.add_argument("--output-path", type=str, default=os.path.join(BASE_DIR, "Data", "training_data", "sim_data.npy"), help="Output path for sim_data.npy")

    # trainmodel
    parser_train = subparsers.add_parser("trainmodel", help="Train the denoising model")
    parser_train.add_argument("--dataset-path", type=str, default=os.path.join(BASE_DIR, "Data", "training_data", "sim_data.npy"))
    parser_train.add_argument("--epochs", type=int, default=100)
    parser_train.add_argument("--batch-size", type=int, default=10)
    parser_train.add_argument("--lr", type=float, default=1e-4)
    parser_train.add_argument("--num-examples", type=int, default=10000)
    parser_train.add_argument("--save-path", type=str, default=os.path.join(BASE_DIR, "wideband_denoise.pth"))

    # runmodel
    parser_run = subparsers.add_parser("runmodel", help="Run the trained model on input data to denoise it. Please ensure the output data is copied to a new directory for each testing method to avoid overwriting results.")
    parser_run.add_argument("--model-path", type=str, default=os.path.join(BASE_DIR, "wideband_denoise.pth"), help="Path to the trained model")
    parser_run.add_argument("--input-path", type=str, required=True, help="Path to input data file to denoise")
    parser_run.add_argument("--output-path", type=str, required=True, help="Path to save denoised data file")

    # timeindividual
    parser_time_ind = subparsers.add_parser("timeindividual", help="Time a specific pulsar file")
    parser_time_ind.add_argument("path", type=str, help="Path to the file to time")

    # timeall
    parser_time_all = subparsers.add_parser("timeall", help="Time all files in the testing data directory")

    # getresults
    parser_results = subparsers.add_parser("getresults", help="Calculate metrics and output to a CSV")
    parser_results.add_argument("--output-csv", type=str, default=os.path.join(BASE_DIR, "results.csv"))

    args = parser.parse_args()

    if args.command == "preprocessindividual":
        print("Preprocessing individual file:", args.path)
        preprocess_individual(args.path)

    elif args.command == "preprocessall":
        print("Preprocessing all files in directory:", args.path)
        preprocess_all(args.path)

    elif args.command == "createdataset":
        print(f"Generating {args.num_examples} examples...")
        templates = [os.path.join(dp, f) for dp, dn, filenames in os.walk(args.template_path) for f in filenames if f.endswith('.fit') or f.endswith('.std')]
        if not templates:
            print("No templates found in", args.template_path)
            return
        
        all_data = []
        for i in range(args.num_examples):
            template = np.random.choice(templates)
            res = generate_individual(template, args.num_examples, args.output_path, i, template)
            all_data.append(res)
            if (i+1) % 100 == 0:
                print(f"Generated {i+1}/{args.num_examples}")
        
        final_data = np.zeros((2 * args.num_examples, all_data[0][0].shape[0], all_data[0][0].shape[1]))
        for i, res in enumerate(all_data):
            final_data[2*i] = res[0]
            final_data[2*i+1] = res[1]
        
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        np.save(args.output_path, final_data)
        print(f"Saved dataset to {args.output_path}")

    elif args.command == "trainmodel":
        if not os.path.exists(args.dataset_path):
            print(f"Dataset not found at {args.dataset_path}")
            return
        print(f"Training model on {args.dataset_path} for {args.epochs} epochs...")
        train(args.dataset_path, args.epochs, args.batch_size, args.lr, args.num_examples, args.save_path)
        print(f"Model saved to {args.save_path}")

    elif args.command == "runmodel":
        if not os.path.exists(args.input_path):
            print(f"Input file not found at {args.input_path}")
            return
        print(f"Running model {args.model_path} on {args.input_path}...")
        runner = ModelRunner(model_path=args.model_path)
        runner.run_model(args.input_path, args.output_path)
        print(f"Saved denoised data to {args.output_path}")

    elif args.command == "timeindividual":
        print(f"Timing individual file: {args.path}")
        runner = ModelRunner()
        runner.time_individual(args.path)

    elif args.command == "timeall":
        print("Timing all files...")
        runner = ModelRunner()
        runner.time_all()
        runner.calculate_snrs()

    elif args.command == "getresults":
        print("Calculating metrics...")
        results = comparative_results()
        if not results:
            print("No results to save.")
            return
        save_results_to_csv(results, args.output_csv)

if __name__ == "__main__":
    main()
