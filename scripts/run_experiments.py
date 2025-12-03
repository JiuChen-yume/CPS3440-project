import argparse
import os
import subprocess
import sys
import json


def run_cmd(cmd_list):
    print("$", " ".join(cmd_list))
    proc = subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd_list)}")


def main():
    parser = argparse.ArgumentParser(description="Run baselines and models for experiments.")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--run_baselines', action='store_true')
    parser.add_argument('--run_mlp', action='store_true')
    parser.add_argument('--run_gnn', action='store_true')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--mlp_features', nargs='*', default=['coords+diff'], choices=['coords', 'coords+diff'], help='Feature sets to train for MLP')
    parser.add_argument('--gen_plots', action='store_true', help='Generate figures after evaluation')
    args = parser.parse_args()

    # Train MLP for each requested feature set
    if args.run_mlp:
        for feat in args.mlp_features:
            run_cmd([sys.executable, '-m', 'src.training.train_mlp', '--data_dir', args.data_dir,
                     '--epochs', str(args.epochs), '--hidden_dim', str(args.hidden_dim), '--lr', str(args.lr),
                     '--device', args.device, '--features', feat])

    # Train GNN (if environment supports Torch; train_gnn itself may skip/handle)
    if args.run_gnn:
        run_cmd([sys.executable, '-m', 'src.training.train_gnn', '--data_dir', args.data_dir,
                 '--epochs', str(args.epochs), '--device', args.device])

    # Evaluate baselines/MLP/GNN
    if args.run_baselines or args.run_mlp or args.run_gnn:
        run_cmd([sys.executable, '-m', 'src.evaluation.evaluate', '--data_dir', args.data_dir, '--device', args.device])
        summary_path = os.path.join(args.data_dir, 'artifacts', 'evaluation_summary.json')
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                print("Evaluation Summary:")
                print(json.dumps(json.load(f), indent=2))

    # Generate plots
    if args.gen_plots:
        run_cmd([sys.executable, '-m', 'scripts.plot_results', '--data_dir', args.data_dir])


if __name__ == '__main__':
    main()
