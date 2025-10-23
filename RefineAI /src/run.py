import os
import sys
import argparse

# Assume your custom modules are in src/, so adjust path if needed
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


from running_experiments import RunningExperiments

def main(data_path, train_name, measure, metric, steps, show_plots, save_plots):
    exp = RunningExperiments(
        name=train_name, 
        measure=measure, 
        metric = metric,
        data_path=data_path,
        show_plots=show_plots,
        save_plots=save_plots
    )

    if 'build' in steps:
        exp.build_samples()
    if 'results' in steps:
        exp.generate_results()
    if 'reports' in steps:
        exp.generate_reports()
    if 'plot' in steps:
        exp.plot_performance_distribution()
    if 'grid' in steps:
        exp.search_thresholds_grid()    
    if 'heuristic' in steps:
        exp.search_thresholds_heuristic()    
    if 'brute' in steps:
        exp.search_thresholds_brute_force()
    if 'annealing' in steps:
        exp.search_thresholds_annealing()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--train_name", type=str, required=True)
    parser.add_argument("--measure", type=str, choices=["ih", "if"], default="ih")
    parser.add_argument("--steps", nargs="+", default=["build", "results", "reports", "plot", "grid", "heuristic", "brute", "annealing"])
    parser.add_argument("--show_plots", action="store_true", help="Show plots in window")
    parser.add_argument("--no_save_plots", action="store_true", help="Don't save plots to file")
    args = parser.parse_args()

    main(
        data_path=args.data_path,
        train_name=args.train_name,
        measure=args.measure,
        steps=args.steps,
        show_plots=args.show_plots,
        save_plots=not args.no_save_plots
    )



