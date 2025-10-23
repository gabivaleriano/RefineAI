import sys
import os

from samples_builder import SamplesBuilder  
from results_generator import ResultsGenerator  
from reports_generator import ReportsGenerator  
from threshold_selector import ThresholdSelector
from graphs_plotter import GraphsPlotter
from config import filtering_threshold, thresholds

class RunningExperiments:
    def __init__(self, name, metric, measure, data_path, show_plots=True, save_plots=True):
        self.name = name
        self.measure = measure
        self.metric = metric
        self.data_path = data_path
        self.show_plots = show_plots
        self.save_plots = save_plots

        self.train_file = os.path.join(data_path, f"{name}.csv")
        self.splits_dir = f"splits_{name}"
        self.results_dir = f"results_{name}"
        self.reports_dir = f"reports_{name}"
        self.graphs_dir = f"graphs_{name}"

        self.builder = SamplesBuilder(
            csv_path=self.train_file, 
            name=self.name, 
            splits_dir=self.splits_dir

        )

        self.results = ResultsGenerator(
            name=self.name, 
            splits_dir=self.splits_dir, 
            results_dir=f"results_{name}",
            measure=self.measure
        )

        self.reports = ReportsGenerator(
            name=self.name, 
            measure=self.measure, 
            results_dir=self.results_dir,
            reports_dir=self.reports_dir
        )

        self.plotter = GraphsPlotter(
            name=self.name,
            metric = self.metric,
            reports_dir=self.reports_dir,
            graphs_dir=self.graphs_dir,
            measure=self.measure,
            show_plots=self.show_plots,
            save_plots=self.save_plots
        )

        self.selector = ThresholdSelector(
            name=self.name, 
            measure=self.measure, 
            results_dir=self.results_dir, 
            splits_dir=self.splits_dir, 
            reports_dir=self.reports_dir
        )

    def build_samples(self):
        print("Building samples...")
        self.builder.split_samples()

    def generate_results(self):
        print("Generating results...")
        self.results.get_results()

    def generate_reports(self):
        print("Generating reports...")
        self.reports.average()

    def plot_performance_distribution(self):
        print("Plotting performance distribution...")
        self.plotter.binary_graphs()

    def search_thresholds_grid(self):
        print("Grid searching thresholds...")
        df_confidence = self.selector.min_cost(criteria = 'confidence', T = filtering_threshold)      
        df_uncertainty = self.selector.min_cost(criteria = 'uncertainty', T = filtering_threshold)
        #self.plotter.cost_vs_tvalue(df_confidence, criteria = 'confidence', strategy = 'grid')
        #self.plotter.cost_vs_tvalue(df_uncertainty, criteria = 'uncertainty', strategy = 'grid')

    def search_thresholds_heuristic(self):
        print("\nHeuristic searching thresholds...")
        df_confidence = self.selector.select_threshold(criteria = 'confidence')
        print("\nConfidence-based rejection")
        self.selector.print_min_cost_thresholds(df_confidence, criteria = 'confidence')
        self.plotter.cost_vs_tvalue(df_confidence, criteria = 'confidence', strategy = 'heuristic')
        df_uncertainty = self.selector.select_threshold(criteria = 'uncertainty')
        print("\nUncertainty-based rejection")
        self.selector.print_min_cost_thresholds(df_uncertainty, criteria = 'uncertainty')
        self.plotter.cost_vs_tvalue(df_uncertainty, criteria = 'uncertainty', strategy = 'heuristic')

    def search_thresholds_brute_force(self):
        print("\nBrute force searching thresholds...")
        brute_force_T = [round(1 - 0.01 * i, 2) for i in range(50)]
        df_confidence = self.selector.brute_force(brute_force_T = brute_force_T, criteria = 'confidence')
        df_uncertainty = self.selector.brute_force(brute_force_T = brute_force_T, criteria = 'uncertainty')    
        #self.plotter.cost_vs_tvalue(df_confidence, criteria = 'confidence', strategy = 'brute' )
        #self.plotter.cost_vs_tvalue(df_uncertainty, criteria = 'uncertainty', strategy = 'brute')

    def search_thresholds_annealing(self):
        print("\nSimulated annealing searching thresholds...")
        df_confidence = self.selector.simulated_annealing(criteria = 'confidence')
        df_uncertainty = self.selector.simulated_annealing(criteria = 'uncertainty')        
        #self.plotter.cost_vs_tvalue(df_confidence, criteria = 'confidence', strategy = 'sa')
        #self.plotter.cost_vs_tvalue(df_uncertainty, criteria = 'uncertainty', strategy = 'sa')