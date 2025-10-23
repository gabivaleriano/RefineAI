import math
import random

import pandas as pd
import numpy as np

from config import filtering_threshold, thresholds

from results_generator import ResultsGenerator
from reports_generator import ReportsGenerator

class ThresholdSelector:
    def __init__(self, name, measure, w_performance=4, w_rejection=1, w_confidence=1, 
                 results_dir="results_dir", splits_dir="splits_dir",
                 reports_dir="reports_dir", metric='macro_f1', confidence_type='general', T=None, 
                 min_conf=0, max_rej=1.0):
        """
        Initialize ThresholdSelector with experiment configuration and parameters.

        Parameters:
        - name (str): Experiment or dataset name.
        - measure (str): Instance hardness or other measure to identify hard instances.
        - w_performance (float): Weight of performance metric in cost function.
        - w_rejection (float): Weight of rejection rate in cost function.
        - w_confidence (float): Weight of confidence in cost function.
        - results_dir (str): Directory where intermediate results are stored.
        - splits_dir (str): Directory where data splits are stored.
        - reports_dir (str): Directory where reports are stored.
        - metric (str): Performance metric to optimize (e.g., 'macro_f1').
        - confidence_type (str): Confidence metric type ('general', 'class_0', 'class_1').
        - T (list or None): List of filtering thresholds T to evaluate.
        - min_conf (float): Minimum allowed confidence to consider a threshold.
        - max_rej (float): Maximum allowed rejection rate to consider a threshold.

        If T is None, default to some global or predefined list of filtering thresholds.
        """
        # Use a default T list if not provided
        if T is None:
            self.T = filtering_threshold.copy()
        else:
            self.T = T.copy() if isinstance(T, list) else [T]
        
        # Load average results from a CSV report (must be generated before)
        #self.average = pd.read_csv(f'{reports_dir}/classification_average_{name}_{measure}.csv')

        self.reports_dir = reports_dir
        self.name = name

        # Weights for cost calculation
        self.wp = w_performance
        self.wr = w_rejection
        self.wc = w_confidence

        self.metric = metric
        self.measure = measure
        self.confidence_type = confidence_type

        self.splits_dir = splits_dir
        self.results_dir = results_dir
        self.max_rej = max_rej
        self.min_conf = min_conf

        # Map metric codes to human-readable names (expand as needed)
        self.metric_names = {
            "precision_class_0": "Precision Negative Class",
            "precision_class_1": "Precision Positive Class",
            "recall_class_0": "Recall Negative Class",
            "recall_class_1": "Recall Positive Class",
            "macro_f1": "Macro F1",
        }

    def measure_cost(self, criteria, T = filtering_threshold.copy()):
        """
        Compute the cost metric combining performance, rejection rate, and confidence
        for all combinations of filtering threshold T and classifier confidence thresholds.

        Returns:
        - DataFrame: The `average` DataFrame updated with a new 'cost' column.
        """

        if not isinstance(T, (list, tuple)):  
            T = [T]

        data = pd.read_csv(f'{self.reports_dir}/classification_average_{self.name}_{self.measure}_{criteria}.csv')
        metric_col = f"{self.metric}_mean"

        # Normalize support to rejection rate: higher support means less rejection
        max_support = data['total_support_mean'].max()
        data['rate_rejected_mean'] = 1 - (data['total_support_mean'] / max_support)

        # Initialize cost column
        data[f"cost_{criteria}"] = np.nan

        # Determine which confidence column to use based on confidence_type
        confidence_column = 'mean_conf_uncert_mean'
        if self.confidence_type == 'class_0':
            confidence_column = 'mean_conf_uncert_class_0_mean'
        elif self.confidence_type == 'class_1':
            confidence_column = 'mean_conf_uncert_class_1_mean'

        # Calculate cost for each combination of T and threshold
        for t_value in T:
            for threshold in thresholds:
                mask = (data['thresholds'] == threshold) & (data['T_value'] == t_value)

                if not mask.any():
                    print(f"No data found for threshold={threshold} and T_value={t_value}")
                    continue  # Skip to the next iteration if no data matches
                subset = data.loc[mask]
                metric_value = subset[metric_col].values[0]
                rejection_rate = subset['rate_rejected_mean'].values[0]
                mean_confidence = subset[confidence_column].values[0]

                # Apply constraints for minimum confidence and maximum rejection
                if (rejection_rate < self.max_rej) and (mean_confidence > self.min_conf):
                    cost = (self.wp * (1 - metric_value) +
                            self.wr * rejection_rate +
                            self.wc * (1 - mean_confidence))
                    data.loc[mask, f"cost_{criteria}"] = cost

        return data

    def min_cost(self, criteria, T = filtering_threshold.copy()):
        """
        For each filtering threshold T, select the threshold value that minimizes the cost.

        Returns:
        - DataFrame with columns ['cost', 'threshold', 'T_value'] representing minimal costs per T.
        """
        if not isinstance(T, (list, tuple)):  
            T = [T]
                    
        data = self.measure_cost(criteria, T)
        mini = []

        # For each T, find threshold with minimum cost
        for t in T:            
            subset = data[data['T_value'] == t]
            if subset.empty:
                continue
            min_index = subset[f"cost_{criteria}"].idxmin()
            mini.append([
                subset.loc[min_index, f"cost_{criteria}"],
                subset.loc[min_index, 'thresholds'],
                subset.loc[min_index, 'T_value']
            ])
        mini = pd.DataFrame(mini, columns=[f"cost_{criteria}", 'threshold', 'T_value'])

        return mini


    def explore_interval(self, criteria, delta, epsilon, mini, min_T_cost, min_T_index, min_T, 
                             new_min_T_cost, min_neighbour_T_value):
        """
        Recursively explore filtering thresholds T within an interval to find
        better T values by bisecting intervals where improvements are possible.

        Parameters:
        - delta (float): Minimum improvement in cost to continue exploring.
        - epsilon (float): Minimum interval size to stop exploration.
        - mini (DataFrame): DataFrame with minimal costs per T.
        - min_T_cost (float): Current minimal cost at min_T.
        - min_T_index (int): Index of current minimal T in mini.
        - min_T (float): Current minimal T value.
        - new_min_T_cost (float): Cost at new candidate T.
        - min_neighbour_T_value (float): Neighboring T value for interval exploration.

        Returns:
        - Updated mini DataFrame including explored points.
        """

        condition = True
        while condition == True:

            new_T = round((min_T + min_neighbour_T_value) / 2, 5)
            #print(f"T={new_T}, previous_cost={new_min_T_cost}")

            condition = (abs(min_T_cost - new_min_T_cost) >= delta and
                    abs(min_T - new_T) >= epsilon)
            
            #print('delta ' + str(abs(min_T_cost - new_min_T_cost)))
            #print('epsilon ' + str(abs(min_T - new_T)))
            #print(condition)
            
            # Use previous classes to generate new results and reports for this T
            results = ResultsGenerator(name=self.name, T=new_T, measure=self.measure, splits_dir=self.splits_dir, results_dir=self.results_dir)
            results.get_results()
            
            reports = ReportsGenerator(name=self.name, T=new_T, measure=self.measure, results_dir=self.results_dir, reports_dir=self.reports_dir)
            reports.rejection(append = True)
            average = reports.average_reports()
            #self.average = average
            self.T.append(new_T)

            # Recalculate costs with the new average results
            average = self.measure_cost(criteria, new_T)

            # Find minimal cost for new_T
            new_min_T_index = average.loc[average['T_value'] == new_T, f"cost_{criteria}"].idxmin()
            new_min_T_cost = average.loc[new_min_T_index, f"cost_{criteria}"]
            new_min_threshold = average.loc[new_min_T_index, 'thresholds']
            
            # Append new minimal cost to mini
            new_row = pd.DataFrame([[new_min_T_cost, new_min_threshold, new_T]], columns=mini.columns)
            mini = pd.concat([mini, new_row], ignore_index=True)

            # Determine which neighbor has the maximum cost (for next exploration)
            min_neighbour_index = mini.loc[[mini.index[-1], min_T_index], f"cost_{criteria}"].idxmax()
            min_neighbour_T_value = mini.loc[min_neighbour_index, 'T_value']

            # Determine which has the minimum cost for next min_T
            min_T_index = mini.loc[[mini.index[-1], min_T_index], f"cost_{criteria}"].idxmin()
            min_T = mini.loc[min_T_index, 'T_value']
            min_T_cost = mini.loc[min_T_index, f"cost_{criteria}"]

            new_min_T_cost = mini.loc[min_neighbour_index, f"cost_{criteria}"]

        return mini
    
    def select_threshold(self, criteria, delta=0.005, epsilon=0.01):
        """
        Select the optimal filtering threshold T by minimizing the cost function.
        Uses recursive interval exploration to refine the choice of T.

        Parameters:
        - delta (float): Minimum improvement threshold for recursive exploration.
        - epsilon (float): Minimum difference between T values to stop exploration.

        Returns:
        - DataFrame with minimal costs and corresponding thresholds and T values.
        """
        self.T = filtering_threshold.copy()
        #print(f"filtering_threshold = {filtering_threshold}")
        #print(f"self.T = {self.T}")
        mini = self.min_cost(criteria = criteria)
        original_mini = mini.copy()

        # Initial best T and cost
        min_T_index = mini[f"cost_{criteria}"].idxmin()
        min_T_cost = mini.loc[min_T_index, f"cost_{criteria}"]
        min_T = mini.loc[min_T_index, 'T_value']

        new_min_T_cost = 1000  # initialize large cost

        # Explore interval forward (next neighbor)
        if min_T_index + 1 < len(mini):
            next_index = mini.index[min_T_index + 1]
            min_neighbour_T_value = mini.loc[next_index, 'T_value']
            mini = self.explore_interval(criteria = criteria, delta = delta, epsilon = epsilon, mini = mini,
                                         min_T_cost = min_T_cost, min_T_index = min_T_index, min_T = min_T, 
                                         new_min_T_cost= new_min_T_cost, min_neighbour_T_value = min_neighbour_T_value)

        
        # Explore interval backward (previous neighbor)
        min_T_index = original_mini[f"cost_{criteria}"].idxmin()
        min_T_cost = original_mini.loc[min_T_index, f"cost_{criteria}"]
        min_T = original_mini.loc[min_T_index, 'T_value']

        new_min_T_cost = 1000  # initialize large cost

        if min_T_index - 1 >= 0:
            prev_index = mini.index[min_T_index - 1]
            min_neighbour_T_value = mini.loc[prev_index, 'T_value']
            mini = self.explore_interval(criteria = criteria, delta = delta, epsilon = epsilon, mini = mini,
                                         min_T_cost = min_T_cost, min_T_index = min_T_index, min_T = min_T, 
                                         new_min_T_cost= new_min_T_cost, min_neighbour_T_value = min_neighbour_T_value)

        return mini

    def brute_force(self, brute_force_T, criteria):

        results = ResultsGenerator(name=self.name, measure = self.measure, T = brute_force_T, splits_dir=self.splits_dir, results_dir=self.results_dir)
        results.get_results()

        reports = ReportsGenerator(name=self.name, measure = self.measure, T = brute_force_T, results_dir=self.results_dir, reports_dir=self.reports_dir)
        reports.rejection(append=False)
        
        average = reports.average_reports()
        #self.average = average

        # Recalculate costs with the new average results
        mini = self.min_cost(T = brute_force_T, criteria = criteria)

        return mini

    # Simulated Annealing algorithm
    def simulated_annealing(self, criteria, initial_filtering = 0.75, cooling_rate = 0.95, initial_temp = 100, iterations = 25, lower_bound = 0.5, upper_bound = 1):

        results = ResultsGenerator(name=self.name,  measure = self.measure, T = initial_filtering, splits_dir=self.splits_dir, results_dir = self.results_dir)
        results.get_results()
        reports = ReportsGenerator(name=self.name,  measure = self.measure, T = initial_filtering, results_dir=self.results_dir, reports_dir = self.reports_dir)
        reports.average(append=False)

        mini = self.min_cost(criteria = criteria, T=initial_filtering)
        #print(mini)
        current_tf = initial_filtering    
        current_cost = mini.loc[mini['T_value'] == initial_filtering, f"cost_{criteria}"].item()
        current_tr = mini.loc[mini['T_value'] == initial_filtering, 'threshold']
        best_x = initial_filtering
        best_cost = current_cost
        temp = initial_temp

        mini_annealing = pd.DataFrame(columns=[f"cost_{criteria}", "threshold", "T_value"])

        new_row = pd.DataFrame([[current_cost, current_tr, current_tf]], columns=mini_annealing.columns)
        mini_annealing = pd.concat([mini_annealing, new_row], ignore_index=True)
    
        for i in range(iterations):
            #print("Searching, iteration " + str(i))
            # Neighbor: random small step
            new_tf = current_tf + random.uniform(-0.25, 0.25)
            new_tf = round(new_tf, 5)
            new_tf = max(min(new_tf, upper_bound), lower_bound)  # keep within bounds

            results = ResultsGenerator(name=self.name,  measure = self.measure, T = new_tf, splits_dir=self.splits_dir, results_dir = self.results_dir)
            results.get_results()
            reports = ReportsGenerator(name=self.name,  measure = self.measure, T = new_tf, results_dir=self.results_dir, reports_dir = self.reports_dir)
            reports.rejection(append=False)

            average = reports.average_reports()
            #self.average = average

            # Recalculate costs with the new average results
            average = self.measure_cost(criteria, new_tf)
            
            # Find minimal cost for new_T
            new_cost_index = average.loc[average['T_value'] == new_tf,f"cost_{criteria}"].idxmin()
            new_cost = average.loc[new_cost_index, f"cost_{criteria}"]
            new_min_threshold = average.loc[new_cost_index, 'thresholds']                   

            # Acceptance probability
            delta = new_cost - current_cost
            if delta < 0 or math.exp(-delta / temp) > random.random():
                current_tf = new_tf
                current_cost = new_cost
                if new_cost < best_cost:
                    best_tf = new_tf
                    best_cost = new_cost
    
            # Append new minimal cost to mini
            new_row = pd.DataFrame([[new_cost, new_min_threshold, new_tf]], columns=mini.columns)
            mini_annealing = pd.concat([mini_annealing, new_row], ignore_index=True)  
            temp *= cooling_rate  # Cool down
    
        return mini_annealing

    @staticmethod
    def print_min_cost_thresholds(df, criteria):
        """
        Prints the filtering (T_value) and rejecting (threshold) thresholds
        corresponding to the minimum cost value, along with the cost.
        """
        # Find the minimum cost
        min_cost = df[f"cost_{criteria}"].min()
        
        # Filter rows with that cost
        min_rows = df[df[f"cost_{criteria}"] == min_cost]
        
        # Get thresholds and T_values
        reject_thresholds = min_rows['threshold'].tolist()
        filter_thresholds = min_rows['T_value'].tolist()
        
        # Format output
        reject_str = ", ".join(map(str, reject_thresholds))
        filter_str = ", ".join(map(str, filter_thresholds))
        
        print(f"Filtering threshold(s): {filter_str}")
        print(f"Rejecting threshold(s): {reject_str}")
        print(f"Associated minimum cost: {min_cost:.6f}")


