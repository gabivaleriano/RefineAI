import os
import pandas as pd
from sklearn.metrics import classification_report
from config import filtering_threshold, thresholds

class ReportsGenerator:
    def __init__(self, name, measure, T = filtering_threshold, results_dir="results_dir", reports_dir="reports_dir"):
        
        """
        Initialize the ReportsGenerator with experiment details and directories.

        Parameters:
        - name (str): Name of the experiment or dataset.
        - measure (str): The measure used ('ih' or 'if').
        - T (float or list): Filtering threshold(s) used in the experiment.
        - results_dir (str): Directory where results CSV files are stored.
        - reports_dir (str): Directory to save the generated reports.
        """
        self.results_dir = results_dir
        self.reports_dir = reports_dir
        self.name = name
        self.T = T
        self.measure = measure

        # Create the reports directory if it does not exist
        os.makedirs(self.reports_dir, exist_ok=True)

    def rejection(self, append=False):            
        """
        Generate classification reports for each seed and save to CSV files.
    
        Parameters:
        - append (bool): If True, append to existing CSV files instead of overwriting.
        """        
        for seed in [42, 43, 44, 45, 46]:
            classification_report_list_confidence = [] 
            classification_report_list_uncertainty  = [] 
            
            # Collect reports for different filtering thresholds for the given seed
            reports_confidence, reports_uncertainty = self.iterate_filtering_thresholds(seed) 
            
            classification_report_list_confidence.extend(reports_confidence)
            classification_report_list_uncertainty.extend(reports_uncertainty)
    
            # Convert list of reports to a DataFrame
            classification_report_df_confidence = pd.DataFrame(classification_report_list_confidence)
            classification_report_df_uncertainty = pd.DataFrame(classification_report_list_uncertainty)
    
            # Construct file paths
            file_confidence = f'{self.reports_dir}/classification_seed_{seed}_{self.name}_{self.measure}_confidence.csv'
            file_uncertainty = f'{self.reports_dir}/classification_seed_{seed}_{self.name}_{self.measure}_uncertainty.csv'
    
            # Determine mode and whether to include headers
            mode = 'a' if append else 'w'
            header_confidence = not (append and os.path.exists(file_confidence))
            header_uncertainty = not (append and os.path.exists(file_uncertainty))
    
            # Save the DataFrames to CSV
            classification_report_df_confidence.to_csv(file_confidence, mode=mode, header=header_confidence, index=False)
            classification_report_df_uncertainty.to_csv(file_uncertainty, mode=mode, header=header_uncertainty, index=False)

    def iterate_filtering_thresholds(self, seed):
        """
        Iterate over all filtering thresholds, read corresponding results, and generate reports.

        Parameters:
        - seed (int): Random seed identifier.

        Returns:
        - list_reports (list): List of classification report dictionaries for all thresholds.
        """
        list_reports_confidence = []    
        list_reports_uncertainty = []

        # Ensure T is a list for iteration
        if not isinstance(self.T, (list, tuple)):  
            self.T = [self.T]
        
        # Loop over each filtering threshold
        for t in self.T:   
            # Load the results CSV for the given seed, threshold, and measure
            results = pd.read_csv(f'{self.results_dir}/results_seed_{seed}_{self.name}_{t}_{self.measure}.csv')
            
            # Generate reports for different confidence thresholds
            reports_confidence = self.compute_testing_thresholds(results, t, 'Confidence_Scores')
            reports_uncertainty = self.compute_testing_thresholds(results, t, 'Uncertainty')
            
            # Accumulate the reports
            list_reports_confidence.extend(reports_confidence)     
            list_reports_uncertainty.extend(reports_uncertainty) 

        return list_reports_confidence, list_reports_uncertainty           
        
    @staticmethod
    def compute_testing_thresholds(results, t, criteria):
        list_reports = []
        
        for t2 in thresholds: 
    
            # Filter results by confidence score > threshold t2
            filtered_results = results[results[criteria] > t2]
        
            if filtered_results.empty:
                # If no samples pass the threshold, fill metrics with zeros
                metrics_dict = {
                    'T_value': t,
                    'thresholds': t2,
                    'precision_class_0': 0,
                    'precision_class_1': 0,
                    'recall_class_0': 0,
                    'recall_class_1': 0,
                    'f1_class_0': 0,
                    'f1_class_1': 0,
                    'support_class_0': 0,
                    'support_class_1': 0,
                    'mean_conf_uncert_class_0': 0,
                    'mean_conf_uncert_class_1': 0,
                    'mean_conf_uncert': 0
                }
            else:
                # Calculate mean confidence for each class and overall
                mean_conf_uncert_class_0 = filtered_results[filtered_results['True_Labels'] == 0][criteria].mean()
                mean_conf_uncert_class_1 = filtered_results[filtered_results['True_Labels'] == 1][criteria].mean()
                mean_conf_uncert = filtered_results[criteria].mean()
    
                # Replace NaNs with 0 if any class is missing in filtered results
                mean_conf_uncert_class_0 = mean_conf_uncert_class_0 if not pd.isna(mean_conf_uncert_class_0) else 0
                mean_conf_uncert_class_1 = mean_conf_uncert_class_1 if not pd.isna(mean_conf_uncert_class_1) else 0
                meanconf_uncert = mean_conf_uncert if not pd.isna(mean_conf_uncert) else 0
                
                # Track which classes are present
                unique_classes = filtered_results['True_Labels'].unique()
                metrics_dict = {
                    'T_value': t,
                    'thresholds': t2,
                    'mean_conf_uncert_class_0': mean_conf_uncert_class_0,
                    'mean_conf_uncert_class_1': mean_conf_uncert_class_1,
                    'mean_conf_uncert': mean_conf_uncert
                }
    
                # Calculate precision, recall, f1-score, and support for each class
                for class_label in [0, 1]:
                    if class_label in unique_classes:
                        labels = [class_label]
    
                        # Generate classification report dict for this class
                        report = classification_report(
                            filtered_results['True_Labels'],
                            filtered_results['Predicted_Class'],
                            output_dict=True,
                            labels=labels,
                            zero_division=0
                        )
    
                        metrics_dict.update({
                            f'precision_class_{class_label}': report[str(class_label)]['precision'],
                            f'recall_class_{class_label}': report[str(class_label)]['recall'],
                            f'f1_class_{class_label}': report[str(class_label)]['f1-score'],
                            f'support_class_{class_label}': report[str(class_label)]['support']
                        })
                    else:
                        # No samples for this class: set metrics to zero
                        metrics_dict.update({
                            f'precision_class_{class_label}': 0,
                            f'recall_class_{class_label}': 0,
                            f'f1_class_{class_label}': 0,
                            f'support_class_{class_label}': 0
                        })
                
            # Append the collected metrics for this threshold
            list_reports.append(metrics_dict)

        return list_reports

    @staticmethod        
    def testing_thresholds(results, t):
        """
        For each confidence threshold, compute classification metrics and confidence statistics.

        Parameters:
        - results (DataFrame): Results data with true labels, predictions, and confidence scores.
        - t (float): Filtering threshold value used for this results batch.

        Returns:
        - list_reports (list): List of metric dictionaries, one for each confidence threshold.
        """
        list_reports_confidence = ReportsGenerator.compute_testing_thresholds(results, t, 'Confidence_Scores')
        list_reports_uncertainty = ReportsGenerator.compute_testing_thresholds(results, t, 'Uncertainty')

        return list_reports_confidence, list_reports_uncertainty   

    @staticmethod        
    def testing_thresholds_final(results, t_confidence, t_uncertainty, threshold):
        """
        For each confidence threshold, compute classification metrics and confidence statistics.

        Parameters:
        - results (DataFrame): Results data with true labels, predictions, and confidence scores.
        - t (float): Filtering threshold value used for this results batch.

        Returns:
        - list_reports (list): List of metric dictionaries, one for each confidence threshold.
        
        list_reports_confidence = ReportsGenerator.final_testing_thresholds(results, t_confidence, 'Confidence_Scores', threshold)
        list_reports_uncertainty = ReportsGenerator.final_testing_thresholds(results, t_uncertainty, 'Uncertainty', threshold)
        list_reports_zero = ReportsGenerator.final_testing_thresholds(results, 0, 'Uncertainty', threshold)

        return list_reports_confidence, list_reports_uncertainty, list_reports_zero
        """
    def average_reports(self):
        """
        Compute mean and standard deviation of all metrics across all seeds.

        Returns:
        - average_df (DataFrame): DataFrame with average and std of metrics.
        """
        # List of metric columns to aggregate
        metrics_columns = ['mean_conf_uncert_class_0', 'mean_conf_uncert_class_1',
                           'mean_conf_uncert', 'precision_class_0', 
                           'precision_class_1', 'recall_class_0', 'recall_class_1', 
                           'f1_class_0', 'f1_class_1', 'support_class_0', 'support_class_1']
        average_dfs = {}
        
        for criteria in ['confidence', 'uncertainty']:
        
            # List of CSV file paths for each seed
            csv_files = [f'{self.reports_dir}/classification_seed_{seed}_{self.name}_{self.measure}_{criteria}.csv' for seed in [42, 43, 44, 45, 46]]
            
            # Dictionary to accumulate metric values per seed
            metrics_data = {metric: [] for metric in metrics_columns}
            
            for file_path in csv_files:
                # Load data from CSV
                data = pd.read_csv(file_path)
                
                # Append metric values for each column
                for metric in metrics_columns:
                    metrics_data[metric].append(data[metric])
            
            # DataFrame to store averages and std deviations
            average_df = pd.DataFrame()
            
            # Calculate mean and std for each metric across seeds
            for metric, metric_data in metrics_data.items():
                metric_concat = pd.concat(metric_data, axis=1)  # Combine columns for each seed
                average_df[f'{metric}_mean'] = metric_concat.mean(axis=1)
                average_df[f'{metric}_std'] = metric_concat.std(axis=1)
            
            # Calculate macro F1-score mean and std as average of class-wise F1s
            f1_columns = ['f1_class_0', 'f1_class_1']
            support_columns = ['support_class_0', 'support_class_1']
            
            average_df['macro_f1_mean'] = average_df[[f'{col}_mean' for col in f1_columns]].mean(axis=1)
            average_df['macro_f1_std'] = average_df[[f'{col}_std' for col in f1_columns]].mean(axis=1)
            
            # Calculate total support mean and std as sum of class-wise support
            average_df['total_support_mean'] = average_df[[f'{col}_mean' for col in support_columns]].sum(axis=1)
            average_df['total_support_std'] = average_df[[f'{col}_std' for col in support_columns]].sum(axis=1)
            
            # Include T_value and thresholds columns for reference
            average_df['T_value'] = data['T_value']
            average_df['thresholds'] = data['thresholds']
    
            average_df.to_csv(f'{self.reports_dir}/classification_average_{self.name}_{self.measure}_{criteria}.csv', index=False)

            average_dfs[f'average_df_{criteria}'] = average_df
            
        return average_dfs['average_df_confidence'], average_dfs['average_df_uncertainty']
        
    def average(self, append = False):
        self.rejection(append = append)
        self.average_reports()

    @staticmethod
    def final_testing_thresholds(results, t, criteria, thresholds):
        list_reports = []

        # Filter results by confidence score > threshold t2
        filtered_results = results[results[criteria] > thresholds]
    
        if filtered_results.empty:
            # If no samples pass the threshold, fill metrics with zeros
            metrics_dict = {
                'T_value': t,
                'thresholds': thresholds,
                'precision_class_0': 0,
                'precision_class_1': 0,
                'recall_class_0': 0,
                'recall_class_1': 0,
                'f1_class_0': 0,
                'f1_class_1': 0,
                'support_class_0': 0,
                'support_class_1': 0,
                'mean_conf_uncert_class_0': 0,
                'mean_conf_uncert_class_1': 0,
                'mean_conf_uncert': 0
            }
        else:
            # Calculate mean confidence for each class and overall
            mean_conf_uncert_class_0 = filtered_results[filtered_results['True_Labels'] == 0][criteria].mean()
            mean_conf_uncert_class_1 = filtered_results[filtered_results['True_Labels'] == 1][criteria].mean()
            mean_conf_uncert = filtered_results[criteria].mean()

            # Replace NaNs with 0 if any class is missing in filtered results
            mean_conf_uncert_class_0 = mean_conf_uncert_class_0 if not pd.isna(mean_conf_uncert_class_0) else 0
            mean_conf_uncert_class_1 = mean_conf_uncert_class_1 if not pd.isna(mean_conf_uncert_class_1) else 0
            meanconf_uncert = mean_conf_uncert if not pd.isna(mean_conf_uncert) else 0
            
            # Track which classes are present
            unique_classes = filtered_results['True_Labels'].unique()
            metrics_dict = {
                'T_value': t,
                'thresholds':thresholds,
                'mean_conf_uncert_class_0': mean_conf_uncert_class_0,
                'mean_conf_uncert_class_1': mean_conf_uncert_class_1,
                'mean_conf_uncert': mean_conf_uncert
            }

            # Calculate precision, recall, f1-score, and support for each class
            for class_label in [0, 1]:
                if class_label in unique_classes:
                    labels = [class_label]

                    # Generate classification report dict for this class
                    report = classification_report(
                        filtered_results['True_Labels'],
                        filtered_results['Predicted_Class'],
                        output_dict=True,
                        labels=labels,
                        zero_division=0
                    )

                    metrics_dict.update({
                        f'precision_class_{class_label}': report[str(class_label)]['precision'],
                        f'recall_class_{class_label}': report[str(class_label)]['recall'],
                        f'f1_class_{class_label}': report[str(class_label)]['f1-score'],
                        f'support_class_{class_label}': report[str(class_label)]['support']
                    })
                else:
                    # No samples for this class: set metrics to zero
                    metrics_dict.update({
                        f'precision_class_{class_label}': 0,
                        f'recall_class_{class_label}': 0,
                        f'f1_class_{class_label}': 0,
                        f'support_class_{class_label}': 0
                    })
                
            # Append the collected metrics for this threshold
            list_reports.append(metrics_dict)

        return list_reports
