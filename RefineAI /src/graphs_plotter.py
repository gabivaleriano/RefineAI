import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import filtering_threshold, thresholds

# Set global font for plots to sans-serif for a cleaner appearance
plt.rcParams["font.family"] = "sans-serif"

class GraphsPlotter:
    def __init__(self, name, measure, metric = 'macro-f1', filtering_threshold = filtering_threshold, reports_dir="output_reports", graphs_dir="output_graphs",font_size=18, spine_color='grey', show_plots=True, save_plots=True):
        """
        Initialize the ReportsAnaliser.

        Parameters:
        - name (str): Name identifier for the experiment.
        - reports_dir (str): Directory where classification reports are saved.
        - graphs_dir (str): Directory where output graphs will be saved.
        """
        self.metric = metric
        self.reports_dir = reports_dir
        self.graphs_dir = graphs_dir
        self.name = name
        self.save_plots = save_plots
        self.show_plots = show_plots
        self.measure = measure
        self.font_size = font_size
        self.spine_color = spine_color

        # Ensure the directory for saving graphs exists
        os.makedirs(self.graphs_dir, exist_ok=True)

        # Set color palette for plots (assumes len(T) exists in global scope)
        self.colors = sns.color_palette('dark')

        # Human-readable metric names for plotting titles and labels
        self.metric_names = {
            "precision_class_0": "Precision Negative Class",
            "precision_class_1": "Precision Positive Class",
            "recall_class_0": "Recall Negative Class",
            "recall_class_1": "Recall Positive Class",
            "macro_f1": "Macro F1",
            # Add more mappings if needed
        }



        plt.rcParams.update({
            'axes.titlesize': font_size + 4,
            'axes.labelsize': font_size,
            'xtick.labelsize': font_size,
            'ytick.labelsize': font_size,
            'legend.fontsize': font_size,
            'figure.titlesize': font_size + 6
        })

    def binary_graphs(self, metric="macro_f1", blocks=len(filtering_threshold)):
        """
        Generate and optionally save a multi-block plot showing the selected performance
        metric and rate of accepted instances over rejection thresholds.
    
        Parameters:
        - metric (str): Name of the metric to plot (e.g., 'macro_f1').
        - blocks (int): Number of experiment blocks to show (e.g., len(T)).
        - save (bool): Whether to save the plot as a PDF file.
        """
        uncertainty = pd.read_csv(f'{self.reports_dir}/classification_average_{self.name}_{self.measure}_uncertainty.csv')
        confidence = pd.read_csv(f'{self.reports_dir}/classification_average_{self.name}_{self.measure}_confidence.csv')
    
        datasets = {
            'uncertainty': uncertainty,
            'confidence': confidence
        }
    
        for dataset_name, data in datasets.items():
        
            # Normalize support to get rate of accepted instances
            max_support = data['total_support_mean'].max()
            data['rate_accepted_mean'] = data['total_support_mean'] / max_support
            data['rate_accepted_std'] = data['total_support_std'] / max_support
    
            # Prepare column names for metric mean and standard deviation
            metric_mean = f"{metric}_mean"
            metric_std = f"{metric}_std"
            columns_to_plot = [metric_mean, 'rate_accepted_mean']
            stdev_columns = [metric_std, 'rate_accepted_std']
            x_column = 'Rejecting threshold'
    
            block_size = len(thresholds)  # assumes thresholds is defined globally
            num_blocks = blocks
    
            # Determine y-axis limits across all blocks for consistency
            y_max = data[columns_to_plot].max().max()
            y_min = data[columns_to_plot].min().min()
    
            # Create subplots: one row per block, two columns for metrics
            fig, axes = plt.subplots(num_blocks, len(columns_to_plot), figsize=(7, 2 * num_blocks))   
    
            for i in range(num_blocks):
                # Slice the data for the current block
                data_ = data[i * block_size: (i + 1) * block_size].copy()
                data_['Rejecting threshold'] = thresholds 
    
                for j, column in enumerate(columns_to_plot):
                    # Scatter plot with error bars
                    axes[i, j].scatter(data_[x_column], data_[column], color=self.colors[1], s=20)                     
                    axes[i, j].errorbar(data_[x_column], data_[column],
                                        yerr=data_[stdev_columns[j]], fmt='none', ecolor='gray')
    
                    # Hide x-axis labels except for the last row
                    if i != num_blocks - 1:
                        axes[i, j].set_xlabel('')
                        axes[i, j].tick_params(axis='x', bottom=False, top=False, labelbottom=False)
                    else:
                        axes[i, j].set_xlabel(x_column, fontsize=self.font_size)
    
                    # Set title for the top row only
                    if i == 0:
                        axes[i, j].set_title(column.replace('_mean', '').replace('_', ' ').capitalize(), fontsize=self.font_size)
    
                    # Set consistent y-axis limits
                    axes[i, j].set_ylim(y_min, y_max)
                    axes[i, j].tick_params(axis='both')
    
                    # Hide y-axis labels for columns other than the first
                    if j != 0:
                        axes[i, j].set_ylabel('')
                        axes[i, j].tick_params(axis='y', left=False, right=False, labelleft=False)
                    else:
                        axes[i, j].set_ylabel('Metric value', fontsize=self.font_size)
    
                    # Add a legend with T value to the last column
                    if j == len(columns_to_plot) - 1:
                        axes[i, j].legend(title=('Tf = ' + str(round(1 - filtering_threshold[i], 2))), fontsize=self.font_size)  
    
            # Adjust layout and optionally save figure
            plt.tight_layout()
            if self.save_plots:
                plt.savefig(f"{self.graphs_dir}/binary_distribution_{self.metric}_{self.measure}_{dataset_name}_.pdf")
            if self.show_plots:
                plt.show()
            plt.close()
       

    def cost_vs_tvalue(self, mini, criteria, strategy = ' '):
        """
        Plot cost values across different filtering thresholds.

        Parameters:
        - mini (pd.DataFrame): DataFrame containing 'T_value' and 'cost' columns.
        - save (bool): Whether to save the plot as a PDF file.
        """
        plt.figure(figsize=(8, 5))

        # Ensure data is sorted by T_value for proper line plotting
        mini_sorted = mini.sort_values(by="T_value")
        
        # Plot cost vs (1 - T_value) with markers and lines
        plt.plot(1 - mini_sorted['T_value'], mini_sorted[f"cost_{criteria}"],
                 marker='o', markersize=8, linestyle='-', color=self.colors[1])
        
        # Set axis labels and title with consistent font size
        plt.xlabel('Filtering threshold', fontsize=self.font_size + 2)
        plt.ylabel(f"Cost {criteria}", fontsize=self.font_size + 2)
        plt.title('Cost value per filtering threshold', fontsize=self.font_size + 2)
        
        # Add horizontal grid lines
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        
        # Customize plot border colors and thickness
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_color(self.spine_color)
            spine.set_linewidth(1.5)
        
        # Ticks styling
        ax.tick_params(axis='both', labelsize=self.font_size)
        
        # Save or show the figure
        plt.tight_layout()
        if self.save_plots:
            plt.savefig(f"{self.graphs_dir}/cost_vs_threshold_{self.metric}_{self.measure}_{criteria}_{strategy}.pdf")
        if self.show_plots:
            plt.show()
        plt.close()


    def plot_uncertainty_confidence_distribution(results: pd.DataFrame, T_value: float):
        """
        Plot the distributions of Uncertainty and Confidence_Scores from the results DataFrame.
    
        Parameters:
        - results: pd.DataFrame containing columns 'Uncertainty' and 'Confidence_Scores'.
        - T_value: float, threshold value used in filtering (used in plot title for clarity).
        """
        plt.figure(figsize=(12, 5))
    
        # Plot distribution of Confidence Scores
        plt.subplot(1, 2, 1)
        sns.histplot(results['Confidence_Scores'], bins=30, kde=True, color=self.colors[1])
        plt.title(f'Confidence Scores Distribution (T={T_value})')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
    
        # Plot distribution of Uncertainty
        plt.subplot(1, 2, 2)
        sns.histplot(results['Uncertainty'], bins=30, kde=True, color=self.colors[1])
        plt.title(f'Uncertainty Distribution (T={T_value})')
        plt.xlabel('Uncertainty (Entropy)')
        plt.ylabel('Frequency')
    
        plt.tight_layout()
        # Save or show the figure
        plt.tight_layout()
        if self.save_plots:
            plt.savefig(f"{self.graphs_dir}/distribution_conf_unc{self.metric}.pdf")
        if self.show_plots:
            plt.show()
        plt.close()
