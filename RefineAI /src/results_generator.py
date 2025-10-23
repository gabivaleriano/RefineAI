import os
import numpy as np
import pandas as pd

from scipy.stats import entropy

from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from samples_builder import SamplesBuilder
from config import filtering_threshold

base_bst = XGBClassifier(
learning_rate=0.01,
n_estimators=1000,
max_depth=8,
min_child_weight=1,
gamma=0,
subsample=0.8,
colsample_bytree=0.8,
objective='binary:logistic',
eval_metric='logloss',
scale_pos_weight=1,
seed=27
)

class ResultsGenerator:
    def __init__(self, name, measure, T = filtering_threshold, model=base_bst, 
                 splits_dir="splits_dir", results_dir="results_dir", ):
        """
        Initialize the ResultsGenerator with experiment parameters and directories.

        Parameters:
        - name (str): name of the experiment (used in filenames)
        - T (float or list of floats): filtering thresholds for instance hardness/influence
        - model: base ML model to be used inside CalibratedClassifierCV
        - splits_dir (str): directory where train/validation splits are saved
        - results_dir_ih (str): output directory for results based on 'ih' measure
        - results_dir_if (str): output directory for results based on 'if' measure
        """
        self.splits_dir = splits_dir
        self.results_dir = results_dir
        self.name = name
        self.T = T
        self.model = model
        self.measure = measure

        # Create output directories if they don't exist, avoiding errors later
        os.makedirs(self.results_dir, exist_ok=True)

    @staticmethod
    def get_easiest(group, t, measure):
        """
        Select the top 't' fraction of easiest instances based on a measure within a class group.

        Parameters:
        - group (DataFrame): subset of data belonging to one class
        - t (float): fraction of instances to select (e.g., 0.1 means top 10%)
        - measure (str): column name to sort by (e.g., 'ih' or 'if')

        Returns:
        - DataFrame of the easiest instances by the measure
        """
        top_easy = int(t * len(group))  # Calculate number of easiest instances to select
        
        # Get the indices of the n smallest values
        #smallest_indices = group[measure].nsmallest(top_easy).index

        smallest_indices = set(group[measure].nsmallest(top_easy).index)
    
        # Return rows in their original order
        return group[group.index.isin(smallest_indices)]
    
        # Return the rows in their original order
        #return group.loc[smallest_indices]
        #return group.nsmallest(top_easy, measure)

        #return group.nsmallest(top_easy, measure).sample(frac=1, random_state=42).reset_index(drop=True)  # Return those with smallest values of measure

        # Get the n smallest indices without full sorting

        #kth = max(top_easy - 1, 0)  # ensure it's non-negative
        #indices = np.argpartition(group[measure].values, kth)[:top_easy]

        #indices = np.argpartition(group[measure].values, top_easy)[:top_easy]
        #subset = group.iloc[indices]
        
        # Shuffle directly (faster than sample for small subsets)
        #subset = subset.sample(frac=1,andom_state=42, ignore_index=True)
        #return subset

    @staticmethod
    def normalize(x):
        return (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else np.zeros_like(x)

    @staticmethod
    def compute_uncertainty_ensemble(X_train, y_train, X_val):
        """
        Train an ensemble of XGB models on X_train, y_train and compute uncertainty
        scores (entropy + variance of predicted class probabilities) for X_val.
    
        Returns:
        - entropy_uncertainty: np.array of shape (n_samples,), entropy scores
        - variance_uncertainty: np.array of shape (n_samples,), average std deviation across class probs
        """
    
        params_list = [
            {'max_depth': 3, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 0},
            {'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.7, 'colsample_bytree': 1.0, 'random_state': 1},
            {'max_depth': 5, 'learning_rate': 0.2, 'subsample': 1.0, 'colsample_bytree': 0.6, 'random_state': 2},
            {'max_depth': 6, 'learning_rate': 0.1, 'subsample': 0.9, 'colsample_bytree': 0.9, 'random_state': 3},
            {'max_depth': 3, 'learning_rate': 0.3, 'subsample': 0.6, 'colsample_bytree': 0.7, 'random_state': 4},
            {'max_depth': 4, 'learning_rate': 0.15, 'subsample': 0.85, 'colsample_bytree': 1.0, 'random_state': 5},
            {'max_depth': 5, 'learning_rate': 0.07, 'subsample': 0.75, 'colsample_bytree': 0.85, 'random_state': 6},
            {'max_depth': 6, 'learning_rate': 0.1, 'subsample': 0.65, 'colsample_bytree': 0.9, 'random_state': 7},
            {'max_depth': 3, 'learning_rate': 0.2, 'subsample': 0.95, 'colsample_bytree': 0.6, 'random_state': 8},
            {'max_depth': 4, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.95, 'random_state': 9},
        ]
    
        probas = []
        for params in params_list:
            model = XGBClassifier(eval_metric='logloss', n_estimators=15, **params)
            model.fit(X_train, y_train)
            probas.append(model.predict_proba(X_val))
    
        probas = np.array(probas)  # shape: (n_models, n_samples, n_classes)
    
        mean_proba = probas.mean(axis=0)  # shape: (n_samples, n_classes)
    
        # Entropy of mean predictions
        entropy_uncertainty = entropy(mean_proba.T)  # shape: (n_samples,)
    
        # Variance-based uncertainty: mean std across ensemble models for each instance
        #std_dev = np.std(probas, axis=0)  # shape: (n_samples, n_classes)
        #variance_uncertainty = np.mean(std_dev, axis=1)  # shape: (n_samples,)
    
        entropy_norm = ResultsGenerator.normalize(entropy_uncertainty)
        #variance_norm = normalize(variance_uncertainty)

        #combined_uncertainty = 0.5 * entropy_norm + 0.5 * variance_norm

        confidence = 1.0 - entropy_norm  # Higher = more confident
        
        scaled_confidence = 0.5 + 0.5 * confidence 
    
        return scaled_confidence

    def get_results(self):
        """
        Main method to process data splits, filter instances, train model, and save results.

        For each seed and threshold:
        - Load training and validation splits
        - For each measure ('ih', 'if'), select easiest training instances by measure
        - Resample the training data to balance classes
        - Train calibrated classifier on resampled data
        - Predict on validation set and record true labels, predicted classes, and confidence scores
        - Save results to CSV files in respective output directories
        """
        # Ensure self.T is a list (even if a single float is provided)
        if not isinstance(self.T, (list, tuple)):  
            self.T = [self.T]
        
        # Loop over predefined seeds for reproducibility
        for seed in [42, 43, 44, 45, 46]:
            
            # Loop over filtering thresholds (fractions of easiest instances to select)
            for t in self.T:
                # Load train and validation data for the current seed and experiment name
                data_train = pd.read_csv(f'{self.splits_dir}/train_{self.name}_seed_{seed}.csv')
                data_test = pd.read_csv(f'{self.splits_dir}/validation_{self.name}_seed_{seed}.csv')                   

                measure = self.measure
        
                # Identify target column (assuming it's third last column)
                target_feature = data_train.columns[-3]
        
                # Select easiest instances per class group by measure
                filtered_train = data_train.groupby(target_feature, group_keys=False).apply(
                    ResultsGenerator.get_easiest, t=t, measure=measure)
        
                # Reset index after filtering
                filtered_train.reset_index(drop=True, inplace=True)
           
                # Prepare feature matrix X and target vector y for training
                X_train = filtered_train.drop(columns=[target_feature, 'ih', 'if'])
                y_train = filtered_train[target_feature]
                
                # Prepare features and target for testing (validation)
                X_test = data_test.drop(columns=[target_feature])
                y_test = data_test[target_feature]
        
                # Resample training data to balance classes (handles class imbalance)
                X_resampled, y_resampled = SamplesBuilder.data_sample(X_train, y_train)

                cv = StratifiedKFold(n_splits=5)
                # Create calibrated classifier using the specified base model
                bst = CalibratedClassifierCV(estimator=self.model, method='sigmoid', cv=cv,
                                            ensemble=False, n_jobs=None) 
    
                # Train the classifier on resampled training data
                bst.fit(X_resampled, y_resampled)
                    
                # Predict class probabilities on validation data
                y_pred_proba = bst.predict_proba(X_test)
                y_pred_positive = y_pred_proba[:, 1]  # Probability for positive class
                y_pred_negative = y_pred_proba[:, 0]  # Probability for negative class
                
                # Determine predicted class by taking class with highest probability
                predicted_class = np.argmax(y_pred_proba, axis=1)
                
                # Assign confidence score based on predicted class's probability
                confidence_scores = np.where(predicted_class == 1, y_pred_positive, y_pred_negative)
                
                # Compute uncertainty using the ensemble function
                uncertainty = self.compute_uncertainty_ensemble(X_resampled, y_resampled, X_test)
            
                # Create results DataFrame including uncertainty column
                results = pd.DataFrame({
                    'True_Labels': y_test, 
                    'Predicted_Class': predicted_class,
                    'Confidence_Scores': confidence_scores,
                    'Uncertainty': uncertainty  # <-- add here
                })  

                results.to_csv(f'{self.results_dir}/results_seed_{seed}_{self.name}_{t}_{self.measure}.csv', index = False)

    @staticmethod
    def filter_and_train(train, test, t, measure):
        
        data_train = train
        data_test = test
                
        target_feature = data_train.columns[-3]

         # Select easiest instances per class group by measure
        filtered_train = data_train.groupby(target_feature, group_keys=False).apply(ResultsGenerator.get_easiest, t=t, measure=measure)

        # Reset the index of the filtered DataFrame
        filtered_train.reset_index(drop=True, inplace=True)
   
        # Split X and y
        X_train = filtered_train.drop(columns=[target_feature, 'ih','if'])
        y_train = filtered_train[target_feature]
        X_test = data_test.drop(columns=[target_feature])
        y_test = data_test[target_feature]

        # Resample
        X_resampled, y_resampled = SamplesBuilder.data_sample(X_train, y_train)

        cv = StratifiedKFold(n_splits=5)
        bst = CalibratedClassifierCV(estimator=base_bst, cv = cv, method='sigmoid', ensemble=False, n_jobs=None ) 
        # Fit model
        bst.fit(X_resampled, y_resampled)
            
        y_pred_proba = bst.predict_proba(X_test)
        y_pred_positive = y_pred_proba[:, 1]  # Predicted probabilities for positive class
        y_pred_negative = y_pred_proba[:, 0]  # Predicted probabilities for negative class
        
        # Determine which class is predicted for each instance
        predicted_class = np.argmax(y_pred_proba, axis=1)  # Predicted class (0 for negative, 1 for positive)
        
        # Combine positive and negative confidence scores based on predicted class
        confidence_scores = np.where(predicted_class == 1, y_pred_positive, y_pred_negative)
        
        # Compute uncertainty using the ensemble function
        uncertainty = ResultsGenerator.compute_uncertainty_ensemble(X_resampled, y_resampled, X_test)
    
        # Create results DataFrame including uncertainty column
        results = pd.DataFrame({
            'True_Labels': y_test, 
            'Predicted_Class': predicted_class,
            'Confidence_Scores': confidence_scores,
            'Uncertainty': uncertainty  
        })  
        return results

