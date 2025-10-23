import os
import pandas as pd
import numpy as np

from scipy.special import expit

from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.metrics import log_loss

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from collections import defaultdict
from numpy.linalg import inv

from pyhard.classification import ClassifiersPool

clf = LogisticRegression(fit_intercept=True, solver='lbfgs', max_iter=1000)
scaler = StandardScaler()
imputer = KNNImputer(n_neighbors=3)

class SamplesBuilder:
    """
    Class for generating train/validation splits, imputing missing values,
    balancing class distributions, computing instance hardness and influence.
    """

    def __init__(self, csv_path, name, splits_dir="output_splits"):
        """
        Initialize the SamplesBuilder.

        Args:
            csv_path (str): Path to the CSV dataset.
            name (str): Identifier used for naming output files.
            splits_dir (str): Directory where splits will be saved.
        """
        self.csv_path = csv_path
        self.splits_dir = splits_dir
        self.data = pd.read_csv(csv_path)
        self.splits = {}
        self.name = name

        os.makedirs(self.splits_dir, exist_ok=True)

    @staticmethod
    def data_sample(X, y):
        """
        Balance data using under- or over-sampling based on class imbalance.
        """
        undersample = RandomUnderSampler(sampling_strategy='majority', random_state=1)
        oversample = RandomOverSampler(sampling_strategy=0.2, random_state=1)

        count_1 = (y == 1).sum()
        count_0 = (y == 0).sum()
        ratio = min(count_1, count_0) / max(count_1, count_0)

        if ratio > 0.6:
            return X, y
        elif ratio <= 0.2:
            X, y = oversample.fit_resample(X, y)
            return undersample.fit_resample(X, y)
        else:
            return undersample.fit_resample(X, y)

    @staticmethod
    def ih_measure(train):
        """Compute instance hardness using ClassifiersPool."""
        train = pd.DataFrame(imputer.fit_transform(train), columns=train.columns, index=train.index)

        pool = ClassifiersPool(train)
        pool.run_all(
            metric='logloss',
            n_folds=5,
            algo_list=[           
                'svc_linear',
                'svc_rbf',
                'random_forest',
                'gradient_boosting',
                'knn',
                'logistic_regression',
                'mlp',
            ], 
            n_iter=5,
            resampling='under',
            hyper_param_optm=False,
        )

        ih_series = pd.Series(pool.estimate_ih().flatten(), name='ih', index=train.index)
        return pd.concat([train.reset_index(drop=True), ih_series.reset_index(drop=True)], axis=1)

    @staticmethod
    def sigmoid(z):
        return expit(z)

    @staticmethod
    def compute_hessian(X, y, theta):
        """Compute Hessian matrix of logistic loss."""
        preds = SamplesBuilder.sigmoid(X @ theta)
        W = np.diag(preds * (1 - preds))
        return X.T @ W @ X

    @staticmethod
    def grad_loss(X, y, theta):
        """Compute gradient of logistic loss."""
        preds = SamplesBuilder.sigmoid(X @ theta)
        return (preds - np.array(y))[:, np.newaxis] * X

    @staticmethod
    def compute_influence_scores(X_train, y_train, X_val, y_val, instance_indices):
        """
        Compute harmful influence scores: how much each training instance contributes to higher validation loss.
        Higher values = more harmful. Scores scaled to [0, 1] per fold.
        """
        clf = LogisticRegression(solver='liblinear')
        clf.fit(X_train, y_train)
    
        theta = np.concatenate([clf.intercept_, clf.coef_.flatten()])
        X_train_aug = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        X_val_aug = np.hstack([np.ones((X_val.shape[0], 1)), X_val])
    
        H = SamplesBuilder.compute_hessian(X_train_aug, y_train, theta)
        H_inv = inv(H + 1e-5 * np.eye(H.shape[0]))  # Regularization
    
        grads_train = SamplesBuilder.grad_loss(X_train_aug, y_train, theta)
        grads_val = SamplesBuilder.grad_loss(X_val_aug, y_val, theta)
    
        val_probs = clf.predict_proba(X_val)
        logloss = log_loss(y_val, val_probs)
    
        raw_scores = {}
        for i, idx in enumerate(instance_indices):
            grad_i = grads_train[i].reshape(-1, 1)
            influence = sum(grad_val @ (H_inv @ grad_i) for grad_val in grads_val)
            influence = -influence.item() / len(y_val) # Flip sign so higher = more harmful
            influence *= logloss
            raw_scores[idx] = influence  
    
        # Per-iteration min-max normalization
        values = np.array(list(raw_scores.values()))
        min_score = values.min()
        max_score = values.max()
    
        if max_score == min_score:
            scaled_scores = {k: 0.0 for k in raw_scores}
        else:
            scaled_scores = {
                k: (v - min_score) / (max_score - min_score)
                for k, v in raw_scores.items()
            }
    
        return scaled_scores

    @staticmethod
    def influence_measure(train):
        """Compute influence of each training point using influence functions (higher = more harmful)."""
        initial_index = train.index.copy()
    
        ih_column = None
        if 'ih' in train.columns:
            ih_column = train.pop('ih')
    
        target = train.columns[-1]
        X = train.drop(columns=[target])
        y = train[target]
    
        if X.isnull().any().any():
            X = SamplesBuilder.impute_missing(X)
    
        all_scores = defaultdict(list)
        used_indices = set()
    
        for repeat_seed in range(5):
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=repeat_seed)
    
            for train_idx, val_idx in skf.split(X, y):
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
                X_train, y_train = SamplesBuilder.data_sample(X_train, y_train)
                used_indices.update(X_train.index)
                resampled_index = X_train.index
    
                X_train = imputer.fit_transform(X_train)
                X_val = imputer.transform(X_val)
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)
    
                scores = SamplesBuilder.compute_influence_scores(X_train, y_train, X_val, y_val, resampled_index)
                for idx, score in scores.items():
                    all_scores[idx].append(score)
    
        # Recover unused samples
        unused_indices = set(X.index) - used_indices
        if unused_indices:
            print(f"Recovering {len(unused_indices)} samples missed during under-sampling...")
    
            X_unused, y_unused = X.loc[unused_indices], y.loc[unused_indices]
            X_, y_ = X.loc[~X.index.isin(unused_indices)], y.loc[~y.index.isin(unused_indices)]
    
            X_train, X_val, y_train, y_val = train_test_split(X_, y_, test_size=0.2, stratify=y_, random_state=42)
            X_train, y_train = SamplesBuilder.data_sample(X_train, y_train)
            X_train = pd.concat([X_train, X_unused])
            y_train = pd.concat([y_train, y_unused])
            resampled_index = X_train.index
    
            X_train = imputer.fit_transform(X_train)
            X_val = imputer.transform(X_val)
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
    
            scores = SamplesBuilder.compute_influence_scores(X_train, y_train, X_val, y_val, resampled_index)
            for idx, score in scores.items():
                all_scores[idx].append(score)
    
        # Final score: average of already normalized values
        final_scores = pd.Series({idx: np.mean(scores) for idx, scores in all_scores.items()})

        for idx, score_list in all_scores.items():
            if any(s < 0 or s > 1 for s in score_list):
                print(f"Out-of-range scores for index {idx}: {score_list}")

        min_val = final_scores.min()
        max_val = final_scores.max()
        if max_val != min_val:
            final_scores = (final_scores - min_val) / (max_val - min_val)
        else:
            final_scores[:] = 0.0

        #print("Final influence score range:", final_scores.min(), final_scores.max())
    
        X_result = X.copy()
        X_result[target] = y
        X_result['if'] = final_scores
    
        if ih_column is not None:
            X_result['ih'] = ih_column
    
        return X_result.reindex(initial_index)


    def split_samples(self, seeds=[42, 43, 44, 45, 46]):
        """
        Create multiple train/validation splits, compute IH and IF scores, and save results.

        Args:
            seeds (list): List of seeds for different random splits.
        """
        y = self.data.iloc[:, -1]
        
        for seed in seeds:
            train, val = train_test_split(self.data, test_size=0.3, random_state=seed, stratify=y) #
            train = self.ih_measure(train)
            train = self.influence_measure(train)

            self.splits[seed] = {'train': train, 'validation': val}

            train.to_csv(os.path.join(self.splits_dir, f"train_{self.name}_seed_{seed}.csv"), index=False)
            val.to_csv(os.path.join(self.splits_dir, f"validation_{self.name}_seed_{seed}.csv"), index=False)

        print(f"Splits saved in '{self.splits_dir}' for seeds: {seeds}")