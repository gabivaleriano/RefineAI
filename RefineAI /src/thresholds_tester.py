import pandas as pd  # Assuming pandas is required
import os

from samples_builder import SamplesBuilder
from reports_generator import ReportsGenerator
from results_generator import ResultsGenerator

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=3)


class ThresholdsTester: 


    def __init__(self, train_name, test_name, metric, data_path, ih_uncertainty, ih_confidence, if_uncertainty, if_confidence):
        """
        Initialize the ThresholdsTester with dataset paths and uncertainty values.
        """
        self.name = train_name        
        self.train_file = os.path.join(data_path, f"{train_name}.csv")
        self.test_file = os.path.join(data_path, f"{test_name}.csv")
        
        self.splits_dir = f"splits_{train_name}"
        self.results_dir = f"results_{train_name}"
        self.reports_dir = f"reports_{train_name}"
        self.graphs_dir = f"graphs_{train_name}"
        self.test_path = data_path
        self.test_name = test_name
        self.metric = metric

        # Store uncertainties and confidences in dictionaries
        self.uncertainty = {
            "ih": ih_uncertainty,
            "if": if_uncertainty
        }
        
        self.confidence = {
            "ih": ih_confidence,
            "if": if_confidence
        }

    def final_measures(self):
  
        train_path = os.path.join(self.splits_dir, f"train_{self.name}_seed_42.csv")
        val_path = os.path.join(self.splits_dir, f"validation_{self.name}_seed_42.csv")    
        train = pd.read_csv(train_path, index_col=None)
        val = pd.read_csv(val_path, index_col=None)
        train = train.drop('ih', axis = 1)
        train = train.drop('if', axis = 1)
        train = pd.concat([train, val], ignore_index=True)  
                
        target_feature = train.columns[-1]      
        
        test_path = os.path.join(self.test_path, f'{self.test_name}.csv')
        data_test = pd.read_csv(test_path, index_col=None)

        imputer.fit(train)
        data_test =  pd.DataFrame(imputer.transform(data_test), columns=data_test.columns, index=data_test.index)

        data_train = SamplesBuilder.ih_measure(train)
        data_train = SamplesBuilder.influence_measure(data_train)

        data_train.to_csv(os.path.join(self.splits_dir, f"train_{self.name}_complete_measures.csv"), index=False)     
        

    def final_results(self):

        data_train = pd.read_csv(os.path.join(self.splits_dir, f"train_{self.name}_complete_measures.csv"))           
        train = data_train.drop(columns = ['ih', 'if'])
        test_path = os.path.join(self.test_path, f'{self.test_name}.csv')
        data_test = pd.read_csv(test_path, index_col=None)

        imputer.fit(train)
        data_test =  pd.DataFrame(imputer.transform(data_test), columns=data_test.columns, index=data_test.index)

        
        classification_report_c = {
            'if': [],
            'ih': []
        }

        classification_report_u = {
            'if': [],
            'ih': []
        }

        classification_report_z = {
            'if': [],
            'ih': []
        }

        final_df_list = []

        # for each measure and each threshold 
        
        for measure in ['ih', 'if']:

            T_values = [
                ("confidence", self.confidence[measure][0]),
                ("uncertainty", self.uncertainty[measure][0]),
                ("one", 1),
            ]
            #T_value = [1, self.confidence[measure][0], self.uncertainty[measure][0]]

            for tag, T in T_values: 
            
                results = ResultsGenerator.filter_and_train(data_train, data_test, T, measure)     
                results = results.reset_index(drop=True)

                t_confidence = self.confidence[measure][1]
                t_uncertainty = self.uncertainty[measure][1]
    
                if tag == "confidence":

                    list_confidence = ReportsGenerator.final_testing_thresholds(results, T, 'Confidence_Scores', 0)
                    classification_report_c[measure].extend(list_confidence)
                
                    list_confidence = ReportsGenerator.final_testing_thresholds(results, T, 'Confidence_Scores', t_confidence)
                    classification_report_c[measure].extend(list_confidence)
                    
                elif tag == "uncertainty":    

                    list_uncertainty = ReportsGenerator.final_testing_thresholds(results, T, 'Uncertainty', 0)
                    classification_report_u[measure].extend(list_uncertainty)

                    list_uncertainty = ReportsGenerator.final_testing_thresholds(results, T, 'Uncertainty', t_uncertainty)
                    classification_report_u[measure].extend(list_uncertainty)                
                    
                elif tag == "one": 
                    
                    list_confidence = ReportsGenerator.final_testing_thresholds(results, T, 'Confidence_Scores', t_confidence)
                    list_uncertainty = ReportsGenerator.final_testing_thresholds(results, T, 'Uncertainty', t_uncertainty)                        
                    classification_report_c[measure].extend(list_confidence)
                    classification_report_u[measure].extend(list_uncertainty)

                    if measure == 'if': 
                        list_zero = ReportsGenerator.final_testing_thresholds(results, T, 'Confidence_Scores', 0)
                        classification_report_z[measure].extend(list_zero)
                        list_zero = ReportsGenerator.final_testing_thresholds(results, T, 'Uncertainty', 0)
                        classification_report_z[measure].extend(list_zero)
    
            # After all T values processed for this measure, build DataFrame
            df_c = pd.DataFrame(classification_report_c[measure])
            df_u = pd.DataFrame(classification_report_u[measure])
            df_z = pd.DataFrame(classification_report_z[measure])
    
            # Add measure column
            df_c['measure'] = measure
            df_u['measure'] = measure
            df_z['measure'] = measure

            # Add measure column
            df_c['metric'] = 'C'
            df_u['metric'] = 'U'

            if not df_z.empty:
                df_z['metric'] = ['C', 'U']
        
            # Append to final list
            final_df_list.extend([df_c, df_u, df_z])
        
        # Concatenate all DataFrames
        final = pd.concat(final_df_list, ignore_index=True)

        final['rate_accep_0'] = final['support_class_0'] / max(final['support_class_0'])
        final['rate_accep_1'] = final['support_class_1'] / max(final['support_class_1'])
        
        desired_order = ['measure', 'metric', 'T_value', 'thresholds', 
                        'precision_class_0', 'precision_class_1', 
                        'recall_class_0', 'recall_class_1', 
                        'rate_accep_0', 'rate_accep_1', 'mean_conf_uncert']
        
        final['T_value'] = 1- final['T_value']
        
        final = final[desired_order]
        final = final.round(3)
        
        macro = final

        macro['f1_0'] = 2*final['precision_class_0']*final['recall_class_0']/                            (final['precision_class_0']+final['recall_class_0'])
        macro['f1_1'] = 2*final['precision_class_1']*final['recall_class_1']/                    (final['precision_class_1']+final['recall_class_1'])

        macro['f1-macro'] = (macro['f1_0']+ macro['f1_1'])/2
        macro['acp_a'] = (macro['rate_accep_0']+ macro['rate_accep_1'])/2
        
        return final, macro
