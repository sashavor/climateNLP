from typing import Text, Dict, Any, List
from allennlp.training.trainer import BatchCallback, EpochCallback
from allennlp.data.dataloader import TensorDict
from allennlp.models import basic_classifier
import socket
import os
import comet_ml
import toml
import pandas as pd


@EpochCallback.register('cometml_epochCallback')
class CometML_EpochCallback(EpochCallback):
    """
    Registers metrics to cometML before/after every epoch:
    """
    
    def __init__(self,
                 api_key: str = None,
                 project_name: str = None,
                 workspace: str = None):
        print("Initializing cometml_epochCallback ...")
        
        # define specifications to save expr, and model_config_file
        self._project_name = project_name
        self._api_key = api_key
        self._workspace = workspace
        # define experiment based on api key, project name, and workspace
        print("api_key:", api_key)
        print("self_api_key:", self._api_key)
        self._experiment = comet_ml.Experiment(self._api_key, project_name=self._project_name, workspace=self._workspace, auto_output_logging=False)
            
        # log current slurm log file
        slurm_log_file = os.environ.get("SLURM_LOG_FILE")
        if slurm_log_file is not None:
                self._experiment.log_asset(slurm_log_file, overwrite=True)
            
         # log model configs/params (if they are not None)
        model_config_file = os.environ.get("MODEL_CONFIG_FILE")
        if model_config_file is not None:
            self._experiment.log_asset(model_config_file)
            with open(model_config_file) as f:
                self._conf = toml.load(f)
            for key, val in self._conf["params"].items():
                self._experiment.log_parameter(key, val)
            self._experiment.add_tag(self._conf["name"])
            
        # log experiment host name
        self._experiment.log_other("hostname", socket.gethostname())
            
    # defines calling function
    def __call__(self,
                trainer: "GradientDescentTrainer",
                metrics: Dict[str, Any],
                epoch: int,
                is_master: bool
                ) -> None:
        print("Calling cometml_epochCallback ...")
        # sets usable epoch number
        epoch = epoch + 1
        
        # logs all metrics to the experiment
        for name, val in metrics.items():
            self._experiment.log_metric(name, val, epoch=epoch)
#             print("name:", name)
#             print("val:", val)
            
        
        
        # if have already done at least one epoch, create confusion matrices, log them
        if epoch > 0:
#             print(metrics["training_tp"])
#             print(type(metrics["training_tp"]))
            
            training_confusion_matrix = [[metrics["training_tp"], metrics["training_fn"]],[metrics["training_fp"], metrics["training_tn"]]]
            fname = "training-confusion-matrix-epoch-%i.json" %(epoch)
            self._experiment.log_confusion_matrix(matrix=training_confusion_matrix, labels=["Positive", "Negative"], file_name=fname)
            
            validation_confusion_matrix = [[metrics["validation_tp"], metrics["validation_fn"]],[metrics["validation_fp"], metrics["validation_tn"]]]
            fname = "validation-confusion-matrix-epoch-%i.json" % (epoch)
            self._experiment.log_confusion_matrix(matrix=validation_confusion_matrix, labels=["Positive", "Negative"], file_name=fname)
            
            best_validation_confusion_matrix = [[metrics["best_validation_tp"], metrics["best_validation_fn"]],[metrics["best_validation_fp"], metrics["best_validation_tn"]]]
            fname = "best-validation-confusion-matrix-epoch-%i.json" %(epoch)
            self._experiment.log_confusion_matrix(matrix=best_validation_confusion_matrix, labels=["Positive", "Negative"], file_name=fname)
            
            
        # logs current slurm log file, replacing old log
        slurm_log_file = os.environ.get("SLURM_LOG_FILE")
        if slurm_log_file is not None:
            self._experiment.log_asset(slurm_log_file, overwrite=True)
            
            
            
@BatchCallback.register('savePreds_batchCallback')
class SavePredsBatchCallback(BatchCallback):
    """
        Extracts and stores predictions, in order to perform error analysis
    """
    
    def __init__(self,
                output_fname: str = "preds.tsv",
                input_fname: str = None,
                tsv: bool = True):
        print("Initializing savePreds_batchCallback ...")
        
        self._output_fname = output_fname
        self._input_fname = input_fname
        
        self._output_label_list = []
        self._output_token_list = []
        self._output_tokenid_list = []
        self._output_probabilities_list = []
        
        if input_fname is not None:
            if bool:
                self._input_df = pd.read_csv(input_fname, sep="\t")
            else:
                self._input_df = pd.read_csv(input_fname)
            self._input_df_num_rows = self._input_df.shape[0]
        else:
            self._input_df = None
    
    
    def __call__(
        self,
        trainer: "GradientDescentTrainer",
        batch_inputs: List[List[TensorDict]],
        batch_outputs: List[Dict[str, Any]],
        epoch: int,
        batch_number: int,
        is_training: bool,
        is_master: bool,
    ) -> None:
        
        print("Calling batch callback ...")
        
        
        
        # if evaluating only:
        if not is_training:
            print("not training batch callback ...")
            # make output human readable
            output_dict = make_output_human_readable(batch_outputs)
            
            # add to tokens
            self._output_token_list = self._output_token_list + output_dict["tokens"]
            self._output_tokenid_list = self._output_tokenid_list + output_dict["token_ids"]
            self._output_label_list = self._output_label_list + output_dict["label"]
            
            # get probabilities into list instead of tensor
            predictions = output_dict["probs"]
            if predictions.dim() == 2:
                predictions_list = [predictions[i] for i in range(predictions.shape[0])]
            else:
                predictions_list = [predictions]
                
            self._output_probabilities_list = self._output_probabilities_list + predictions_list
            
            # if is last batch, append columns to df and save df to output file
            if len(self._output_probabilities_list) == self._input_df_num_rows:
                # add columns
                output_df = self._input_df.assign({"OUT_TOKENS": self._output_token_list,
                                     "OUT_TOKEN_IDS": self._output_tokenid_list,
                                     "OUT_LABELS": self._output_label_list,
                                     "OUT_PROBABILITIES": self._output_probabilities_list})
                
                # save df
                output_df.to_csv(self._output_fname, sep="\t", index=False)
            else:
                print("lengths didn't match!")
            
                
            