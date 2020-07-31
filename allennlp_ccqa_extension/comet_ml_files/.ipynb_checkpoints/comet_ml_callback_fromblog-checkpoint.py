from typing import Text
import socket
import os
import comet_ml
import toml
from allennlp.training.callbacks.callback import Callback, handle_event
from allennlp.training.callbacks.events import Events


@Callback.register("log_to_comet")
class LogToComet(Callback):
    def __init__(self, project_name: Text = None):
        self._project_name = project_name
        model_config_file = os.environ.get("MODEL_CONFIG_FILE")
        if project_name is None or model_config_file is None:
            self._experiment = None
            self._conf = None
        else:
            self._experiment = comet_ml.Experiment(project_name=self._project_name)
            slurm_log_file = os.environ.get("SLURM_LOG_FILE")
            if slurm_log_file is not None:
                self._experiment.log_asset(slurm_log_file, overwrite=True)
            model_config_file = os.environ.get("MODEL_CONFIG_FILE")
            if model_config_file is not None:
                self_experiment.log_asset(model_config_file) 
                with open(model_config_file) as f:
                    self._conf = toml.load(f)
                for key, val in self._conf["params"].items():
                    self._experiment.log_parameter(key, val)
                self._experiment.add_tag(self._conf["name"])
            self._experiment.log_other("hostname", socket.gethostname())
    
    @handle_event(Events.TRAINING_END)
    def training_end(self, _):
        if self._experiment is not None:
            self._experiment.add_tag("COMPLETED")
    
    @handle_event(Events.EPOCH_END)
    def epoch_end_logging(self, trainer):
        if self._experiment is not None:
            epoch = trainer.epoch_number + 1
            for key, val in trainer.train_metrics.items():
                self._experiment.log_metric(f"train_{key}", val, epoch=epoch)
            
            for key, val in trainer.val_metrics.items():
                self._experiment.log_metric(f"val_{key}", val, epoch=epoch)
            slurm_log_file = os.environ.get("SLURM_LOG_FILE")
            if slurm_log_file is not None:
                self._experiment.log_asset(slurm_log_file, overwrite=True)
            
    @handle_event(Events.ERROR)
    def mark_run_failure(self, _):
        if self._experiment is not None:
            self._experiment.add_tag("FAILED")