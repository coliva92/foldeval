from abc import ABC, abstractmethod





class PredictionAlgorithm(ABC):
  
  def __init__(self):
    global sequence
    self.plddt_scores = None
    self.tqdm_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} ' + \
                       '[elapsed: {elapsed} remaining: {remaining}]'

  @abstractmethod
  def setup(self) -> None: pass
  
  @abstractmethod
  def do_msa(self) -> None: pass

  @abstractmethod
  def make_prediction(self) -> str: pass

  @abstractmethod
  def display_predicted_model(self) -> None: pass

  @abstractmethod
  def prepare_results_folder(self) -> str:
    global jobname
    %shell cp metrics.txt $jobname
    return jobname
