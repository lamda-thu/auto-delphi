import yaml
import random
from typing import Dict, List
from experiment import DecisionMakerBase

class RandomDecisionMaker(DecisionMakerBase):
    """ The decision maker that randomly picks an alternative for each decision."""
    def __init__(self, config_file_path: str):
        """
        Parameters
        ----------
        config_file_path : str
            The path to the config file.
        """
        with open(config_file_path, 'r') as f:
            config = yaml.safe_load(f)
        random.seed(config['seed'])

    def getName(self):
        return "random"

    def makeDecision(self, text: str, decision_alternatives: Dict[str, List[str]], context_dict: Dict[str, str]) -> Dict[str, str]:
        """
        For each decision variable, randomly pick one alternative.
        """
        return {
            decision_variable: decision_alternatives[decision_variable][random.randint(0, len(decision_alternatives[decision_variable]) - 1)] for decision_variable in decision_alternatives.keys()
        }