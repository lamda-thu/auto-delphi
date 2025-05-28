import yaml
import random
from typing import Dict, List
from .dellmaAgent.agent import DecisionMakerBase, DeLLMaAgent

class DellmaDecisionMaker(DecisionMakerBase):
    def __init__(self, language_model, config_file_path: str):
        """
        Parameters
        ----------
        config_file_path : str
            The path to the config file.
        """
        super().__init__()
        with open(config_file_path, 'r') as f:
            config = yaml.safe_load(f)
        random.seed(config['seed'])
        self.__language_model = language_model

    def getName(self):
        return f"dellma-{self.__language_model.__class__.__name__}"

    def makeDecision(self, text: str, decision_alternatives: Dict[str, List[str]], context_dict: Dict[str, str]) -> Dict[str, str]:
        """
        Warnings
        --------
        The current implementation only allows a single decision node.
        """
        # get the first decision variable
        # store its values
        decision_variable = list(decision_alternatives.keys())[0]
        decision_choices = decision_alternatives[decision_variable]
        decision_choices = [str(choice) for choice in decision_choices]

        agent = DeLLMaAgent(
            choices=decision_choices,
            context=text,
            temperature=0.0,
            llm=self.__language_model
        )
        try:
            agent.initialize_workflow(regenerate_states=True, regenerate_beliefs=True)
            decision = agent.make_decision_with_expected_utility()
            choice = decision['decision']
            if choice not in decision_choices:
                raise ValueError(f"PARSING ERROR: {self.__class__.__name__}")

        except Exception as e:
            print(f"Error: {e}")
            choice = None
        return {
            decision_variable: choice
        }
