"""
DecisionMaker with self-consistency.
"""

from typing import Dict, List
from src.agent.decisionAgent import DecisionAgent
from experiment import DecisionMakerBase

class ScDecisionMaker(DecisionMakerBase):
    def __init__(self, language_model):
        super().__init__()
        self.__decision_agent = DecisionAgent(language_model)

    def getName(self) -> str:
        return f"sc-{self.__decision_agent.getLanguageModel().__class__.__name__}"

    def makeDecision(self, text: str, decision_alternatives: Dict[str, List[str]], context_dict: Dict[str, str], K: int=5) -> Dict[str, str]:
        try:
            decision = self.__decision_agent.generateSingleDecisionSc(text, decision_alternatives, K)
            # transform all values of decision to string
            decision = {k: str(v) for k, v in decision.items()}
            for decision_variable in decision:
                if decision_variable not in decision_alternatives:
                    raise ValueError(f"Decision variable {decision_variable} not a valid decision variable")
                else:
                    if decision[decision_variable] not in decision_alternatives[decision_variable]:
                        raise ValueError(f"Decision {decision[decision_variable]} not a valid decision for decision variable {decision_variable}")
            return decision
        except Exception as e:
            print(f"Error: {e}")
            return {decision_variable: None for decision_variable in decision_alternatives.keys()}
