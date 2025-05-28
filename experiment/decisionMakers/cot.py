from typing import Dict, List

from experiment import DecisionMakerBase
from src.agent import DecisionAgent

class CotDecisionMaker(DecisionMakerBase):
    """
    The decision maker that utilizes an LLM with Chain-of-Thought for decision-making
    """
    def __init__(self, language_model):
        super().__init__()
        self.__decision_agent = DecisionAgent(language_model)

    def getName(self) -> str:
        return f"cot-{self.__decision_agent.getLanguageModel().__class__.__name__}"

    def makeDecision(self, text: str, decision_alternatives: Dict[str, List[str]], context_dict: Dict[str, str], verbose: bool = False) -> Dict[str, str]:
        try:
            decision = self.__decision_agent.generateSingleDecisionCot(text, decision_alternatives, verbose)
            for decision_variable in decision:
                if decision_variable not in decision_alternatives:
                    raise ValueError(f"Decision variable {decision_variable} not a valid decision variable")
                else:
                    if decision[decision_variable] not in decision_alternatives[decision_variable]:
                        raise ValueError(f"Decision {decision[decision_variable]} not a valid decision for decision variable {decision_variable}")
            return decision
        except:
            print(f"PARSING ERROR: {self.__class__.__name__}")
            return {decision_variable: None for decision_variable in decision_alternatives.keys()}