from typing import Dict, List

from experiment import DecisionMakerBase
from src.agent import DecisionAgent

class VanillaDecisionMaker(DecisionMakerBase):
    """
    The decision maker that utilizes an LLM directly for decision-making
    """
    def __init__(self, language_model):
        super().__init__()
        self.__decision_agent = DecisionAgent(language_model)

    def getName(self) -> str:
        return f"vanilla-{self.__decision_agent.getLanguageModel().__class__.__name__}"

    def makeDecision(self, text: str, decision_alternatives: Dict[str, List[str]], context_dict: Dict[str, str]) -> Dict[str, str]:
        try:
            return self.__decision_agent.generateSingleDecisionVanilla(text, decision_alternatives)
        except:
            print(f"PARSING ERROR: {self.__class__.__name__}")
            return {decision_variable: None for decision_variable in decision_alternatives.keys()}