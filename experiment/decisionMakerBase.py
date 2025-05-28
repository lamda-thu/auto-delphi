from abc import ABC, abstractmethod
from typing import List, Dict


class DecisionMakerBase(ABC):
    """
    The base class for all decision makers in the experiment.
    Each decision maker should implement the `makeDecision` method,\
        which receices input text and decision alternatives, and\
            returns the chosen action for each decision variable.
    """
    def __init__(self):
        pass

    @abstractmethod
    def getName(self) -> str:
        """ 
        Getter method of the name of the Decision Maker.
        Should all be in lower case.
        If LLM is involved, should include the name of the language model.
        """
        raise NotImplementedError

    @abstractmethod
    def makeDecision(self, text: str, decision_alternatives: Dict[str, List[str]], context_dict: Dict[str, str]) -> Dict[str, str]:
        """
        The method for decision making.

        Parameters
        ----------
        text : str
            The text description of the decision-making problem.
        decision_alternatives : Dict[str, List[str]]
            A dictionary mapping from the decision variable name to the list of values that the variable can take.
        context_dict : Dict[str, str]
            A dictionary mapping from the context variable name to the value of the context variable.

        Returns
        -------
        Dict[str, str]
            The chosen action for each decision variable.
            A dictionary mapping from the decision variable name to its value.
        
        Warning
        -------
        The current implementation does not support wait-and-see decisions.
        """
        raise NotImplementedError