"""
DecisionMaker with Automated Influence Diagram generation.
"""

from typing import Dict, List
import concurrent.futures
from src.agent import GenerationMode
from src.agent.graphAgent import GraphAgent
from src.agent.probabilityAgent import ProbabilityAgent
from experiment import DecisionMakerBase


class AidDecisionMaker(DecisionMakerBase):
    def __init__(
        self,
        language_model,
        graph_generation_mode: GenerationMode = GenerationMode.extract,
        probability_generation_mode: GenerationMode = GenerationMode.extract,
        max_retries: int = 5,
        timeout: int = 1200,  # Default timeout of 1200 seconds
    ):
        super().__init__()
        self.__graph_agent = GraphAgent(language_model, mode=graph_generation_mode, max_retries=max_retries)
        self.__probability_agent = ProbabilityAgent(language_model, mode=probability_generation_mode, max_retries=max_retries)
        self.__timeout = timeout

    def getName(self) -> str:
        return f"aid-{self.__graph_agent.getLanguageModel().__class__.__name__}"

    def _execute_with_timeout(self, func, timeout, *args, **kwargs):
        """Execute a function with timeout"""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"Operation exceeded timeout of {timeout} seconds")

    def _solve_with_timeout(self, influence_diagram):
        """Execute influence_diagram.solve() with timeout"""
        return self._execute_with_timeout(influence_diagram.solve, self.__timeout)

    def _joint_generation_with_timeout(self, text, joint_fix=True):
        """Execute graph_agent.jointGeneration with timeout"""
        return self._execute_with_timeout(self.__graph_agent.jointGeneration, self.__timeout, text, joint_fix)

    def _construct_graph_with_timeout(self):
        """Execute graph_agent.constructGraph with timeout"""
        return self._execute_with_timeout(self.__graph_agent.constructGraph, self.__timeout)

    def _assign_cpd_with_timeout(self, text):
        """Execute probability_agent.assignStochasticFunctionCPD with timeout"""
        return self._execute_with_timeout(self.__probability_agent.assignStochasticFunctionCPD, self.__timeout, text)

    def makeDecision(self, text: str, decision_alternatives: Dict[str, List[str]], context_dict: Dict[str, str]) -> Dict[str, str]:
        """
        Parameters
        ----------
        text: str
            The text of the problem.
        decision_alternatives: Dict[str, List[str]]
            The decision alternatives.
        context_dict: Dict[str, str]
            The context dictionary, specifying the values of the context variables.
        """
        try:
            # Apply timeouts to potentially slow operations
            self._joint_generation_with_timeout(text, joint_fix=True)
            influence_diagram = self._construct_graph_with_timeout()
            self.__probability_agent.addGraph(influence_diagram)
            self._assign_cpd_with_timeout(text)
            influence_diagram = self.__probability_agent.getDiagram()
            optimal_policy = self._solve_with_timeout(influence_diagram)

            first_decision = self.__probability_agent.getDiagram().get_valid_order()[0]
            optimal_policy = optimal_policy[first_decision]
            if len(context_dict) > 0:
                # handle when context_dict is not empty
                optimal_policy = optimal_policy.stochastic_function(**context_dict)
                optimal_action = max(optimal_policy, key=optimal_policy.get)
            else:
                # handle when context_dict is empty
                chosen_action_idx = optimal_policy.values.argmax()
                optimal_action = optimal_policy.domain[chosen_action_idx]
        except TimeoutError as te:
            print(f"Timeout error in AidDecisionMaker: {te}")
            optimal_action = None
        except Exception as e:
            print(f"Error in AidDecisionMaker: {e}")
            optimal_action = None
        
        variable_name = list(decision_alternatives.keys())[0]
        return {variable_name: optimal_action}
