"""
Given an influence diagram graph, elicit conditional probability distributions for each node. This Delphi-style process takes two steps:
1. iterate through each node in the graph, and ask (human) user to describe how the values of the node varies with the values of its parents.
2. based on the user's description, prompt LLM to generate a stochstic function CPD for the node.

The `ProbabilityDelphi` class works on variable levels, independent of the influence diagram graph.
The `InfluenceDiagramProbabilityDelphi` class works on influence diagram graph levels, and is a subclass of `ProbabilityDelphi`.
"""

import re
from typing import List, Dict
from pycid.core.cpd import StochasticFunctionCPD
from langchain_core.language_models import LLM
from langchain_core.prompts import PromptTemplate

from .delphi_prompt import DELPHI_PROMPTS


class ProbabilityDelphi(object):
    def __init__(
        self, 
        language_model: LLM,
        max_retries: int = 5
    ):
        self.language_model = language_model
        self.max_retries = max_retries

    def _elicit_cpd_description(self, variable: str, parent_variables: List[str]) -> str:
        """ Elicit user description of the conditional probability distribution for the variable given the parent variables. """
        if len(parent_variables) == 0:
            prompt_text = f"\nPlease describe the probability distribution for variable '{variable}' (which has no parent variables):\n"
        else:
            parent_list_str = "', '".join(parent_variables)
            prompt_text = f"\nPlease describe how variable '{variable}' depends on its parent variables ['{parent_list_str}']:\n"
        
        description = input(prompt_text)
        return description

    def _generate_cpd_params(self, variable: str, parent_variables: List[str], description: str, feedback: str = "") -> str:
        """ Generate a stochastic function CPD parameters for the variable given the parent variables and the user description.
        """
        if feedback == "":
            # First generation
            template = DELPHI_PROMPTS["cpd_generation_from_description"]
            prompt = PromptTemplate(
                template=template,
                input_variables=["variable", "parent_variable_list", "description"]
            )
            chain = prompt | self.language_model
            result = chain.invoke({
                "variable": variable,
                "parent_variable_list": parent_variables,
                "description": description
            })
        else:
            # Improvement with feedback
            template = DELPHI_PROMPTS["cpd_improve_from_feedback"]
            prompt = PromptTemplate(
                template=template,
                input_variables=["variable", "parent_variable_list", "description", "feedback"]
            )
            chain = prompt | self.language_model
            result = chain.invoke({
                "variable": variable,
                "parent_variable_list": parent_variables,
                "description": description,
                "feedback": feedback
            })
        
        # Extract lambda function from code block
        pattern = r'```python(.*?)```'
        match = re.search(pattern, result, re.DOTALL)
        if match:
            lambda_function = match.group(1).strip().lower()
        else:
            lambda_function = "Error in extracting the lambda function. Please use a python code block to wrap the lambda function."
            return lambda_function
        
        # Post-process the lambda function
        lambda_function = self._post_process_lambda(lambda_function)
        
        return lambda_function

    def _post_process_lambda(self, lambda_function: str) -> str:
        """ Post-process the lambda function string to fix common issues. """
        # Convert string numbers to numerical values
        string_numbers = re.findall(r'[\'"]([-]?\d+\.?\d*)[\'"]', lambda_function)
        for num_str in string_numbers:
            lambda_function = lambda_function.replace(f'"{num_str}"', num_str)
            lambda_function = lambda_function.replace(f"'{num_str}'", num_str)
        
        # Handle mathematical expressions in dictionary keys
        pattern = r'\{[\'"](.*?)[\'"]\s*:'
        matches = re.findall(pattern, lambda_function)
        for expr in matches:
            # Skip descriptive labels with format "word (number)"
            if re.search(r'\w+\s+\([^)]+\)', expr):
                continue
            
            # Check if the expression contains numbers or math operators
            if re.search(r'\d|[-+*/()]', expr):
                # Only convert pure numeric expressions
                if re.search(r'[<>]', expr):
                    continue
                lambda_function = lambda_function.replace(f"'{expr}':", f"{expr}:")
                lambda_function = lambda_function.replace(f'"{expr}":', f"{expr}:")
        
        return lambda_function

    def _validate_cpd_params(self, cpd_params: str) -> List[str]:
        """
        Called before generating the CPD from str parameters.
        Validate the parameters are valid for a CPD.
        If there are issues, record the issues and provide suggested fixes.
        Return a list of validation feedback strings.
        """
        issues = []
        
        if cpd_params.startswith("Error") or cpd_params.startswith("error"):
            issues.append(cpd_params)
            return issues
        
        # Check if it's a valid lambda expression
        if not cpd_params.strip().startswith("lambda"):
            issues.append("The CPD parameters should be a lambda function starting with 'lambda'.")
            return issues
        
        # Try to evaluate the lambda function
        try:
            eval(cpd_params)
        except Exception as e:
            issues.append(f"Error in evaluating the lambda function: {e}")
        
        return issues

    def _construct_cpd(self, cpd_params: str) -> StochasticFunctionCPD:
        """ Construct the CPD from the parameters. 
        
        Note: This is a simplified version that returns the lambda function as a string.
        For full CPD construction, use InfluenceDiagramProbabilityDelphi which has access to the diagram.
        """
        # This method needs to be overridden in subclasses that have access to the diagram
        # For now, return the evaluated lambda function
        return eval(cpd_params)

    def run(
        self,
        variable: str,
        parent_variables: List[str]
    ) -> StochasticFunctionCPD:
        """ Run the Delphi process for the variable given the parent variables.

        Run generation at most max_retries times, each time validate after generation. If no issues, return the CPD. Otherwise, try generation again.
        """
        # Elicit the CPD description
        description = self._elicit_cpd_description(variable, parent_variables)

        cpd_params = None
        feedback = ""

        for retry_count in range(self.max_retries):
            # Generate the CPD
            cpd_params = self._generate_cpd_params(variable, parent_variables, description, feedback)

            # Validate the CPD parameters
            validation_feedback = self._validate_cpd_params(cpd_params)
            if len(validation_feedback) == 0:
                break
            else:
                # update the feedback
                feedback = "\n".join(validation_feedback)
                print(f"Validation issues found (retry {retry_count + 1}/{self.max_retries}):")
                print(feedback)
        
        if len(validation_feedback) > 0:
            print(f"Warning: CPD generation failed after {self.max_retries} retries.")
        
        cpd = self._construct_cpd(cpd_params)
        return cpd