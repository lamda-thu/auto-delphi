from utils import getCurrentTimeForFileName
import json
from tqdm import tqdm
from pycid.core.cpd import TabularCPD, StochasticFunctionCPD
from graph import InfluenceDiagram
import graph
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.language_models import LLM
from langchain_core import tools
import langchain_core
from APIKEYS import *
from models import Gpt4Turbo, DeepseekCoder
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple
import os
import sys
import re
from agent.summarizerAgent import SummarizerAgent
from agent.distance import kl_divergence, js_divergence, total_variation_distance, brier_score

from mode import GenerationMode
from prompt import PROMPTS

import langchain_core.exceptions
from functools import reduce

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)


class ProbabilityAgent:
    def __init__(
        self,
        language_model: LLM,
        max_retries: int = 5,
        mode: GenerationMode = GenerationMode.extract
    ):
        self.__diagram: Optional[InfluenceDiagram] = None
        self.language_model = language_model
        self.max_retries = max_retries
        self.local_cpd_list = []
        self.mode = mode

    def addGraph(self, graph: InfluenceDiagram):
        self.__diagram = graph
        # print(self.__diagram.getNodeDict())

    def saveCPD(self, output_dir):
        # output_dir_with_time = os.path.join(output_dir, getCurrentTimeForFileName())
        # if not os.path.exists(output_dir_with_time):
        #    os.makedirs(output_dir_with_time)
        CPD_file_name = os.path.join(output_dir, "cpd.json")
        with open(CPD_file_name, 'w', encoding='utf-8') as f:
            json.dump(self.local_cpd_list, f, indent=4)

    def loadCPD(self, input_dir):
        assert os.path.exists(input_dir)
        # print(f"ProbabilityAgent: Loading CPDs from {input_dir}")
        cpd_file_name = os.path.join(input_dir, "cpd.json")
        if not os.path.exists(cpd_file_name):
            raise Exception(
                f"ProbabilityAgent: CPD file {cpd_file_name} does not exist")

        with open(cpd_file_name, 'r', encoding='utf-8') as f:
            self.local_cpd_list = json.load(f)
        self.__diagram.setCpds(self.local_cpd_list)
        return self.local_cpd_list

    def getCPD(self) -> List[str]:
        """
        Get the list of CPD params.
        """
        return self.local_cpd_list

    def getDiagram(self) -> InfluenceDiagram:
        return self.__diagram

    # TODO: method: generateCPD()
    # choose from stochastic function & tabular CPD generation
    def generateCPD(self):
        raise NotImplementedError

    def addCPD(self, cpd_param: Dict[str, str]):
        """
        Add a CPD to the local list and the diagram.
        If the CPD already exists, it will be overwritten.
        """
        self.local_cpd_list = [cpd for cpd in self.local_cpd_list if cpd.get("variable") != cpd_param.get("variable")]
        self.local_cpd_list.append(cpd_param)
        self.__diagram.addStochasticFunctionCPD(cpd_param)

    # Stochastic function CPD
    def generateStochasticFunctionCPDForVariable(self, variable: str, parent_variable_list: List[str], source_text: Optional[str]) -> str:
        """ LLM to generate CPD expression for variable, without supporting corpus"""
        if self.mode == GenerationMode.random:
            generate_template = PROMPTS["cpd_stochastic_random"]
        elif self.mode == GenerationMode.generate:
            generate_template = PROMPTS["cpd_stochastic_generation"]
        elif self.mode == GenerationMode.extract:
            generate_template = PROMPTS["cpd_stochastic_extraction"]
        else:
            raise ValueError(f"Invalid mode for cpd: {self.mode}")
        prompt = PromptTemplate(
            template=generate_template,
            input_variables=["variable", "parent_variable_list"],
            partial_variables={
                "text": source_text if source_text is not None else "",
                "variable_domain": self.__diagram.node_list.getNode(variable).getVariableValues(),
                "condition_domain": [self.__diagram.node_list.getNode(v).getVariableValues() for v in parent_variable_list],
            }
        )
        chain = prompt | self.language_model
        ori_lambda_function = chain.invoke({
            "parent_variable_list": parent_variable_list,
            "variable": variable
        })
        pattern = r'```python(.*?)```'
        match = re.search(pattern, ori_lambda_function, re.DOTALL)
        issues = None
        if match:
            lambda_function = match.group(1).strip().lower()

        else:
            issues = "Error in extracting the lambda function, Please using a python code block to wrap the lambda function."
            lambda_function = "Error in extracting the lambda function, Please using a python code block to wrap the lambda function."

        # Convert string numbers to numerical values in all nodes
        # Find all string numbers in the function, including negative numbers and decimals
        string_numbers = re.findall(r'[\'"]([-]?\d+\.?\d*)[\'"]', lambda_function)
        for num_str in string_numbers:
            # Replace string numbers with numerical values
            lambda_function = lambda_function.replace(f'"{num_str}"', num_str)
            lambda_function = lambda_function.replace(f"'{num_str}'", num_str)
        
        # Handle mathematical expressions in dictionary keys
        pattern = r'\{[\'"](.*?)[\'"]\s*:'
        matches = re.findall(pattern, lambda_function)
        for expr in matches:
            # Skip descriptive labels with format "word (number)" - typical for categories with ranges
            if re.search(r'\w+\s+\([^)]+\)', expr):
                continue
        
            # Check if the expression contains numbers or math operators
            if re.search(r'\d|[-+*/()]', expr):
                # Only convert pure numeric expressions, keep expressions with operators like <, > as strings
                if re.search(r'[<>]', expr):
                    # Keep expressions with comparison operators as strings
                    continue
                # Replace the string expression with the actual expression
                lambda_function = lambda_function.replace(f"'{expr}':", f"{expr}:")
                lambda_function = lambda_function.replace(f'"{expr}":', f"{expr}:")
        
        # Check for non-numeric values in dictionary values
        # Use more specific pattern that only matches colons within dictionary context
        # This avoids matching the colon in "lambda:" declarations
        dict_value_pattern = r'[{\s,][\'"]?[\w\s()+\-*/]+[\'"]?\s*:\s*([^,\}]+)'
        value_matches = re.findall(dict_value_pattern, lambda_function)
        for value in value_matches:
            value = value.strip()
            # Skip if already a number or valid Python expression
            if re.match(r'^-?\d+\.?\d*$', value) or value in ['True', 'False', 'None'] or re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$', value):
                continue
            # If it's a string representation of a number, convert it
            if re.match(r'^[\'"](-?\d+\.?\d*)[\'"]$', value):
                num = re.match(r'^[\'"](-?\d+\.?\d*)[\'"]$', value).group(1)
                lambda_function = lambda_function.replace(value, num)
        
        if issues is None:
            issues, contents = self.validateStochasticFunctionCPD(
                lambda_function, variable, parent_variable_list)
        retry_count = 0
        while len(issues) > 0 and retry_count < self.max_retries:
            retry_count += 1
            print(
                f"Retrying {retry_count} times in constructing CPD for {variable}")
            # print(issues)
            lambda_function = self.improveStochasticFunctionCPD(
                variable, parent_variable_list, source_text, issues, lambda_function)
            
            issues, contents= self.validateStochasticFunctionCPD(
                    lambda_function, variable, parent_variable_list)
        contents['variable'] = variable
        contents['evidence'] = parent_variable_list
            
        #print(contents)
        return contents

    def validateStochasticFunctionCPD(
        self,
        lambda_function: Dict,
        variable: str,
        parent_variable_list: List[str]
    ) -> Tuple[str, Dict]:
        if lambda_function.startswith("Error") or lambda_function.startswith("error"):
            return lambda_function, lambda_function
        issues = ""
        """ validate the generated CPD """
        contents={}
        contents['stochastic_function'] = lambda_function.lower()
        stochastic_fn = contents['stochastic_function']
        for evidence_name in parent_variable_list:
            evidence_name = evidence_name.replace(" ", "_")
            if evidence_name not in stochastic_fn:
                issue = f"Variable {evidence_name} not found in the stochastic function.\nConsider modifying the arguments of the stochastic function.\n"
                issues += issue
        if "evidence" in stochastic_fn:
            issues += "The variable name 'evidence' is used in the stochastic function. Please use the actual variable name instead.\n"
        try:
            eval(stochastic_fn)
        except Exception as e:
            issues += f"Error in evaluating the stochastic function: {e}\n"
        try:
            contents['variable'] = variable
            contents['evidence'] = parent_variable_list
            cpd = self.__diagram.constructStochasticFunctionCPD(contents)
        except Exception as e:
            issues += f"Error in constructing the CPD: {e}\n"
            #print("Current CPD:")
            #print(self.getCPD())
            issues += f"Additional dict information:{self.__diagram.node_dict}\n" 
        if self.__diagram.node_list.getNode(variable).getVariableType() == "utility":
            pass #TODO
        return issues, contents

    def improveStochasticFunctionCPD(
        self,
        variable: str,
        parent_variable_list: List[str],
        source_text: Optional[str],
        issues: str,
        ori_contents: str
    ) -> str:
        """ LLM to improve CPD expression for variable, without supporting corpus"""
        improve_template = PROMPTS["cpd_stochastic_improve"]
        prompt = PromptTemplate(
            template=improve_template,
            input_variables=["variable", "parent_variable_list", "issues", "ori_contents"],
            partial_variables={
                "text": source_text if source_text is not None else "",
                "variable_domain": self.__diagram.node_list.getNode(variable).getVariableValues(),
                "condition_domain": [self.__diagram.node_list.getNode(v).getVariableValues() for v in parent_variable_list],
            }
        )
        chain = prompt | self.language_model
        new_lambda_function = chain.invoke({
            "parent_variable_list": parent_variable_list,
            "variable": variable,
            "issues": issues,
            "ori_contents": ori_contents
        })
        pattern = r'```python(.*?)```'
        match = re.search(pattern, new_lambda_function, re.DOTALL)
        if match:
            new_lambda_function = match.group(1).strip()
        else:
            new_lambda_function = "Error in extracting the lambda function, Please using a python code block to wrap the lambda function."
        return new_lambda_function

    def assignStochasticFunctionCPD(self, source_text: Optional[str] = None):
        """ add CPD models to the InfluenceDiagram"""
        assert self.__diagram is not None
        variable_names = [
            self.getDiagram()._get_label(node) for node in 
            self.getDiagram().get_valid_order(self.getDiagram().nodes)
        ]
        self.summarizer = SummarizerAgent(language_model=self.language_model)
        for variable_name in variable_names:
            if variable_name not in self.__diagram.getDecompositionDict().keys():
                raise ValueError(f"Undefined variable: {variable_name}")
            elif self.__diagram.getVariableType(variable_name) == "decision":
                continue
            else:
                parent_variable_list = self.__diagram.getDecompositionDict()[
                    variable_name]
            # print(variable_name)
            # print(parent_variable_list)
            # LLM step: generate tabularCPD parameters
            # summarized_txt = self.summarizer.summarize(source_text, variable_name)
            # print(summarized_txt)
            stochasticFunctionCPD_params = self.generateStochasticFunctionCPDForVariable(
                variable_name, parent_variable_list, source_text)
            if stochasticFunctionCPD_params is None:
                raise ValueError(f"Undefined variable: {variable_name}")

            cpd = self.__diagram.constructStochasticFunctionCPD(
                stochasticFunctionCPD_params)
            try:
                self.__diagram.add_cpds(cpd)
                self.local_cpd_list.append(stochasticFunctionCPD_params)
            except KeyError:
                # TODO add retry loop
                raise NotImplementedError
        # self.debug()

    # Tabular CPD

    def generateTabularCPDForVariable(self, variable: str, parent_variable_list: List[str]) -> str:
        """ LLM to generate CPD expression for variable, without supporting corpus"""

        #
        # prompting for P(variable|parent_variable_list)
        #
        generate_template = "Assign the conditional probability table for variable {variable} and each value of the condition {parent_variable_list}. Output should follow the tabular format:\n{format_instructions}"

        parser = JsonOutputParser(pydantic_object=graph.TabularCPD)
        prompt = PromptTemplate(
            template=generate_template,
            input_variables=["variable", "parent_variable_list"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()
            }
        )
        try:
            chain = prompt | self.language_model | parser
            contents = chain.invoke({
                "parent_variable_list": parent_variable_list,
                "variable": variable
            })
        except langchain_core.exceptions.OutputParserException:
            # TODO
            print("ProbabilityAgent: Parsing failed, retrying...")
            fixing_parser = OutputFixingParser.from_llm(
                parser=parser, llm=Gpt4Turbo(), max_retries=self.max_retries)
            completion_chain = prompt | self.language_model
            main_chain = RunnableParallel(
                completion=completion_chain, prompt=prompt
            ) | RunnableLambda(lambda x: fixing_parser.parse_with_prompt(**x))
            contents = main_chain.invoke({
                "parent_variable_list": parent_variable_list,
                "variable": variable
            })
        return contents

    def constructTabularCPD(self, tabularCPD_params: str) -> TabularCPD:
        """ 
        Instantiate a TabularCPD object.
        Parameters
        ----------
        tabularCPD_params: LLM-generated input arguments to TabularCPD __init__ function
        """
        cpd = TabularCPD(
            variable=self.__diagram.node_dict[tabularCPD_params['variable']],
            variable_card=tabularCPD_params['variable_card'],
            values=tabularCPD_params['values'],
            evidence=tabularCPD_params['evidence'],
            evidence_card=tabularCPD_params['evidence_card']
        )
        return cpd

    def assignTabularCPD(self):
        """ add CPD models to the InfluenceDiagram"""
        assert self.__diagram is not None

        for variable_name in tqdm(self.__diagram.getNodeDict().keys()):
            if variable_name not in self.__diagram.getDecompositionDict().keys():
                raise ValueError(f"Undefined variable: {variable_name}")
            else:
                parent_variable_list = self.__diagram.getDecompositionDict()[
                    variable_name]

            # LLM step: generate tabularCPD parameters
            tabularCPD_params = self.generateTabularCPDForVariable(
                variable_name, parent_variable_list)
            tabularCPD_params['variable'] = tabularCPD_params['variable'].replace(
                "_", " ").lower()
            if tabularCPD_params['evidence'] is not None:
                tabularCPD_params['evidence'] = [evidence_name.replace(
                    "_", " ").lower() for evidence_name in tabularCPD_params['evidence']]

            cpd = self.constructTabularCPD(tabularCPD_params)
            try:
                self.__diagram.add_cpds(cpd)
                self.local_cpd_list.append(tabularCPD_params)
            except KeyError:
                # TODO add retry loop
                raise NotImplementedError
        self.debug()

    def debug(self):
        self.__diagram.draw()
        print(self.__diagram.get_cpds())

    def get_joint_factors(self):
        id = self.getDiagram()
        factors = [cpd.to_factor() for cpd in id.get_cpds()]
        # sort factors by variable name
        factors.sort(key=lambda f: f.scope())
        factor_prod = reduce(lambda f1, f2: f1 * f2, factors)
        factor_prod.normalize()
        return factor_prod

    def get_random_joint_factors(self):
        id = self.getDiagram()
        cpds = id.get_random_cpds(id.states)
        factors = [cpd.to_factor() for cpd in cpds]
        factor_prod = reduce(lambda f1, f2: f1 * f2, factors)
        factor_prod.normalize()
        return factor_prod
    
    def distance(self, other=None, method='total_variation'):
        jpd1 = self.get_joint_factors()
        if other is not None:
            jpd2 = other.get_joint_factors()
        else:
            jpd2 = self.get_random_joint_factors()
        if method == 'TVD':
            return total_variation_distance(jpd1, jpd2)
        elif method == 'KL':
            return kl_divergence(jpd1, jpd2)
        elif method == 'JS':
            return js_divergence(jpd1, jpd2)
        elif method == 'BS':
            return brier_score(jpd1, jpd2)
        else:
            raise ValueError(f"Invalid distance method: {method}")

