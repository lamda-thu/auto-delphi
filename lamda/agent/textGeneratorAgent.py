import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import json
from typing import Dict, List, Optional, Tuple

from graph import Node, Edge, NodeList, EdgeList, InfluenceDiagram

from models import Glm4, Gpt4Turbo
import langchain_core
from langchain_core.language_models import LLM


class TextGeneratorAgent:
    """ The agent that generates text from Influence Diagram instance. """
    language_model: LLM = Gpt4Turbo()
    def __init__(
        self, 
        influence_diagram: Optional[InfluenceDiagram],
        cpd_param_list: Optional[List[Dict[str, str]]]=None,
        language_model: LLM = Gpt4Turbo()
    ) -> None:
        """ 
        Generate text from Influence Diagram instance\
            or parameters that can be used to generate an influence diagram.
        The adoption of node_dict, edge_dict, and cpd_param_list for generation facilitates generation from symbolic data (e.g., with placeholders).

        Parameters
        ----------
        influence_diagram: Optional[InfluenceDiagram]
            Influence Diagram instance
        cpd_param_list: Optional[List[Dict[str, str]]]
            List of CPD parameters
            CPDs may not be integrated to the `InfluenceDiagram` object due to *symbolic representations*.
        """
        self.influence_diagram = influence_diagram
        self.cpd_param_list = cpd_param_list
        self.language_model = language_model

    def getDecisionOptions(self) -> Tuple[str, list]:
        """ 
        Get decision options from Influence Diagram instance. 
        The decision variables are in the valid order.
        
        Returns
        -------
        str
            String representation of decision options.
        list[dict]
            List of decision variable names and their valid values.

        Warnings
        --------
        Current version considers a single decision node. 
        Order matters when there are multiple nodes.
        - Consider asking for decision one by one when doing a multi-decision problem.
        """
        if self.influence_diagram is not None:
            __decisions = self.influence_diagram.getDecisions()
            string = ""
            for decision in __decisions:
                string += f"Decision:'{decision['variable_name']}'\nAlternatives:{decision['variable_values']}\n"
            return string, {decision['variable_name']: decision['variable_values'] for decision in __decisions}

    def getInformationVariables(self) -> Tuple[str, list]:
        """ Get information variables from Influence Diagram instance. """
        string = ""
        for decision in self.influence_diagram.getDecisions():
            # get the parent nodes of the decision
            information_variable_list = self.influence_diagram.getDecompositionDict()[decision['variable_name']]
            string += f"Decision:'{decision['variable_name']}':\nInformation Variables:{[node for node in self.influence_diagram.getNodes() if node['variable_name'] in information_variable_list]}\n"
        return string, [{node['variable_name']: node['variable_values']} for node in self.influence_diagram.getNodes() if node['variable_name'] in information_variable_list]
    
    # generate information automatically
    def generateTextFromInfluenceDiagramGraph(self) -> str:
        """ Generate description of decision-making problem from Influence Diagram *Graph* instance"""
        prompt = f"Describe the decision-making problem represented in the Influence Diagram using plain text. The description targets general readers with no expertise in influence diagram or decision analysis, so avoid using decision analysis terminologies such as 'node' or 'edge'. Make sure the descrription is clear and easy to understand. Describe the problem without mentioning the diagram. \n\nInfluence Diagram: {self.influence_diagram}"

        completion = self.language_model.invoke(prompt)
        return completion

    def generateTextFromInfluenceDiagram(self) -> str:
        """ Generate description of decision-making problem from Influence Diagram instance"""
        simplified_cpd_param_list = [{"variable": cpd_param['variable'], "stochastic function cpd": cpd_param['stochastic_function']} for cpd_param in self.cpd_param_list]
        prompt = f"Describe the decision-making problem represented in the Influence Diagram using plain text. The description targets general readers with no expertise in influence diagram or decision analysis, so avoid using decision analysis terminologies such as 'node' or 'edge'. Make sure the descrription is clear and easy to understand. Describe the problem without mentioning the diagram. \n\nInfluence Diagram Graph:\n {self.influence_diagram}\n\n Conditional Probability: \n {simplified_cpd_param_list}. Preserve all relevant quantitative information."
        # TODO: test
        
        completion = self.language_model.invoke(prompt)
        return completion

    def abbreviateText(self, original_text: str) -> str:
        prompt = f"Abbreviate the text description of the decision-making problem. Remove all bullet points and replace them with fluent articulation. Make it shorter while preserving the key information and readability.\n\nText description: {original_text}"

        completion = self.language_model.invoke(prompt)
        return completion


    # specialized generation methods
    def generateBackgroundInformation(self) -> str:
        """
        Generate description of the background of the decision-making problem from Influence Diagram *Graph* instance.
        """
        if self.influence_diagram is None:
            raise ValueError("InfluenceDiagram instance is not provided.")
        else:
            influence_diagram_graph_string = str(self.influence_diagram)
        prompt = f"Describe the background of the decision-making problem represented in the Influence Diagram. Notice that an edge ending in a chance node indicates a probable dependency on the source node of the edge. An edge ending in a decision node indicates the source node is known at the time of that decision.\n\nRequirements:\n\n Do not include detailed description of factors to be considered and relations between them. Use concise plain language. The description targets general readers with no expertise in influence diagram or decision analysis, so avoid using decision analysis terminologies such as 'node' or 'edge', and do not mention the influence diagram. Make sure the description is clear and easy to understand.\n\nInfluence Diagram:\n\n{influence_diagram_graph_string}"

        completion = self.language_model.invoke(prompt)
        return completion
    
    def generateNodeInformation(self) -> str:
        node_list = self.influence_diagram.getNodes()
        prompt = f"Describe the factors to be considered in the decision-making problem.\n\nRequirements:\n\nDo not mention relations between the factors. Use concise plain language. The description targets general readers with no expertise in influence diagram or decision analysis, so avoid using decision analysis terminologies such as 'node' or 'edge', and do not mention the influence diagram. Make sure the description is clear and easy to understand.\n\nInfluence Diagram:\n\n{self.influence_diagram} \n\nFactors:\n\n{[{'variable_name': node['variable_name'], 'variable_type': node['variable_type']} for node in node_list]}"

        completion = self.language_model.invoke(prompt)
        return completion

    def generateGraphInformation(self) -> str:
        node_list = self.influence_diagram.getNodes()
        edge_list = self.influence_diagram.getEdgeNameList()
        prompt = f"Describe the factors to be considered and the relations between the factors for the decision-making problem. Notice that an edge ending in a chance node or a utility node indicates a potential dependency on the source node of the edge. An edge ending in a decision node indicates the source node is known at the time of that decision.\n\nRequirements:\n\nUse concise plain language. The description targets general readers with no expertise in influence diagram or decision analysis, so avoid using decision analysis terminologies such as 'node' or 'edge', and do not mention the influence diagram. Make sure the description is clear and easy to understand.\n\nInfluence Diagram:\n\n{self.influence_diagram} \n\nFactors:\n\n{[{'variable_name': node['variable_name'], 'variable_type': node['variable_type']} for node in node_list]}\n\nRelations:\n\n{edge_list}"

        completion = self.language_model.invoke(prompt)
        return completion

    def generateCpdInformation(self) -> str:
        simplified_cpd_param_list = [{"variable": cpd_param['variable'], "stochastic function cpd": cpd_param['stochastic_function'], "evidence": cpd_param.get('evidence', None)} for cpd_param in self.cpd_param_list]
        prompt = f"Describe the quantitative relations between variables presented in the Conditional Probability list. Each element of the list corresponds to a variable, with the stochastic function describing the conditional probability distribution. Make sure to preserve all relevant quantitative information. If placeholders (e.g., #cost#) presents, just use it as a numerical value. \n\nRequirements:\n\n Make sure all the quantitative information is included in the description. Make sure the description is clear and easy to understand.\n\n Conditional Probability list:\n\n{simplified_cpd_param_list}"

        completion = self.language_model.invoke(prompt)
        return completion

    # controller method for the specialized generation methods
    def generateTextFromRule(self, level: str) -> str:
        """ 
        Parameters
        ----------
        level : str
            The level of information for text generation. Values can be:
            "background", "node", "graph", "quanlitative CPD", "complete"

        Returns
        -------
        str
            Text description of the decision-making problem.
            The level of specification depends on the `level` parameter.
        """

        if level not in ["background", "node", "graph", "quanlitative CPD", "complete"]:
            raise ValueError(f"Invalid level {level}.")
        
        string = ""
        if level == "background":
            string = "Background:\n" + self.generateBackgroundInformation()
        elif level == "node":
            #string = "Background:\n" + self.generateBackgroundInformation()
            string += "Factors to be considered:\n" + self.generateNodeInformation()
        elif level == "graph":
            #string = "Background:\n" + self.generateBackgroundInformation()
            string += "Factors to be considered and their relations:\n" +self.generateGraphInformation()
        elif level == "complete":
            # string = "Background:\n" + self.generateBackgroundInformation()+"\n\n"
            # string += "Factors to be considered:\n" + self.generateNodeInformation()+"\n\n"
            # string += "Factors to be considered and their relations:\n" +self.generateGraphInformation()+"\n\n"
            string += "Conditional probability distributions:\n" + self.generateCpdInformation()+"\n\n"
        else:
            raise NotImplementedError

        return string