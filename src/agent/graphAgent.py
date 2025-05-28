import os
import sys

import langchain_core.exceptions

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import warnings
from typing import List, Dict, Optional, Tuple
from models import Kimi, Llama3, Glm4, Gemma2, DeepseekCoder, Mixtral, Gpt4, Gpt4Turbo, Deepseek_r1
from graph import Node, Edge, NodeList, EdgeList, NodeAndEdgeList, InfluenceDiagram
from enum import Enum
import json
from pydantic import BaseModel, Field

import langchain_core
from langchain_core.language_models import LLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
import re

from utils import getCurrentTimeForFileName
from mode import GenerationMode
from prompt import PROMPTS

class GraphAgent:
    temperature: float = 0.0 #TODO: pass temperature arg to LLMs
    language_model: LLM = Gpt4Turbo()
    mode: GenerationMode = GenerationMode.extract
    def __init__(
        self,
        language_model: LLM,
        max_retries: int = 5,
        mode: GenerationMode = GenerationMode.extract
    ): 
        self.language_model = language_model
        self.max_retries = max_retries
        self.mode = mode

        # cache the latest node_list and edge_list
        self.node_list: Optional[List[Dict[str, str]]] = []
        self.edge_list: Optional[List[Dict[str, str]]] = []

    def reset(self):
        self.node_list = []
        self.edge_list = []

    def setMode(self, newMode: GenerationMode):
        self.mode = newMode

    def saveNodeList(self, output_dir):
        node_file_name = os.path.join(output_dir, "nodes.json")
        with open(node_file_name, 'w', encoding='utf-8') as f:
            json.dump(self.node_list, f, indent=4, ensure_ascii=False)

    def saveEdgeList(self, output_dir):
        edge_file_name = os.path.join(output_dir, "edges.json")
        with open(edge_file_name, 'w', encoding='utf-8') as f:
            json.dump(self.edge_list, f, indent=4, ensure_ascii=False)
    
    def saveNodeAndEdgeList(self, output_dir: str):
        """ project_root/output
        
        Warning
        -------
        The output dir will be given a time stamp.
        """
        output_dir_with_time = os.path.join(output_dir, getCurrentTimeForFileName())
        if not os.path.exists(output_dir_with_time):
            os.makedirs(output_dir_with_time)

        self.saveNodeList(output_dir_with_time)
        self.saveEdgeList(output_dir_with_time)
    
    def loadNodeAndEdgeList(self, input_dir: str):
        # print(input_dir)
        assert os.path.exists(input_dir)
        node_file_name = os.path.join(input_dir, "nodes.json")
        edge_file_name = os.path.join(input_dir, "edges.json")
        with open(node_file_name, 'r', encoding='utf-8') as f:
            self.node_list = json.load(f)
        with open(edge_file_name, 'r', encoding='utf-8') as f:
            self.edge_list = json.load(f)

    def getNodeNameList(self)-> List[str]:
        return [node["variable_name"] for node in self.node_list]

    def setVariableNamesLowercase(self):
        """ Set the variable names to lowercase. """
        if self.node_list is None:
            raise ValueError("Node list is None.")
        for node in self.node_list:
            if "variable_name" in node:
                node["variable_name"] = node["variable_name"].lower()
        for edge in self.edge_list:
            if "cause" in edge:
                edge["cause"] = edge["cause"].lower()
            if "effect" in edge:
                edge["effect"] = edge["effect"].lower()

    def constructGraph(self) -> InfluenceDiagram:
        try:
            self.setVariableNamesLowercase()
            return InfluenceDiagram(self.node_list, self.edge_list)
        except KeyError as e:
            # TODO
            issue = f"Node {e} does not exist. Make remedies by modifying related node names in the edge list."
            #print(issue)

    def verifyGraph(self) -> Tuple[List[str], List[str]]:
        """ Verify the current graph is valid. """
        node_issues = self.verifyNodeList()
        edge_issues = self.verifyEdgeList()

        if len(node_issues) > 0 or len(edge_issues) > 0:
            issues = node_issues + edge_issues
            print(f"VerifyGraph: Found {len(issues)} issues.")
        return node_issues, edge_issues

    def fixGraph(self, node_issues: List[str], edge_issues: List[str], source_text: str, joint_fix: bool = False) -> None:
        """ Fix the graph based on the source text and reflection."""
        if joint_fix == False:
            if len(node_issues) > 0:
                self.improveNodeList(source_text=source_text, issues=node_issues, reflection="")
            if len(edge_issues) > 0:
                self.improveEdgeList(source_text=source_text, issues=edge_issues, reflection="")
        else:
            self.improveJointGeneration(source_text=source_text, node_issues=node_issues, edge_issues=edge_issues, reflection="")

    def verifyAndFixGraph(self, source_text: str, joint_fix: bool = False) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Returns
        -------
        Tuple[List[Dict[str, str]], List[Dict[str, str]]]
            The fixed node list and edge list.
        """
        node_issues, edge_issues = self.verifyGraph()
        if len(node_issues) == 0 and len(edge_issues) == 0:
            return self.node_list, self.edge_list
        else:
            for _ in range(self.max_retries):
                self.fixGraph(node_issues, edge_issues, source_text, joint_fix)
                node_issues, edge_issues = self.verifyGraph()
                if len(node_issues) == 0 and len(edge_issues) == 0:
                    break
            return self.node_list, self.edge_list

    def getLanguageModel(self):
        return self.language_model
    
    #
    # Joint generation
    # 
    def jointGeneration(self, source_text: Optional[str] = None, joint_fix: bool = False) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """Generate a graph from a source text."""
        if self.mode == GenerationMode.generate:
            prompt_template = PROMPTS["joint_generation"]
        elif self.mode == GenerationMode.extract:
            prompt_template = PROMPTS["joint_extraction"]
        else:
            raise ValueError(f"ValueError:Invalid mode for joint generation: {self.mode}")

        parser = JsonOutputParser(pydantic_object=NodeAndEdgeList)

        prompt = PromptTemplate(
            template=prompt_template,
            partial_variables={
                "node_list": self.node_list,
                "edge_list": self.edge_list,
                "format_instructions": parser.get_format_instructions(),
                "text": source_text if source_text is not None else ""
            }
        )
        
        try:
            # First get completion from the language model
            completion = self.language_model.invoke(prompt.invoke({}))
            
            # Preprocess the JSON expressions before parsing
            processed_completion = self.preprocessJsonExpressions(completion)
            
            # If preprocessing returned a dictionary, use it directly
            if isinstance(processed_completion, dict):
                contents = processed_completion
            else:
                # Otherwise, parse the processed string
                contents = parser.parse(processed_completion)
            
            node_list, edge_list = self.removeRedundancyForJointGeneration(contents["node_list"], contents["edge_list"])

            self.node_list.extend(node_list)
            self.edge_list.extend(edge_list)
            
            self.verifyAndFixGraph(source_text=source_text, joint_fix=joint_fix)
            self.setVariableNamesLowercase()
            return self.node_list, self.edge_list
        except Exception as e:
            #print(f"Parsing failed: {str(e)}, retrying...")
            fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=self.language_model, max_retries=self.max_retries)
            
            # Process the completion to handle expressions before parsing
            def get_completion_and_preprocess():
                completion = self.language_model.invoke(prompt.invoke({}))
                return self.preprocessJsonExpressions(completion)
            
            try:
                processed_completion = get_completion_and_preprocess()
                
                # If preprocessing returned a dictionary, use it directly
                if isinstance(processed_completion, dict):
                    contents = processed_completion
                else:
                    # Otherwise, try parsing with the fixing parser
                    contents = fixing_parser.parse(processed_completion)
                
                node_list, edge_list = self.removeRedundancyForJointGeneration(contents["node_list"], contents["edge_list"])

                self.node_list.extend(node_list)
                self.edge_list.extend(edge_list)
                self.setVariableNamesLowercase()
                return contents["node_list"], contents["edge_list"]
            except Exception as e2:
                print(f"Second parsing attempt failed: {str(e2)}")
                # Last resort: Return empty lists to avoid crashing
                return self.node_list, self.edge_list

    def removeRedundantNodes(self, node_list: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """ Remove nodes that are present in self.node_list from the given node_list. """
        if self.node_list is None or len(self.node_list) == 0:
            return node_list
        prompt_template = PROMPTS["remove_redundant_nodes"]
        parser = JsonOutputParser(pydantic_object=NodeList)
        prompt = PromptTemplate(
            template=prompt_template,
            partial_variables={
                "node_list": node_list,
                "original_node_list": self.node_list,
                "format_instructions": parser.get_format_instructions()
            }
        )
        chain = prompt | self.language_model | parser
        try:
            contents = chain.invoke({})
            return contents["node_list"]
        except Exception as e:
            print(e)
            for _ in range(self.max_retries):
                contents = chain.invoke({})
                return contents["node_list"]
            return node_list

    def removeRedundancyForJointGeneration(self, node_list: List[Dict[str, str]], edge_list: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Remove redundant nodes and edges from the given node list and edge list.
        
        Returns
        -------
        Tuple[List[Dict[str, str]], List[Dict[str, str]]]
            The reduced node list and edge list.
        """
        if self.node_list is None or len(self.node_list) == 0:
            return node_list, edge_list
        print(self.node_list)
        node_list = self.removeRedundantNodes(node_list)
        combined_node_list = node_list + self.node_list
        node_names = [node.get("variable_name") for node in combined_node_list]
        assert len(node_names) > 0, "Combined node list is empty." 
        print(len(edge_list))
        if len(self.edge_list) > 0:
            edge_list = [
                edge for edge in edge_list
                if edge["cause"] in node_names and edge["effect"] in node_names
            ]
        return node_list, edge_list

    def reflectJointGeneration(self, source_text: str) -> str:
        """Reflect on the current graph and return a summary of improvement suggestions."""
        if self.mode == GenerationMode.extract:
            prompt_template = PROMPTS["joint_extraction_reflection"]
        elif self.mode == GenerationMode.generate:
            prompt_template = PROMPTS["joint_generation_reflection"]
        else:
            raise NotImplementedError

        prompt = PromptTemplate(
            template=prompt_template,
            partial_variables={
                "node_list": self.node_list,
                "edge_list": self.edge_list,
                "text": source_text
            }
        )
        chain = prompt | self.language_model
        contents = chain.invoke({})
        return contents
    
    def preprocessJsonExpressions(self, completion):
        """Pre-process mathematical expressions in JSON string to evaluate them before parsing."""
        # Handle non-string inputs
        if not isinstance(completion, str):
            # If it's already parsed JSON data, return it as is
            if isinstance(completion, dict):
                return completion
            # For other types, convert to string or return empty dict
            try:
                completion = str(completion)
            except:
                print(f"Error: Cannot preprocess non-string input of type {type(completion)}")
                return {"node_list": [], "edge_list": []}
            
        # Extract the JSON string from markdown if present
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', completion)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = completion
            
        # Find all mathematical expressions in utility node values
        pattern = r'"variable_values": \[(.*?)\]'
        
        def replace_expressions(match):
            values_str = match.group(1)
            try:
                # Split by commas, but not those within expressions
                values = []
                current = ""
                paren_count = 0
                
                for char in values_str:
                    if char == '(' or char == '[':
                        paren_count += 1
                    elif char == ')' or char == ']':
                        paren_count -= 1
                    
                    if char == ',' and paren_count == 0:
                        values.append(current.strip())
                        current = ""
                    else:
                        current += char
                
                if current:
                    values.append(current.strip())
                
                # Evaluate expressions
                evaluated_values = []
                for val in values:
                    val = val.strip()
                    if val.startswith('"') or val.startswith("'") or val in ["true", "false", "null"]:
                        # It's a string or literal, keep as is
                        evaluated_values.append(val)
                    else:
                        try:
                            # Safely evaluate mathematical expressions
                            evaluated = eval(val, {"__builtins__": {}})
                            evaluated_values.append(str(evaluated))
                        except:
                            # If evaluation fails, keep the original
                            evaluated_values.append(val)
                
                return '"variable_values": [' + ', '.join(evaluated_values) + ']'
            except Exception as e:
                # If anything goes wrong, return the original
                #print(f"Error evaluating expression: {e}")
                return match.group(0)
        
        processed_json = re.sub(pattern, replace_expressions, json_str)
        
        # If we extracted from markdown, put it back
        if json_match and not processed_json.startswith('{'):
            processed_json = f"```json\n{processed_json}\n```"
            
        # Try to parse the processed JSON
        try:
            return json.loads(processed_json)
        except json.JSONDecodeError:
            # If parsing fails, return the processed string
            return processed_json

    def improveJointGeneration(self, source_text: str, node_issues: List[str], edge_issues: List[str], reflection: str) -> InfluenceDiagram:
        """Improve the current graph based on the reflection."""
        prompt_template = PROMPTS["joint_improvement"]
        parser = JsonOutputParser(pydantic_object=NodeAndEdgeList)
        prompt = PromptTemplate(
            template=prompt_template,
            partial_variables={
                "source_text": source_text,
                "node_list": self.node_list,
                "edge_list": self.edge_list,
                "issues": node_issues + edge_issues,
                "reflection": reflection,
                "format_instructions": parser.get_format_instructions()
            }
        )
        chain = prompt | self.language_model | parser
        try:
            contents = chain.invoke({})
            # Preprocess the JSON expressions
            processed_contents = self.preprocessJsonExpressions(contents)
            self.node_list = processed_contents["node_list"]
            self.edge_list = processed_contents["edge_list"]
            self.setVariableNamesLowercase()

        except Exception as e:
            #print(f"Parsing failed: {str(e)}, retrying...")
            fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=self.language_model, max_retries=self.max_retries)
            completion_chain = prompt | self.language_model
            
            # Process the completion to handle expressions before parsing
            def preprocess_and_parse(x):
                completion = x["completion"]
                preprocessed = self.preprocessJsonExpressions(completion)
                return fixing_parser.parse_with_prompt(completion=preprocessed, prompt=x["prompt"])
            
            main_chain = RunnableParallel(
                completion=completion_chain, prompt=prompt
            ) | RunnableLambda(lambda x: preprocess_and_parse(x))
            
            contents = main_chain.invoke({})
            processed_contents = self.preprocessJsonExpressions(contents)
            self.node_list = processed_contents["node_list"]
            self.edge_list = processed_contents["edge_list"]
            self.setVariableNamesLowercase()
        return self.node_list, self.edge_list

    #
    # Node
    # 
    def extendNodeList(self, source_text: str) -> List[Dict[str, str]]:
        """Extend the existing NodeList with new context or corpus."""
        if self.mode == GenerationMode.extract:
            prompt_template = PROMPTS["node_extraction"]
        elif self.mode == GenerationMode.generate:
            prompt_template = PROMPTS["node_generation"]
        else:
            raise NotImplementedError
        parser = JsonOutputParser(pydantic_object=NodeList)
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["text"],
            partial_variables={
                "node_list": self.node_list,
                "format_instructions": parser.get_format_instructions()},
        )

        try:
            # First get completion from the language model
            completion = self.language_model.invoke(prompt.invoke({"text": source_text}))
            
            # Preprocess the JSON expressions before parsing
            processed_completion = self.preprocessJsonExpressions(completion)
            
            # If preprocessing returned a dictionary, use it directly
            if isinstance(processed_completion, dict):
                contents = processed_completion
            else:
                # Otherwise, parse the processed string
                contents = parser.parse(processed_completion)
            
            node_list = self.removeRedundantNodes(contents['node_list'])
            self.node_list.extend(node_list)

            issues = self.verifyNodeList()
            retry_count = 0
            while len(issues) > 0 and retry_count < self.max_retries:
                retry_count += 1
                print(f"Retrying {retry_count} times in node list extension.")
                self.improveNodeList(source_text=source_text, issues=issues, reflection="")
                issues = self.verifyNodeList()
        except Exception as e:
            #print(f"Parsing failed: {str(e)}, retrying...")
            fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=self.language_model, max_retries=self.max_retries)
            
            # Process the completion to handle expressions before parsing
            def get_completion_and_preprocess():
                completion = self.language_model.invoke(prompt.invoke({"text": source_text}))
                return self.preprocessJsonExpressions(completion)
            
            try:
                processed_completion = get_completion_and_preprocess()
                
                # If preprocessing returned a dictionary, use it directly
                if isinstance(processed_completion, dict):
                    contents = processed_completion
                else:
                    # Otherwise, try parsing with the fixing parser
                    contents = fixing_parser.parse(processed_completion)
                
                node_list = self.removeRedundantNodes(contents['node_list'])
                self.node_list.extend(node_list)
                self.setVariableNamesLowercase()
                return contents['node_list']
            except Exception as e2:
                print(f"Second parsing attempt failed: {str(e2)}")
                # Last resort: Return empty list to avoid crashing
                return []
                
        # If no exceptions occurred, return the result
        self.setVariableNamesLowercase()
        return contents['node_list']

    def verifyNodeList(self) -> List[str]:
        """ Verify the current node list is valid. """
        issues = []

        # check required fields
        required_fields = ["variable_name", "variable_type", "variable_values"]
        for node in self.node_list:
            for field in required_fields:
                if field not in node:
                    issues.append(f"- ValueError: Node {node.get('variable_name')} missing the required field: {field}. Modify the node to include the required field.\n")

        # check node types
        valid_types = ["chance", "decision", "utility"]
        for node in self.node_list:
            variable_type = node.get("variable_type")
            if variable_type is not None and variable_type not in valid_types:
                issues.append(f"- TypeError: Node {node.get('variable_name')} has an invalid type: {variable_type}. Modify the type to be one of the following: {valid_types}\n")
         
        # check utility nodes are non-empty
        utility_nodes = [node.get("variable_name") for node in self.node_list if node.get("variable_type") == "utility"]
        if len(utility_nodes) == 0:
            issues.append("- ValueError: No utility nodes found in the node list. Consider identifying the utility nodes in the source text.\n")

        # check decision nodes are non-empty
        decision_nodes = [node.get("variable_name") for node in self.node_list if node.get("variable_type") == "decision"]
        if len(decision_nodes) == 0:
            issues.append("- ValueError: No decision nodes found in the node list. Consider identifying the decision nodes in the source text.\n")

        # check node values
        for node in self.node_list:
            if "variable_values" not in node:
                continue
            if len(node.get("variable_values", [])) < 2:
                issues.append(f"- ValueError: Node {node.get('variable_name')} has less than 2 values: {node.get('variable_values')}. Modify the node by identifying at least 2 valid values it can take.\n")

        # check utility node values are numerical
        for node in self.node_list:
            if node.get("variable_type") == "utility":
                for value in node.get("variable_values", []):
                    if not isinstance(value, (int, float)):
                        issues.append(
                            f"- ValueError: Utility node {node.get('variable_name')} has a non-numerical value: {value}.\
                                Modify the node by identifying valid numerical values.\n")
        return issues

    def reflectNodeList(self, source_text: str) -> str:
        """ reflect on the current self.node_list, return a summary of improvement suggestions
        
        Use `source_text` as supporting text for reflection. Reflect on the following issues:
        - add missing nodes that are mentioned in `source_text`
        - remove redundant nodes from the node list
        - make sure node types are proper
        """

        if self.mode == GenerationMode.extract:
            reflect_template = PROMPTS["node_extraction_reflection"]
            prompt = PromptTemplate(
                template=reflect_template,
                partial_variables={
                    "node_list": self.node_list,
                    "source_text": source_text
                }
            )
            chain = prompt | self.language_model
            contents = chain.invoke({})
            return contents
        else:
            raise NotImplementedError

    def improveNodeList(self, source_text: str, issues: list, reflection: str) -> List[Dict[str, str]]:
        """ Modify the node list based on the source text and reflection.
        
        Warning
        -------
        This function does not depend on the mode.
        """
        extract_template = PROMPTS["node_improvement"]
        parser = JsonOutputParser(pydantic_object=NodeList)
        prompt = PromptTemplate(
            template=extract_template,
            input_variables=["reflection"],
            partial_variables={
                "source_text":source_text,
                "node_list": self.node_list,
                "issues": issues,
                "format_instructions": parser.get_format_instructions()},
        )
        try:
            chain = prompt | self.language_model | parser
            contents = chain.invoke({"reflection": reflection})
            _node_list = contents['node_list']
            
        except langchain_core.exceptions.OutputParserException:
            print("Parsing failed, retrying..")
            fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=self.language_model, max_retries=self.max_retries)
            completion_chain = prompt | self.language_model
            main_chain = RunnableParallel(
                completion=completion_chain, prompt=prompt
            ) | RunnableLambda(lambda x: fixing_parser.parse_with_prompt(**x))
            contents = main_chain.invoke({"reflection": reflection})

        self.node_list = contents["node_list"]
        self.setVariableNamesLowercase()
        return contents["node_list"]  

    
    # 
    # Edge
    #
    def extendEdgeList(self, source_text: str) -> List[Dict[str, str]]:
        if self.mode == GenerationMode.extract:
            if len(self.node_list) == 0:
                raise UserWarning("Generating edge with no nodes.")
            prompt_template = PROMPTS["edge_extraction"]
        elif self.mode == GenerationMode.generate:
            prompt_template = PROMPTS["edge_generation"]
        else:
            raise NotImplementedError
        parser = JsonOutputParser(pydantic_object=EdgeList)
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["text"],
            partial_variables={
                "format_instructions": parser.get_format_instructions(),
                "node_list": self.node_list,
                "edge_list": self.edge_list
            }
        )
        try:
            chain = prompt | self.language_model | parser
            contents = chain.invoke({"text": source_text})

            self.edge_list.extend(contents["edge_list"])

            issues = self.verifyEdgeList()
            retry_count = 0
            while len(issues) > 0 and retry_count < self.max_retries:
                retry_count += 1
                print(f"Retrying {retry_count} times in edge list extension.")
                self.improveEdgeList(source_text=source_text, issues=issues, reflection="")
                issues = self.verifyEdgeList()
        except langchain_core.exceptions.OutputParserException:
            print("Parsing failed, retrying..")
            fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=self.language_model, max_retries=3)
            completion_chain = prompt | self.language_model
            main_chain = RunnableParallel(
                completion=completion_chain, prompt=prompt
            ) | RunnableLambda(lambda x: fixing_parser.parse_with_prompt(**x))
            contents = main_chain.invoke({"text": source_text})

        return contents["edge_list"]


    def verifyEdgeList(self) -> List[str]:
        """ Verify the current edge list is valid. """
        issues = []
        
        required_fields = ["cause", "effect"]
        for edge in self.edge_list:
            for field in required_fields:
                if field not in edge:
                    issues.append(f"- ValueError: Edge missing the required field: {field}. Modify the edge to include the required field.\n")

        # 1. edge list consistent with node list
        node_name_list = self.getNodeNameList()
        for edge in self.edge_list:
            cause = edge.get("cause")
            effect = edge.get("effect")
            if cause not in node_name_list and cause is not None:
                issues.append(f"- ValueError: In edge ({cause}, {effect}), cause variable {cause} does not exist in the node list. Consider modifying the cause variable name to be consistent with the node list.\n")
            if effect not in node_name_list and effect is not None:
                issues.append(f"- ValueError: In edge ({cause}, {effect}), effect variable {effect} does not exist in the node list. Consider modifying the effect variable name to be consistent with the node list.\n")
        
        # 2. edge list should be acyclic
        graph = {} # Build a directed graph representation
        for edge in self.edge_list:
            cause = edge.get("cause")
            effect = edge.get("effect")
            if cause is not None:
                if cause not in graph:
                    graph[cause] = []
                graph[cause].append(effect)
            if effect is not None:
                if effect not in graph:
                    graph[effect] = []
        
        # Detect cycles using DFS and report the cycle
        def find_cycle(node, visited, path, cycles_found):
            visited.add(node)
            path.append(node)
            
            # Visit all neighbors
            for neighbor in graph.get(node, []):
                if neighbor in path:
                    # Found a cycle
                    cycle = path[path.index(neighbor):] + [neighbor]
                    # Create a canonical representation of the cycle to avoid duplicates
                    # Sort the cycle to start with the lexicographically smallest node
                    min_idx = cycle.index(min(cycle))
                    canonical_cycle = cycle[min_idx:] + cycle[:min_idx]
                    cycle_str = " -> ".join(canonical_cycle)
                    if cycle_str not in cycles_found:
                        cycles_found.add(cycle_str)
                        return cycle_str
                elif neighbor not in visited:
                    cycle = find_cycle(neighbor, visited, path, cycles_found)
                    if cycle:
                        return cycle
            
            # Remove node from path as we're done exploring it
            path.pop()
            return None
        
        visited = set()
        cycles_found = set()
        
        for node in graph:
            if node not in visited:
                cycle = find_cycle(node, visited, [], cycles_found)
                if cycle:
                    issues.append(f"- CycleError: The edge list contains a cycle: {cycle}\n")

        return issues
    
    def reflectEdgeList(self, source_text: str) -> str:
        """ reflect on the current `self.edge_list`, return a summary of improvement suggestions
        
        Use `source_text` as supporting text for reflection. Reflect on the following issues:

        - add missing edges (probabilistic dependences) that are mentioned or implied in `source_text`
        - remove information edges that do not exist
        - remove edges between nodes that are conditionally independent
        - modify edges by changing the direction of edges
        """
        
        reflect_template = PROMPTS["edge_extraction_reflection"]
        prompt = PromptTemplate(
            template=reflect_template,
            partial_variables={
                "source_text": source_text,
                "node_list": self.node_list,
                "edge_list": self.edge_list
            }
        )
        chain = prompt | self.language_model
        contents = chain.invoke({})
        return contents
    
    def improveEdgeList(self, source_text: str, issues: list, reflection: str) -> List[Dict[str, str]]:
        """ Modify the edge list based on the source text and reflection"""
        extract_template = PROMPTS["edge_improvement"]
        
        parser = JsonOutputParser(pydantic_object=EdgeList)
        prompt = PromptTemplate(
            template=extract_template,
            input_variables=["reflection"],
            partial_variables={
                "source_text": source_text,
                "node_list": self.node_list,
                "edge_list": self.edge_list,
                "issues": issues,
                "format_instructions": parser.get_format_instructions()
            }
        )
        try:
            chain = prompt | self.language_model | parser
            contents = chain.invoke({"reflection": reflection})
            _edge_list = contents["edge_list"]
        except: # langchain_core.exceptions.OutputParserException
            print("Parsing failed, retrying..")
            fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=self.language_model, max_retries=self.max_retries)
            completion_chain = prompt | self.language_model
            main_chain = RunnableParallel(
                completion=completion_chain, prompt=prompt
            ) | RunnableLambda(lambda x: fixing_parser.parse_with_prompt(**x))
            contents = main_chain.invoke({"reflection": reflection})
        self.edge_list = contents["edge_list"]
        return contents["edge_list"]
