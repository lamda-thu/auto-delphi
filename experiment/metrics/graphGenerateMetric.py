"""
This file contains metrics for evaluating graph generation results.
It includes metrics for counting extended nodes, evaluating node relevance,
and comparing the quality of generated graphs against original graphs.
"""

import os
import sys
import random
import json
from typing import Dict, Tuple, List, Any
import re

# Add project root to path
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from src.graph import NodeList, EdgeList
from src.models import Gpt4Turbo
from src.prompt import PROMPTS

# Import Langchain components
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

def countExtendedNodes(original_nodes: NodeList, generated_nodes: NodeList) -> Tuple[int, int]:
    """
    Count the number of extended nodes in the generated graph compared to the original one.
    
    Parameters:
    -----------
    original_nodes: NodeList
        The original node list
    generated_nodes: NodeList
        The generated node list
        
    Returns:
    --------
    Tuple[int, int]: The number of extended nodes and the number of original nodes
    """
    original_node_list = original_nodes.getNodeList()
    generated_node_list = generated_nodes.getNodeList()
    
    return len(generated_node_list) - len(original_node_list), len(original_node_list)

class NodeRelevanceResults(BaseModel):
    """Model for node relevance evaluation results."""
    relevance: Dict[str, bool] = Field(description="A dictionary mapping node names to relevance (True/False)")

def evaluateNodeRelevance(text: str, nodes: NodeList, llm=Gpt4Turbo(), max_retries: int = 3) -> Dict[str, bool]:
    """
    Evaluate the relevance of all nodes in a single LLM call using Langchain.
    
    Parameters:
    -----------
    text: str
        The decision context text
    nodes: NodeList
        The node list to evaluate
    llm:
        The LLM to use for evaluation
    max_retries: int
        Maximum number of retries for parsing failures
        
    Returns:
    --------
    Dict[str, bool]: A dictionary mapping node names to relevance (True/False)
    """
    # Format all nodes as a string
    node_list = nodes.getNodeList()
    nodes_str = "\n".join([
        f"Node {i+1}: Name: {node['variable_name']}, Type: {node['variable_type']}, Values: {node['variable_values']}" 
        for i, node in enumerate(node_list)
    ])
    
    parser = PydanticOutputParser(pydantic_object=NodeRelevanceResults)
    prompt = PromptTemplate(
        template=PROMPTS["node_relevance_evaluation"],
        input_variables=["text", "nodes"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        }
    )

    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({"text": text, "nodes": nodes_str})
        return result.relevance
    except Exception as e:
        for _ in range(max_retries):
            try:
                result = chain.invoke({"text": text, "nodes": nodes_str})
                return result.relevance
            except Exception as e:
                continue
        
        raise Exception(f"evaluateNodeRelevance failed after {max_retries} retries")

class GraphComparisonResult(BaseModel):
    """Model for graph comparison results."""
    preferred: str = Field(description="The preferred graph, either 'A' or 'B'")
    explanation: str = Field(description="Explanation for the preference")

def compareGraphs(
    text: str, 
    original_nodes: NodeList, 
    original_edges: EdgeList,
    generated_nodes: NodeList, 
    generated_edges: EdgeList,
    llm=Gpt4Turbo(),
    max_retries: int = 3
) -> Tuple[str, str]:
    """
    Compare the original and generated graphs using an LLM with Langchain.
    
    Parameters:
    -----------
    text: str
        The decision context text
    original_nodes: NodeList
        The original node list
    original_edges: EdgeList
        The original edge list
    generated_nodes: NodeList
        The generated node list
    generated_edges: EdgeList
        The generated edge list
    llm:
        The LLM to use for evaluation
    max_retries: int
        Maximum number of retries for parsing failures
        
    Returns:
    --------
    Tuple[str, str]: The preferred graph (either "original" or "generated") and the explanation
    """
    # Convert nodes and edges to string representations
    node_list_original = original_nodes.getNodeList()
    node_list_generated = generated_nodes.getNodeList()
    edge_list_original = original_edges.getEdgeList()
    edge_list_generated = generated_edges.getEdgeList()

    nodes_original_str = "\n".join([f"- {node['variable_name']} ({node['variable_type']}): {node['variable_values']}" 
                                   for node in node_list_original])
    edges_original_str = "\n".join([f"- {edge['cause']} -> {edge['effect']}" 
                                for edge in edge_list_original])
    
    nodes_generated_str = "\n".join([f"- {node['variable_name']} ({node['variable_type']}): {node['variable_values']}" 
                                    for node in node_list_generated])
    edges_generated_str = "\n".join([f"- {edge['cause']} -> {edge['effect']}" 
                                    for edge in edge_list_generated])
    
    # Randomize the order of presentation
    if random.random() < 0.5:
        mapping = {"A": "original", "B": "generated"}
        inputs = {
            "text": text,
            "nodes_a": nodes_original_str,
            "edges_a": edges_original_str,
            "nodes_b": nodes_generated_str,
            "edges_b": edges_generated_str
        }
    else:
        mapping = {"A": "generated", "B": "original"}
        inputs = {
            "text": text,
            "nodes_a": nodes_generated_str,
            "edges_a": edges_generated_str,
            "nodes_b": nodes_original_str,
            "edges_b": edges_original_str
        }
    
    # Create prompt template
    parser = PydanticOutputParser(pydantic_object=GraphComparisonResult)
    prompt = PromptTemplate(
        template=PROMPTS["graph_comparison"],
        input_variables=["text", "nodes_a", "edges_a", "nodes_b", "edges_b"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        }
    )
    
    # Create and execute chain
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke(inputs)
    except Exception as e:
        for _ in range(max_retries):
            try:
                result = chain.invoke(inputs)
                return result.preferred, result.explanation
            except Exception as e:
                continue
        raise Exception(f"compareGraphs failed after {max_retries} retries")
    
    preferred = mapping.get(result.preferred, "unclear")
    explanation = result.explanation
    
    return preferred, explanation 