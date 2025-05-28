from typing import List, Tuple, Optional, Dict
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from src.models import Gpt4o
from src.graph import NodeList, EdgeList
from src.prompt import PROMPTS


#
# LLM-based node matching
# with direct edge matching
#

async def matchNodes(
    target_node_list: NodeList,
    generated_node_list: NodeList,
    max_retries: int,
    llm = Gpt4o()
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Match nodes in the generated node list to the target node list.

    Returns
    -------
    Tuple[Dict[str, str], Dict[str, str]]
        A tuple of two dictionaries.
        The first dictionary maps generated node names to target node names.
        The second dictionary maps target node names to generated node names.
    """

    class NodeMatchList(BaseModel):
        node_pairs: List[Tuple[str, str]] = Field(description="A list of tuples, each containing a target node name and a generated node name")

    target_nodes = target_node_list.getNodeList()
    target_nodes_names = [node["variable_name"] for node in target_nodes]
    generated_nodes = generated_node_list.getNodeList()
    generated_nodes_names = [node["variable_name"] for node in generated_nodes]

    prompt_template = PROMPTS["node_list_common_nodes"]    
    parser = PydanticOutputParser(pydantic_object=NodeMatchList)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["target_nodes", "generated_nodes"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        }
    )

    chain = prompt | llm | parser
    try:
        response = await chain.ainvoke({
            "target_nodes": target_nodes_names,
            "generated_nodes": generated_nodes_names
        })
        node_pairs = response.node_pairs

        # check if the node pairs are valid
        issues = verifyMatchNodes(target_node_list, generated_node_list, node_pairs)
        if len(issues) > 0:
            node_pairs = await fixNodePairs(target_node_list, generated_node_list, node_pairs, issues, max_retries)
    except Exception as e:
        print(e)
        # when parsing fails, retry
        for _ in range(max_retries):
            print(f"Retrying ... {_ + 1} / {max_retries}")
            try:
                response = await chain.ainvoke({
                    "target_nodes": target_nodes_names,
                    "generated_nodes": generated_nodes_names
                })
                node_pairs = response.node_pairs

                issues = verifyMatchNodes(target_node_list, generated_node_list, node_pairs)
                if len(issues) > 0:
                    node_pairs = await fixNodePairs(target_node_list, generated_node_list, node_pairs, issues, max_retries)
            except Exception as e:
                print(e)
                continue
        raise ValueError("Failed to parse the response")
    
    # Create dictionaries, ensuring no duplicate entries
    generated2target = {}
    target2generated = {}
    
    for node_pair in node_pairs:
        target, generated = node_pair
        # Only add if neither target nor generated is already in the dictionaries
        if target not in target2generated and generated not in generated2target:
            generated2target[generated] = target
            target2generated[target] = generated
    
    assert len(generated2target) == len(target2generated), f"The number of generated2target and target2generated are not equal:\n{generated2target}\n{target2generated}"
    return generated2target, target2generated


async def fixNodePairs(
    target_node_list: NodeList,
    generated_node_list: NodeList,
    node_pairs: List[Tuple[str, str]],
    issues: List[str],
    max_retries: int = 3
) -> List[Tuple[str, str]]:
    """
    Fix issues with node pairs that were identified during verification.
    Uses an LLM to find correct node names that exist in the respective lists.
    Retries until issues are fixed or max_retries is reached.

    Parameters
    ----------
    target_node_list: NodeList
        The target node list.
    generated_node_list: NodeList
        The generated node list.
    node_pairs: List[Tuple[str, str]]
        The original list of node pairs, some of which may have issues.
    issues: List[str]
        A list of identified issues.
    max_retries: int
        Maximum number of retries to fix issues.

    Returns
    -------
    List[Tuple[str, str]]
        A corrected list of node pairs.
    """
    if not issues:
        return node_pairs  # No issues to fix
    
    target_nodes = target_node_list.getNodeList()
    target_nodes_names = [node["variable_name"] for node in target_nodes]
    generated_nodes = generated_node_list.getNodeList()
    generated_nodes_names = [node["variable_name"] for node in generated_nodes]
    
    class NodeMatchList(BaseModel):
        node_pairs: List[Tuple[str, str]] = Field(description="A list of tuples, each containing a target node name and a generated node name")

    prompt_template = PROMPTS["fix_node_pairs"]
    parser = PydanticOutputParser(pydantic_object=NodeMatchList)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["target_nodes", "generated_nodes", "node_pairs", "issues"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        }
    )

    llm = Gpt4o()
    chain = prompt | llm | parser
    
    current_node_pairs = node_pairs
    current_issues = issues
    
    # Keep track of attempts that failed to reduce issues
    consecutive_failures = 0
    
    for retry in range(max_retries + 1):  # +1 for the initial try
        if not current_issues:
            print(f"All issues fixed after {retry} retries.")
            return current_node_pairs
        
        print(f"Fixing node pairs, attempt {retry + 1}/{max_retries}. Issues: {len(current_issues)}")
        
        try:
            response = await chain.ainvoke({
                "target_nodes": target_nodes_names,
                "generated_nodes": generated_nodes_names,
                "node_pairs": current_node_pairs,
                "issues": current_issues
            })
            
            fixed_node_pairs = response.node_pairs
            
            # Verify the fixed node pairs to ensure they're valid
            fixed_node_pairs = [node_pair for node_pair in fixed_node_pairs 
                               if node_pair[0] in target_nodes_names and 
                                  node_pair[1] in generated_nodes_names]
            
            # Ensure uniqueness (each node appears at most once)
            fixed_node_pairs_unique = []
            target_seen = set()
            generated_seen = set()
            for node_pair in fixed_node_pairs:
                if (node_pair[0] not in target_seen and 
                    node_pair[1] not in generated_seen):
                    fixed_node_pairs_unique.append(node_pair)
                    target_seen.add(node_pair[0])
                    generated_seen.add(node_pair[1])
            
            # Add valid pairs from the original list that weren't part of the issues
            valid_original_pairs = []
            for node_pair in current_node_pairs:
                if (node_pair[0] in target_nodes_names and 
                    node_pair[1] in generated_nodes_names and
                    node_pair[0] not in target_seen and
                    node_pair[1] not in generated_seen):
                    valid_original_pairs.append(node_pair)
                    target_seen.add(node_pair[0])
                    generated_seen.add(node_pair[1])
            
            current_node_pairs = fixed_node_pairs_unique + valid_original_pairs
            
            # Check if there are still issues
            current_issues = verifyMatchNodes(target_node_list, generated_node_list, current_node_pairs)
            
            if len(current_issues) >= len(issues):
                consecutive_failures += 1
                if consecutive_failures >= 2:
                    print(f"Failed to reduce issues after {consecutive_failures} consecutive attempts.")
                    break
            else:
                consecutive_failures = 0
                
        except Exception as e:
            print(f"Error in retry {retry + 1}: {e}")
            if retry >= max_retries:
                break
            
    # If we're out of retries or consecutive failures, return the best solution we have
    # Filter out invalid pairs
    final_pairs = []
    target_seen = set()
    generated_seen = set()
    for node_pair in current_node_pairs:
        if (node_pair[0] in target_nodes_names and 
            node_pair[1] in generated_nodes_names and
            node_pair[0] not in target_seen and
            node_pair[1] not in generated_seen):
            final_pairs.append(node_pair)
            target_seen.add(node_pair[0])
            generated_seen.add(node_pair[1])
    
    if current_issues:
        print(f"Warning: {len(current_issues)} issues remain unresolved after {max_retries + 1} attempts.")
        
    return final_pairs

def verifyMatchNodes(
    target_node_list: NodeList,
    generated_node_list: NodeList,
    node_pairs: List[Tuple[str, str]]
) -> List[str]:
    """
    Verify the matched node names exist in the original and target node lists.
    Existence is checked by exact string matching.

    Returns
    -------
    List[str]
        A list of strings, specifying the non-existent node names.
    """
    issues = []
    target_nodes = target_node_list.getNodeList()
    target_nodes_names = [node["variable_name"] for node in target_nodes]
    generated_nodes = generated_node_list.getNodeList()
    generated_nodes_names = [node["variable_name"] for node in generated_nodes]
    for node_pair in node_pairs:
        target, generated = node_pair
        if target not in target_nodes_names:
            issues.append(f"The target node `{target}` does not exist in the target node list.\nCheck for issues or typos.")
        if generated not in generated_nodes_names:
            issues.append(f"The generated node `{generated}` does not exist in the generated node list.\nCheck for issues or typos.")
    return issues
    
def matchEdgesWithNodeMatches(
    target_edge_list: EdgeList,
    generated_edge_list: EdgeList,
    generated2target: Dict[str, str],
    target2generated: Dict[str, str]
) -> Tuple[List[Tuple[dict, dict]], List[Tuple[dict, dict]]]:
    """
    Match edges in the generated edge list to the target edge list.
    This function utiltizes the matched nodes to match the edges.

    Returns
    -------
    Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]
        A tuple of two lists.
        The first list contains the matched edges.
        The second list contains the matched edges with the opposite direction.
    """
    target_edges = target_edge_list.getEdgeList()
    generated_edges = generated_edge_list.getEdgeList()

    matched_edges = []
    matched_edges_with_opposite_direction = []
    for generated_edge in generated_edges:
        cause = generated_edge["cause"]
        effect = generated_edge["effect"]
        if cause in generated2target and effect in generated2target:
            target_cause = generated2target[cause]
            target_effect = generated2target[effect]
            target_edge = {
                "cause": target_cause,
                "effect": target_effect
            }
            if target_edge in target_edges:
                matched_edges.append((target_edge, generated_edge))
            
            target_edge_opposite_direction = {
                "cause": target_effect,
                "effect": target_cause
            }
            if target_edge_opposite_direction in target_edges:
                matched_edges_with_opposite_direction.append((target_edge_opposite_direction, generated_edge))

    return matched_edges, matched_edges_with_opposite_direction


#
# LLM-based node and edge matching
# Nodes and edges are evaluated separately
#

async def nodeListCommonNodesLLM(
    target_node_list: NodeList, 
    generated_node_list: NodeList,
    max_retries: int,
    llm = Gpt4o()
) -> List[Tuple[str, str]]:
    """
    Find common nodes between two NodeList objects.

    Parameters
    ----------
    target_node_list: NodeList
        The target node list.
    generated_node_list: NodeList
        The generated node list.

    Returns
    -------
    List[Tuple[str, str]]
        A list of tuples, each containing a target node name and a generated node name.
    """
    target_nodes = target_node_list.getNodeList()
    target_nodes_names = [node["variable_name"] for node in target_nodes]
    generated_nodes = generated_node_list.getNodeList()
    generated_nodes_names = [node["variable_name"] for node in generated_nodes]

    class NodeMatchList(BaseModel):
        node_pairs: List[Tuple[str, str]] = Field(description="A list of tuples, each containing a target node name and a generated node name")

    prompt_template = PROMPTS["node_list_common_nodes"]    
    parser = PydanticOutputParser(pydantic_object=NodeMatchList)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["target_nodes", "generated_nodes"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        }
    )

    chain = prompt | llm | parser
    try:
        response = await chain.ainvoke({
            "target_nodes": target_nodes_names,
            "generated_nodes": generated_nodes_names
        })
        node_pairs = response.node_pairs
        node_pairs = [node_pair for node_pair in node_pairs if node_pair[0] in target_nodes_names and node_pair[1] in generated_nodes_names]
        # each node appears at most once in the output, check both elements of the pair
        # if it appears multiple times, remove all but the first occurrence
        node_pairs_unique = []
        target_seen = set()
        generated_seen = set()
        for node_pair in node_pairs:
            if node_pair[0] not in target_seen and node_pair[1] not in generated_seen:
                node_pairs_unique.append(node_pair)
                target_seen.add(node_pair[0])
                generated_seen.add(node_pair[1])
        assert len(node_pairs_unique) <= len(target_nodes_names), "The number of node pairs is greater than the number of target nodes"
        assert len(node_pairs_unique) <= len(generated_nodes_names), "The number of node pairs is greater than the number of generated nodes"
        return node_pairs_unique
    except Exception as e:
        print(e)
        # when parsing fails, retry
        for _ in range(max_retries):
            try:
                response = await chain.ainvoke({
                    "target_nodes": target_nodes_names,
                    "generated_nodes": generated_nodes_names
                })
                node_pairs = response.node_pairs
                node_pairs = [node_pair for node_pair in node_pairs if node_pair[0] in target_nodes_names and node_pair[1] in generated_nodes_names]
                # each node appears at most once in the output, check both elements of the pair
                # if it appears multiple times, remove all but the first occurrence
                node_pairs_unique = []
                target_seen = set()
                generated_seen = set()
                for node_pair in node_pairs:
                    if node_pair[0] not in target_seen and node_pair[1] not in generated_seen:
                        node_pairs_unique.append(node_pair)
                        target_seen.add(node_pair[0])
                        generated_seen.add(node_pair[1])
                assert len(node_pairs_unique) <= len(target_nodes_names), "The number of node pairs is greater than the number of target nodes"
                assert len(node_pairs_unique) <= len(generated_nodes_names), "The number of node pairs is greater than the number of generated nodes"
                return node_pairs_unique
            except Exception as e:
                print(e)
                continue
        raise ValueError("Failed to parse the response")


def nodePrecisionRecallLLM(
    target_node_list: NodeList,
    target_edge_list: EdgeList,
    generated_node_list: NodeList,
    generated_edge_list: EdgeList,
    verbose: bool = False,
    node_pairs: Optional[List[Tuple[str, str]]] = None
) -> Tuple[float, float, float]:
    """
    Calculate the precision and recall of the generated node list.
    Use LLM-as-a-judge to determine if the generated node list is correct.

    Parameters
    ----------
    target_node_list: NodeList
        The target node list.
    generated_node_list: NodeList
        The generated node list.
    
    Returns
    -------
    Tuple[float, float, float]
        A tuple of precision, recall, and f1 score.
    """
    if node_pairs is None:
        # This should not happen in async contexts as node_pairs should be provided
        raise ValueError("node_pairs must be provided when using async version")
    if verbose:
        print(target_node_list.getNodeList())
        print(generated_node_list.getNodeList())
        print(node_pairs)

    precision = len(node_pairs) / len(generated_node_list) if len(generated_node_list) > 0 else 0
    recall = len(node_pairs) / len(target_node_list) if len(target_node_list) > 0 else 0
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


async def edgeListCommonEdgesLLM(
    target_edge_list: EdgeList,
    generated_edge_list: EdgeList,
    max_retries: int,
    llm = Gpt4o()
) -> List[Tuple[str, str]]:
    """
    Find common edges between two EdgeList objects.

    Parameters
    ----------
    target_edge_list: EdgeList
        The target edge list.
    generated_edge_list: EdgeList
        The generated edge list.
    
    Returns
    -------
    List[Tuple[Tuple[str, str], Tuple[str, str]]]
        A list of tuples, each containing a target edge and a generated edge.
    """
    target_edges = target_edge_list.getEdgeList()
    target_edges = ["->".join([edge["cause"], edge["effect"]]) for edge in target_edges]
    generated_edges = generated_edge_list.getEdgeList()
    generated_edges = ["->".join([edge["cause"], edge["effect"]]) for edge in generated_edges]

    class EdgeMatchList(BaseModel):
        edge_pairs: List[Tuple[str, str]] = Field(description="A list of tuples, each containing a target edge and a generated edge")

    prompt_template = PROMPTS["edge_list_common_edges"]
    parser = PydanticOutputParser(pydantic_object=EdgeMatchList)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["target_edges", "generated_edges"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        }
    )

    chain = prompt | llm | parser
    try:
        response = await chain.ainvoke({
            "target_edges": target_edges,
            "generated_edges": generated_edges
        })
        edge_pairs = response.edge_pairs
        edge_pairs = [edge_pair for edge_pair in edge_pairs if edge_pair[0] in target_edges and edge_pair[1] in generated_edges]
        # each edge appears at most once in the output, check both elements of the pair
        # if it appears multiple times, remove all but the first occurrence
        edge_pairs_unique = []
        for edge_pair in edge_pairs:
            if edge_pair[0] not in [edge_pair[0] for edge_pair in edge_pairs_unique] and edge_pair[1] not in [edge_pair[1] for edge_pair in edge_pairs_unique]:
                edge_pairs_unique.append(edge_pair)
        assert len(edge_pairs_unique) <= len(target_edges), "The number of edge pairs is greater than the number of target edges"
        assert len(edge_pairs_unique) <= len(generated_edges), "The number of edge pairs is greater than the number of generated edges"
        return edge_pairs_unique
    except Exception as e:
        print(e)
        # when parsing fails, retry
        for _ in range(max_retries):
            try:
                response = await chain.ainvoke({
                    "target_edges": target_edges,
                    "generated_edges": generated_edges
                })
                edge_pairs = response.edge_pairs
                edge_pairs = [edge_pair for edge_pair in edge_pairs if edge_pair[0] in target_edges and edge_pair[1] in generated_edges]
                # each edge appears at most once in the output, check both elements of the pair
                # if it appears multiple times, remove all but the first occurrence
                edge_pairs_unique = []
                for edge_pair in edge_pairs:
                    if edge_pair[0] not in [edge_pair[0] for edge_pair in edge_pairs_unique] and edge_pair[1] not in [edge_pair[1] for edge_pair in edge_pairs_unique]:
                        edge_pairs_unique.append(edge_pair)
                assert len(edge_pairs_unique) <= len(target_edges), "The number of edge pairs is greater than the number of target edges"
                assert len(edge_pairs_unique) <= len(generated_edges), "The number of edge pairs is greater than the number of generated edges"
                return edge_pairs_unique
            except Exception as e:
                print(e)
                continue
        raise ValueError("Failed to parse the response")


def edgePrecisionRecallLLM(
    target_node_list: NodeList,
    target_edge_list: EdgeList,
    generated_node_list: NodeList,
    generated_edge_list: EdgeList,
    verbose: bool = False,
    edge_pairs: Optional[List[Tuple[str, str]]] = None,
    node_pairs: Optional[List[Tuple[str, str]]] = None
) -> Tuple[float, float, float]:
    """
    Calculate the precision and recall of the generated edge list.
    Use LLM-as-a-judge to determine if the generated edge list is correct.
    """
    if edge_pairs is None:
        # This should not happen in async contexts as edge_pairs should be provided
        raise ValueError("edge_pairs must be provided when using async version")
    if verbose:
        print(target_edge_list.getEdgeList())
        print(generated_edge_list.getEdgeList())
        print(edge_pairs)

    # if node_pairs is not None:
    #     # consider only the edges between the matched nodes
    #     matched_nodes = [node_pair[0] for node_pair in node_pairs] + [node_pair[1] for node_pair in node_pairs]
    #     generated_edge_subset = []
    #     for edge in generated_edge_list.getEdgeList():
    #         if edge["cause"] in matched_nodes and edge["effect"] in matched_nodes:
    #             generated_edge_subset.append(edge)
    #     generated_edge_list = generated_edge_subset

    #     target_edge_subset = []
    #     for edge in target_edge_list.getEdgeList():
    #         if edge["cause"] in matched_nodes and edge["effect"] in matched_nodes:
    #             target_edge_subset.append(edge)
    #     target_edge_list = target_edge_subset
        
    precision = len(edge_pairs) / len(generated_edge_list) if len(generated_edge_list) > 0 else 0
    recall = len(edge_pairs) / len(target_edge_list) if len(target_edge_list) > 0 else 0
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

# Sync wrapper for backward compatibility
def matchNodes_sync(
    target_node_list: NodeList,
    generated_node_list: NodeList,
    max_retries: int,
    llm = Gpt4o()
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Synchronous wrapper for matchNodes"""
    import asyncio
    return asyncio.run(matchNodes(target_node_list, generated_node_list, max_retries, llm))

# verifyMatchNodes is already synchronous, so no wrapper needed

# Sync wrapper for backward compatibility
def matchEdgesWithNodeMatches_sync(
    target_edge_list: EdgeList,
    generated_edge_list: EdgeList,
    generated2target: Dict[str, str],
    target2generated: Dict[str, str]
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Synchronous wrapper for matchEdgesWithNodeMatches (though it's already synchronous)"""
    return matchEdgesWithNodeMatches(target_edge_list, generated_edge_list, generated2target, target2generated)

# Sync wrapper for backward compatibility
def nodeListCommonNodesLLM_sync(
    target_node_list: NodeList, 
    generated_node_list: NodeList,
    max_retries: int,
    llm = Gpt4o()
) -> List[Tuple[str, str]]:
    """Synchronous wrapper for nodeListCommonNodesLLM"""
    import asyncio
    return asyncio.run(nodeListCommonNodesLLM(target_node_list, generated_node_list, max_retries, llm))

# Sync wrapper for backward compatibility
def edgeListCommonEdgesLLM_sync(
    target_edge_list: EdgeList,
    generated_edge_list: EdgeList,
    max_retries: int,
    llm = Gpt4o()
) -> List[Tuple[str, str]]:
    """Synchronous wrapper for edgeListCommonEdgesLLM"""
    import asyncio
    return asyncio.run(edgeListCommonEdgesLLM(target_edge_list, generated_edge_list, max_retries, llm))

# Sync wrapper for backward compatibility
def fixNodePairs_sync(
    target_node_list: NodeList,
    generated_node_list: NodeList,
    node_pairs: List[Tuple[str, str]],
    issues: List[str],
    max_retries: int = 3
) -> List[Tuple[str, str]]:
    """Synchronous wrapper for fixNodePairs"""
    import asyncio
    return asyncio.run(fixNodePairs(target_node_list, generated_node_list, node_pairs, issues, max_retries))
