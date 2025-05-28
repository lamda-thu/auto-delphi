from typing import Tuple, Optional, List

from .nodeEdgeClassification import nodeListCommonNodesLLM, edgeListCommonEdgesLLM
from src.graph import NodeList, EdgeList

async def graphEditDistanceLLM(
    target_node_list: NodeList,
    target_edge_list: EdgeList,
    generated_node_list: NodeList,
    generated_edge_list: EdgeList,
    node_substitution_cost: float = 1.0,
    edge_substitution_cost: float = 1.0,
    node_deletion_cost: float = 1.0,
    edge_deletion_cost: float = 1.0,
    node_insertion_cost: float = 1.0,
    edge_insertion_cost: float = 1.0,
    max_retries: int = 3,
    verbose: bool = False,
    node_pairs: Optional[List[Tuple[str, str]]] = None,
    edge_pairs: Optional[List[Tuple[str, str]]] = None,
    edge_pairs_with_opposite_direction: Optional[List[Tuple[str, str]]] = None
) -> float:
    """
    Calculate the graph edit distance between the target and generated graphs.

    Warning
    -------
    The node_substitution_cost is set to 0.0, not considering non-exact node matches as costs.
    """
    if node_pairs is None:
        node_pairs = await nodeListCommonNodesLLM(target_node_list, generated_node_list, max_retries=max_retries)
    if edge_pairs is None:
        edge_pairs = await edgeListCommonEdgesLLM(target_edge_list, generated_edge_list, max_retries=max_retries)

    # node operations
    node_deletions = len(generated_node_list) - len(node_pairs)
    node_insertions = len(target_node_list) - len(node_pairs)

    node_substitutions = 0
    # substitute nodes with different variable types
    for node_pair in node_pairs:
        if target_node_list.getNode(node_pair[0]) is None or generated_node_list.getNode(node_pair[1]) is None:
            node_substitutions += 1
            print("GED: node not found.")
        elif target_node_list.getNode(node_pair[0]).getVariableType() != generated_node_list.getNode(node_pair[1]).getVariableType():
            node_substitutions += 1

    # edge operations
    edge_deletions = len(generated_edge_list) - len(edge_pairs)
    edge_insertions = len(target_edge_list) - len(edge_pairs)

    if edge_pairs_with_opposite_direction is None:
        edge_substitutions = 0
    else:
        edge_substitutions = len(edge_pairs_with_opposite_direction)

    
    graph_edit_distance = (
        node_deletions * node_deletion_cost + node_insertions * node_insertion_cost + node_substitutions * node_substitution_cost +
        edge_deletions * edge_deletion_cost + edge_insertions * edge_insertion_cost + edge_substitutions * edge_substitution_cost
    )
    max_graph_edit_distance = (
        len(target_node_list) + len(target_edge_list) + len(generated_node_list) + len(generated_edge_list)
    )
    if max_graph_edit_distance == 0:
        normalized_graph_edit_distance = 0
        print("Warning: empty graph")
    else:
        normalized_graph_edit_distance = graph_edit_distance / max_graph_edit_distance
    return normalized_graph_edit_distance

# Synchronous wrapper for backward compatibility
def graphEditDistanceLLM_sync(
    target_node_list: NodeList,
    target_edge_list: EdgeList,
    generated_node_list: NodeList,
    generated_edge_list: EdgeList,
    node_substitution_cost: float = 1.0,
    edge_substitution_cost: float = 1.0,
    node_deletion_cost: float = 1.0,
    edge_deletion_cost: float = 1.0,
    node_insertion_cost: float = 1.0,
    edge_insertion_cost: float = 1.0,
    max_retries: int = 3,
    verbose: bool = False,
    node_pairs: Optional[List[Tuple[str, str]]] = None,
    edge_pairs: Optional[List[Tuple[str, str]]] = None,
    edge_pairs_with_opposite_direction: Optional[List[Tuple[str, str]]] = None
) -> float:
    """Synchronous wrapper for graphEditDistanceLLM"""
    import asyncio
    return asyncio.run(graphEditDistanceLLM(
        target_node_list, target_edge_list, 
        generated_node_list, generated_edge_list,
        node_substitution_cost, edge_substitution_cost,
        node_deletion_cost, edge_deletion_cost,
        node_insertion_cost, edge_insertion_cost,
        max_retries, verbose, node_pairs, edge_pairs, edge_pairs_with_opposite_direction
    ))
    