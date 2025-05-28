"""
This file contains the metrics for the graph.
Include the graph edit distance, graph precision, graph recall, and graph F1 score.
Difference to nodeEdgeClassification.py:
    - Use LLM-as-a-judge once for all metrics.
"""

from typing import Tuple

from .nodeEdgeClassification import nodePrecisionRecallLLM, edgePrecisionRecallLLM, matchNodes, matchEdgesWithNodeMatches
from .graphEditDistance import graphEditDistanceLLM

MAX_RETRIES = 5
async def graphMetricsLLM(
    target_node_list,
    target_edge_list,
    generated_node_list,
    generated_edge_list,
    node_substitution_cost: float = 0.0,
    edge_substitution_cost: float = 1.0,
    node_deletion_cost: float = 1.0,
    edge_deletion_cost: float = 1.0,
    node_insertion_cost: float = 1.0,
    edge_insertion_cost: float = 1.0,
    max_retries: int = MAX_RETRIES,
    verbose: bool = False
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Calculate the graph precision, recall, and F1 score.

    Returns
    -------
    node_precision: float
        The precision of the generated node list.
    node_recall: float
        The recall of the generated node list.
    node_f1: float
        The F1 score of the generated node list.
    edge_precision: float
        The precision of the generated edge list.
    edge_recall: float
        The recall of the generated edge list.
    edge_f1: float
        The F1 score of the generated edge list.
    normalized_graph_edit_distance: float
        The normalized graph edit distance between the target and generated graphs.

    Warning
    -------
    For the node_pairs and edge_pairs, only the length of the lists are used.

    Warning
    -------
    The generated_edge_list_with_opposite_direction is not used currently.
    """
    generated2target, target2generated = await matchNodes(target_node_list, generated_node_list, max_retries=max_retries)
    node_pairs = [(node, target2generated[node]) for node in target2generated]

    matched_edges, matched_edges_with_opposite_direction = matchEdgesWithNodeMatches(target_edge_list, generated_edge_list, generated2target, target2generated)

    node_precision, node_recall, node_f1 = nodePrecisionRecallLLM(target_node_list, target_edge_list, generated_node_list, generated_edge_list, verbose, node_pairs)
    edge_precision, edge_recall, edge_f1 = edgePrecisionRecallLLM(target_node_list, target_edge_list, generated_node_list, generated_edge_list, verbose, matched_edges, node_pairs)
    
    normalized_graph_edit_distance = await graphEditDistanceLLM(
        target_node_list=target_node_list,
        target_edge_list=target_edge_list,
        generated_node_list=generated_node_list,
        generated_edge_list=generated_edge_list,
        node_substitution_cost=node_substitution_cost,
        edge_substitution_cost=edge_substitution_cost,
        node_deletion_cost=node_deletion_cost,
        edge_deletion_cost=edge_deletion_cost,
        node_insertion_cost=node_insertion_cost,
        edge_insertion_cost=edge_insertion_cost,
        max_retries=max_retries,
        verbose=verbose,
        node_pairs=node_pairs,
        edge_pairs=matched_edges,
        edge_pairs_with_opposite_direction=matched_edges_with_opposite_direction
    )
    return node_precision, node_recall, node_f1, edge_precision, edge_recall, edge_f1, normalized_graph_edit_distance

# Sync wrapper for compatibility with existing code
def graphMetricsLLM_sync(
    target_node_list,
    target_edge_list,
    generated_node_list,
    generated_edge_list,
    node_substitution_cost: float = 1.0,
    edge_substitution_cost: float = 1.0,
    node_deletion_cost: float = 1.0,
    edge_deletion_cost: float = 1.0,
    node_insertion_cost: float = 1.0,
    edge_insertion_cost: float = 1.0,
    max_retries: int = MAX_RETRIES,
    verbose: bool = False
) -> Tuple[float, float, float, float, float, float, float]:
    """Synchronous wrapper for graphMetricsLLM"""
    import asyncio
    return asyncio.run(graphMetricsLLM(
        target_node_list, target_edge_list,
        generated_node_list, generated_edge_list,
        node_substitution_cost, edge_substitution_cost, 
        node_deletion_cost, edge_deletion_cost,
        node_insertion_cost, edge_insertion_cost,
        max_retries, verbose
    ))
    
    