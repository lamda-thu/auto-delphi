#!/usr/bin/env python3
"""
Minimal example script for testing probability extraction functionality.

This script demonstrates how to:
1. Load an existing influence diagram from data files
2. Extract conditional probability distributions (CPDs) from text
3. Solve the influence diagram for optimal policy

Usage:
    python scripts/extractProbability.py

The script uses a predefined dataset and extracts probabilities to create
a complete influence diagram that can be solved for decision making.
"""

import os
import sys

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src.models import Gpt4o
from src.agent import GraphAgent, ProbabilityAgent, GenerationMode


if __name__ == '__main__':
    # Use default dataset for testing
    data_path = "data/gpt-07-recruitment/recruitment-cpd1"
    
    # Initialize GraphAgent and load existing graph structure
    graphAgent = GraphAgent(
        language_model=Gpt4o()
    )
    graphAgent.loadNodeAndEdgeList(data_path)
    influence_diagram = graphAgent.constructGraph()

    with open(os.path.join(data_path, "complete-1.txt"), 'r') as f:
        text = f.read()


    # Initialize ProbabilityAgent to extract CPDs
    probabilityAgent = ProbabilityAgent(
        language_model=Gpt4o(), 
        mode=GenerationMode.extract
    )    
    # Add the graph structure to probability agent
    probabilityAgent.addGraph(influence_diagram)

    probabilityAgent.assignStochasticFunctionCPD(text)
    
    print("\nExtracted CPDs:")
    print(probabilityAgent.getDiagram().get_cpds())
    
    # Get the complete influence diagram
    influence_diagram = probabilityAgent.getDiagram()
    print(f"\nComplete Influence Diagram:")
    print(influence_diagram)
    
    # Check if the diagram has sufficient information for solving
    print(f"\nSufficient recall: {influence_diagram.sufficient_recall()}")

    # With CPDs, we can solve the influence diagram for optimal policy
    optimal_policy = influence_diagram.solve()
    print(f"\nOptimal Policy:")
    print(optimal_policy)
    
    # Update model with optimal policy
    influence_diagram.model.update(**influence_diagram.solve())

