#!/usr/bin/env python3
"""
Minimal example script for testing graph extraction functionality.

This script demonstrates how to:
1. Extract nodes and edges from text corpus using GraphAgent
2. Generate influence diagram from extracted components
3. Display the resulting diagram

Usage:
    python scripts/extractGraph.py

The script uses a predefined corpus about chemical usage decision-making
and extracts the causal structure to create an influence diagram.
"""

import os
import sys

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src.models import Gpt4o
from src.agent import GraphAgent, GenerationMode


if __name__ == '__main__':
    # Sample corpus for testing
    corpus = ("Let us suppose that a chemical having some benefits is also suspected of possible carcinogenicity. "
              "We wish to determine whether to ban, restrict, or permit its use. "
              "The economic value of the product and the cancer cost attributed to it both depend on the decision regarding usage of the chemical. "
              "The economic value given the usage decision is independent of the human exposure, carcinogenic activity, and the cancer cost. "
              "However, the cancer cost is dependent on the usage decision as well as on both the carcinogenic activity and human exposure levels of the chemical. "
              "The net value of the chemical given the economic value and the cancer cost is independent of the other variables. "
              "Also, human exposure and carcinogenic activity are independent.")
    corpus = " ".join(corpus)
    print(corpus)

    # Initialize GraphAgent
    graphAgent = GraphAgent(        
        language_model=Gpt4o(),
        max_retries=3
    )

    # Set generation mode and extract graph
    graphAgent.setMode(GenerationMode.extract)
    graphAgent.jointGeneration(corpus, joint_fix=True)
    
    # Construct and display the influence diagram
    influence_diagram = graphAgent.constructGraph()
    print("Generated Influence Diagram:")
    print(influence_diagram)