#!/usr/bin/env python3
"""
Minimal example script for testing InfluenceDiagram functionality.

This script demonstrates how to:
1. Load args for a influence diagram from JSON files
2. Create and display the diagram

Usage:
    # Use default dataset (recruitment example)
    python scripts/createInfluenceDiagram.py
    
    # Specify a custom dataset path
    python scripts/createInfluenceDiagram.py ./data/da-02-carryUmbrella/carryUmbrella-cpd1
    
Required Files:
    The data directory should contain:
    - nodes.json: List of nodes with variable_name, variable_type, and variable_values
    - edges.json: List of edges with cause and effect relationships
    - cpd.json: Conditional probability distributions for the nodes (optional)

Example Dataset Structure:
    data/
    ├── gpt-07-recruitment/
    │   └── recruitment-cpd1/
    │       ├── nodes.json
    │       ├── edges.json
    │       └── cpd.json
    └── da-02-carryUmbrella/
        └── carryUmbrella-cpd1/
            ├── nodes.json
            ├── edges.json
            └── cpd.json
"""

import os
import sys
import json

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src.graph import InfluenceDiagram


if __name__ == '__main__':
    data_path="./data/gpt-07-recruitment/recruitment-cpd1"

    # load the required lists from JSON files
    with open(os.path.join(data_path, "nodes.json"), 'r', encoding='utf-8') as f:
        nodes = json.load(f)
    
    with open(os.path.join(data_path, "edges.json"), 'r', encoding='utf-8') as f:
        edges = json.load(f)
    try:
        with open(os.path.join(data_path, "cpd.json"), 'r', encoding='utf-8') as f:
            cpds = json.load(f)
    except:
        cpds = None

    influence_diagram = InfluenceDiagram(nodes, edges, cpds)
    print(influence_diagram)