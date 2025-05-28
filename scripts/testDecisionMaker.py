if __name__ == '__main__':
    import os
    import sys

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    sys.path.append(project_root)
    import json
    from src.graph.influenceDiagram import InfluenceDiagram
    from src.models import Qwen7B, Qwen72B, Gpt4o
    from experiment.decisionMakers import RandomDecisionMaker, VanillaDecisionMaker, CotDecisionMaker, DellmaDecisionMaker, ScDecisionMaker, AidDecisionMaker

    # load the instance
    instance_name = "gpt-08-financialManagement/financialManagement-cpd1"
    node_path = os.path.join(project_root, "data", instance_name, "nodes.json")
    edge_path = os.path.join(project_root, "data", instance_name, "edges.json")
    instance_path = os.path.join(project_root, "data", instance_name, "complete-1.txt")

    with open(node_path, "r") as f:
        nodes = json.load(f)
    with open(edge_path, "r") as f:
        edges = json.load(f)
    influence_diagram = InfluenceDiagram(nodes, edges)

    text = open(instance_path, "r").read()

    decision = influence_diagram.getDecisions()[0] # only the first decision
    decision_alternatives = {decision["variable_name"]: decision["variable_values"]}
    decision_alternatives = {k: [str(element) for element in v] for k, v in decision_alternatives.items()}

    # decision-makers receive as input: text & decision_alternatives
    # here we present a random decision-maker for testing
    # you can try other decision-makers by uncommenting the following lines
    dm = RandomDecisionMaker("config.yaml")
    #dm = VanillaDecisionMaker(Gpt4o())
    #dm = CotDecisionMaker(Qwen7B())
    #dm = ScDecisionMaker(Gpt4o(temperature=0.5))
    #dm = DellmaDecisionMaker(Gpt4o(), "config.yaml")
    #dm = AidDecisionMaker(Gpt4o(), max_retries=3, timeout=1200)
    action = dm.makeDecision(text, decision_alternatives, {})

    print(action)