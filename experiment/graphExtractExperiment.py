"""
Graph Experiment
Runs graph extraction and graph generation on all base instances 
    with information level background, node, graph.
Experiment results saved in ./output/exp-graph/{method_name}_{model_name}_{time_stamp}_replicate{n_replicate}/{instance_name}_{information_level}_{replicate_id}
    which includes two json files: nodes.json and edges.json
    - Method name and model name should include no underscores.
    - If reflection is used, it should be included in the method name.
        e.g. "joint_qwen2.5" or "sequential_deepseek-r1:7b"
"""


import os
import sys

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    sys.path.append(project_root)


from langchain_core.language_models import LLM
import concurrent.futures

from src.agent import GraphAgent, GenerationMode
from src.utils import getCurrentTimeForFileName

from llmNameDict import LLM_NAME_DICT


def runGraphExperiment(
    method_name: str,
    model_name: str,
    n_replicate: int=1
):
    time_stamp = getCurrentTimeForFileName(format="%Y%m%d-%H%M")

    data_dir = os.path.join(project_root, "data")
    instance_names = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    instance_paths = [os.path.join(data_dir, instance_name) for instance_name in instance_names]

    levels = ["background", "node", "graph"]

    def process_instance(instance_index, instance_path, level):
        try:
            print(f"Run {time_stamp}:{instance_names[instance_index]}-{level}.")
            runGraphExperimentTrial(method_name, model_name, time_stamp, instance_path, level, n_replicate)
        except Exception as e:
            print(f"Error in {instance_names[instance_index]}-{level}: {e}")

    # Run experiments in parallel for resource-intensive models
    if model_name == "gpt-4o" or model_name == "qwen2.5:7b" or model_name == "qwen2.5:72b":
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for i, instance_path in enumerate(instance_paths):
                for level in levels:
                    futures.append(
                        executor.submit(
                            process_instance, i, instance_path, level
                        )
                    )
            concurrent.futures.wait(futures)
    else:
        # Sequential execution for other models
        for i, instance_path in enumerate(instance_paths):
            for level in levels:
                process_instance(i, instance_path, level)


def runGraphExperimentTrial(
    method_name: str,
    model_name: str,
    time_stamp: str,
    instance_path: str,
    level: str,
    n_replicate: int
):
    instance_name = os.path.basename(instance_path)
    for replicate_id in range(n_replicate):
        print(f"Replicate {replicate_id+1} of {n_replicate}.")
        output_dir = os.path.join(
            project_root, "output", "exp-graph",
            f"{method_name}_{model_name}_{time_stamp}_replicate{n_replicate}",
            f"{instance_name}_{level}_{replicate_id}"
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model = LLM_NAME_DICT[model_name]
        graph_agent = GraphAgent(model)

        text_file_path = os.path.join(
            project_root,
            instance_path,
            instance_name.split("-")[-1] + "-base",
            level + ".txt"
        )
        with open(text_file_path, "r") as f:
            text = f.read()

        runGraphExperimentMethod(
            graph_agent,
            method_name,
            text,
            output_dir
        )


def runGraphExperimentMethod(
    graph_agent: GraphAgent,
    method_name: str,
    text: str,
    output_dir: str
) -> None:
    graph_agent.reset() # empty the node and edge list
    if method_name == "joint-extraction":
        graph_agent.setMode(GenerationMode.extract)
        graph_agent.jointGeneration(text, joint_fix=True)
    elif method_name == "sequential-extraction":
        graph_agent.setMode(GenerationMode.extract)
        graph_agent.extendNodeList(text)
        graph_agent.extendEdgeList(text)
        graph_agent.verifyAndFixGraph(text, joint_fix=False)
    else:
        raise ValueError(f"Method {method_name} not found.")
    
    
    graph_agent.saveNodeList(output_dir)
    graph_agent.saveEdgeList(output_dir)


if __name__ == "__main__":
    runGraphExperiment(
        "joint-extraction",
        "gpt-4o",
        n_replicate=5
    )
    exit()
    runGraphExperimentTrial(
        "joint-extraction",
        "qwen2.5:7b",
        "20250307-1200",
        os.path.join(
            "data",
            "da-01-longTermCareInsurance"
        ),
        "background"
    )