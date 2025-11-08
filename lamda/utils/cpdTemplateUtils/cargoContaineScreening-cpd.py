import os
import sys

if __name__=='__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    sys.path.append(project_root)
    print(sys.path)

import json
import itertools

from util import copyInstanceJsonFile

def loadTemplate(
    template_value_range,
    input_dir):
    combinations = [dict(zip(template_value_range.keys(), combo)) for combo in list(itertools.product(*template_value_range.values()))]
    combinations = [combo for combo in combinations if combo["#cost of screening#"] < combo["#cost of inspection#"]]
    for cnt, combo in enumerate(combinations):
        print(combo)
        # TODO change output dir name
        output_dir = os.path.join("data", "da-04-cargoContainerScreening", f"cargoContainerScreening-cpd{cnt+1}")
        copyInstanceJsonFile(input_dir, output_dir)
        cpd_file_name = os.path.join(output_dir, "cpd.json")

        with open(cpd_file_name, 'r', encoding='utf-8') as f:
            local_cpd_template_list = json.load(f)
            for i, local_cpd_template in enumerate(local_cpd_template_list):
                # FILL IN THE CPD TEMPLATE
                stochastic_function = local_cpd_template["stochastic_function"]
                for k, v in combo.items():
                    stochastic_function = stochastic_function.replace(k, str(v)) 
                local_cpd_template_list[i]["stochastic_function"] = stochastic_function

        with open(cpd_file_name, 'w', encoding='utf-8') as f:
            json.dump(local_cpd_template_list, f, indent=4)

        node_file_name = os.path.join(output_dir, "nodes.json")
        with open(node_file_name, 'r', encoding='utf-8') as f:
            node_list = json.load(f)
            for i, node in enumerate(node_list):
                if node["variable_name"] in ["cost of terrorist attack", "cost of inspection", "cost of screening"]:
                    variable_values = node["variable_values"]
                    for k, v in combo.items():
                        variable_values = [variable_value.replace(k, str(v)) for variable_value in variable_values]
                    node_list[i]["variable_values"] = [
                        eval(variable_value) if any(op in variable_value for op in ['+', '-', '*', '/']) else float(variable_value)
                        for variable_value in variable_values
                    ]
        with open(node_file_name, 'w', encoding='utf-8') as f:
            json.dump(node_list, f, indent=4)

if __name__ == "__main__":
    # manipulate:
    # "number of people one contacts", "infection"
    template_value_range = {
        "#deterrence#": [1.0, 5.0],
        "#cost of screening#": [6, 40],
        "#cost of inspection#": [600, 1000],
        "#cost of attack#": [1e10, 4e10, 1e11]
    }
    input_dir = os.path.join("data", "da-04-cargoContainerScreening", "cargoContainerScreening-base")
    loadTemplate(
        template_value_range,
        input_dir
    )