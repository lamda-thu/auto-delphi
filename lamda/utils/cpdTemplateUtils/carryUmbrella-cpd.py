import os
import sys

if __name__=='__main__':

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    sys.path.append(project_root)

import json
import itertools

from util import copyInstanceJsonFile

def loadTemplate(
    template_value_range,
    input_dir):
    combinations = [dict(zip(template_value_range.keys(), combo)) for combo in list(itertools.product(*template_value_range.values()))]
    
    for cnt, combo in enumerate(combinations):
        print(combo)
        # TODO change output dir name
        output_dir = os.path.join("data", "da-02-carryUmbrella", f"carryUmbrella-cpd{cnt+1}")
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
        
        nodes_file_name = os.path.join(output_dir, "nodes.json")
        with open(nodes_file_name, 'r', encoding='utf-8') as f:
            node_list = json.load(f)
            for i, node in enumerate(node_list):
                if node["variable_name"] == "cost":
                    variable_values = node["variable_values"]
                    for k, v in combo.items():
                        variable_values = [
                            str(variable_value).replace(k, str(v)) for variable_value in variable_values
                        ]
                    variable_values = [
                        float(variable_value) for variable_value in variable_values
                    ]
                    node_list[i]["variable_values"] = variable_values

        with open(nodes_file_name, 'w', encoding='utf-8') as f:
            json.dump(node_list, f, indent=4)


if __name__ == "__main__":
    # manipulate:
    # "number of people one contacts", "infection"
    template_value_range = {
        "#rain_rate#": [0.1, 0.4],
        "#true_positive_prob#": [0.6, 0.9],
        "#true_negative_prob#": [0.6, 0.9],
        "#high_cost#": [5, 25] # TODO: check whether use 25 for random guess & seldom rain case
    }
    loadTemplate(
        template_value_range,
        input_dir= os.path.join("data", "da-02-carryUmbrella", "carryUmbrella-base")
    )