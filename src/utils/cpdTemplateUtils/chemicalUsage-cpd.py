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
    ## ADD CONSTRAINTS
    combinations = [combo for combo in combinations if combo["#low_economic_value#"] < combo["#high_economic_value#"]]

    for cnt, combo in enumerate(combinations):
        print(combo)
        # TODO change output dir name
        output_dir = os.path.join("data", "da-05-chemicalUsage", f"chemicalUsage-cpd{cnt+1}")

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
            nodes_template_list = json.load(f)
            for i, nodes_template in enumerate(nodes_template_list):
                # FILL IN THE NODES TEMPLATE
                variable_values = nodes_template["variable_values"]
                # skip if not all elements are str
                if nodes_template["variable_name"] == "economic value" or nodes_template["variable_name"] == "net value":
                    for k, v in combo.items():
                        variable_values = [variable_value.replace(k, str(v)) for variable_value in variable_values]
                    nodes_template_list[i]["variable_values"] = [
                        eval(variable_value) if any(op in variable_value for op in ['+', '-', '*', '/']) else float(variable_value)
                        for variable_value in variable_values
                    ]

        with open(nodes_file_name, 'w', encoding='utf-8') as f:
            json.dump(nodes_template_list, f, indent=4)

if __name__ == "__main__":
    # manipulate:

    template_value_range = {
        "#ban_economic_value#": [-1, 0],
        "#low_economic_value#": [1, 5, 10, 50],
        "#high_economic_value#": [5, 20, 100],
    }
    loadTemplate(
        template_value_range,
        input_dir=os.path.join("data", "da-05-chemicalUsage", "chemicalUsage-base")
    )
