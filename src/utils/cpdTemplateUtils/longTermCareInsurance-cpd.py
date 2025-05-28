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
        output_dir = os.path.join("data", "da-01-longTermCareInsurance", f"longTermCareInsurance-cpd{cnt+1}")
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
                if node["variable_name"] == "long term care expense per day":
                    variable_values = node["variable_values"]
                    for k, v in combo.items():
                        variable_values = [
                            value.replace(k, str(v))
                            for value in variable_values
                        ]
                    variable_values = [
                        eval(variable_value) if any(op in variable_value for op in ['+', '-', '*', '/']) else float(variable_value)
                        for variable_value in variable_values
                    ]
                    print(variable_values)

                    node_list[i]["variable_values"] = variable_values

        with open(node_file_name, 'w', encoding='utf-8') as f:
            json.dump(node_list, f, indent=4)
                


if __name__ == "__main__":
    # manipulate:
    # "number of people one contacts", "infection"
    template_value_range = {
        "#age#": [65, 75],
        "#gender#": ["'male'", "'female'"],
        "#gender_factor#": [0.8, 1.25],
        "#age_factor#": [3, 9, 27, 42],
    }
    loadTemplate(
        template_value_range,
        input_dir=os.path.join("data", "da-01-longTermCareInsurance", "longTermCareInsurance-base")
    )