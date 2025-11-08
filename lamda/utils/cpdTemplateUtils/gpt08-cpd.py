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
    input_dir,
    output_base_dir):
    combinations = [dict(zip(template_value_range.keys(), combo)) for combo in list(itertools.product(*template_value_range.values()))]

    for cnt, combo in enumerate(combinations):
        print(combo)
        # TODO change output dir name
        output_dir = f"{output_base_dir}{cnt+1}"

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

        

            

if __name__ == "__main__":
    # manipulate:
    # "number of people one contacts", "infection"
    template_value_range = {
        "#risk#": [0, 0.1, 0.2],
        "#condition#": [0, 0.1, 0.2]
    }
    loadTemplate(
        template_value_range,
        input_dir=os.path.join("data", "gpt-08-financialManagement", "financialManagement-base"),
        output_base_dir=os.path.join("data", "gpt-08-financialManagement", "financialManagement-cpd")
    )