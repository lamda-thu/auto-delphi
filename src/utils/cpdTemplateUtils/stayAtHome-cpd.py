import os
import sys

if __name__=='__main__':

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    sys.path.append(project_root)

import json
import itertools

from util import copyInstanceJsonFile

def loadTemplate(template_value_range, input_dir=os.path.join("data", "da-06-stayAtHome", "stayAtHome-base")):
    
    combinations = [dict(zip(template_value_range.keys(), combo)) for combo in list(itertools.product(*template_value_range.values()))]
    
    for cnt, combo in enumerate(combinations):
        # TODO change output dir name
        print(combo)
        output_dir = os.path.join("data", "da-06-stayAtHome", f"stayAtHome-cpd{cnt+1}")
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
        "#age#": [38, 58],
        "#income#": [40000],
        "#population_density#": [0.0153], 
        "#duration#": [7],
        "#low_num_contact#": [1, 5, 9],
        "#high_num_contact#": [42, 52, 62],
        "#p_infect#": [0.01, 0.055, 0.1],
    }
    loadTemplate(
        template_value_range,
        input_dir=os.path.join("data", "da-06-stayAtHome", "stayAtHome-base")
    )
