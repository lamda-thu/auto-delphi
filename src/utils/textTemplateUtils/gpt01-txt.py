import os
import sys

if __name__=='__main__':

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    sys.path.append(project_root)

import json
import itertools

from util import copyInstanceTxtFile

def loadTemplate(
    template_value_range,
    input_dir,
    output_base_dir):
    combinations = [dict(zip(template_value_range.keys(), combo)) for combo in list(itertools.product(*template_value_range.values()))]

    for cnt, combo in enumerate(combinations):
        print(combo)
        # TODO change output dir name
        output_dir = f"{output_base_dir}{cnt+1}"

        copyInstanceTxtFile(input_dir, output_dir)

        # replace cpd-related values
        complete_info_filename = os.path.join(output_dir, "complete.txt")

        with open(complete_info_filename, 'r', encoding='utf-8') as f:
            complete_text = f.read()
            for k, v in combo.items():
                complete_text = complete_text.replace(k, str(v))

        with open(complete_info_filename, 'w', encoding='utf-8') as f:
            f.write(complete_text)
        
        # For context variables with no parents
        # their values are specified in the cpd
        # the context value description copies value from the cpd `combo`.
        context_cnt = 0
        context_combo = {}
        for file_prefix in ["complete", "graph", "node", "background"]:
            context_info_filename = os.path.join(output_dir, f"{file_prefix}-{context_cnt+1}.txt") #target path
            print(context_info_filename)
            template_info_filename = os.path.join(output_dir, f"{file_prefix}.txt") # template path
            with open(template_info_filename, 'r', encoding='utf-8') as f:
                template_text = f.read()
            for k, v in context_combo.items():
                template_text = template_text.replace(k, str(v))
            
            with open(context_info_filename, 'w', encoding='utf-8') as f:
                f.write(template_text)

        

            

if __name__ == "__main__":
    # manipulate:
    # "number of people one contacts", "infection"
    template_value_range = {
        "#danger_factor#": [0.1, 0.2, 0.3],
        "#basic_reaction#": [0.1, 0.2, 0.3]
    }
    loadTemplate(
        template_value_range,
        input_dir=os.path.join("data", "gpt-01-disasterManagement", "disasterManagement-base"),
        output_base_dir=os.path.join("data", "gpt-01-disasterManagement", "disasterManagement-cpd")
    )