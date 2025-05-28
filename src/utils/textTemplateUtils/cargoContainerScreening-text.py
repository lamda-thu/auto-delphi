import os
import sys

if __name__=='__main__':

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    sys.path.append(project_root)

import itertools

from util import copyInstanceTxtFile

def loadTemplate(
    template_value_range,
    context_value_range,
    input_dir):
    combinations = [dict(zip(template_value_range.keys(), combo)) for combo in list(itertools.product(*template_value_range.values()))]
    
    for cnt, combo in enumerate(combinations):
        print(combo)

        # TODO change output dir name
        output_dir = os.path.join("data", "da-04-cargoContainerScreening", f"cargoContainerScreening-cpd{cnt+1}")
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
        context_combo = {k: combo[k] for k in context_value_range.keys()}
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


if __name__=='__main__':
    template_value_range = {
        "#deterrence#": [1.0, 5.0],
        "#cost of screening#": [6, 40],
        "#cost of inspection#": [600, 1000],
        "#cost of attack#": [1e10, 4e10, 1e11]
    }
    context_value_range = {
    }
    input_dir = os.path.join("data", "da-04-cargoContainerScreening", "cargoContainerScreening-base")
    loadTemplate(
        template_value_range,
        context_value_range,
        input_dir=input_dir
    )