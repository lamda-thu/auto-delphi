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
        # TODO change output dir name
        output_dir = os.path.join("data", "da-02-carryUmbrella", f"carryUmbrella-cpd{cnt+1}")
        copyInstanceTxtFile(input_dir, output_dir)

        # replace cpd-related values
        complete_info_filename = os.path.join(output_dir, "complete.txt")
        with open(complete_info_filename, 'r', encoding='utf-8') as f:
            complete_text = f.read()
            for k, v in combo.items():
                complete_text = complete_text.replace(k, str(v))
        
        with open(complete_info_filename, 'w', encoding='utf-8') as f:
            f.write(complete_text)
        
        # replace context values
        # this involves multiple text files including `background.txt` etc
        context_combinations = [dict(zip(context_value_range.keys(), context_combo)) for context_combo in list(itertools.product(*context_value_range.values()))]
        
        for context_cnt, context_combo in enumerate(context_combinations):
            print(context_combo)
            for file_prefix in ["complete", "graph", "node", "background"]:
                context_info_filename = os.path.join(output_dir, f"{file_prefix}-{context_cnt+1}.txt") #target path
                template_info_filename = os.path.join(output_dir, f"{file_prefix}.txt") # template path
                with open(template_info_filename, 'r', encoding='utf-8') as f:
                    template_text = f.read()
                for k, v in context_combo.items():
                    template_text = template_text.replace(k, str(v))
                
                with open(context_info_filename, 'w', encoding='utf-8') as f:
                    f.write(template_text)


if __name__=='__main__':
    template_value_range = {
        "#rain_rate#": [0.1, 0.4],
        "#true_positive_prob#": [0.6, 0.9],
        "#true_negative_prob#": [0.6, 0.9],
        "#high_cost#": [5, 25]
    }
    context_value_range = {
        "#forecast#": ['rain', 'no rain']
    }
    loadTemplate(
        template_value_range,
        context_value_range,
        input_dir=os.path.join("data", "da-02-carryUmbrella", "carryUmbrella-base")
    )