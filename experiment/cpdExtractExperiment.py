"""
CPD Experiment
Runs cpd extraction on all base instances 
    with information level complete.
Experiment results saved in ./output/exp-cpd/{method_name}_{model_name}_{time_stamp}/{instance_name}/{information_level}
    which includes three json files: nodes.json, edges.json and cpd.json
    - Method name and model name should include no underscores.
    - If reflection is used, it should be included in the method name.
        e.g. "joint_qwen2.5" or "sequential_deepseek-r1:7b"

A config file is also saved in the `{method_name}_{model_name}_{time_stamp}` directory.
"""


import os
import sys

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    sys.path.append(project_root)


import json
import pandas as pd
from langchain_core.language_models import LLM

from src.agent import GraphAgent, GenerationMode, ProbabilityAgent
from src.models import Qwen7B
from src.utils import getCurrentTimeForFileName

from llmNameDict import LLM_NAME_DICT
import math
import statistics


def runCPDExperiment(
    model_name: str,
):
    time_stamp = getCurrentTimeForFileName(format="%Y%m%d-%H%M")

    data_dir = os.path.join(project_root, "data")
    instance_names = [f for f in os.listdir(
        data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    instance_paths = [os.path.join(data_dir, instance_name)
                      for instance_name in instance_names]

    import concurrent.futures

    def process_instance(instance_name, instance_path, cpd_instance):
        try:
            # print(f"Run {time_stamp}:{instance_name}-cpd{cpd_instance}.")
            runCPDExperimentTrial(
                model_name, time_stamp, instance_path, cpd_instance
            )
        except Exception as e:
            print(f"Error in {instance_name}-cpd{cpd_instance}: {e}")

    def eval_instance(instance_name, instance_path, cpd_instance):
        try:
            # print(f"Evaluate {time_stamp}:{instance_name}-cpd{cpd_instance}.")
            tvd, kl, js, bs, optimal_action = evalCPDExperimentTrial(
                model_name, time_stamp, instance_path, cpd_instance
            )
            return tvd, kl, js, bs, optimal_action
        except Exception as e:
            # print(f"Error in {instance_name}-cpd{cpd_instance}: {e}")
            return 1, 9999999, math.log(2), 1, -9999999

    # run all instances
    # if model_name == "gpt-4o" or model_name == "deepseek-r1":
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         futures = []
    #         for i, instance_path in enumerate(instance_paths):
    #             for cpd_instance in range(1, len(os.listdir(instance_path))):
    #                 futures.append(
    #                     executor.submit(
    #                         process_instance, instance_names[i], instance_path, cpd_instance
    #                     )
    #                 )
    #         concurrent.futures.wait(futures)
    # else:
    #     for i, instance_path in enumerate(instance_paths):
    #         for cpd_instance in range(1, len(os.listdir(instance_path))):
    #             process_instance(instance_names[i], instance_path, cpd_instance)

    # evaluate all instances
    data_frame = pd.DataFrame(
        columns=["instance", "cpd_instance", "tvd", "kl", "js", "bs", "optimal_action"])
    tvd_dict = {}
    kl_dict = {}
    js_dict = {}
    bs_dict = {}
    tvd_list = []
    kl_list = []
    js_list = []
    bs_list = []
    all = 0
    wrong = 0 
    for i, instance_path in enumerate(instance_paths):
        inst_tvd_list = []
        inst_kl_list = []
        inst_js_list = []
        inst_bs_list = []
        for cpd_instance in range(1, len(os.listdir(instance_path))):
            # process_instance(instance_names[i], instance_path, cpd_instance)
            all += 1
            tvd, kl, js, bs, optimal_action = eval_instance(
                instance_names[i], instance_path, cpd_instance)
            #保留4位小数
            tvd = round(tvd, 3)
            kl = round(kl, 3)
            js = round(js, 3)
            bs = round(bs, 3)
            new_row = {
                "instance": instance_names[i],
                "cpd_instance": cpd_instance,
                "tvd": tvd,
                "kl": kl,
                "js": js,
                "bs": bs,
                "optimal_action": optimal_action
            }
            data_frame = data_frame._append(new_row, ignore_index=True)
            if kl!= 9999999:
                inst_tvd_list.append(tvd)
                inst_kl_list.append(kl)
                inst_js_list.append(js)
                inst_bs_list.append(bs)
                tvd_list.append(tvd)
                kl_list.append(kl)
                js_list.append(js)
                bs_list.append(bs)
            else:
                wrong +=1
        if len(inst_tvd_list) != 0:
            inst_avg_tvd = statistics.mean(inst_tvd_list)
            inst_avg_kl = statistics.mean(inst_kl_list)
            inst_avg_js = statistics.mean(inst_js_list)
            inst_avg_bs = statistics.mean(inst_bs_list)
            if len(inst_tvd_list) > 1:
                inst_std_tvd = statistics.stdev(inst_tvd_list)
                inst_std_kl = statistics.stdev(inst_kl_list)
                inst_std_js = statistics.stdev(inst_js_list)
                inst_std_bs = statistics.stdev(inst_bs_list)
            else:
                inst_std_tvd = -1
                inst_std_kl = -1
                inst_std_js = -1
                inst_std_bs = -1
            tvd_dict[instance_names[i]] = {
                "avg": inst_avg_tvd,
                "std": inst_std_tvd
            }
            kl_dict[instance_names[i]] = {
                "avg": inst_avg_kl,
                "std": inst_std_kl
            }
            js_dict[instance_names[i]] = {
                "avg": inst_avg_js,
                "std": inst_std_js
            }
            bs_dict[instance_names[i]] = {
                "avg": inst_avg_bs,
                "std": inst_std_bs
            }
    micro_avg_tvd = statistics.mean(tvd_list)
    micro_avg_kl = statistics.mean(kl_list)
    micro_avg_js = statistics.mean(js_list)
    micro_avg_bs = statistics.mean(bs_list)
    micro_std_tvd = statistics.stdev(tvd_list)
    micro_std_kl = statistics.stdev(kl_list)
    micro_std_js = statistics.stdev(js_list)
    micro_std_bs = statistics.stdev(bs_list)
    macro_avg_tvd_list = [i["avg"] for i in tvd_dict.values()]
    macro_avg_kl_list = [i["avg"] for i in kl_dict.values()]
    macro_avg_js_list = [i["avg"] for i in js_dict.values()]
    macro_avg_bs_list = [i["avg"] for i in bs_dict.values()]
    macro_avg_tvd = statistics.mean(macro_avg_tvd_list)
    macro_avg_kl = statistics.mean(macro_avg_kl_list)
    macro_avg_js = statistics.mean(macro_avg_js_list)
    macro_avg_bs = statistics.mean(macro_avg_bs_list)
    macro_std_tvd = statistics.stdev(macro_avg_tvd_list)
    macro_std_kl = statistics.stdev(macro_avg_kl_list)
    macro_std_js = statistics.stdev(macro_avg_js_list)
    macro_std_bs = statistics.stdev(macro_avg_bs_list)
    error = wrong / all
    tvd_dict["macro"] = {
        "avg": macro_avg_tvd,
        "std": macro_std_tvd
    }
    kl_dict["macro"] = {
        "avg": macro_avg_kl,
        "std": macro_std_kl
    }
    js_dict["macro"] = {
        "avg": macro_avg_js,
        "std": macro_std_js
    }
    bs_dict["macro"] = {
        "avg": macro_avg_bs,
        "std": macro_std_bs
    }
    tvd_dict["micro"] = {
        "avg": micro_avg_tvd,
        "std": micro_std_tvd
    }
    kl_dict["micro"] = {
        "avg": micro_avg_kl,
        "std": micro_std_kl
    }
    js_dict["micro"] = {
        "avg": micro_avg_js,
        "std": micro_std_js
    }
    bs_dict["micro"] = {
        "avg": micro_avg_bs,
        "std": micro_std_bs
    }
    # dict sort
    tvd_dict = dict(sorted(tvd_dict.items(), key=lambda item: item[0]))
    kl_dict = dict(sorted(kl_dict.items(), key=lambda item: item[0]))
    js_dict = dict(sorted(js_dict.items(), key=lambda item: item[0]))
    bs_dict = dict(sorted(bs_dict.items(), key=lambda item: item[0]))
    # save the results
    with open(os.path.join(
        project_root, "output", "exp-cpd",
        f"{model_name}",
        f"tvd.json"
    ), 'w') as f:
        json.dump(tvd_dict, f, indent=4)
    with open(os.path.join(
        project_root, "output", "exp-cpd",
        f"{model_name}",
        f"kl.json"
    ), 'w') as f:
        json.dump(kl_dict, f, indent=4)
    with open(os.path.join(
        project_root, "output", "exp-cpd",
        f"{model_name}",
        f"js.json"
    ), 'w') as f:
        json.dump(js_dict, f, indent=4)
    with open(os.path.join(
        project_root, "output", "exp-cpd",
        f"{model_name}",
        f"bs.json"
    ), 'w') as f:
        json.dump(bs_dict, f, indent=4)
    with open(os.path.join(
        project_root, "output", "exp-cpd",
        f"{model_name}",
        f"error.json"
    ), 'w') as f:
        json.dump({"error": error}, f, indent=4)
    data_frame.to_csv(
        os.path.join(
            project_root, "output", "exp-cpd",
            f"{model_name}",
            f"jpd.csv"
        )
    )


def runCPDExperimentTrial(
    model_name: str,
    time_stamp: str,
    instance_path: str,
    cpd_instance: int,
):
    instance_name = os.path.basename(instance_path)
    output_dir = os.path.join(
        project_root, "output", "exp-cpd",
        f"{model_name}",
        f"{instance_name}",
        f"cpd{cpd_instance}",
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        if os.path.exists(os.path.join(output_dir, "cpd.json")):
            # print(f"Run {time_stamp}:{instance_name}-cpd{cpd_instance} already exists.")
            return
    print(f"Run {time_stamp}:{instance_name}-cpd{cpd_instance}.")
    # return
    graphAgent = GraphAgent(LLM_NAME_DICT[model_name])

    data_path = os.path.join(
        project_root,
        instance_path,
        instance_name.split("-")[-1] + "-cpd"+str(cpd_instance),
    )
    graphAgent.loadNodeAndEdgeList(data_path)
    influence_diagram = graphAgent.constructGraph()
    probabilityAgent = ProbabilityAgent(
        language_model=LLM_NAME_DICT[model_name], mode=GenerationMode.extract, max_retries=5)
    probabilityAgent.addGraph(influence_diagram)
    with open(os.path.join(data_path, "complete.txt"), 'r') as f:
        text = f.read()
    probabilityAgent.assignStochasticFunctionCPD(text)
    probabilityAgent.saveCPD(output_dir)


def evalCPDExperimentTrial(
    model_name: str,
    time_stamp: str,
    instance_path: str,
    cpd_instance: int,
):
    instance_name = os.path.basename(instance_path)
    predict_dir = os.path.join(
        project_root, "output", "exp-cpd",
        f"{model_name}",
        f"{instance_name}",
        f"cpd{cpd_instance}",
    )
    truth_dir = os.path.join(
        project_root, "data", instance_name,
        instance_name.split("-")[-1] + "-cpd"+str(cpd_instance),
    )
    graphAgent = GraphAgent(LLM_NAME_DICT[model_name])
    graphAgent.loadNodeAndEdgeList(truth_dir)
    influence_diagram = graphAgent.constructGraph()
    probabilityAgent_predict = ProbabilityAgent(
        language_model=LLM_NAME_DICT[model_name], mode=GenerationMode.extract, max_retries=5)
    probabilityAgent_truth = ProbabilityAgent(
        language_model=LLM_NAME_DICT[model_name], mode=GenerationMode.extract, max_retries=5)
    probabilityAgent_predict.addGraph(influence_diagram)
    probabilityAgent_truth.addGraph(influence_diagram.copy_ID_without_cpds())
    probabilityAgent_predict.loadCPD(predict_dir)
    probabilityAgent_truth.loadCPD(truth_dir)
    try:
        id = probabilityAgent_predict.getDiagram()
        optimal_policy = id.solve()
        first_decision = id.get_valid_order()[0]
        optimal_policy = optimal_policy[first_decision]
        chosen_action_idx = optimal_policy.values.argmax()
        optimal_action = optimal_policy.domain[chosen_action_idx]
        # print(optimal_action)
    except Exception as e:
        print(e)
        optimal_action = -9999999
    tvd = probabilityAgent_predict.distance(probabilityAgent_truth, "TVD")
    kl = probabilityAgent_predict.distance(probabilityAgent_truth, "KL")
    js = probabilityAgent_predict.distance(probabilityAgent_truth, "JS")
    bs = probabilityAgent_predict.distance(probabilityAgent_truth, "BS")
    tvd0 = probabilityAgent_truth.distance(method="TVD")
    kl0 = probabilityAgent_truth.distance(method="KL")
    js0 = probabilityAgent_truth.distance(method="JS")
    bs0 = probabilityAgent_truth.distance(method="BS")
    print(f"{instance_name}-cpd{cpd_instance}: TVD={tvd}, KL={kl}, JS={js}, BS={bs}")
    print(f"{instance_name}-cpd{cpd_instance}: TVD0={tvd0}, KL0={kl0}, JS0={js0}, BS0={bs0}")
    return tvd, kl, js, bs, optimal_action


if __name__ == "__main__":
    # runCPDExperiment(
    #     "deepseek-r1"
    # )
    runCPDExperiment(
        "qwen2.5:7b"
    )
    # runCPDExperiment(
    #     "qwen2.5:72b"
    # )
    # runCPDExperiment(
    #     "gpt-4o"
    # )
    exit()
    runCPDExperimentTrial(
        "qwen2.5:7b",
        "20250311-1200",
        os.path.join(
            "data",
            "da-01-longTermCareInsurance"
        ),
        1,
        "complete"
    )
