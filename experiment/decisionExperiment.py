"""
Decision Experiment
Runs decision making on all base instances 
    with information level complete.
Experiment results saved in ./output/exp-decision/{decision_maker_name}_{time_stamp}_replicate{n_replicate}/{instance_name}_{information_level}_{replicate_id}.csv
"""

import os
import sys

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    sys.path.append(project_root)

import csv
import json
import asyncio
import time
from typing import List, Dict, Tuple, Any

from experiment import DecisionMakerBase
from experiment.decisionMakers import (
    RandomDecisionMaker,
    VanillaDecisionMaker, CotDecisionMaker, ScDecisionMaker, DellmaDecisionMaker,
    AidDecisionMaker)

from src.graph import InfluenceDiagram
from src.agent import TextGeneratorAgent
from src.utils import getCurrentTimeForFileName
from src.models import Llama3, Gemma2, Qwen7B, Qwen72B, Phi4, Gpt4Turbo, DeepseekCoder, O1preview, Gpt4o, Deepseek_r1

async def runDecisionExperimentTrial(
        instance_path: str,
        decision_maker_factory,
        level = "complete",
        replicate_id = 0,
        instance_name = None,
        context_dict = None
) -> dict:
    """
    Run a single trial with one instance and one decision maker.

    Parameters
    ----------
    instance_path : str
        path to the instance folder
    decision_maker_factory : callable
        Factory function that creates a fresh DecisionMakerBase instance
    level : str
        The level of information for text generation. Values can be:
        "background", "node", "graph", "quanlitative CPD", "complete"
    replicate_id : int
        The ID of the current replicate run
    instance_name : str, optional
        The name of the instance. If None, derived from the path.

    Returns
    -------
    context_decision_dict : dict
        dictionary of decisions
        [context_id: dict[variable_name: decision]]
    """

    if instance_name is None:
        instance_name = os.path.basename(instance_path)
    instance_id = instance_name.split("-")[-1]
    with open(os.path.join(instance_path, "nodes.json"), 'r', encoding='utf-8') as f:
        nodes = json.load(f)
    with open(os.path.join(instance_path, "edges.json"), 'r', encoding='utf-8') as f:
        edges = json.load(f)

    influence_diagram = InfluenceDiagram(nodes, edges)
    decision = influence_diagram.getDecisions()[0] # only the first decision
    decision_alternatives = {decision["variable_name"]: decision["variable_values"]}
    decision_alternatives = {k: [str(element) for element in v] for k, v in decision_alternatives.items()}

    # load text of `level`
    file_list = os.listdir(instance_path)
    file_list = [file for file in file_list if file.endswith(".txt") and file.startswith(level) and file != f"{level}.txt"]
    context_decision_dict = {}
    
    for file in file_list:
        context_id = int(file.split("-")[-1].split(".")[0])
        instance_context_key = (instance_id, level, context_id)
        with open(os.path.join(instance_path, file), 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Create a fresh decision maker for each context
        fresh_decision_maker = decision_maker_factory()
        
        # Allow other async tasks to run while making decisions
        decision_dict = await make_decision_async(fresh_decision_maker, text, decision_alternatives, context_dict.get(context_id, {}))
        context_decision_dict[instance_context_key] = decision_dict
    
    return context_decision_dict

async def make_decision_async(decision_maker, text, decision_alternatives, context_values: dict):
    """Wrapper to make CPU-bound decision making non-blocking"""
    # Use asyncio.to_thread for Python 3.9+ to run CPU-bound work without blocking
    # For earlier Python versions, this could be replaced with loop.run_in_executor with None executor
    return await asyncio.to_thread(decision_maker.makeDecision, text, decision_alternatives, context_values)

async def saveDecisionsAsync(output_dir: str, instance_name: str, instance_decision_dict: dict) -> None:
    """Save decisions for the instance at `instancePath` asynchronously."""
    # Create all necessary directories in the path
    output_path = os.path.join(output_dir, instance_name+".csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # set headers for the csv
    # append decision variable names
    headers = ["instance", "context"]
    decision_variables = set()
    for decision_dict in instance_decision_dict.values():
        decision_variables.update(decision_dict.keys())
    headers.extend(decision_variables)

    # Check if file exists and read existing data
    existing_data = {}
    if os.path.exists(output_path):
        # Read existing file content
        await asyncio.to_thread(read_existing_csv, output_path, existing_data)
    
    # Merge with new data - this preserves previously written instance IDs
    merged_data = {**existing_data, **instance_decision_dict}
    
    # Use asyncio.to_thread to avoid blocking I/O when writing
    await asyncio.to_thread(write_csv, output_path, headers, merged_data)

def read_existing_csv(output_path, existing_data):
    """Read existing CSV file and populate the existing_data dictionary"""
    try:
        with open(output_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)  # Skip header row
            
            for row in reader:
                if len(row) >= 3:  # Must have at least instance, context, and one decision
                    instance = row[0]
                    context = row[1]
                    
                    # Create decision dictionary from remaining columns
                    decision_dict = {}
                    for i in range(2, len(headers)):
                        if i < len(row) and row[i]:  # Skip empty values
                            decision_dict[headers[i]] = row[i]
                    
                    # Add to existing data
                    instance_context_key = (instance, "complete", context)  # Assuming level is "complete"
                    existing_data[instance_context_key] = decision_dict
    except Exception as e:
        print(f"Error reading existing CSV: {e}")
        # If there's an error reading, just continue with an empty existing_data

def write_csv(output_path, headers, instance_decision_dict):
    """Helper function to write CSV without blocking the event loop"""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for instance_context_key, decision_dict in instance_decision_dict.items():
            instance, level, context = instance_context_key
            row = [instance, context]
            for decision_variable in headers[2:]:
                row.append(decision_dict.get(decision_variable, ""))
            writer.writerow(row)

async def runDecisionExperiment(
    data_dir: str,
    instance_name_list: list,
    context_dict_list: list,
    decision_maker_factory_list: list,
    level: str = "complete",
    n_replicate: int = 1,
    concurrency_limit: int = 5,  # Limit concurrent tasks
    task_timeout: int = 1800  # Default 30-minute timeout per task
):
    """
    Run experiment trials asynchronously for all instances

    Parameters
    ----------
    data_dir: str
        The location of all instances.
    instance_name_list: list
        The list of instances to be tested.
    context_dict_list: list
        The list of context dictionaries.
        Matches context id with a dict of context variable values.
    decision_maker_factory_list : list
        List of functions that create decision makers.
        Each function should take no arguments and return a fresh DecisionMakerBase instance.
    level : str, optional
        The level of information, by default "complete"
    n_replicate : int, optional
        Number of replications to run for each configuration, by default 1
    concurrency_limit: int, optional
        Maximum number of concurrent tasks, by default 5
    task_timeout: int, optional
        Maximum time in seconds to wait for a task before considering it timed out, by default 1800 (30 min)
    """
    time_stamp = getCurrentTimeForFileName()
    
    for decision_maker_factory in decision_maker_factory_list:
        # Create a sample decision maker to get its name
        sample_decision_maker = decision_maker_factory()
        decision_maker_name = sample_decision_maker.getName()
        
        print(f"Testing decision maker: {decision_maker_name}")
        
        # Dictionary to collect results by instance_name and replicate
        consolidated_results = {}  # {(base_instance_name, replicate_id): instance_decision_dict}
        
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrency_limit)
        
        # Create a list to store all task coroutines and a list for failed tasks to retry
        all_tasks = []
        failed_tasks = []
        
        # Define a worker function that respects the semaphore
        async def bounded_trial_task(instance_path, instance_name, replicate_id, context_dict, retry_count=0):
            max_retries = 2  # Maximum number of retries for a failed task
            
            try:
                async with semaphore:
                    start_time = time.time()
                    print(f"Starting {instance_name} replicate {replicate_id+1}" + 
                          (f" (retry {retry_count})" if retry_count > 0 else ""))

                    try:
                        # Wrap the execution in a timeout
                        result = await asyncio.wait_for(
                            runDecisionExperimentTrial(
                                instance_path,
                                decision_maker_factory=decision_maker_factory,
                                level=level,
                                replicate_id=replicate_id,
                                instance_name=instance_name,
                                context_dict=context_dict
                            ),
                            timeout=task_timeout
                        )
                        
                        elapsed = time.time() - start_time
                        print(f"Completed {instance_name} replicate {replicate_id+1} in {elapsed:.2f}s")
                        return (instance_name, replicate_id, result)
                    
                    except asyncio.TimeoutError:
                        elapsed = time.time() - start_time
                        print(f"TIMEOUT: {instance_name} replicate {replicate_id+1} after {elapsed:.2f}s")
                        
                        # If we haven't exceeded max retries, add to the failed tasks list
                        if retry_count < max_retries:
                            failed_tasks.append((instance_path, instance_name, replicate_id, context_dict, retry_count + 1))
                        else:
                            print(f"Giving up on {instance_name} replicate {replicate_id+1} after {retry_count+1} attempts")
                        
                        return None
            
            except Exception as e:
                print(f"ERROR in {instance_name} replicate {replicate_id+1}: {str(e)}")
                # Add to retry if we haven't exceeded max retries
                if retry_count < max_retries:
                    failed_tasks.append((instance_path, instance_name, replicate_id, context_dict, retry_count + 1))
                return None
        
        # Queue up all the tasks
        for i, instance_name in enumerate(instance_name_list):
            print(f"Processing {instance_name}")
            instance_root = os.path.join(data_dir, instance_name)
            
            # enumerate all sub-directories in the directory
            for instance_folder_name in os.listdir(instance_root):
                instance_num = instance_folder_name.split("-")[-1]
                if instance_num == "base":
                    continue
                
                instance_path = os.path.join(instance_root, instance_folder_name)
                instance_full_name = f"{instance_name}-{instance_num}"
                
                # Create tasks for all replicates
                for replicate_id in range(n_replicate):
                    task = bounded_trial_task(instance_path, instance_full_name, replicate_id, context_dict_list[i])
                    all_tasks.append(task)
        
        # Create output directory
        output_dir = os.path.join(
            project_root, "output", "exp-decision",
            f"{decision_maker_name}_{time_stamp}_replicate{n_replicate}"
        )
        os.makedirs(output_dir, exist_ok=True)
        
        # Execute tasks and process results as they complete
        results = []
        for task_coro in asyncio.as_completed(all_tasks):
            result = await task_coro
            if result:  # Only add non-None results
                results.append(result)
        
        # Process the results
        for instance_name, replicate_id, instance_decision_dict in results:
            # Extract base instance name without the ID suffix
            base_instance_name = instance_name.rsplit('-', 1)[0]
            
            # Use base_instance_name as part of the key to group by base instance
            key = (base_instance_name, replicate_id)
            if key not in consolidated_results:
                consolidated_results[key] = {}

            # Store all results in the consolidated_results dictionary
            for context_key, decision_dict in instance_decision_dict.items():
                consolidated_results[key][context_key] = decision_dict
        
        # Handle any failed tasks that need to be retried
        if failed_tasks:
            print(f"\n===== Retrying {len(failed_tasks)} failed tasks =====\n")
            
            # Create retry tasks
            retry_task_coros = [bounded_trial_task(*task_args) for task_args in failed_tasks]
            
            # Process retry results
            for task_coro in asyncio.as_completed(retry_task_coros):
                result = await task_coro
                if result:  # Only add non-None results
                    instance_name, replicate_id, instance_decision_dict = result
                    
                    # Extract base instance name
                    base_instance_name = instance_name.rsplit('-', 1)[0]
                    
                    # Update consolidated results
                    key = (base_instance_name, replicate_id)
                    if key not in consolidated_results:
                        consolidated_results[key] = {}
                    
                    for context_key, decision_dict in instance_decision_dict.items():
                        consolidated_results[key][context_key] = decision_dict
        
        # Save all results after all instances are processed
        print("All instances processed. Saving results...")
        for (base_instance_name, replicate_id), instance_decision_dict in consolidated_results.items():
            file_name = f"{base_instance_name}_{level}_{replicate_id}"
            print(f"Saving {file_name}.csv")
            await saveDecisionsAsync(output_dir, file_name, instance_decision_dict)
        
        print(f"All tasks completed for {decision_maker_name}")

if __name__ == '__main__':
    data_dir = os.path.join(project_root, "data")
    TIME_OUT = 3600
    with open(os.path.join(data_dir, "dataInfo.json"), 'r', encoding='utf-8') as f:
        data_info = json.load(f)
    instance_name_list = [
        "-".join([data_info[i]["id"], data_info[i]["name"]]) for i in range(len(data_info))
    ]
    print(instance_name_list)
    context_dict_list = [
        data_info[i]["contexts"] for i in range(len(data_info))
    ] # a list of lists, each list contains dicts with keys "id" and "values"
    context_dict_list = [
        {context["id"]: context["values"] for context in context_dict} for context_dict in context_dict_list
    ]

    # initilaize language models
    def backbone_llm():
        return Gpt4o(temperature=0)
    def sc_llm():
        return Gpt4o(temperature=0.5)
    config_path = "config.yaml"

    # Create factory functions for decision makers
    def create_random_dm():
        return RandomDecisionMaker(config_path)
    
    def create_vanilla_dm():
        return VanillaDecisionMaker(backbone_llm())
    
    def create_cot_dm():
        return CotDecisionMaker(backbone_llm())
    
    def create_sc_dm():
        return ScDecisionMaker(sc_llm())
    
    def create_dellma_dm():
        return DellmaDecisionMaker(backbone_llm(), config_path)
    
    def create_aid_dm():
        return AidDecisionMaker(backbone_llm(), timeout=TIME_OUT, max_retries=3)

    # List of decision maker factory functions
    dm_factory_list = [
        create_dellma_dm
    ]

    # Run the async function with asyncio.run
    asyncio.run(runDecisionExperiment(
        data_dir,
        instance_name_list,
        context_dict_list,
        dm_factory_list,  # passing factory functions instead of instances
        "complete",
        n_replicate=5,
        concurrency_limit=64,
        task_timeout=TIME_OUT
    ))
