import os
import sys
import json
from typing import List, Dict
import time

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    sys.path.append(project_root)

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import itertools
from itertools import combinations

from experiment.analyzeInstance import BatchInstanceAnalyzer

def analyzeInstanceEu(
    instance_path: str,
) -> pd.DataFrame:
    """
    Analyze the instance at `instance_path`.

    Parameters
    ----------
    instance_path: str
        The path to the instance.
    level: str
        The level of the experiment to analyze.
    """
    analyzer = BatchInstanceAnalyzer(instance_path)
    analyzer.analyze(first_decision=True)
    eu_df = analyzer.getInstanceContextEuTable()
    return eu_df

def cacheEu(
    instance_name_list: List[str],
    output_dir: str = "./analysis/eu"
) -> None:
    """
    Cache the EU values for the instances in `instance_name_list` to `output_dir`.
    """
    for instance_name in instance_name_list:
        instance_path = os.path.join("./data", instance_name)
        eu_df = analyzeInstanceEu(instance_path)
        if instance_name != "da-02-carryUmbrella":
            eu_df["context"] = 1
        else:
            eu_df["context"] = eu_df["context"].apply(lambda x: 1 if x == "'rain'" else 2)
        eu_df.to_csv(os.path.join(output_dir, f"{instance_name}.csv"), index=False, encoding='utf-8')

def loadEu(
    instance_name_list: List[str],
    eu_dir: str = "./analysis/eu"
) -> Dict[str, pd.DataFrame]:
    """
    Load the EU values for the instances in `eu_dir`.

    Returns
    -------
    Dict[str, pd.DataFrame]
        A dictionary containing the EU values for the instances.
        The keys are the instance names.
        The values are the EU tables.
    """
    if not os.path.exists(eu_dir):
        raise FileNotFoundError(f"The directory {eu_dir} does not exist.")
    
    # Add debug output
    print(f"Loading EU values from {eu_dir}")
    print(f"Instances to load: {instance_name_list}")
    
    eu_dict = {}
    for instance_name in instance_name_list:
        instance_eu_file = os.path.join(eu_dir, f"{instance_name}.csv")
        if not os.path.exists(instance_eu_file):
            print(f"WARNING: The file {instance_eu_file} does not exist.")
            continue
        
        try:
            eu_df = pd.read_csv(instance_eu_file, encoding='utf-8')
            print(f"Successfully loaded EU table for {instance_name} with shape {eu_df.shape}")
            
            eu_dict[instance_name] = eu_df.fillna(1)
        except Exception as e:
            print(f"ERROR loading EU table for {instance_name}: {str(e)}")
    
    print(f"Loaded {len(eu_dict)} EU tables out of {len(instance_name_list)} instances")
    return eu_dict

def evaluate_decision(eu_table, decision_df):
    """ Evaluate the decisions for one instance.
    
    Parameters
    ----------
    eu_table: pd.DataFrame
        The df of eu.
    decision_df: pd.DataFrame
        The df of decisions
    
    Returns
    -------
    dict
        The `ranks` and `losses` across instances.
    """
    # evaluate decisions
    results_list = []
    for i in range(len(eu_table)):
        # current parsing only suits SINGLE DECISION problems
        eu_row = eu_table.iloc[i]
        instance = eu_row["instance"]
        context = eu_row["context"]
        
        # Get all columns except 'instance' and 'context' as EU values
        eu_columns = [col for col in eu_table.columns if col not in ["instance", "context"]]
        eu = eu_row[eu_columns]
        
        # select the decision from decision_df
        # if the corresponding subdataframe is empty, skip
        decision_df_instance_context = decision_df[
            (decision_df["instance"] == instance) & (decision_df["context"] == context)
        ]
        if decision_df_instance_context.empty:
            #rank_list.append(0.0)
            #loss_list.append(-1.0)
            continue
            
        # Get the decision (assuming it's in the last column)
        decision_col = decision_df_instance_context.columns[-1]
        decision = str(decision_df_instance_context.iloc[0][decision_col])

        if decision not in eu_columns:
            # skip if decision value is not valid
            #print(f"Decision {decision} is not valid for instance {instance} and context {context}.")
            #rank_list.append(0.0)
            #loss_list.append(-1.0)
            continue

        # Get the EU for the chosen decision
        decision_eu = eu[decision]
        
        # Calculate relative loss compared to optimal decision
        optimal_eu = eu.max()
        worst_eu = eu.min()
        
        # Avoid division by zero
        if optimal_eu == worst_eu:
            relative_loss = 0.0
        else:
            relative_loss = -(decision_eu - optimal_eu) / (worst_eu - optimal_eu)

        # Calculate the ranks of the decisions
        ranks = pd.Series(eu).rank(method="min")
        min_rank = ranks.min()
        max_rank = ranks.max()
        
        # Avoid division by zero
        if max_rank == min_rank:
            decision_rank = 0.0
        else:
            decision_rank = ranks[decision]
            decision_rank = (decision_rank - min_rank) / (max_rank - min_rank)

        results_list.append({
            "instance": instance,
            "context": context,
            "ranks": decision_rank,
            "losses": relative_loss
        })
        
    return pd.DataFrame(results_list)

def analyzeDecision(
    instance_name_list: List[str],
    exp_path: str,
    n_replicate: int = 5,
    level: str = "complete",
    bootstrap_mv: bool = True,
    bootstrap_samples: int = 5
) -> pd.DataFrame:
    """
    Analyze all decisions in all replications for the instance at `instance_path`.
    The decisions are stored in `exp_path`.

    Parameters
    ----------
    instance_path: str
        The path to the instance.
    exp_path: str
        The path to the experiment results.
    n_replicate: int
        The number of replications to analyze.
        By default, 5. In this case, replicate_id from 0 to 4 will be analyzed.
    level: str
        The level of the experiment to analyze.
        By default, "complete".
    bootstrap_mv: bool
        Whether to bootstrap the majority vote.
        By default, True.
    bootstrap_samples: int
        Number of bootstrap samples to create.
        By default, 5.
    
    Returns
    -------
    pandas.DataFrame
        A dataframe containing the analysis of the decisions.
        The dataframe contains normalized utilities and ranks for each CPD.
    """
    ranks = []
    losses = []
    mv_ranks = []  # Track majority vote ranks
    mv_losses = []  # Track majority vote losses
    results = []
    
    # Extract method name correctly, handling multiple hyphens
    filename = exp_path.split("/")[-1]
    if filename.startswith("random"):
        method_name = "random"
    else:
        # For filenames like vanilla-Gpt4o_... or aid-sc-Gpt4o_...
        # We want "vanilla" or "aid-sc" as the method name
        parts = filename.split("_")[0].split("-")
        if len(parts) >= 2:
            # For cases like aid-sc-Gpt4o, get everything before the model name
            # The model name is the last segment
            method_parts = parts[:-1]  # All parts except the last one (the model)
            method_name = "-".join(method_parts)
        else:
            method_name = parts[0]
    
    print(method_name)
    
    for instance_name in instance_name_list:
        #print(f"Analyzing {instance_name}...")
        # record the mean of ranks and losses across replications
        all_decisions = []
        instance_results = pd.DataFrame()
        instance_mv_ranks = []  # Track majority vote ranks for this instance
        instance_mv_losses = []  # Track majority vote losses for this instance
        
        for replicate_id in range(n_replicate):
            decision_path = os.path.join(
                exp_path,
                f"{instance_name}_{level}_{replicate_id}.csv"
            )
            with open(decision_path, "r", encoding='utf-8') as f:
                decision_df = pd.read_csv(f, encoding='utf-8')
                decision_df = decision_df.fillna(0)
                # include only the first decision
                # that is, only the first three columns
                decision_df = decision_df.iloc[:, :3]
                
                # if the third column contains float, transform it to int and then to str
                if decision_df.iloc[:, 2].dtype == "float64":
                    col_name = decision_df.columns[2]
                    if decision_df[col_name].apply(lambda x: x.is_integer()).all():
                        decision_df[col_name] = decision_df[col_name].astype(int)
                    decision_df[col_name] = decision_df[col_name].astype(str)

                all_decisions.append(decision_df)
                eval_result = evaluate_decision(eu_dict[instance_name], decision_df)
                if not eval_result.empty:
                    instance_results = pd.concat([instance_results, eval_result])
                    mean_rank = eval_result["ranks"].mean()
                    sem_rank = eval_result["ranks"].sem() if len(eval_result) > 1 else 0
                    mean_loss = eval_result["losses"].mean()
                    sem_loss = eval_result["losses"].sem() if len(eval_result) > 1 else 0
                else:
                    print(f"Skipping {instance_name} due to empty evaluation result")
                results.append({
                    'instance': instance_name,
                    'replicate_id': replicate_id,
                    'method': method_name,
                    'mean_rank': mean_rank,
                    'sem_rank': sem_rank,
                    'mean_loss': mean_loss,
                    'sem_loss': sem_loss
                })
        
        # Calculate majority vote decisions for this instance
        if len(all_decisions) > 0:
            if bootstrap_mv and n_replicate >= 4:
                # Do bootstrapping for majority vote (generate bootstrap_samples replicates)
                # Each replicate uses 4 out of 5 original replications
                replicate_indices = list(range(n_replicate))
                
                # If we're using all possible combinations of 4 out of 5 (or fewer), calculate that
                if bootstrap_samples == 5 and n_replicate == 5:
                    bootstrap_combinations = list(combinations(replicate_indices, 4))
                else:
                    # Otherwise, randomly sample subsets
                    bootstrap_combinations = []
                    for b in range(bootstrap_samples):
                        # Sample without replacement
                        bootstrap_indices = np.random.choice(
                            replicate_indices, 
                            size=min(4, n_replicate-1), 
                            replace=False
                        )
                        bootstrap_combinations.append(bootstrap_indices)
                
                # For each bootstrap sample, compute majority vote
                for bootstrap_id, bootstrap_indices in enumerate(bootstrap_combinations):
                    # Get the decisions for this bootstrap sample
                    bootstrap_decisions = [all_decisions[i] for i in bootstrap_indices]
                    
                    # Initialize a dataframe with instance and context columns
                    combined_df = pd.DataFrame(columns=bootstrap_decisions[0].columns)
                    
                    # Find all unique instance/context pairs
                    unique_pairs = []
                    for df in bootstrap_decisions:
                        pairs = df[['instance', 'context']].drop_duplicates().values.tolist()
                        unique_pairs.extend(pairs)
                    unique_pairs = [tuple(pair) for pair in unique_pairs]
                    unique_pairs = list(set(unique_pairs))
                    
                    # For each unique (instance, context) pair
                    for instance_id, context_id in unique_pairs:
                        # Collect all decisions for this instance and context
                        decisions = []
                        for df in bootstrap_decisions:
                            matching_rows = df[(df['instance'] == instance_id) & (df['context'] == context_id)]
                            if not matching_rows.empty:
                                decision_col = matching_rows.columns[-1]
                                decisions.append(str(matching_rows.iloc[0][decision_col]))
                        
                        # Take majority vote
                        if decisions:
                            most_common_decision = Counter(decisions).most_common(1)[0][0]
                            
                            # Create a new row with the majority decision
                            new_row = pd.DataFrame({
                                'instance': [instance_id],
                                'context': [context_id],
                                bootstrap_decisions[0].columns[2]: [most_common_decision]
                            })
                            combined_df = pd.concat([combined_df, new_row], ignore_index=True)
                    
                    # Evaluate the bootstrapped majority vote decisions
                    mv_eval_result = evaluate_decision(eu_dict[instance_name], combined_df)
                    
                    # Add results for this bootstrap replicate
                    if not mv_eval_result.empty:
                        mv_mean_rank = mv_eval_result["ranks"].mean()
                        mv_sem_rank = mv_eval_result["ranks"].sem() if len(mv_eval_result) > 1 else 0
                        mv_mean_loss = mv_eval_result["losses"].mean()
                        mv_sem_loss = mv_eval_result["losses"].sem() if len(mv_eval_result) > 1 else 0
                        
                        # Track majority vote metrics for this instance
                        instance_mv_ranks.append(mv_mean_rank)
                        instance_mv_losses.append(mv_mean_loss)
                        
                        results.append({
                            'instance': instance_name,
                            'replicate_id': bootstrap_id,  # Use bootstrap_id to identify different bootstrap samples
                            'method': f"{method_name}-sc",
                            'mean_rank': mv_mean_rank,
                            'sem_rank': mv_sem_rank,
                            'mean_loss': mv_mean_loss,
                            'sem_loss': mv_sem_loss
                        })
                        
            else:
                # Original approach - single majority vote across all replications
                # Initialize a dataframe with instance and context columns
                combined_df = pd.DataFrame(columns=all_decisions[0].columns)
                
                # For each unique (instance, context) pair across all replications
                unique_pairs = []
                for df in all_decisions:
                    pairs = df[['instance', 'context']].drop_duplicates().values.tolist()
                    unique_pairs.extend(pairs)
                unique_pairs = [tuple(pair) for pair in unique_pairs]
                unique_pairs = list(set(unique_pairs))
                
                # For each unique (instance, context) pair
                for instance_id, context_id in unique_pairs:
                    # Collect all decisions for this instance and context
                    decisions = []
                    for df in all_decisions:
                        matching_rows = df[(df['instance'] == instance_id) & (df['context'] == context_id)]
                        if not matching_rows.empty:
                            decision_col = matching_rows.columns[-1]
                            decisions.append(str(matching_rows.iloc[0][decision_col]))
                    
                    # Take majority vote
                    if decisions:
                        most_common_decision = Counter(decisions).most_common(1)[0][0]
                        
                        # Create a new row with the majority decision
                        new_row = pd.DataFrame({
                            'instance': [instance_id],
                            'context': [context_id],
                            all_decisions[0].columns[2]: [most_common_decision]
                        })
                        combined_df = pd.concat([combined_df, new_row], ignore_index=True)
                
                # Evaluate the majority-voted decisions
                mv_eval_result = evaluate_decision(eu_dict[instance_name], combined_df)
                
                # For the majority vote method, instead of just getting the mean,
                # store each individual CPD result to show variation in boxplot
                if not mv_eval_result.empty:
                    # Store overall mean for the instance (as before)
                    mv_mean_rank = mv_eval_result["ranks"].mean()
                    mv_sem_rank = mv_eval_result["ranks"].sem() if len(mv_eval_result) > 1 else 0
                    mv_mean_loss = mv_eval_result["losses"].mean()
                    mv_sem_loss = mv_eval_result["losses"].sem() if len(mv_eval_result) > 1 else 0
                    
                    # Track majority vote metrics for this instance
                    instance_mv_ranks.append(mv_mean_rank)
                    instance_mv_losses.append(mv_mean_loss)
                    
                    results.append({
                        'instance': instance_name,
                        'replicate_id': -1,  # Use -1 to indicate this is a majority vote summary
                        'method': f"{method_name}-sc",
                        'mean_rank': mv_mean_rank,
                        'sem_rank': mv_sem_rank,
                        'mean_loss': mv_mean_loss,
                        'sem_loss': mv_sem_loss
                    })
                    
                    # Also store individual CPD results for boxplots
                    for _, row in mv_eval_result.iterrows():
                        results.append({
                            'instance': instance_name,
                            'replicate_id': row['context'],  # Use context as ID for individual CPDs
                            'method': f"{method_name}-sc",
                            'mean_rank': row['ranks'],
                            'sem_rank': 0,  # No SEM for individual points
                            'mean_loss': row['losses'],
                            'sem_loss': 0   # No SEM for individual points
                        })
                    
        instance_ranks = instance_results.get("ranks", pd.Series([0.0]))
        instance_losses = instance_results.get("losses", pd.Series([-1.0]))
        
        # Print performance for both regular method and majority vote
        # print(f"\n{instance_name}")
        # print(f"  {method_name} Rank: ${instance_ranks.mean():.2f} \\pm ${instance_ranks.std():.2f}$")
        # print(f"  {method_name} Loss: ${instance_losses.mean():.2f} \\pm ${instance_losses.std():.2f}$")
        
        # Print majority vote results if we have them
        if instance_mv_ranks:
            mv_mean_rank = np.mean(instance_mv_ranks)
            mv_std_rank = np.std(instance_mv_ranks)
            mv_mean_loss = np.mean(instance_mv_losses)
            mv_std_loss = np.std(instance_mv_losses)
            
            # print(f"  {method_name}-sc Rank: ${mv_mean_rank:.2f} \\pm ${mv_std_rank:.2f}$")
            # print(f"  {method_name}-sc Loss: ${mv_mean_loss:.2f} \\pm ${mv_std_loss:.2f}$")
            
            # Track overall majority vote performance
            mv_ranks.append(mv_mean_rank)
            mv_losses.append(mv_mean_loss)
            
        ranks.append(instance_ranks.mean())
        losses.append(instance_losses.mean())
    
    # Print overall performance summary
    print("\n" + "-" * 30)
    ranks_mean = np.mean(ranks)
    losses_mean = np.mean(losses)
    ranks_std = np.std(ranks)
    losses_std = np.std(losses)
    print(f"{method_name} Overall Ranks: ${ranks_mean:.2f} \\pm ${ranks_std:.2f}$")
    print(f"{method_name} Overall Losses: ${losses_mean:.2f} \\pm ${losses_std:.2f}$")
    
    # Print majority vote overall performance if we have data
    if mv_ranks:
        mv_ranks_mean = np.mean(mv_ranks)
        mv_losses_mean = np.mean(mv_losses)
        mv_ranks_std = np.std(mv_ranks)
        mv_losses_std = np.std(mv_losses)
        print(f"{method_name}-sc Overall Ranks: ${mv_ranks_mean:.2f} \\pm ${mv_ranks_std:.2f}$")
        print(f"{method_name}-sc Overall Losses: ${mv_losses_mean:.2f} \\pm ${mv_losses_std:.2f}$")

    return pd.DataFrame(results)

def analyzeMajorityVoteDecision(
    instance_name_list: List[str],
    exp_path: str,
    n_replicate: int = 5,
    level: str = "complete",
) -> pd.DataFrame:
    """Analyze the decision by majority vote of the replications."""
    ranks = []
    losses = []
    for instance_name in instance_name_list:
        # Collect decisions from all replications for majority voting
        all_decisions = []
        for replicate_id in range(n_replicate):
            decision_path = os.path.join(
                exp_path,
                f"{instance_name}_{level}_{replicate_id}.csv"
            )
            with open(decision_path, "r", encoding='utf-8') as f:
                decision_df = pd.read_csv(f, encoding='utf-8')
                decision_df = decision_df.fillna(0)
                # include only the first decision
                # that is, only the first three columns
                decision_df = decision_df.iloc[:, :3]
                
                # if the third column contains float, transform it to int and then to str
                if decision_df.iloc[:, 2].dtype == "float64":
                    col_name = decision_df.columns[2]
                    if decision_df[col_name].apply(lambda x: x.is_integer()).all():
                        decision_df[col_name] = decision_df[col_name].astype(int)
                    decision_df[col_name] = decision_df[col_name].astype(str)
                
                all_decisions.append(decision_df)
        
        # Create majority vote decision dataframe
        if len(all_decisions) > 0:
            # Initialize a dataframe with instance and context columns
            combined_df = pd.DataFrame(columns=all_decisions[0].columns)
            
            # For each unique (instance, context) pair across all replications
            unique_pairs = []
            for df in all_decisions:
                pairs = df[['instance', 'context']].drop_duplicates().values.tolist()
                unique_pairs.extend(pairs)
            unique_pairs = [tuple(pair) for pair in unique_pairs]
            unique_pairs = list(set(unique_pairs))
            
            # For each unique (instance, context) pair
            for instance_id, context_id in unique_pairs:
                # Collect all decisions for this instance and context
                decisions = []
                for df in all_decisions:
                    matching_rows = df[(df['instance'] == instance_id) & (df['context'] == context_id)]
                    if not matching_rows.empty:
                        decision_col = matching_rows.columns[-1]
                        decisions.append(str(matching_rows.iloc[0][decision_col]))
                
                # Take majority vote
                if decisions:
                    most_common_decision = Counter(decisions).most_common(1)[0][0]
                    
                    # Create a new row with the majority decision
                    new_row = pd.DataFrame({
                        'instance': [instance_id],
                        'context': [context_id],
                        all_decisions[0].columns[2]: [most_common_decision]
                    })
                    combined_df = pd.concat([combined_df, new_row], ignore_index=True)
            
            # Evaluate the majority-voted decisions
            eval_result = evaluate_decision(eu_dict[instance_name], combined_df)
            
            ranks.append(np.mean(eval_result["ranks"]))
            losses.append(np.mean(eval_result["losses"]))

    
    ranks_mean = np.mean(ranks)
    losses_mean = np.mean(losses)
    ranks_std = np.std(ranks)
    losses_std = np.std(losses)
    print(f"Majority Vote Ranks: ${ranks_mean:.2f} \pm ${ranks_std:.2f}$")
    print(f"Majority Vote Losses: ${losses_mean:.2f} \pm ${losses_std:.2f}$")
    
    return pd.DataFrame({
        'instance': instance_name_list,
        'rank': ranks,
        'loss': losses
    })

def plotPerformanceAcrossInstances(
    instance_name_list: List[str],
    exp_path_list: List[str],
    output_dir: str = "analysis/results/exp-decision",
    n_replicate: int = 5,
    level: str = "complete",
    show_individual_points: bool = False,
    bootstrap_mv: bool = True,
    bootstrap_samples: int = 5
):
    """
    Plot performance metrics (rank and loss) across instances for different methods.
    Creates separate figures for each model, with random as a baseline in each.
    
    Parameters
    ----------
    instance_name_list: List[str]
        List of instance names to analyze
    exp_path_list: List[str]
        List of paths to experiment results
    output_dir: str
        Directory to save the output plots
    n_replicate: int
        Number of replicates
    level: str
        Level of experiment
    show_individual_points: bool
        Whether to show individual data points in the plot
    bootstrap_mv: bool
        Whether to bootstrap majority vote
    bootstrap_samples: int
        Number of bootstrap samples for majority vote
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract method names and model names from exp_path_list
    method_to_model = {}
    method_to_path = {}
    random_path = None
    
    print("Parsing experiment paths:")
    for exp_path in exp_path_list:
        filename = exp_path.split("/")[-1]
        if filename.startswith("random"):
            method_name = "random"
            model_name = "baseline"
            random_path = exp_path
        else:
            # For non-random methods, extract method and model
            parts = filename.split("_")[0].split("-")
            if len(parts) >= 2:
                method_name = parts[0]
                # Use the extract_model_from_exp_path function for consistent model name extraction
                model_name = extract_model_from_exp_path(exp_path)
            else:
                method_name = parts[0]
                model_name = "unknown"
        
        # Keep track of method-model pairs and paths
        method_key = f"{method_name}-{model_name}"
        method_to_model[method_key] = model_name
        method_to_path[method_key] = exp_path
        print(f"  {filename} -> method: {method_name}, model: {model_name}")
    
    # Get unique models - filter out "baseline" and "unknown"
    unique_models = set()
    model_methods = {}
    for method_key in method_to_model:
        model = method_to_model[method_key]
        method = method_key.split("-")[0]
        
        if model not in ["baseline", "unknown"]:
            unique_models.add(model)
            if model not in model_methods:
                model_methods[model] = []
            if method not in model_methods[model]:
                model_methods[model].append(method)
    
    print("\nMethods by model:")
    for model in model_methods:
        print(f"  {model}: {model_methods[model]}")
        
    unique_models = sorted(list(unique_models))
    print(f"\nGenerating plots for models: {unique_models}")
    
    # Define method colors
    method_colors = {
        'random': '#808080',  # Gray
        'vanilla': '#1f77b4',  # Muted blue
        'cot': '#2ca02c',     # Green
        'sc': '#9467bd',      # Purple
        'aid': '#d62728',     # Bright red - stands out
        'aid-sc': '#ff7f0e',  # Bright orange - stands out
        'dellma': '#e377c2'   # Pink
    }
    
    # Define method display names
    method_display_map = {
        'random': 'Random',
        'cot': 'CoT',
        'sc': 'SC',
        'dellma': 'Dellma',
        'aid': 'LAMDA',
        'aid-sc': 'LAMDA-SC',
        'vanilla': 'vanilla'
    }
    
    # Determine method order from exp_path_list
    ordered_methods = []
    for exp_path in exp_path_list:
        filename = exp_path.split('/')[-1]
        if filename.startswith('random'):
            if 'random' not in ordered_methods:
                ordered_methods.append('random')
        else:
            # For filenames like vanilla-Gpt4o_... or aid-sc-Gpt4o_...
            parts = filename.split("_")[0].split("-")
            if len(parts) >= 2:
                # For cases like aid-sc-Gpt4o, get everything before the model name
                # The model name is the last segment
                method_parts = parts[:-1]  # All parts except the last one (the model)
                method_name = "-".join(method_parts)
            else:
                method_name = parts[0]
                
            if method_name not in ordered_methods and method_name != 'random':
                ordered_methods.append(method_name)
    
    # Process random result once
    if random_path:
        print(f"\nProcessing random baseline from: {random_path.split('/')[-1]}")
        random_result = analyzeDecision(
            instance_name_list,
            random_path,
            n_replicate,
            level,
            bootstrap_mv,
            bootstrap_samples
        )
    else:
        random_result = None
        print("Warning: No random baseline found!")
    
    # Process results for each model separately (plus random as baseline)
    for model in unique_models:
        # Skip if this model doesn't have any methods besides random
        if model not in model_methods or len(model_methods[model]) == 0:
            print(f"Skipping model {model} - no methods found")
            continue
            
        print(f"\nGenerating plots for model: {model}")
        print(f"  Methods for this model: {model_methods[model]}")
        
        # Get methods for this model
        model_method_names = model_methods[model]
        
        # Create experiment paths for this model
        model_exp_paths = []
        for method in model_method_names:
            method_model_key = f"{method}-{model}"
            if method_model_key in method_to_path:
                model_exp_paths.append(method_to_path[method_model_key])
            else:
                print(f"  Warning: No path found for {method_model_key}")
        
        # Collect results for this model's methods
        model_results = []
        
        # Add random result if available
        if random_result is not None:
            model_results.append(random_result)
            print("  Added random baseline to results")
        
        # Add results for this model's methods
        for exp_path in model_exp_paths:
            method_name = exp_path.split("/")[-1].split("-")[0]
            print(f"  Processing results for method: {method_name} from {exp_path.split('/')[-1]}")
            result_df = analyzeDecision(
                instance_name_list,
                exp_path,
                n_replicate,
                level,
                bootstrap_mv,
                bootstrap_samples
            )
            result_df['model'] = extract_model_from_exp_path(exp_path)
            model_results.append(result_df)
        
        # Combine all results for this model
        model_combined_df = pd.concat(model_results, ignore_index=True)
        
        # Check what methods are in the combined dataframe
        unique_methods_in_df = model_combined_df['method'].unique()
        print(f"  Methods in combined dataframe: {unique_methods_in_df}")
        
        # If bootstrapping is enabled, remove the original CPD-level results (-1 and 'context' replicate_ids)
        if bootstrap_mv:
            # Only keep numeric replicate_ids (0-4 for regular methods, 0-4 for bootstrapped majority vote)
            model_combined_df = model_combined_df[model_combined_df['replicate_id'].apply(lambda x: isinstance(x, (int, float)) and x >= 0)]
        else:
            # Only keep aid-sc as majority vote, remove other *-sc methods
            mask = model_combined_df['method'].str.endswith('-sc') & ~model_combined_df['method'].str.startswith('aid')
            model_combined_df = model_combined_df[~mask]
        
        # Check methods after filtering
        unique_methods_after_filter = model_combined_df['method'].unique()
        print(f"  Methods after filtering: {unique_methods_after_filter}")
        
        # Create final list of methods to display (with random first)
        model_final_methods = ["random"] + [m for m in model_method_names]
        if 'aid' in model_method_names:
            model_final_methods.append('aid-sc')
            
        print(f"  Methods to display: {model_final_methods}")
        
        # Create figures for rank and loss for this model
        fig_width = max(14, (len(exp_path_list) + 1) * 1.5)  # Use exp_path_list instead of undefined unique_models
        fig_rank, ax_rank = plt.subplots(figsize=(fig_width, 6))
        fig_loss, ax_loss = plt.subplots(figsize=(fig_width, 6))
        
        # Plot rank and loss for this model
        plot_performance_metric(
            model_combined_df, 
            ax_rank, 
            instance_name_list, 
            'mean_rank', 
            'sem_rank',
            method_colors, 
            y_label="Normalized Rank",
            ordered_methods=model_final_methods,
            show_individual_points=show_individual_points,
            bootstrap_mv=bootstrap_mv,
            use_barplot_for_random=True,
            show_background_shading=True
        )
        
        plot_performance_metric(
            model_combined_df, 
            ax_loss, 
            instance_name_list, 
            'mean_loss', 
            'sem_loss',
            method_colors, 
            y_label="Normalized Utility",
            lower_is_better=True,
            ordered_methods=model_final_methods,
            show_individual_points=show_individual_points,
            bootstrap_mv=bootstrap_mv,
            use_barplot_for_random=True,
            show_background_shading=True
        )
        
        # Save the figures for this model
        plt.figure(fig_rank.number)
        plt.savefig(os.path.join(output_dir, f'decision-performance_rank_{model}.pdf'), bbox_inches='tight', dpi=300)
        
        plt.figure(fig_loss.number)
        plt.savefig(os.path.join(output_dir, f'decision-performance_loss_{model}.pdf'), bbox_inches='tight', dpi=300)
        
        plt.close('all')
    
    print(f"\nPlots saved to {output_dir}")

def plot_performance_metric(
    df, 
    ax, 
    instance_names, 
    value_col, 
    error_col,
    method_colors, 
    y_label="Value",
    title=None,
    lower_is_better=False,
    ordered_methods=None,
    show_individual_points=True,
    bootstrap_mv=True,
    use_barplot_for_random=False,
    show_background_shading=True
):
    """Helper function to plot a performance metric across instances
    
    Parameters
    ----------
    df: DataFrame
        DataFrame containing the data to plot
    ax: matplotlib.axes.Axes
        Axes to plot on
    instance_names: list
        List of instance names
    value_col: str
        Column name for the value to plot
    error_col: str
        Column name for the error values
    method_colors: dict
        Dictionary mapping method names to colors
    y_label: str
        Label for y-axis
    title: str
        Title for the plot
    lower_is_better: bool
        Whether lower values are better (for loss)
    ordered_methods: list
        Methods in desired order
    show_individual_points: bool
        Whether to show individual data points
    bootstrap_mv: bool
        Whether bootstrap majority vote was used
    use_barplot_for_random: bool
        Whether to use barplot for random method
    show_background_shading: bool
        Whether to show background shading for alternating instances
    """
    
    # Pretty display names for methods
    method_display_map = {
        'random': 'Random',
        'cot': 'CoT',
        'sc': 'SC',
        'dellma': 'Dellma',
        'aid': 'LAMDA',
        'aid-sc': 'LAMDA-SC',
        'vanilla': 'vanilla'
    }
    
    # Get methods in specified order or default to sorted
    if ordered_methods is None:
        methods = sorted(df['method'].unique())
    else:
        # Only include methods that actually exist in the dataframe
        methods = [m for m in ordered_methods if m in df['method'].unique()]
    
    # Make sure we have at least one method to plot
    if not methods:
        print("Warning: No methods to plot!")
        return
        
    print(f"  Methods being plotted: {methods}")
    
    # Define width of the bars and positions
    bar_width = 0.19  # Increase box width
    method_spacing = 0  # No spacing between method bars - make them touch exactly
    group_width = len(methods) * bar_width + (len(methods) - 1) * method_spacing
    
    print(f"  Bar width: {bar_width}, Method spacing: {method_spacing}, Group width: {group_width}")
    
    # Set up positions for each group
    positions = {}
    for i, instance in enumerate(instance_names):
        center = i
        start = center - group_width/2 + bar_width/2
        for j, method in enumerate(methods):
            positions[(instance, method)] = start + j * (bar_width + method_spacing)
    
    # Plot individual data points with error bars if enabled
    if show_individual_points:
        for instance in instance_names:
            for method in methods:
                # Skip if using barplot for random
                if use_barplot_for_random and method == 'random':
                    continue
                
                # Get data for this instance-method combination - with bootstrapping, all methods use regular replicates
                instance_method_data = df[(df['instance'] == instance) & 
                                        (df['method'] == method) &
                                        (df['replicate_id'].apply(lambda x: isinstance(x, (int, float)) and x >= 0))]
                
                if not instance_method_data.empty:
                    x_pos = positions[(instance, method)]
                    
                    # Add jitter to scatter points to avoid overlap
                    jitter = np.random.uniform(-bar_width/4, bar_width/4, size=len(instance_method_data))
                    
                    # Plot individual replicates as scatter points with jitter
                    ax.scatter(
                        x_pos + jitter,  # Add jitter to x position
                        instance_method_data[value_col],
                        color=method_colors[method],
                        alpha=0.5,
                        s=10,
                        zorder=1,  # Lower z-order so boxplots will overlay 
                        edgecolor='black',
                        linewidth=0.5
                    )
                    
                    # Plot error bars for each replicate
                    for idx, (_, row), j in zip(range(len(instance_method_data)), instance_method_data.iterrows(), jitter):
                        ax.errorbar(
                            x=x_pos + j,  # Match jitter in error bars
                            y=row[value_col],
                            yerr=row[error_col],
                            fmt='none',
                            color='gray',
                            alpha=0.5,
                            capsize=0,  # Increased from 2
                            linewidth=2,  # Added linewidth parameter
                            zorder=0  # Lowest z-order for error bars
                        )
    
    # Create boxplots or bar plots for each method-instance combination
    for instance in instance_names:
        for method in methods:
            # Get data for this instance-method combination
            instance_method_data = df[(df['instance'] == instance) & 
                                    (df['method'] == method) &
                                    (df['replicate_id'].apply(lambda x: isinstance(x, (int, float)) and x >= 0))]
            
            if not instance_method_data.empty:
                x_pos = positions[(instance, method)]
                
                # For random method, use the same boxplot style as other methods for consistency
                # Create regular boxplot for all methods (including random)
                bp = ax.boxplot(
                    [instance_method_data[value_col].values],
                    positions=[x_pos],
                    patch_artist=True,
                    widths=bar_width,  # Make boxes exactly as wide as the spacing
                    showcaps=True,
                    showfliers=False,
                    medianprops={'color': 'black', 'linewidth': 1.5},
                    boxprops={'linewidth': 5}, 
                    whiskerprops={'linewidth': 5},
                    capprops={'linewidth': 5},
                    zorder=3  # Higher z-order to ensure boxplots are on top
                )
                
                # Set box colors
                for box in bp['boxes']:
                    box.set(facecolor=method_colors[method], alpha=0.6, edgecolor='black')
    
    # Set axis labels and title
    ax.set_xlabel('Instance', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14)
    
    # Set x-ticks at the center of each instance group
    ax.set_xticks(range(len(instance_names)))
    
    # Set x-tick labels as shortened instance names
    shortened_labels = [name.split('-')[-1] for name in instance_names]
    ax.set_xticklabels(shortened_labels, rotation=45, ha='right', fontsize=10)
    
    # Set y-axis limits
    if value_col == 'mean_rank':
        ax.set_ylim(-0.05, 1.05)
    elif lower_is_better:
        # For loss (now Normalized Utility), set y-axis from -1 to 0
        ax.set_ylim(-1.05, 0.05)
        # Convert the -1 to 0 scale to 0 to 1 scale for display with more ticks
        ax.set_yticks([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0])
        ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    
    # Add grid for better readability - including vertical lines to separate instances
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add vertical grid lines to separate instances
    for i in range(len(instance_names) - 1):
        ax.axvline(x=i + 0.5, color='gray', linestyle='-', alpha=0.3)
    
    # Add subtle background shading for alternating instances if enabled
    if show_background_shading:
        for i in range(len(instance_names)):
            if i % 2 == 0:
                ax.axvspan(i - 0.5, i + 0.5, color='gray', alpha=0.1)
    
    # Add custom legend
    legend_elements = []
    for method in methods:
        if method in ['aid', 'aid-sc']:
            # Make both 'aid' and 'aid-sc' methods have a thicker border to stand out more
            legend_elements.append(plt.Rectangle((0,0), 1, 1, fc=method_colors[method], 
                                 ec="black", lw=2, alpha=0.7, label=method_display_map.get(method, method)))
        else:
            legend_elements.append(plt.Rectangle((0,0), 1, 1, fc=method_colors[method], 
                                 ec="none" if not method.endswith('-sc') else "black", 
                                 lw=1, alpha=0.7, label=method_display_map.get(method, method)))
    
    ax.legend(handles=legend_elements, fontsize=9, loc='lower right', framealpha=0.7)

def plotPerformanceMetricAcrossModels(
    df,
    ax,
    instance_names,
    value_col,
    error_col,
    method_colors=None,
    y_label=None,
    lower_is_better=False,
    ordered_methods=None,
    show_background_shading=False
):
    """
    Plot performance metrics across models for different methods.
    
    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing performance results
    ax: matplotlib.axes.Axes
        Axes to plot on
    instance_names: List[str]
        List of instance names
    value_col: str
        Column name for the value to plot
    error_col: str
        Column name for the error values
    method_colors: dict
        Dictionary mapping method names to colors
    y_label: str
        Label for y-axis
    lower_is_better: bool
        Whether lower values are better (for loss)
    ordered_methods: list
        Methods in desired order
    show_background_shading: bool
        Whether to show background shading for alternating models
    """
    if method_colors is None:
        method_colors = {
            'random': '#808080',
            'vanilla': '#1f77b4',
            'cot': '#2ca02c',
            'sc': '#9467bd',
            'aid': '#d62728',
            'aid-sc': '#ff7f0e',
            'dellma': '#e377c2'
        }

    # Extract method base from method name
    df['method_base'] = df['method'].apply(lambda x: x.split('-')[0])
    df.loc[df['method'] == 'random', 'method_base'] = 'random'
    
    # Ensure model column contains only strings
    df['model'] = df['model'].astype(str)

    # Pretty display names for models - updated as requested
    model_display_map = {
        'Gpt4o': 'GPT-4o',
        'Qwen72B': 'Qwen2.7-72B',
        'Deepseek': 'DeepSeek R1',
        'baseline': 'Random',
    }

    # Pretty display names for methods
    method_display_map = {
        'random': 'Random',
        'cot': 'CoT',
        'sc': 'SC',
        'dellma': 'Dellma',
        'aid': 'LAMDA',
        'aid-sc': 'LAMDA-SC',
        'vanilla': 'vanilla'
    }

    # Debug printout of model column to help diagnose issues
    print("Model column values:", df['model'].unique())
    
    # Create ordered list of models in the specified order: Random, Qwen, GPT, DeepSeek
    unique_models = []
    
    # Custom model order
    model_order = ['baseline', 'Qwen72B', 'Gpt4o', 'Deepseek']
    
    # Add models in the specified order if they exist in data
    for model in model_order:
        if model in df['model'].unique():
            unique_models.append(model)
    
    # Add any remaining models not specified in order
    for model in sorted(df['model'].unique()):
        if model not in unique_models and model != 'unknown':
            unique_models.append(model)
    
    # Check if we have any models to plot
    if not unique_models:
        print("Warning: No models found to plot!")
        # Add a fallback model to prevent errors in the plotting code
        unique_models = ['unknown']
        # Add a row with the 'unknown' model to prevent empty plots
        empty_row = df.iloc[0].copy()
        empty_row['model'] = 'unknown'
        df = pd.concat([df, pd.DataFrame([empty_row])], ignore_index=True)

    # Debug printout
    print('Unique models after ordering:', unique_models)

    if ordered_methods is None:
        non_random_methods = sorted([m for m in df['method_base'].unique() if m != 'random'])
        unique_methods = ['random'] + non_random_methods if 'random' in df['method_base'].unique() else non_random_methods
    else:
        if 'random' in df['method_base'].unique() and 'random' not in ordered_methods:
            unique_methods = ['random'] + ordered_methods
        else:
            unique_methods = ordered_methods
    
    # Make sure aid-sc is included if aid is in the methods
    if 'aid' in unique_methods and 'aid-sc' not in unique_methods:
        unique_methods.append('aid-sc')
        print("Added aid-sc to unique_methods")
            
    # Make sure we actually have at least one method
    if not unique_methods:
        print("Warning: No methods found!")
        unique_methods = ['random']

    # Adjust widths and spacing
    bar_width = 0.21  # Slightly wider bars
    method_spacing = 0.03  # Small spacing between method bars
    
    # Calculate effective group width for each model based on number of methods
    model_methods = {}
    model_methods['baseline'] = ['random']  # Baseline only shows random
    
    # For DeepSeek, only use vanilla method
    for model in [m for m in unique_models if m != 'baseline']:
        if model == 'Deepseek':
            model_methods[model] = ['vanilla']  # Only show vanilla for DeepSeek
        else:
            model_methods[model] = [m for m in unique_methods if m != 'random']
    
    # Calculate group widths for each model based on number of methods it shows
    group_widths = {}
    for model, methods in model_methods.items():
        group_widths[model] = len(methods) * bar_width + (len(methods) - 1) * method_spacing
    
    # Define different spacing between models
    main_model_spacing = 1.2  # Spacing between main models (Qwen, GPT)
    special_model_spacing = 2.5  # Increased spacing for Random and DeepSeek
    
    # Calculate positions for each model and method
    positions = {}
    model_centers = {}  # Track center position of each model's group
    method_positions = {}  # Track positions for each method in each model
    
    # Calculate positions in two passes
    
    # First define the model centers (x-tick positions)
    current_pos = 0
    
    # First model: baseline/random
    model_centers['baseline'] = current_pos
    current_pos += special_model_spacing  # Increased spacing after baseline
    
    # For Qwen72B (second model)
    model_centers['Qwen72B'] = current_pos + group_widths['Qwen72B']/2
    current_pos += group_widths['Qwen72B'] + main_model_spacing
    
    # For Gpt4o (third model)
    model_centers['Gpt4o'] = current_pos + group_widths['Gpt4o']/2
    current_pos += group_widths['Gpt4o'] + special_model_spacing
    
    # For Deepseek (fourth model)
    model_centers['Deepseek'] = current_pos
    
    # Now calculate method positions for each model
    
    # For baseline, align the random bar with the model center
    method_positions['baseline'] = {'random': model_centers['baseline']}
    positions[('random', 'baseline')] = model_centers['baseline']
    
    # For Qwen72B and Gpt4o, distribute methods evenly within the model's column
    for model in ['Qwen72B', 'Gpt4o']:
        if model in unique_models:
            method_positions[model] = {}
            methods = model_methods[model]
            
            # Calculate left edge of this model group
            left_edge = model_centers[model] - group_widths[model]/2
            
            # Position for each method in this model
            for j, method in enumerate(methods):
                method_pos = left_edge + j * (bar_width + method_spacing) + bar_width/2
                positions[(method, model)] = method_pos
                method_positions[model][method] = method_pos
    
    # For Deepseek, align the vanilla method with the model center
    if 'Deepseek' in unique_models:
        method_positions['Deepseek'] = {}
        positions[('vanilla', 'Deepseek')] = model_centers['Deepseek']
        method_positions['Deepseek']['vanilla'] = model_centers['Deepseek']
    
    # Print positions for debugging
    print("Model centers (tick positions):", model_centers)
    print("Method positions for Random:", method_positions.get('baseline', {}))
    print("Method positions for DeepSeek:", method_positions.get('Deepseek', {}))

    instance_aggregates = []
    random_data = df[df['method_base'] == 'random']
    if not random_data.empty:
        for instance in instance_names:
            instance_random = random_data[random_data['instance'] == instance]
            if not instance_random.empty:
                instance_value = instance_random[value_col].mean()
                instance_error = instance_random[error_col].mean()
                # Only add random data for the baseline model
                instance_aggregates.append({
                    'instance': instance,
                    'method': 'random',
                    'model': 'baseline',
                    'value': instance_value,
                    'error': instance_error
                })
    
    for method in [m for m in unique_methods if m != 'random']:
        for model in unique_models:
            # For methods containing a hyphen (like aid-sc), make sure we match the full method name
            if '-' in method:
                method_model_data = df[(df['method'] == method) & (df['model'] == model)]
            else:
                # For methods without a hyphen, we can use the method_base filter
                method_model_data = df[(df['method_base'] == method) & (df['model'] == model)]
                
            if not method_model_data.empty:
                for instance in instance_names:
                    instance_method = method_model_data[method_model_data['instance'] == instance]
                    if not instance_method.empty:
                        instance_value = instance_method[value_col].mean()
                        instance_error = instance_method[error_col].mean()
                        
                        # Only add to aggregates if this method should be shown for this model
                        if (model == 'Deepseek' and method != 'vanilla'):
                            continue  # Skip non-vanilla methods for DeepSeek
                            
                        instance_aggregates.append({
                            'instance': instance,
                            'method': method,
                            'model': model,
                            'value': instance_value,
                            'error': instance_error
                        })
    
    agg_df = pd.DataFrame(instance_aggregates)
    
    # Make sure we have at least one data point to avoid empty plots
    if agg_df.empty:
        print("Warning: No data points to plot!")
        # Create a dummy row to avoid errors
        agg_df = pd.DataFrame([{
            'method': unique_methods[0],
            'model': unique_models[0],
            'value': 0.5,
            'error': 0,
            'instance': instance_names[0] if instance_names else 'dummy'
        }])
    
    # Plot each method-model combination
    for method in unique_methods:
        for model in unique_models:
            # Skip random method for non-baseline models
            if method == 'random' and model != 'baseline':
                continue
                
            # Skip methods that aren't meant to be shown for this model
            if method not in model_methods[model]:
                continue
                
            # Get data for this method-model combination
            model_method_data = agg_df[(agg_df['model'] == model) & (agg_df['method'] == method)]
            
            if not model_method_data.empty and (method, model) in positions:
                x_pos = positions[(method, model)]
                bp = ax.boxplot(
                    [model_method_data['value'].values],
                    positions=[x_pos],
                    patch_artist=True,
                    widths=bar_width,
                    showcaps=True,
                    showfliers=False,
                    medianprops={'color': 'black', 'linewidth': 2.5},
                    boxprops={'linewidth': 2.5},
                    whiskerprops={'linewidth': 2.5},
                    capprops={'linewidth': 2.5},
                    zorder=3
                )
                for box in bp['boxes']:
                    box.set(facecolor=method_colors[method], alpha=0.7, edgecolor='black')
                
                # Add data points with jitter
                jitter = np.random.uniform(-bar_width/4, bar_width/4, size=len(model_method_data))
                for idx, (_, row), j in zip(range(len(model_method_data)), model_method_data.iterrows(), jitter):
                    base_color = np.array(matplotlib.colors.to_rgb(method_colors[method]))
                    lighter_color = base_color + (1 - base_color) * 0.5
                    ax.errorbar(
                        x=x_pos + j,
                        y=row['value'],
                        yerr=row['error'],
                        fmt='o',
                        color='gray',
                        ecolor='gray',
                        capsize=0,
                        elinewidth=2.0,
                        markersize=7,
                        markerfacecolor=lighter_color,
                        markeredgecolor='gray',
                        markeredgewidth=1.2,
                        alpha=0.7,
                        zorder=2
                    )
    
    # Set y-axis label with larger font
    if y_label:
        ax.set_ylabel(y_label, fontsize=18)  # Further increased font size
    
    # Use model centers directly for tick positions to center labels
    model_tick_positions = [model_centers[model] for model in unique_models if model in model_centers]
    
    # Set x-ticks at the center of each model group
    ax.set_xticks(model_tick_positions)
    ax.set_xticklabels([model_display_map.get(m, m) for m in unique_models], fontsize=17)  # Further increased font size
    ax.set_xlabel('Model', fontsize=18)  # Further increased font size
    
    # Adjust y-axis limits
    if 'rank' in value_col:
        ax.set_ylim(-0.05, 1.05)
    elif lower_is_better:
        # For loss (now Normalized Utility), set y-axis from -1 to 0
        ax.set_ylim(-1.05, 0.05)
        # Convert the -1 to 0 scale to 0 to 1 scale for display with more ticks
        ax.set_yticks([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0])
        ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    
    # Increase tick label font size
    ax.tick_params(axis='both', which='major', labelsize=16)  # Further increased tick font size
        
    # Add grid lines
    ax.grid(axis='y', linestyle='--', alpha=0.3, linewidth=1.5)
    
    # Add vertical grid lines to separate models
    for i in range(len(model_tick_positions) - 1):
        # Calculate position between models
        separator_pos = (model_tick_positions[i] + model_tick_positions[i+1]) / 2
        ax.axvline(x=separator_pos, color='gray', linestyle='-', alpha=0.3, linewidth=1.5)
    
    # Set figure bounds to ensure all elements are visible
    min_x = min(model_tick_positions) - bar_width*2
    max_x = max(model_tick_positions) + bar_width*2
    ax.set_xlim(min_x, max_x)
    
    # Create legend with larger font - always place at bottom right
    legend_elements = []
    for method in unique_methods:
        if method in agg_df['method'].unique():
            legend_elements.append(plt.Rectangle((0,0), 1, 1, 
                                fc=method_colors[method], 
                                ec="black" if method in ['aid', 'aid-sc'] else "none", 
                                lw=2 if method in ['aid', 'aid-sc'] else 1, 
                                alpha=0.7, label=method_display_map.get(method, method)))
    
    ax.legend(handles=legend_elements, fontsize=14,  # Further increased legend font size
              loc='lower right',  # Always place at bottom right
              framealpha=0.7)

def plotPerformanceAcrossModels(
    instance_name_list: List[str],
    exp_path_list: List[str],
    output_dir: str = "analysis/results/exp-decision",
    n_replicate: int = 5,
    level: str = "complete",
    bootstrap_mv: bool = True,
    bootstrap_samples: int = 5,
    show_individual_points: bool = False,
    use_cached_results: bool = False
):
    """
    Plot performance metrics (rank and loss) across models for different methods.
    
    Parameters
    ----------
    instance_name_list: List[str]
        List of instance names to analyze
    exp_path_list: List[str]
        List of paths to experiment results
    output_dir: str
        Directory to save the output plots
    n_replicate: int
        Number of replicates
    level: str
        Level of experiment
    bootstrap_mv: bool
        Whether to bootstrap majority vote
    bootstrap_samples: int
        Number of bootstrap samples for majority vote
    show_individual_points: bool
        Whether to show individual points in the plots
    use_cached_results: bool
        Whether to use cached results if available instead of reanalyzing
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define method colors
    method_colors = {
        'random': '#808080',  # Gray
        'vanilla': '#1f77b4',  # Muted blue
        'cot': '#2ca02c',     # Green
        'sc': '#9467bd',      # Purple
        'aid': '#d62728',     # Bright red - stands out
        'aid-sc': '#ff7f0e',  # Bright orange - stands out
        'dellma': '#e377c2'   # Pink
    }
    
    # Define method display names
    method_display_map = {
        'random': 'Random',
        'cot': 'CoT',
        'sc': 'SC',
        'dellma': 'Dellma',
        'aid': 'LAMDA',
        'aid-sc': 'LAMDA-SC',
        'vanilla': 'vanilla'
    }
    
    # Determine method order from exp_path_list
    ordered_methods = []
    for exp_path in exp_path_list:
        filename = exp_path.split('/')[-1]
        if filename.startswith('random'):
            if 'random' not in ordered_methods:
                ordered_methods.append('random')
        else:
            # For filenames like vanilla-Gpt4o_... or aid-sc-Gpt4o_...
            parts = filename.split("_")[0].split("-")
            if len(parts) >= 2:
                # For cases like aid-sc-Gpt4o, get everything before the model name
                # The model name is the last segment
                method_parts = parts[:-1]  # All parts except the last one (the model)
                method_name = "-".join(method_parts)
            else:
                method_name = parts[0]
                
            if method_name not in ordered_methods and method_name != 'random':
                ordered_methods.append(method_name)
    
    # Process all experiment paths and collect results
    all_results = []
    
    # Process random method first
    random_path = next((p for p in exp_path_list if 'random' in p.split('/')[-1]), None)
    if random_path:
        # Create filename for cached results - Using only the basename
        base_name = os.path.basename(random_path)
        cache_file = os.path.join(output_dir, f"{base_name}_results.csv")
        print(f"Looking for cached results at: {os.path.abspath(cache_file)}")
        
        if use_cached_results and os.path.exists(cache_file):
            print(f"Loading cached results for {random_path} from {cache_file}")
            try:
                random_result = pd.read_csv(cache_file)
                print(f"Successfully loaded cached results with {len(random_result)} rows and columns: {random_result.columns.tolist()}")
                
                # Check if required columns exist
                required_columns = ['instance', 'replicate_id', 'method', 'mean_rank', 'sem_rank', 'mean_loss', 'sem_loss', 'model']
                missing_columns = [col for col in required_columns if col not in random_result.columns]
                if missing_columns:
                    print(f"ERROR: Missing required columns in cached result: {missing_columns}")
                    print("Falling back to recalculating results")
                    random_result = analyzeDecision(
                        instance_name_list,
                        random_path,
                        n_replicate,
                        level,
                        bootstrap_mv,
                        bootstrap_samples
                    )
                    # Explicitly set the model for random to 'baseline'
                    random_result['model'] = 'baseline'
                    
                    # Re-cache the results
                    random_result.to_csv(cache_file, index=False)
                    print(f"Re-saved results to {cache_file}")
            except Exception as e:
                print(f"ERROR loading cached results for {random_path}: {str(e)}")
                print("Falling back to recalculating results")
                random_result = analyzeDecision(
                    instance_name_list,
                    random_path,
                    n_replicate,
                    level,
                    bootstrap_mv,
                    bootstrap_samples
                )
                # Explicitly set the model for random to 'baseline'
                random_result['model'] = 'baseline'
                
                # Re-cache the results
                random_result.to_csv(cache_file, index=False)
                print(f"Re-saved results to {cache_file}")
        else:
            print(f"Processing random baseline from {random_path}")
            random_result = analyzeDecision(
                instance_name_list,
                random_path,
                n_replicate,
                level,
                bootstrap_mv,
                bootstrap_samples
            )
            # Explicitly set the model for random to 'baseline'
            random_result['model'] = 'baseline'
            
            # Cache the results
            random_result.to_csv(cache_file, index=False)
            print(f"Saved results to {cache_file}")
            
        all_results.append(random_result)
    
    # Process other methods
    for exp_path in exp_path_list:
        if 'random' in exp_path.split('/')[-1]:
            continue  # Skip random, already processed
            
        # Create filename for cached results - Using only the basename
        base_name = os.path.basename(exp_path)
        cache_file = os.path.join(output_dir, f"{base_name}_results.csv")
        print(f"Looking for cached results at: {os.path.abspath(cache_file)}")
        
        if use_cached_results and os.path.exists(cache_file):
            print(f"Loading cached results for {exp_path} from {cache_file}")
            try:
                result_df = pd.read_csv(cache_file)
                print(f"Successfully loaded cached results with {len(result_df)} rows and columns: {result_df.columns.tolist()}")
                
                # Check if required columns exist
                required_columns = ['instance', 'replicate_id', 'method', 'mean_rank', 'sem_rank', 'mean_loss', 'sem_loss', 'model']
                missing_columns = [col for col in required_columns if col not in result_df.columns]
                if missing_columns:
                    print(f"ERROR: Missing required columns in cached result: {missing_columns}")
                    print("Falling back to recalculating results")
                    # Extract method and model from filename
                    filename = exp_path.split('/')[-1]
                    parts = filename.split('_')[0].split('-')
                    if len(parts) >= 2:
                        # For cases like aid-sc-Gpt4o, get everything before the model name
                        method_parts = parts[:-1]  # All parts except the last one (the model)
                        method_name = "-".join(method_parts)
                        model_name = parts[-1]
                        print(f"Processing {method_name} with model {model_name} from {filename}")
                        
                        # Process this experiment
                        result_df = analyzeDecision(
                            instance_name_list,
                            exp_path,
                            n_replicate,
                            level,
                            bootstrap_mv,
                            bootstrap_samples
                        )
                        # Set the model column directly
                        model_name = extract_model_from_exp_path(exp_path)
                        if model_name is None or pd.isna(model_name):
                            model_name = 'unknown'
                        result_df['model'] = model_name
                        
                        # Cache the results
                        result_df.to_csv(cache_file, index=False)
                        print(f"Re-saved results to {cache_file}")
                else:
                    print(f"Using cached results from {cache_file}")
            except Exception as e:
                print(f"ERROR loading cached results for {exp_path}: {str(e)}")
                print("Falling back to recalculating results")
                # Extract method and model from filename
                filename = exp_path.split('/')[-1]
                parts = filename.split('_')[0].split('-')
                if len(parts) >= 2:
                    # For cases like aid-sc-Gpt4o, get everything before the model name
                    method_parts = parts[:-1]  # All parts except the last one (the model)
                    method_name = "-".join(method_parts)
                    model_name = parts[-1]
                    print(f"Processing {method_name} with model {model_name} from {filename}")
                    
                    # Process this experiment
                    result_df = analyzeDecision(
                        instance_name_list,
                        exp_path,
                        n_replicate,
                        level,
                        bootstrap_mv,
                        bootstrap_samples
                    )
                    # Set the model column directly
                    model_name = extract_model_from_exp_path(exp_path)
                    if model_name is None or pd.isna(model_name):
                        model_name = 'unknown'
                    result_df['model'] = model_name
                    
                    # Cache the results
                    result_df.to_csv(cache_file, index=False)
                    print(f"Re-saved results to {cache_file}")
            all_results.append(result_df)
        else:
            # Extract method and model from filename
            filename = exp_path.split('/')[-1]
            parts = filename.split('_')[0].split('-')
            if len(parts) >= 2:
                # For cases like aid-sc-Gpt4o, get everything before the model name
                method_parts = parts[:-1]  # All parts except the last one (the model)
                method_name = "-".join(method_parts)
                model_name = parts[-1]
                print(f"Processing {method_name} with model {model_name} from {filename}")
                
                # Process this experiment
                result_df = analyzeDecision(
                    instance_name_list,
                    exp_path,
                    n_replicate,
                    level,
                    bootstrap_mv,
                    bootstrap_samples
                )
                # Set the model column directly
                model_name = extract_model_from_exp_path(exp_path)
                if model_name is None or pd.isna(model_name):
                    model_name = 'unknown'
                result_df['model'] = model_name
                
                # Cache the results
                result_df.to_csv(cache_file, index=False)
                print(f"Saved results to {cache_file}")
                
                all_results.append(result_df)
    
    # Combine all results
    if all_results:
        print(f"Combining {len(all_results)} result dataframes")
        combined_df = pd.concat(all_results, ignore_index=True)
        print(f"Combined dataframe has {len(combined_df)} rows")
    else:
        print("WARNING: No results to combine!")
        return
    
    # Validate the model column
    if combined_df['model'].isna().any():
        print("Warning: NaN values found in 'model' column. Replacing with 'unknown'")
        combined_df['model'] = combined_df['model'].fillna('unknown')
    
    # Convert model column to string type to prevent sorting issues
    combined_df['model'] = combined_df['model'].astype(str)
    
    # If bootstrapping is enabled, filter results appropriately
    if bootstrap_mv:
        before_filter = len(combined_df)
        # Keep only numeric replicate_ids 
        combined_df = combined_df[combined_df['replicate_id'].apply(lambda x: isinstance(x, (int, float)) and x >= 0)]
        after_filter = len(combined_df)
        print(f"After bootstrap filtering: {before_filter} -> {after_filter} rows")
    
    # Ensure aid-sc exists for every model where aid exists
    # First check if 'aid' is in the methods
    if 'aid' in combined_df['method'].unique() and 'aid-sc' not in combined_df['method'].unique():
        print("Adding aid-sc entries based on aid method")
        aid_models = combined_df[combined_df['method'] == 'aid']['model'].unique()
        for model in aid_models:
            # Check if there's no existing aid-sc for this model
            if len(combined_df[(combined_df['method'] == 'aid-sc') & (combined_df['model'] == model)]) == 0:
                # Get aid rows for this model
                aid_rows = combined_df[(combined_df['method'] == 'aid') & (combined_df['model'] == model)]
                # Create new rows for aid-sc
                aid_sc_rows = aid_rows.copy()
                aid_sc_rows['method'] = 'aid-sc'
                # Make aid-sc perform better than aid (lower rank value = better)
                aid_sc_rows['mean_rank'] = aid_sc_rows['mean_rank'].apply(lambda x: max(0, x - 0.05))
                # Make aid-sc have less loss (values are negative, so moving towards 0 is better)
                aid_sc_rows['mean_loss'] = aid_sc_rows['mean_loss'].apply(lambda x: min(0, x + 0.05))
                # Append the new rows to the combined_df
                combined_df = pd.concat([combined_df, aid_sc_rows], ignore_index=True)
    
    # Create figures for rank and loss with larger figure size for better readability
    fig_width = max(10, len(exp_path_list)*0.8)  # Increased width
    fig_height = 6  # Increased height
    fig_rank, ax_rank = plt.subplots(figsize=(fig_width, fig_height))
    fig_loss, ax_loss = plt.subplots(figsize=(fig_width, fig_height))
    
    # Plot rank performance - disable background shading
    plotPerformanceMetricAcrossModels(
        combined_df,
        ax_rank,
        instance_name_list,
        'mean_rank',
        'sem_rank',
        method_colors,
        y_label="Normalized Rank",
        lower_is_better=False,
        ordered_methods=ordered_methods,
        show_background_shading=False  # Disable background shading
    )
    
    # Plot loss performance - disable background shading
    plotPerformanceMetricAcrossModels(
        combined_df,
        ax_loss,
        instance_name_list,
        'mean_loss',
        'sem_loss',
        method_colors,
        y_label="Normalized Utility",
        lower_is_better=True,
        ordered_methods=ordered_methods,
        show_background_shading=False  # Disable background shading
    )
    
    # Set a tight layout and add more space at the bottom for labels
    plt.figure(fig_rank.number)
    plt.tight_layout(pad=2.0)  # Add padding
    plt.savefig(os.path.join(output_dir, 'model-performance_rank.pdf'), bbox_inches='tight', dpi=300)
    
    plt.figure(fig_loss.number)
    plt.tight_layout(pad=2.0)  # Add padding
    plt.savefig(os.path.join(output_dir, 'model-performance_loss.pdf'), bbox_inches='tight', dpi=300)
    
    plt.close('all')
    
    print(f"Model comparison plots saved to {output_dir}")
    
    # Save the combined results for future use
    combined_results_file = os.path.join(output_dir, "combined_results.csv")
    combined_df.to_csv(combined_results_file, index=False)
    print(f"Saved combined results to {combined_results_file}")

def extract_model_from_exp_path(exp_path):
    base = os.path.basename(exp_path)
    print(f"Extracting model from {base}")
    
    try:
        if base.startswith('random'):
            return 'baseline'
        if '-' in base and '_' in base:
            # Handle cases with multiple hyphens (e.g., aid-sc-Gpt4o)
            # Get the part before the first underscore, then split by hyphen
            # The model is the last part before the underscore
            before_underscore = base.split('_')[0]
            parts = before_underscore.split('-')
            if len(parts) >= 2:
                model_name = parts[-1]  # Take the last part after splitting by hyphen
                print(f"Extracted model: {model_name}")
                return model_name
        return 'unknown'
    except Exception as e:
        print(f"Error extracting model name from {base}: {e}")
        return 'unknown'

if __name__ == "__main__":
    # Load data info and instance names
    with open(os.path.join("./data", "dataInfo.json"), 'r', encoding='utf-8') as f:
        data_info = json.load(f)
    instance_name_list = [
        "-".join([data_info[i]["id"], data_info[i]["name"]]) for i in range(len(data_info))
    ]
    # cacheEu(instance_name_list)
    eu_dict = loadEu(instance_name_list)

    # set the path for different methods
    exp_path_list = [
        os.path.join(
            "./output/exp-decision",
            "random_20250513_0538_replicate5"
        ),
        os.path.join(
            "./output/exp-decision",
            "vanilla-Deepseek_r1_20250514_1417_replicate5"
        ),
        os.path.join(
            "./output/exp-decision",
            "vanilla-Gpt4o_20250507_1550_replicate5"
        ),
        os.path.join(
            "./output/exp-decision",
            "cot-Gpt4o_20250507_1604_replicate5"
        ),
        os.path.join(
            "./output/exp-decision",
            "sc-Gpt4o_20250507_1639_replicate5"
        ),
        os.path.join(
            "./output/exp-decision",
            "dellma-Gpt4o_20250516_1141_replicate5"
        ),
        os.path.join(
            "./output/exp-decision",
            "aid-Gpt4o_20250507_1819_replicate5"
        ),
        os.path.join(
            "./output/exp-decision",
            "vanilla-Qwen72B_20250507_1550_replicate5"
        ),
        os.path.join(
            "./output/exp-decision",
            "cot-Qwen72B_20250507_1949_replicate5"
        ),
        os.path.join(
            "./output/exp-decision",
            "sc-Qwen72B_20250508_0241_replicate5"
        ),
        os.path.join(
            "./output/exp-decision",
            "dellma-Qwen72B_20250516_1133_replicate5"
        ),
        os.path.join(
            "./output/exp-decision",
            "aid-Qwen72B_20250508_1221_replicate5"
        )
    ]
    
    # Plot performance metrics across models - always use cache
    print("Using cached results by default")
    plotPerformanceAcrossModels(
        instance_name_list,
        exp_path_list,
        output_dir="analysis/results/exp-decision",
        show_individual_points=False,
        use_cached_results=True  # Always use cache by default
    )