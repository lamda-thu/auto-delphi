import os
import json
import sys
import warnings
from itertools import product
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Callable, Union

import numpy as np
import yaml

script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir)

sys.path.append(PROJECT_ROOT)
sys.path.insert(0, os.path.abspath('.'))

from experiment.decisionMakerBase import DecisionMakerBase

@dataclass
class ActionConfig:
    action_enum_mode: str = "base"


@dataclass
class StateConfig:
    state_enum_mode: str = "base"
    states: Dict[str, str] = field(default_factory=dict)
    topk: int = 3


@dataclass
class PreferenceConfig:
    pref_enum_mode: str = "base"
    sample_size: int = 50  # number of states to be sampled from the belief distribution
    minibatch_size: int = 50  # size of the minimbatch
    overlap_pct: float = 0.2  # percentage of overlap between minibatches


class DeLLMaAgent:
    temperature: float = 0.0
    belief2score: Dict[str, float] = {
        "very likely": 6,
        "likely": 5,
        "somewhat likely": 4,
        "somewhat unlikely": 3,
        "unlikely": 2,
        "very unlikely": 1,
    }

    def __init__(
        self,
        choices: List[str],
        context: str,
        temperature: float = 0.0,
        utility_prompt: Optional[str] = "I want to make the optimal decision that maximizes expected utility.",
        state_config: Optional[StateConfig] = StateConfig(
            state_enum_mode="sequential",
            topk=3
        ),
        action_config: Optional[ActionConfig] = ActionConfig(
            action_enum_mode="base"
        ),
        preference_config: Optional[PreferenceConfig] = PreferenceConfig(
            pref_enum_mode="rank-minibatch",
            minibatch_size=8,
            sample_size=4,
            overlap_pct=0.25
        ),
        llm=None,
        max_retries: int = 3,
    ):
        """
        Initialize the DeLLMaAgent.
        
        Parameters:
        -----------
        choices: List[str]
            A list of available choices/actions for the agent
        context: str
            The raw context
        temperature: float
            Temperature for model inference
        utility_prompt: Optional[str]
            Custom utility prompt for the agent
        state_config: Optional[StateConfig]
            Configuration for state enumeration
        action_config: Optional[ActionConfig]
            Configuration for action enumeration
        preference_config: Optional[PreferenceConfig]
            Configuration for preference elicitation
        llm: Optional
            Language model to use for inference, if applicable
        max_retries: int
            Maximum number of retries for LLM calls
        """
       
        # In-memory storage as direct attributes
        self.factors = None
        self.states = None
        
        # Set default configurations if not provided
        if state_config is None:
            state_config = StateConfig(state_enum_mode="sequential")
        if action_config is None:
            action_config = ActionConfig()
        if preference_config is None:
            preference_config = PreferenceConfig(pref_enum_mode="rank")
        if utility_prompt is None:
            utility_prompt = "I want to make the optimal decision that maximizes expected utility."

        self.choices = choices
        self.action_strs = self.choices
        self.context = context
        self.temperature = temperature
        self.utility_prompt = utility_prompt
        self.state_config = state_config
        self.action_config = action_config
        self.preference_config = preference_config
        self.llm = llm
        self.max_retries = max_retries
        
        # Initialized but will be populated during workflow
        self.state_beliefs = None
        self.belief_dist = None
        self.decision = None
    
    def initialize_workflow(self, regenerate_states=False, regenerate_beliefs=False):
        """
        Initialize the complete DeLLMa workflow.
        
        Parameters:
        -----------
        regenerate_states: bool
            Whether to regenerate state variables even if in memory
        regenerate_beliefs: bool
            Whether to regenerate belief distribution even if in memory
            
        Returns:
        --------
        self: The initialized agent ready for decision making
        """
        
        # Step 1: Generate state variables
        # print("Step 1:\n Generating state variables...")
        if regenerate_states or self.factors is None:
            self.state_config.states = self.format_state_dict(llm=self.llm)
        else:
            self.state_config.states = self._factors_to_state_dict(self.factors)
        
        # print(f"STATE ENUMERATION: {len(self.state_config.states)} state variables.")
        
        # Step 2: Generate belief distribution
        # print("Step 2:\n Generating belief distribution over states...")
        
        if regenerate_beliefs or self.states is None:
            belief_prompt = self.prepare_belief_dist_generation_prompt()
            
            if self.llm:
                # print("Using LLM to generate belief distribution...")
                
                # Retry mechanism for belief distribution generation
                for attempt in range(self.max_retries):
                    try:
                        belief_distribution_response = self.llm.invoke(
                            belief_prompt,
                            temperature=self.temperature,
                        )
                        # Parse the response
                        if isinstance(belief_distribution_response, str):
                            parsed_response = self.extract_json_from_response(belief_distribution_response)
                        else:
                            # Assume response is already in the correct format (dict)
                            parsed_response = belief_distribution_response
                            
                        # Store belief distribution in memory
                        self.states = parsed_response
                        break
                    except json.JSONDecodeError:
                        if attempt < self.max_retries - 1:
                            print(f"Invalid JSON response. Retrying (attempt {attempt+1}/{self.max_retries})...")
                        else:
                            raise ValueError("Failed to generate valid belief distribution after multiple retries")
            else:
                raise Exception("No LLM provided. Cannot generate belief distribution.")
        
        # Load belief distribution
        self.belief_dist = self.load_state_beliefs()
        # print(f"STATE BELIEF DISTRIBUTION: {len(self.belief_dist)} state variables.")
        
        return self

    def transform_preferences_to_utility(
        self, 
        preferences,
        mode="pairwise", 
        alpha=0.01, 
        temperature=1.0,
        softmax_mode="full"
    ):
        """
        Transform pairwise or rank preferences into utility values.
        
        Parameters:
        -----------
        preferences: Dict
            Dictionary containing preference data (decision, rank or pair)
        mode: str
            Mode for utility calculation, either "pairwise" or "top1"
        alpha: float
            Regularization parameter for utility calculation
        temperature: float
            Temperature parameter for softmax normalization
        softmax_mode: str
            How to apply softmax: 'full' (over all scores) or 'action' (per action)
        
        Returns:
        --------
        Tuple[Dict[str, float], List[Tuple]]: 
            - Dictionary mapping choices to utility values
            - List of state-action pairs with their utility scores
        """
        # Import dependencies if not already imported
        try:
            import choix
            from scipy.special import softmax
        except ImportError:
            raise ImportError("This function requires choix and scipy packages. Please install them.")
        
        # Extract state-action pairs and preferences
        state_action_pairs = []
        
        # Check if state_action_pairs exist in the preferences
        if "state_action_pairs" not in preferences:
            raise ValueError("Preferences must contain 'state_action_pairs' key")
        
        # Parse the preferences based on format
        if "rank" in preferences:
            # Handle rank-based preferences
            ranks = preferences["rank"]
            data = []
            
            # Create indexed state-action pairs
            for i, rank_pos in enumerate(ranks):
                if i < len(preferences["state_action_pairs"]):
                    state_action_str = preferences["state_action_pairs"][i]
                    state_action_pairs.append((i, self._parse_state_action_pair(state_action_str)))
                else:
                    # print(f"Warning: Missing state-action pair for rank position {i+1}")
                    # Create a dummy state-action pair if missing
                    pass
                    # state_action_pairs.append((i, (["Unknown state"], "Unknown action")))
            
            if mode == "top1":
                # Format for top1: [top_item, [other_items]]
                for i in range(len(ranks) - 1):
                    data.append([ranks[i], [ranks[j] for j in range(i+1, len(ranks))]])
            elif mode == "pairwise":
                # Format for pairwise: pairs of [preferred, less_preferred]
                data = []
                for i in range(len(ranks) - 1):
                    for j in range(i+1, len(ranks)):
                        data.append([ranks[i], ranks[j]])
        
        elif "pair" in preferences:
            # Handle pairwise comparison-based preferences
            pairs = preferences["pair"]
            data = pairs
            
            # Create indexed state-action pairs
            for pair in pairs:
                for idx in pair:
                    idx_adj = idx - 1  # Adjust index (pairs are 1-indexed)
                    if idx_adj < len(preferences["state_action_pairs"]):
                        state_action_str = preferences["state_action_pairs"][idx_adj]
                        # Check if this state-action pair is already in our list
                        pair_exists = any(p[0] == idx_adj for p in state_action_pairs)
                        if not pair_exists:
                            state_action_pairs.append((idx_adj, self._parse_state_action_pair(state_action_str)))
                    else:
                        # print(f"Warning: Missing state-action pair for index {idx}")
                        # Create a dummy state-action pair if missing
                        pass
                        # state_action_pairs.append((idx_adj, (["Unknown state"], "Unknown action")))
        
        else:
            raise ValueError("Preferences must contain either 'rank' or 'pair' key")
        
        # Calculate utility scores
        n_items = len(state_action_pairs)
        
        if mode == "pairwise":
            util_func = lambda data: choix.ilsr_pairwise(
                n_items=n_items, 
                data=data,
                alpha=alpha,
                max_iter=10_000
            )
        elif mode == "top1":
            util_func = lambda data: choix.ilsr_top1(
                n_items=n_items, 
                data=data,
                alpha=alpha,
                max_iter=10_000
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Calculate raw utility scores
        raw_scores = util_func(data)
        
        # Apply softmax normalization
        if softmax_mode == "full":
            scores = softmax(raw_scores / temperature)
        elif softmax_mode == "action":
            scores = np.copy(raw_scores)
            action_groups = {}
            
            # Group scores by action
            for i, (_, state_action) in enumerate(state_action_pairs):
                action = state_action[1]  # Assuming state_action is (state, action)
                if action not in action_groups:
                    action_groups[action] = []
                action_groups[action].append(i)
            
            # Apply softmax per action group
            for action, indices in action_groups.items():
                action_scores = [scores[i] for i in indices]
                normalized = softmax(np.array(action_scores) / temperature)
                for idx, norm_score in zip(indices, normalized):
                    scores[idx] = norm_score
        
        # Create utility dictionary per choice (action)
        choice_utilities = {}
        action_scores = {}
        
        for i, (_, state_action) in enumerate(state_action_pairs):
            state, action = state_action
            if action not in action_scores:
                action_scores[action] = []
            action_scores[action].append(scores[i])
        
        # Average scores per action
        for action, action_score_list in action_scores.items():
            choice_utilities[action] = np.mean(action_score_list)
        
        # Create state-action utility list
        state_action_utilities = []
        for i, (_, state_action) in enumerate(state_action_pairs):
            state_action_utilities.append((state_action, scores[i]))
        
        return choice_utilities, state_action_utilities

    def _parse_state_action_pair(self, state_action_str):
        """
        Parse a state-action pair string into state and action components.
        
        Parameters:
        -----------
        state_action_str: str
            String describing a state-action pair
            
        Returns:
        --------
        Tuple[List[str], str]: State description and action
        """
        # Expected format: "State-Action Pair N. State: state1, state2, ...; Action"
        parts = state_action_str.split("State:")
        if len(parts) < 2:
            raise ValueError(f"Invalid state-action pair format: {state_action_str}")
        
        # Extract action from end of string
        action_parts = parts[1].split(";")
        if len(action_parts) < 2:
            raise ValueError(f"Cannot extract action from: {state_action_str}")
        
        action = action_parts[1].strip()
        
        # Extract state components
        state_str = action_parts[0].strip()
        state_components = [s.strip() for s in state_str.split(",")]
        
        return (state_components, action)

    def calculate_expected_utility(self, state_action_utilities):
        """
        Calculate the expected utility of each choice using state belief distribution and utilities.
        
        Parameters:
        -----------
        state_action_utilities: List[Tuple]
            List of ((state, action), utility) tuples
            
        Returns:
        --------
        Dict[str, float]: Expected utility for each choice/action
        """
        if not self.belief_dist:
            # print(self.belief_dist)
            raise ValueError("Belief distribution not initialized.")
        
        # Initialize expected utility for each action
        expected_utilities = {action: 0.0 for action in self.action_strs}
        
        # Count samples per action for averaging
        action_samples = {action: 0 for action in self.action_strs}
        
        # For each state-action pair and its utility
        for (state_components, action), utility in state_action_utilities:
            # Calculate the probability of this state according to belief distribution
            state_prob = 1.0
            
            # For each state component, find its probability in the belief distribution
            for state_desc in state_components:
                # Parse the state description (format: "state_name: state_value")
                parts = state_desc.split(":")
                if len(parts) < 2:
                    continue
                    
                state_name = parts[0].strip()
                state_value = parts[1].strip()
                
                # If this state is in our belief distribution
                if state_name in self.belief_dist:
                    values, probs = self.belief_dist[state_name]
                    
                    # Find the probability of this state value
                    try:
                        value_idx = values.index(state_value)
                        state_prob *= probs[value_idx]
                    except ValueError:
                        # If value not found, use a small probability
                        state_prob *= 0.01
            
            # Add weighted utility to expected utility for this action
            expected_utilities[action] += utility * state_prob
            action_samples[action] += 1
        
        # Normalize by number of samples for each action
        for action in expected_utilities:
            if action_samples[action] > 0:
                expected_utilities[action] /= action_samples[action]
        
        # Add a method to choose the best action based on expected utility
        best_action = max(expected_utilities.items(), key=lambda x: x[1])[0]
        
        return expected_utilities, best_action

    def make_decision_with_expected_utility(self, mode="pairwise", alpha=0.01, temperature=1.0):
        """
        Make a decision using expected utility calculation.
        
        This method:
        1. Elicits preferences from the LLM
        2. Transforms preferences to utility values
        3. Calculates expected utility for each choice
        4. Selects the action with highest expected utility
        
        Parameters:
        -----------
        mode: str
            Mode for utility calculation, either "pairwise" or "top1"
        alpha: float
            Regularization parameter for utility calculation
        temperature: float
            Temperature parameter for softmax normalization
            
        Returns:
        --------
        Dict: The decision, including chosen action, and expected utilities
        """
        # print("Step 3:\n Eliciting preferences...")
        
        # Generate the preference elicitation prompt
        preference_prompt = self.prepare_preference_prompt()
        
        if self.llm:
            # Handle list of prompts for batch mode
            if isinstance(preference_prompt, list):
                all_responses = []
                for i, prompt in enumerate(preference_prompt):
                    # print(f"Processing batch {i+1}/{len(preference_prompt)}...")
                    # Retry mechanism for batch preferences
                    for attempt in range(self.max_retries):
                        try:
                            response = self.llm.invoke(prompt, temperature=self.temperature)

                            if isinstance(response, str):
                                parsed_response = self.extract_json_from_response(response)
                                # Add state-action pairs to the response
                                if "rank" in self.preference_config.pref_enum_mode:
                                    # Extract state-action pairs from the prompt
                                    state_action_lines = [line for line in prompt.split('\n') if line.strip().startswith('- State-Action Pair')]
                                    parsed_response["state_action_pairs"] = state_action_lines
                                    assert len(parsed_response["rank"]) == len(state_action_lines)
                                if "pair" in self.preference_config.pref_enum_mode:
                                    # Extract state-action pairs from the prompt
                                    state_action_lines = [line for line in prompt.split('\n') if line.strip().startswith('- State-Action Pair')]
                                    parsed_response["state_action_pairs"] = state_action_lines
                                all_responses.append(parsed_response)
                            else:
                                # Response is already in expected format, but may need state-action pairs
                                if "state_action_pairs" not in response and self.preference_config.pref_enum_mode == "rank":
                                    state_action_lines = [line for line in prompt.split('\n') if line.strip().startswith('- State-Action Pair')]
                                    response["state_action_pairs"] = state_action_lines
                                    assert len(response["rank"]) == len(state_action_lines)
                                if "state_action_pairs" not in response and self.preference_config.pref_enum_mode == "pair":
                                    state_action_lines = [line for line in prompt.split('\n') if line.strip().startswith('- State-Action Pair')]
                                    response["state_action_pairs"] = state_action_lines
                                all_responses.append(response)
                            break
                        except Exception as e:
                            if attempt < self.max_retries - 1:
                                print(f"Failed to parse: {e}")
                            else:
                                raise Exception(f"Failed to parse: {e}")
            else:
                # Single prompt processing
                for attempt in range(self.max_retries):
                    try:
                        response = self.llm.invoke(preference_prompt, temperature=self.temperature)

                        if isinstance(response, str):
                            preferences = self.extract_json_from_response(response)
                            
                            # Add state-action pairs to the response
                            if self.preference_config.pref_enum_mode == "rank":
                                state_action_lines = [line for line in preference_prompt.split('\n') if line.strip().startswith('- State-Action Pair')]
                                preferences["state_action_pairs"] = state_action_lines
                        else:
                            preferences = response
                            if "state_action_pairs" not in preferences and self.preference_config.pref_enum_mode == "rank":
                                state_action_lines = [line for line in preference_prompt.split('\n') if line.strip().startswith('- State-Action Pair')]
                                preferences["state_action_pairs"] = state_action_lines
                                assert len(preferences["rank"]) == len(preferences["state_action_pairs"])
                        all_responses = [preferences]
                        break
                    except Exception as e:
                        if attempt < self.max_retries - 1:
                            print(f"Failed to parse: {e}")
                        else:
                            # On maximum retries, create a default response
                            raise Exception(f"Failed to parse: {e}")
        else:
            raise Exception("No LLM provided. Cannot elicit preferences.")
        
        # Combine preferences from all batches
        preferences = self._combine_batch_preferences(all_responses)

        # print("Step 4:\n Transforming preferences to utility...")
        choice_utilities, state_action_utilities = self.transform_preferences_to_utility(
            preferences,
            mode=mode,
            alpha=alpha,
            temperature=temperature
        )

        # print("Step 5:\n Calculating expected utility...")
        expected_utilities, best_action = self.calculate_expected_utility(state_action_utilities)
        
        # Prepare decision object

        decision = {
            "decision": best_action,
            "expected_utilities": expected_utilities,
            "raw_utilities": choice_utilities
        }
        
        self.decision = decision
        # print(f"Decision made using expected utility: {best_action}")
        return decision

    def _combine_batch_preferences(self, batch_responses):
        """
        Combine preferences from multiple batches.
        
        Parameters:
        -----------
        batch_responses: List[Dict]
            List of response dictionaries from each batch
            
        Returns:
        --------
        Dict: Combined preferences
        """
        # Combine rank-based preferences
        if all("rank" in response for response in batch_responses):
            all_ranks = []
            all_state_action_pairs = []
            offset = -1
            
            for response in batch_responses:
                # Adjust ranks to account for different batches
                adjusted_ranks = [r + offset for r in response["rank"]]
                all_ranks.extend(adjusted_ranks)
                
                # Get state-action pairs if they exist
                if "state_action_pairs" in response:
                    all_state_action_pairs.extend(response["state_action_pairs"])
                
                offset += len(response["rank"])
            return {
                "rank": all_ranks,
                "state_action_pairs": all_state_action_pairs,
            }
        
        # Combine pairwise preferences
        elif all("pair" in response for response in batch_responses):
            all_pairs = []
            all_state_action_pairs = []
            offset = -1
            for response in batch_responses:
                # Adjust pair indices to account for different batches
                adjusted_pairs = [
                    [p[0] + offset, p[1] + offset] for p in response["pair"]
                ]
                all_pairs.extend(adjusted_pairs)
                
                # Get state-action pairs if they exist
                if "state_action_pairs" in response:
                    all_state_action_pairs.extend(response["state_action_pairs"])
                
                offset += len(response["pair"])
            
            return {
                "pair": all_pairs,
                "state_action_pairs": all_state_action_pairs,
            }

        # Fallback to first batch only if formats don't match
        else:
            # print("Warning: Inconsistent batch response formats. Using first batch only.")
            response = batch_responses[0]
            result = {}
            
            if "rank" in response:
                result["rank"] = response["rank"]
            elif "pair" in response:
                result["pair"] = response["pair"]
            else:
                raise ValueError("Invalid batch response formats. Must contain either 'rank' or 'pair' key.")
            
            # Include state-action pairs if they exist
            if "state_action_pairs" in response:
                result["state_action_pairs"] = response["state_action_pairs"]
            
            return result

    # generate and format state dict
    def format_state_dict(self, llm=None):
        """
        Generate factors from context and format them as state variables.
        
        Parameters:
        -----------
        llm: Optional
            An LLM inference function/object to use for generating states.
            If None, assumes the agent or caller handles inference.

        Returns:
        --------
        Dict[str, str]: A dictionary mapping state names to their descriptions
        """
        # Check if states are already in memory
        if self.factors is not None:
            return self._factors_to_state_dict(self.factors)
        
        # Generate prompt for LLM to extract factors from context
        prompt = self._format_state_generation_prompt()
        
        # Try to generate states with retries if needed
        for attempt in range(self.max_retries):
            try:
                response_raw = llm.invoke(prompt, temperature=self.temperature)
            
                # Parse response based on its type
                if isinstance(response_raw, str):
                    response = self.extract_json_from_response(response_raw)
                else:
                    response = response_raw

                # Process valid response
                if isinstance(response, dict) and "factors" in response:
                    # Store in memory
                    self.factors = response["factors"]
                    state_dict = self._factors_to_state_dict(response["factors"])
                    return state_dict
                else:
                    raise ValueError("Response missing 'factors' key")
                
            except (json.JSONDecodeError, ValueError) as e:
                if attempt < self.max_retries - 1:
                    print(f"Warning: Attempt {attempt+1}/{self.max_retries} failed: {e}. Retrying...")
                else:
                    raise Exception(f"Failed to generate states after {self.max_retries} retries")
    
    def _format_state_generation_prompt(self):
        """
        Format a prompt to ask the LLM to identify important factors (states)
        for decision making based on the context.
        
        Returns:
        --------
        str: The formatted prompt
        """
        query = f"Below is a context for a decision-making task:\n\n{self.context}\n\n"
        format_instruction = f"""Please analyze this context and identify the key factors that would influence the decision between the following options: {', '.join(self.choices)}.

You should format your response as a JSON object. The JSON object should contain the following keys:
- 'factors': a list of strings that enumerates the factors that should be considered when making this decision, based on the context provided. Each factor should be a concise phrase (5-10 words) describing a specific variable or consideration. Include at least 7 factors, ranked in decreasing order of importance.

For example:
{{
  "factors": [
    "Expected rainfall during the growing season",
    "Market price forecasts for each crop",
    "Production costs for different options",
    "Labor requirements and availability",
    "Risk of disease or pest outbreaks",
    "Government subsidy programs",
    "Storage and transportation considerations"
  ]
}}"""
        
        return f"{query}\n\n{format_instruction}"
    
    def _factors_to_state_dict(self, factors):
        """
        Convert a list of factors into a properly formatted state dictionary
        
        Parameters:
        -----------
        factors: List[str]
            List of factor strings from LLM response
            
        Returns:
        --------
        Dict[str, str]: State dictionary mapping state names to descriptions
            Keys are ids, values are factor names (str)
        """
        state_dict = {}
        
        # Process each factor into a state variable
        for i, factor in enumerate(factors):
            # Create a normalized key from the factor
            key = f"factor_{i+1}"
            
            # Store the factor as the description
            state_dict[key] = factor
            
            # For top factors, also create choice-specific variants
            #if i < 3:  # Only for the top 3 factors
            #    for choice in self.choices:
            #        choice_key = f"{choice}_{key}"
            #        state_dict[choice_key] = f"Impact of '{factor}' specifically on {choice}"

        return state_dict

    def load_state_beliefs(self) -> Dict[str, Tuple[List[str], List[float]]]:
        """
        Load and convert natural language beliefs to probabilities
        
        Returns:
        --------
        belief_dist: Dict[str, Tuple[List[str], List[float]]]
            key: names of state variables that are relevant to the agent
            value: tuple of lists of state values and their corresponding probabilities
        """
        if self.states is None:
            raise ValueError("State beliefs not generated yet. Run initialize_workflow first.")

        self.belief_dist = {}
        for state, val2belief in self.states.items():
            if state in self.state_config.states:
                total_score = sum([self.belief2score[v] for v in val2belief.values()])
                self.belief_dist[state] = (
                    list(val2belief.keys()),
                    [self.belief2score[v] / total_score for v in val2belief.values()],
                )

        return self.belief_dist

    # sampling utils
    def sample_state(self) -> List[str]:
        """Sample a state from the belief distribution self.belief_dist
        
        Returns:
        --------
        sampled_state: List[str]
            a list of strings that describe the sampled state
        """
        if not hasattr(self, "belief_dist"):
            self.load_state_beliefs()
        sampled_state = []
        for state, (vals, probs) in self.belief_dist.items():
            sampled_state.append(f"{state}: {np.random.choice(vals, p=probs)}")
        return sampled_state

    def sample_state_action_pairs_batch(self) -> List[str]:
        """Sample batches of state-action pairs for preference elicitation."""
        # sample_size * len(action_strs)
        state_batch = [
            self.sample_state() for _ in range(self.preference_config.sample_size)
        ] * len(self.action_strs)
        state_batch = np.array(state_batch)
        action_batch = np.repeat(self.action_strs, self.preference_config.sample_size)
        action_batch = np.array(action_batch)
        # shuffle state and action using the same index
        idx = np.arange(len(state_batch))
        np.random.shuffle(idx)
        state_batch = state_batch[idx]
        action_batch = action_batch[idx]
        stride = int(
            self.preference_config.minibatch_size
            * (1 - self.preference_config.overlap_pct)
        )
        state_action_batch = []
        for i in range(0, len(state_batch), stride):
            minibatch = []
            j = min(
                len(state_batch),
                i + self.preference_config.minibatch_size,
            )
            for k in range(i, j):
                sampled_state = state_batch[k]
                sampled_action = action_batch[k]
                minibatch.append(
                    f"- State-Action Pair {k+1-i}. State: {', '.join(sampled_state)}; {sampled_action}"
                )
            state_action_batch.append(minibatch)
            if j == len(state_batch):
                break
        return state_action_batch

    def sample_state_action_pairs(self) -> str:
        """Sample a set of state-action pairs for preference elicitation.
        
        Returns:
        --------
        state_action_pairs: List[str]
            a list of strings describing sampled state-action pairs
        """
        state_action_pairs = []
        for i in range(self.preference_config.sample_size):
            sampled_state = self.sample_state()
            sampled_action = np.random.choice(self.action_strs)
            state_action_pairs.append(
                f"- State-Action Pair {i+1}. State: {', '.join(sampled_state)}; {sampled_action}"
            )
        return state_action_pairs

    # prompts
    def prepare_context(self) -> str:
        """Format context into a string"""
        return self.context

    def prepare_actions(self) -> str:
        """
        Format actions into a string
        """
        if self.action_config is None:
            raise ValueError("Action config not found.")
        action_enum_mode = self.action_config.action_enum_mode
        if action_enum_mode != "base":
            raise NotImplementedError

        # implement base actions
        choices = getattr(self, "choices", [])
        merged_action_str = "\n".join(choices)
        return f"Below are the actions I can take:\n{merged_action_str}"

    def prepare_state_prompt(self) -> Dict[str, List[str]]:
        """Prompt the model with context and state variables, and return the model's belief distribution over the state variables
        N.B. This function is NOT used for the final DeLLMa decision prompt, but is used to generate the state belief distribution
        """
        if self.state_config is None:
            raise ValueError("State config not found.")
        
        state_enum_mode = self.state_config.state_enum_mode
        if state_enum_mode == "base":
            if len(self.state_config.states) > 0:
                warnings.warn("States are not used in base mode but provided.")
            state_prompt = ""
        elif state_enum_mode == "sequential":
            state_prompt = f"""I would like to adopt a decision making under uncertainty framework to make my decision. The goal of you, the decision maker, is to choose an optimal action, while accounting for uncertainty in the unknown state. Previously, you have already provided a forecast of future state variables relevant to decisions. The state is a vector of {len(self.state_config.states)} elements, each of which is a random variable. The state variables (and their most probable values) are enumerated below:"""
            
            if self.states is None:
                raise ValueError("State beliefs not yet generated.")
            
            for state in self.state_config.states.keys():
                state_prompt += f"\n- {state}: {self.states[state]}"
        else:
            raise NotImplementedError
        return state_prompt

    def prepare_preference_prompt(self) -> Union[str, List[str]]:
        """Prompt the model with context, actions, (and potentially states), and return the model's preference/decision over the actions

        Available preference enum modes:
        - base: the model is asked to make a decision based on the context and actions WITHOUT any state information
        - rank: the model is asked to rank the state-action pairs sampled from the state belief distribution and action space
        """
        if self.preference_config is None:
            raise ValueError("Preference config not found.")
        pref_enum_mode = self.preference_config.pref_enum_mode

        if pref_enum_mode == "base":
            preference_prompt = "I would like to know which action I should take based on the information provided above."
            format_instruction = f"""You should format your response as a JSON object. The JSON object should contain the following keys:
- decision: a string that describes the action you recommend to take. The output format should be the same as the format of the actions listed above, e.g. {self.action_strs[0]}"""
            
            return "\n".join([
                preference_prompt,
                format_instruction
            ])
        else:
            preference_prompt = "Below, I have sampled a set of state-action pairs, wherein states are sampled from the state belief distribution you provided and actions are sampled uniformly from the action space. I would like to construct a utility function from your comparisons of state-action pairs\n\n"

            if "rank" in pref_enum_mode:
                format_instruction = f"""You should format your response as a JSON object. The JSON object should contain the following keys:
- decision: a string that describes the state-action pair you recommend to take. The output format should be the same as the format of the state-action pairs listed above, e.g. State-Action Pair 5.
- rank: a list of integers that ranks the state-action pairs in decreasing rank of preference. For example, if you think the first state-action pair is the most preferred, the second state-action pair is the second most preferred, and so on. For example, [1, 2, 3, 4, 5]."""
            elif "pair" in pref_enum_mode:
                format_instruction = f"""You should format your response as a JSON object. The JSON object should contain the following keys:
- decision: a string that describes the state-action pair you recommend to take. The output format should be the same as the format of the state-action pairs listed above, e.g. State-Action Pair 5.
- pair: a list of lists of integers that describes your pairwise preference between state-action pairs. For example, if you think the first state-action pair is more preferred than the second state-action pair, the second state-action pair is more preferred than the third state-action pair, and so on. For example, [[1, 2], [2, 3], [3, 4], [4, 5]]."""
        if "minibatch" in pref_enum_mode:
            state_action_batch = self.sample_state_action_pairs_batch()
        else:
            state_action_batch = [self.sample_state_action_pairs()]

        preference_prompts = []
        state_action_pairs_list = []
        for state_action_pairs in state_action_batch:
            preference_prompts.append(
                "\n".join([
                    preference_prompt + "\n\n".join(state_action_pairs) + "\n\n",
                    format_instruction
                ])
            )
            state_action_pairs_list.append(state_action_pairs)
        return preference_prompts

    def prepare_belief_dist_generation_prompt(self) -> str:
        """Prompt the model with context and state variables to generate the model's belief distribution over the state variables
        N.B. This function is NOT used for the final DeLLMa decision prompt, but is used to generate the state belief distribution
        """
        context = self.prepare_context()

        state_prompt = ""
        if self.state_config is None:
            raise ValueError("State config not found.")
        
        state_enum_mode = self.state_config.state_enum_mode
        if state_enum_mode == "base":
            raise ValueError(
                "State enum mode must be cannot be base for belief dist generation."
            )
        elif state_enum_mode == "sequential":
            # enumerate through each state dimension in tandem
            state_prompt = f"""I would like to adopt a decision making under uncertainty framework to make my decision. The goal of you, the decision maker, is to choose an optimal action, while accounting for uncertainty in the unknown state. The first step of this procedure is for you to produce a belief distribution over the future state. The state is a vector of {len(self.state_config.states)} elements, each of which is a random variable. The state variables are enumerated below:"""
            
            if not self.state_config.states and hasattr(self, 'format_state_dict'):
                self.state_config.states = self.format_state_dict()

            for state, desc in self.state_config.states.items():
                state_prompt += f"\n- {state}: {desc}"

            format_instruction = f"""You should format your response as a JSON object with {len(self.state_config.states.keys())} keys, wherein each key should be a state variable from the list above. 

Each key should map to a JSON object with {self.state_config.topk} keys, each of which is a string that describes the value of the state variable. Together, these keys should enumerate the top {self.state_config.topk} most likely values of the state variable. Each key should map to your belief verbalized in natural language. If the state variable is continuous (e.g. changes to a quantity), you should discretize it into {self.state_config.topk} bins.

You should strictly choose your belief from the following list: 'very likely', 'likely', 'somewhat likely', 'somewhat unlikely', 'unlikely', 'very unlikely'.
For example, if one of the state variable is 'climate condition', and the top 3 most likely values are 'drought', 'heavy precipitation', and 'snowstorm', then your response should be formatted as follows:
{{
    "climate condition": {{
        "drought": "somewhat likely",
        "heavy precipitation": "very likely",
        "snowstorm": "unlikely"
    }},
    ...
}}
"""
            state_prompt = "\n".join([
                state_prompt,
                format_instruction
            ])
        else:
            raise NotImplementedError

        return f"{context}\n\n{state_prompt}"

    def extract_json_from_response(self, response_str):
        """
        Extract JSON content from a response string that may be wrapped in markdown code blocks.
        
        Parameters:
        -----------
        response_str: str
            The raw response string from the LLM
        
        Returns:
        --------
        dict: The parsed JSON data
        
        Raises:
        -------
        json.JSONDecodeError: If valid JSON cannot be extracted
        """
        # First try parsing the entire string as JSON
        try:
            return json.loads(response_str)
        except json.JSONDecodeError:
            # Check for JSON within code blocks
            if "```json" in response_str and "```" in response_str:
                # Extract content between json code block markers
                start_idx = response_str.find("```json") + 7
                end_idx = response_str.find("```", start_idx)
                
                if start_idx > 7 and end_idx > start_idx:
                    json_content = response_str[start_idx:end_idx].strip()
                    return json.loads(json_content)
                
            # Try to find any JSON-like structure using a simple heuristic
            if '{' in response_str and '}' in response_str:
                start_idx = response_str.find('{')
                # Find the matching closing brace
                brace_count = 0
                for i in range(start_idx, len(response_str)):
                    if response_str[i] == '{':
                        brace_count += 1
                    elif response_str[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                
                if brace_count == 0:  # Found a complete JSON object
                    json_content = response_str[start_idx:end_idx].strip()
                    return json.loads(json_content)
                
            # If we get here, no valid JSON was found
            raise json.JSONDecodeError("No valid JSON found in response", response_str, 0)


if __name__ == "__main__":
    from src.models.utils import Qwen7B
    
    # Sample data for demonstration
    sample_context = """
    The 2023 agricultural outlook shows mixed signals. Corn prices are expected to remain high due to continued export demand, though domestic use for ethanol might decrease. Soybean prices have been volatile but are trending upward due to increased demand from China. Cotton production has been affected by drought conditions in key growing regions. Weather forecasts indicate a potential El Niño pattern, which could bring increased rainfall to the southern states and drier conditions to the northern grain belt. Input costs, particularly for fertilizers and fuel, have risen by 15% since last year. Labor shortages continue to impact harvesting operations in labor-intensive crops.
    """
    
    choices = ["Plant Corn", "Plant Soybeans", "Plant Cotton"]
    
    # Initialize the LLM
    # print("Initializing LLM...")
    llm = Qwen7B()
    
    # Create the agent
    # print("Creating DeLLMaAgent...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    pair_preference_config = PreferenceConfig(
        pref_enum_mode="pairwise-minibatch",
        sample_size=32,
    )
    rank_preference_config = PreferenceConfig(
        pref_enum_mode="rank-minibatch",  # Rank-based preference elicitation
        minibatch_size=8,
        sample_size=32,
        overlap_pct=0.25
    )
    agent = DeLLMaAgent(
        choices=choices,
        context=sample_context,
        temperature=0.0,
        llm=llm,
        state_config=StateConfig(
            state_enum_mode="sequential",
            topk=3
        ),
        preference_config=pair_preference_config
    )
    # Run the complete workflow
    # print("\nStarting DeLLMa workflow...")
    agent.initialize_workflow(regenerate_states=False, regenerate_beliefs=False)
    # Make the final decision
    decision = agent.make_decision_with_expected_utility()
    
    # # print the results
    # print("\n===== FINAL DECISION =====")
    # print(f"Chosen action: {decision['decision']}")

    exit()