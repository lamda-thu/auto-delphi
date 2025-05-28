PROMPTS = {}

PROMPTS["joint_extraction"] = """
You are an expert decision analyst specializing in decision-making under uncertainty and influence diagram modelling. You will be provided with a source text. Your goal is to EXTRACT an influence diagram that STRICTLY represents the decision problem AS DESCRIBED in the text. 

Steps:
1. Node extraction: Extract ONLY decision-related variables that are EXPLICITLY MENTIONED or CLEARLY IMPLIED in the text. Each variable must correspond to uncertainty, action or utility.
    - For each node, provide a name (using terminology from the source text), a type (decision, chance, utility), and valid values.
    - You must be able to cite or reference the specific part of the text that supports each node.

2. Edge extraction: Extract ONLY influences and causal relations that are EXPLICITLY STATED or CLEARLY IMPLIED in the text.
    - Each edge must be supported by specific text evidence.
    - Edge types: 
        a) Information edges (ending in decision nodes): Shows what information is known when making decisions
        b) Dependency edges (between other nodes): Shows probabilistic dependencies

Text:

{text}

The original node list:

{node_list}

The original edge list:

{edge_list}

Requirements:
1. Do not include nodes or edges in the original lists or have similar meanings to exising ones.
2. The nodes and edges should be directly mentioned or clearly implied in the text.
3. The node names should be concise and meaningful, and should come from the source text if possible.
4. Utility node values should be numerical. We aim to maximize the utility, so if utility node is cost-related, the values should be negative. If utility node is benefit-related, the values should be positive.
5. No cycles are allowed. For example, if the influence with condition 'a' to variable 'b' is present, there cannot be another influence with condition 'b' to variable 'a'. 
6. All cause and effect variables (nodes) used in the edge list should be members of the node list, either in the original or the added list.
7. Output using the following format:

{format_instructions}
"""

PROMPTS["joint_generation"] = """
You are an expert decision analyst specializing in decision-making under uncertainty and influence diagram modelling. You will be provided with a text describing a decision context. Your goal is to GENERATE a COMPREHENSIVE influence diagram that captures ALL RELEVANT aspects of this type of decision problem, even those not explicitly mentioned in the text.

Steps:
1. Node generation: Using your expertise, identify ALL RELEVANT decision-related variables that would affect this type of decision problem:
    - Consider standard variables common to this domain
    - Include implicit factors that would realistically affect the decision
    - Add variables needed for a complete decision model, even if not mentioned in the text

2. Edge generation: Using domain knowledge, identify ALL PLAUSIBLE influences between variables:
    - Edge types: 
        a) Information edges (ending in decision nodes): Shows what information is known when making decisions
        b) Dependency edges (between other nodes): Shows probabilistic dependencies
    - Consider standard causal relationships in this domain
    - Include practical information flows that would exist in reality
    - Add edges needed for a complete and realistic model

Text:

{text}

The original node list:

{node_list}

The original edge list:

{edge_list}

Requirements:
1. Do not include nodes or edges in the original lists or have similar meanings to exising ones.
2. The node names should be concise and meaningful.
3. Utility node values should be numerical. We aim to maximize the utility, so if utility node is cost-related, the values should be negative. If utility node is benefit-related, the values should be positive.
4. No cycles are allowed. For example, if the influence with condition 'a' to variable 'b' is present, there cannot be another influence with condition 'b' to variable 'a'. 
5. All cause and effect variables (nodes) used in the edge list should be members of the node list, either in the original or the added list.
6. Output using the following format:

{format_instructions}
"""

PROMPTS["joint_generation_reflection"] = """
You are an expert decision analyst specializing in decision-making under uncertainty and influence diagram modelling. You will be provided with a source text and a list of nodes and edges in an influence diagram that were generated to comprehensively model the decision problem.

Your task is to critically evaluate the generated influence diagram and provide constructive feedback:

1. Evaluate the completeness of the node list:
   - Are there any important variables missing that would be relevant to this type of decision problem?
   - Are there any redundant or irrelevant nodes that should be removed?
   - Are the node types (chance, decision, utility) properly assigned?

2. Evaluate the edge list:
   - Are there any important influences or dependencies missing?
   - Are there any edges that are implausible or incorrect?
   - Are information flows to decision nodes appropriate and realistic?

Source text:

{text}

Node list:

{node_list}

Edge list:

{edge_list}

Provide your reflection as a list of specific, actionable suggestions for improving the influence diagram.
"""

PROMPTS["joint_extraction_reflection"] = """
You are an expert decision analyst specializing in decision-making under uncertainty and influence diagram modelling. You will be provided with a source text and a list of nodes and edges that were extracted from the text to create an influence diagram.

Your task is to critically evaluate the extracted influence diagram and provide constructive feedback:

1. Evaluate the node list:
   - Are there any important variables explicitly mentioned in the text that were missed?
   - Are all extracted nodes actually present or clearly implied in the text?
   - Are the node types (chance, decision, utility) properly assigned based on the text?

2. Evaluate the edge list:
   - Are there any influences or dependencies explicitly mentioned in the text that were missed?
   - Are all extracted edges actually supported by the text?
   - Do the edges accurately represent the relationships described in the text?

Source text:

{text}

Node list:

{node_list}

Edge list:

{edge_list}

Provide your reflection as a list of specific, actionable suggestions for improving the accuracy and completeness of the extracted influence diagram.
"""

PROMPTS["joint_improvement"] = """
You are an expert decision analyst specializing in decision-making under uncertainty and influence diagram modelling. You will be provided with a source text and a list of nodes and edges in an influence diagram. Your goal is to improve the influence diagram by modifying the node list, fixing existing issues, and taking into account a list of expert suggestions and criticisms. Output the modified list of nodes and edges.

Source text:

{source_text}

Node list:

{node_list}

Edge list:

{edge_list}

Issues:

{issues}

Expert suggestions:

{reflection}

Requirements:
1. Address all reported issues.
2. Address the suggestions.
3. Output using the following format:

{format_instructions}
"""

PROMPTS["remove_redundant_nodes"] = """
You are an expert decision analyst specializing in decision-making under uncertainty and influence diagram modelling. You will be provided with a list of nodes. Your goal is to remove nodes that are semantically redundant with those in the original node list, and output only the truly novel nodes.

Node list:

{node_list}

Original node list:

{original_node_list}

Requirements:
1. Remove nodes that are semantically equivalent to any node in the original node list.
2. Focus on the semantic meaning of nodes, not just literal text matching. If two nodes have different phrasing but represent the same concept or variable, consider them redundant. For example, "economic cost" and "financial impact" might be differently phrased but semantically redundant.
3. Analyze both the variable name and type to determine semantic equivalence.
4. Output using the following format:

{format_instructions}
"""


# node
PROMPTS["node_generation"] = """
You are an expert decision analyst specializing in decision-making under uncertainty and influence diagram modelling. You will be provided with a text describing a decision context. For the decision problem as described in the text, identify every decision-related variable, that supplements the original list. The variables may correspond to uncertainty, action or utility.
Your goal is to GENERATE a COMPREHENSIVE influence diagram that captures ALL RELEVANT aspects of this type of decision problem, even those not explicitly mentioned in the text.
Output the added list of nodes. 

Text:

{text}

The original node list:

{node_list}

Requirements:
1. Do not include nodes in the original node list or have similar meanings to exising ones in the node list.
2. Utility node values should be numerical. We aim to maximize the utility, so if utility node is cost-related, the values should be negative. If utility node is benefit-related, the values should be positive.
3. Output using the following format:

{format_instructions}
"""

PROMPTS["node_extraction"] = """
You are an expert decision analyst specializing in decision-making under uncertainty and influence diagram modelling.
For the following text, extract every decision-related variable, that supplements the original list. The variables may correspond to uncertainty, action or utility.
Output the added list of nodes. 

Text:

{text}

The original node list:

{node_list}

Requirements:
1. Do not include nodes in the original node list or have similar meanings to exising ones in the node list.
2. The variables should be directly mentioned or clearly implied in the text.
3. The variable names should be concise and meaningful, and should come from the source text if possible.
4. Utility node values should be numerical. We aim to maximize the utility, so if utility node is cost-related, the values should be negative. If utility node is benefit-related, the values should be positive.
5. Output using the following format:

{format_instructions}
"""

PROMPTS["node_generation_reflection"] = """
You are an expert decision analyst specializing in decision-making under uncertainty and influence diagram modelling. You will be provided with a source text and a list of nodes in the constructed influence diagram. Your goal is to help improve the influence diagram by suggesting modifications to the node list.
Your task is to carefully inspect the node list, then give constructive criticism and helpful suggestions. 

Node list:

{node_list}

Source text:

{source_text}

First, analyze the current node list by identifying its strengths and weaknesses. Then provide specific suggestions for improvement.
When writing suggestions, pay attention to the following aspects:

(i) Completeness. Are there any critical variables (nodes) that are relevant to the decision problem described in the source text but are not in the node list? You could suggest adding them. Do not add nodes that share similar meanings to existing nodes in the list.

(ii) Minimality. Are there any duplicate nodes that share similar meanings, or have opposite meanings but refer to the same variable? For example, 'health' and 'disease' may refer to the same variable. You could suggest aggregating some nodes into a single one, for example, some (binary) nodes may be considered values of a single variable. The suggestions should be contingent on the source text. Do not remove nodes only because they are not present in the source text.

(iii) Properness. Are the node types proper? The type of a node should match the role of the variable in practice:
   - Decision node: A variable that the decision maker can directly control or choose
   - Utility node: A variable that represents the decision maker's objectives or preferences
   - Chance node: A variable with uncertainty that the decision maker cannot directly control

(iv) Understandability. Are the node names understandable? The node names should be concise and meaningful, and should come from the source text if possible.

Examples:
- Good node: {"variable_name": "Treatment option", "variable_type": "decision", "variable_values": ["medication", "surgery", "no treatment"]}
- Poor node: {"name": "Med", "type": "decision", "values": ["Yes", "No"]}

Write a numbered list of specific, helpful and constructive suggestions for improving the node list. Each suggestion should address one specific part of the node list. Prioritize your suggestions from most to least important. Keep each suggestion concise Output only the suggestions and nothing else.
"""

PROMPTS["node_extraction_reflection"] = """
You are an expert decision analyst specializing in decision-making under uncertainty and influence diagram modelling. You will be provided with a source text and a list of nodes in the constructed influence diagram. Your goal is to help improve the influence diagram by suggesting modifications to the node list.
Your task is to carefully inspect the node list, then give constructive criticism and helpful suggestions. 

Node list:

{node_list}

Source text:

{source_text}

First, analyze the current node list by identifying its strengths and weaknesses. Then provide specific suggestions for improvement.
When writing suggestions, pay attention to the following aspects:

(i) Completeness. Are there any nodes that are directly mentioned or clearly implied in the source text but are not in the node list? You could suggest adding them, and provide a brief explanation. Do not add nodes that share similar meanings to existing nodes in the list. Do not add nodes that are not mentioned in the source text.

(ii) Minimality. Are there any duplicate nodes that share similar meanings, or have opposite meanings but refer to the same variable? For example, 'health' and 'disease' may refer to the same variable. You could suggest aggregating some nodes into a single one, for example, some (binary) nodes may be considered values of a single variable. The suggestions should be contingent on the source text. Do not remove nodes only because they are not present in the source text.

(iii) Properness. Are the node types proper? The type of a node should match the role of the variable in practice:
   - Decision node: A variable that the decision maker can directly control or choose
   - Utility node: A variable that represents the decision maker's objectives or preferences
   - Chance node: A variable with uncertainty that the decision maker cannot directly control

(iv) Understandability. Are the node names understandable? The node names should be concise and meaningful, and should come from the source text if possible.

Examples:
- Good node: {"variable_name": "Treatment option", "variable_type": "decision", "variable_values": ["medication", "surgery", "no treatment"]}
- Poor node: {"name": "Med", "type": "decision", "values": ["Yes", "No"]}

Write a numbered list of specific, helpful and constructive suggestions for improving the node list. Each suggestion should address one specific part of the node list. Prioritize your suggestions from most to least important. Keep each suggestion concise Output only the suggestions and nothing else.
"""

PROMPTS["node_improvement"] = """
You are an expert decision analyst specializing in decision-making under uncertainty and influence diagram modelling. You will be provided with a source text and a list of nodes in an influence diagram. Your goal is to improve the influence diagram by modifying the node list, fixing existing issues, and taking into account a list of expert suggestions and criticisms. Output the modified list of nodes.

The original node list:

{node_list}

Source text:

{source_text}

Issues:

{issues}

Expert suggestions:

{reflection}

Requirements:
1. Address all reported issues.
2. Address the suggestions.
2. Output using the following format:

{format_instructions}
"""


# edge
PROMPTS["edge_generation"] = """
You are an expert decision analyst specializing in decision-making under uncertainty and influence diagram modelling.
For the decision problem as described in the text, list every influence or causal relation (edge) between variables (nodes) that supplements the original edge list.
Your goal is to GENERATE a COMPREHENSIVE influence diagram that captures ALL RELEVANT aspects of this type of decision problem, even those not explicitly mentioned in the text.
Output the added list of edges.

Understanding edges in influence diagrams:
- An edge ending in a decision node specifies information available when making that decision
- An edge ending in a chance node or a utility node specifies probabilistic dependencies
- The absence of edges between nodes implies conditional independence

Examples:
- Edge {{'cause':'weather forecast', 'effect':'take umbrella'}} ends in a decision node 'take umbrella', meaning the value of 'weather forecast' is known when deciding whether to take the umbrella.
- Edge {{'cause': 'weather', 'effect': 'weather forecast'}} ends in a chance node 'weather forecast', meaning the distribution of weather forecast depends on the weather (conditional probability P(weather forecast | weather)).

Text:

{text}

The original edge list:

{edge_list}

Requirements:
1. Do not include edges in the original edge list or edges with similar meanings.
2. No cycles are allowed. For example, if there is an edge from 'a' to 'b', there cannot be another edge from 'b' to 'a'.
3. All cause and effect variables (nodes) used in the edge list must be members of the node list: 

{node_list}

4. Output using the following format:

{format_instructions}
"""

PROMPTS["edge_extraction"] = """
You are an expert decision analyst specializing in decision-making under uncertainty and influence diagram modelling.
For the following text, extract every influence or causal relation (edge) between variables (nodes) that supplements the original edge list.
Output the added list of edges.

Understanding edges in influence diagrams:
- An edge ending in a decision node specifies information available when making that decision
- An edge ending in a chance node or a utility node specifies probabilistic dependencies
- The absence of edges between nodes implies conditional independence

Examples:
- Edge {{'cause':'weather forecast', 'effect':'take umbrella'}} ends in a decision node 'take umbrella', meaning the value of 'weather forecast' is known when deciding whether to take the umbrella.
- Edge {{'cause': 'weather', 'effect': 'weather forecast'}} ends in a chance node 'weather forecast', meaning the distribution of weather forecast depends on the weather (conditional probability P(weather forecast | weather)).

Text:

{text}

The original edge list:

{edge_list}

Requirements:
1. Do not include edges in the original edge list or edges with similar meanings.
2. The edges should be directly mentioned or clearly implied in the text.
3. No cycles are allowed. For example, if there is an edge from 'a' to 'b', there cannot be another edge from 'b' to 'a'.
4. All cause and effect variables (nodes) used in the edge list must be members of the node list: 

{node_list}

5. Output using the following format:

{format_instructions}
"""

PROMPTS["edge_generation_reflection"] = """
You are an expert decision analyst specializing in decision-making under uncertainty and influence diagram modelling. You will be provided with a source text, a list of nodes, and a list of edges in the constructed influence diagram. Your goal is to improve the influence diagram by suggesting modifications to the edge list.

Your task is to carefully inspect the edge list, then give constructive criticism and helpful suggestions.

The node list:

{node_list}

The edge list:

{edge_list}

Source text:

{source_text}

First, analyze the current edge list by identifying its strengths and weaknesses. Then provide specific suggestions for improvement.
When writing suggestions, pay attention to whether there are ways to improve the edge list in the following aspects:

(i) Completeness. Are there any critical causal or probabilistic relations (edges) that are relevant to the decision problem described in the source text but are not present in the edge list? Are there any information known at the time of decision (information edges) that are missing? You could suggest adding them.

(ii) Minimality. Are there any edges between nodes that are have no probabilistic dependence? You could suggest removing them, and provide a brief explanation (why these nodes are not causally related or conditionally independent).

(iii) Practicality. Are there any information edges that do not exist in practice? Information edges are those that end in (whose "effect" is) a "decision" type variable, and imply the value of the "cause" variable is known when the decision is made. You could suggest removing them, and provide a brief explanation.

Write a list of specific, helpful and constructive suggestions for improving the edge list. Each suggestion should address one specific part of the edge list. Output only the suggestions and nothing else.
"""

PROMPTS["edge_extraction_reflection"] = """
You are an expert decision analyst specializing in decision-making under uncertainty and influence diagram modelling. You will be provided with a source text, a list of nodes, and a list of edges in the constructed influence diagram. Your goal is to improve the influence diagram by suggesting modifications to the edge list.

Your task is to carefully inspect the edge list of an influence diagram, then give constructive criticism and helpful suggestions to improve the edge list for the influence diagram.

The node list:

{node_list}

The edge list:

{edge_list}

Source text:

{source_text}

First, analyze the current edge list by identifying its strengths and weaknesses. Then provide specific suggestions for improvement.
When writing suggestions, pay attention to whether there are ways to improve the edge list in the following aspects:

(i) Completeness. Are there any causal or probabilistic relations (edges) that are directly mentioned or clearly implied in the source text but are not present in the edge list? Are there any information known at the time of decision (information edges) that are missing? You could suggest adding them, and provide a brief explanation.

(ii) Minimality. Are there any edges between nodes that are have no probabilistic dependence? You could suggest removing them, and provide a brief explanation.

(iii) Practicality. Are there any information edges that do not exist in practice? Information edges are those that end in (whose "effect" is) a "decision" type variable, and imply the value of the "cause" variable is known when the decision is made. You could suggest removing them, and provide a brief explanation.

Write a list of specific, helpful and constructive suggestions for improving the edge list. Each suggestion should address one specific part of the edge list. Output only the suggestions and nothing else.
"""

PROMPTS["edge_improvement"] = """
You are an expert decision analyst specializing in decision-making under uncertainty and influence diagram modelling. You will be provided with a source text, a list of nodes and a list of edges in the constructed influence diagram. Your goal is to improve the influence diagram by modifying the edge list, taking into account a list of expert suggestions and criticisms. Output the modified list of edges.

The original edge list:

{edge_list}

Source text:

{source_text}

Issues:

{issues}

Expert suggestions:

{reflection}

Requirements:
1. Address all reported issues.
2. The edges should be directly mentioned or clearly implied in the text.
3. No cycles are allowed. For example, if the influence with condition 'a' to variable 'b' is present, there cannot be another influence with condition 'b' to variable 'a'. 
4. All cause and effect variables (nodes) used in the edge list should be members of the node list: 

{node_list}

5. Output using the following format:

{format_instructions}
"""


# cpds
PROMPTS["cpd_generation"] = """ """

PROMPTS["cpd_extraction"] = """ """

PROMPTS["cpd_stochastic_random"] = """
You are an expert decision analyst specializing in decision-making under uncertainty and influence diagram modelling. 
You will be provided with a variable and its evidences.
Output should follow the *JSON* format: 
{format_instructions}


Assign the conditional probability for variable "{variable}" and each value of the evidence "{parent_variable_list}".
The values of the variable and the evidence variables *must* be valid and complete.
The valid values of the variable: "{variable_domain}"
The valid values of the evidence variables:"{condition_domain}"

Utility node values must be numerical.
We aim to maximize the utility, so if utility node is cost-related, the values should be negative. If utility node is benefit-related, the values should be positive.
"""

PROMPTS["cpd_stochastic_generation"] = """
You are an expert decision analyst specializing in decision-making under uncertainty and influence diagram modeling.

You will be provided with:
1. A source text:  
   {text}
2. A target variable:  
   {variable}
3. Evidence variables (parents):  
   {parent_variable_list}

Your task is to:
1. **Generate the conditional probability distribution (CPD)** for {variable} based on the provided **source text** and common sense.
2. Assign probabilities **for every possible value** of {variable} given **each possible combination** of the valid evidence values.
3. Adhere to the valid domains of:
   - **Target variable** {variable_domain}
   - **Evidence variables** {condition_domain}
4. Utility node values must be numerical.
5. We aim to maximize the utility, so if utility node is cost-related, the values should be negative. If utility node is benefit-related, the values should be positive.
6. **Output** must strictly follow the lambda string format.

**Required Output**  
Please produce the **conditional probability distribution (CPD)** using a lambda function. All variables should be in lowercase.
Please provide the following Python lambda function example in a code block, and return only the code (no additional text or explanation). 
The function is: ```python lambda ...: ... ```

1. **Function Signature**  
   - If the variable has no parents (unconditional), write a lambda with no arguments.  
   - If the variable has one or more parents (conditional), include them as parameters.  
2. **Dictionary Keys:**  
   - Must be all valid values that the variable can take (e.g., 0, 1, `'rain'`, etc.). 
   - if the values are numerical, they should be integers or floats, not strings.
3. **Dictionary Values:**  
   - Must be probabilities (floats) in the range **[0, 1]**.  
   - For each parent condition, these probabilities **must sum to 1**.
4. **Conditional Logic:**  
   - **All** `if...else` statements must occur **outside** the dictionary, selecting the appropriate dictionary as a whole.  
   - **Not** allowed:  
     {{0: 0.5 if parent else 0.2, 1: 0.5 if parent else 0.8}}  # Disallowed
   - **Allowed** (top-level branching):  
     lambda evidence_var1, evidence_var2: {{0: 0.1, 1: 0.9}} if evidence_var1 == 1 and evidence_var2 == 0 else {{0: 0.9, 1: 0.1}}
5. **No Escaped Newlines/Characters:**  
   - Keep the lambda on one line; avoid using `\n` or other escape sequences within the lambda.
6. **Example:**  
   - **Without parents:**  
     lambda: {{0: 0.1, 1: 0.9}}
   - **With parents and multiple conditions:**  
     lambda evidence_var1, evidence_var2: {{0: 0.1, 1: 0.9}} if evidence_var1 == 1 and evidence_var2 == 0 else {{0: 0.9, 1: 0.1}}
"""

# PROMPTS["cpd_stochastic_extraction"] = """
# You are an expert decision analyst specializing in decision-making under uncertainty and influence diagram modelling. 
# You will be provided with a source text, a variable and its evidences. 
# Output should follow the *JSON* format (make sure to check the "Parentheses"  "Brackets"  "Braces"): 
# {format_instructions}


# Assign the conditional probability for variable "{variable}" and each value of the evidence "{parent_variable_list}".
# The values of the variable and the evidence variables *must* be valid and complete.
# The valid values of the variable: "{variable_domain}"
# The valid values of the evidence variables:"{condition_domain}"

# Please extract the conditional probability distribution (CPD) from the text.
# {text}
# """

PROMPTS["cpd_stochastic_extraction"] = """
You are an expert decision analyst specializing in decision-making under uncertainty and influence diagram modeling.

You will be provided with:
1. A source text:  
   {text}
2. A target variable:  
   {variable}
3. Evidence variables (parents):  
   {parent_variable_list}

Your task is to:
1. **Extract the conditional probability distribution (CPD)** for {variable} based on the provided **source text**.
2. Assign probabilities **for every possible value** of {variable} given **each possible combination** of the valid evidence values.
3. Adhere to the valid domains of:
   - **Target variable** {variable_domain}
   - **Evidence variables** {condition_domain}
4. Utility node values must be numerical.
5. We aim to maximize the utility, so if utility node is cost-related, the values should be negative. If utility node is benefit-related, the values should be positive.
6. **Output** must strictly follow the lambda string format.

**Required Output**  
Please produce the **conditional probability distribution (CPD)** using a lambda function. All variables should be in lowercase.
Please provide the following Python lambda function example in a code block, and return only the code (no additional text or explanation). 
The function is: ```python lambda ...: ... ```

1. **Function Signature**  
   - If the variable has no parents (unconditional), write a lambda with no arguments.  
   - If the variable has one or more parents (conditional), include them as parameters.  
2. **Dictionary Keys:**  
   - Must be all valid values that the variable can take (e.g., `0`, `1`, `'rain'`, etc.).  
3. **Dictionary Values:**  
   - Must be probabilities (floats) in the range **[0, 1]**.  
   - For each parent condition, these probabilities **must sum to 1**.
4. **Conditional Logic:**  
   - **All** `if...else` statements must occur **outside** the dictionary, selecting the appropriate dictionary as a whole.  
   - **Not** allowed:  
     {{0: 0.5 if parent else 0.2, 1: 0.5 if parent else 0.8}}  # Disallowed
   - **Allowed** (top-level branching):  
     lambda evidence_var1, evidence_var2: {{0: 0.1, 1: 0.9}} if evidence_var1 == 1 and evidence_var2 == 0 else {{0: 0.9, 1: 0.1}}
5. **No Escaped Newlines/Characters:**  
   - Keep the lambda on one line; avoid using `\n` or other escape sequences within the lambda.
6. **Example:**  
   - **Without parents:**  
     lambda: {{0: 0.1, 1: 0.9}}
   - **With parents and multiple conditions:**  
     lambda evidence_var1, evidence_var2: {{0: 0.1, 1: 0.9}} if evidence_var1 == 1 and evidence_var2 == 0 else {{0: 0.9, 1: 0.1}}
"""

PROMPTS["cpd_stochastic_improve"] = """
You are an expert decision analyst specializing in decision-making under uncertainty and influence diagram modeling.

You will be provided with:
1. A source text:  
   {text}
2. A target variable:  
   {variable}
3. Evidence variables (parents):  
   {parent_variable_list}

Your task is to:
1. **Extract the conditional probability distribution (CPD)** for {variable} based on the provided **source text**.
2. Assign probabilities **for every possible value** of {variable} given **each possible combination** of the valid evidence values.
3. Adhere to the valid domains of:
   - **Target variable** {variable_domain}
   - **Evidence variables** {condition_domain}
4. Utility node values must be numerical.
5. We aim to maximize the utility, so if utility node is cost-related, the values should be negative. If utility node is benefit-related, the values should be positive.
6. **Output** must strictly follow the lambda string format.

**Required Output**  
Please produce the **conditional probability distribution (CPD)** using a lambda function. All variables should be in lowercase.
Please provide the following Python lambda function example in a code block, and return only the code (no additional text or explanation). 
The function is: ```python lambda ...: ... ```

1. **Function Signature**  
   - If the variable has no parents (unconditional), write a lambda with no arguments.  
   - If the variable has one or more parents (conditional), include them as parameters.  
2. **Dictionary Keys:**  
   - Must be all valid values that the variable can take (e.g., `0`, `1`, `'rain'`, etc.).  
3. **Dictionary Values:**  
   - Must be probabilities (floats) in the range **[0, 1]**.  
   - For each parent condition, these probabilities **must sum to 1**. 
   - If the text does not provide exact values, infer them based on the information to ensure consistency.
4. **Conditional Logic:**  
   - Use **if/else** statements **outside** the dictionaries to **choose which dictionary** to return.  
   - **Do not** place `if condition: value` *inside* a single dictionary for partial entries. Instead, return entire dictionaries via `if... else...`.  
   - Example (two parents):  
     lambda evidence_var1, evidence_var2: {{0: 0.1, 1: 0.9}} if evidence_var1 == 1 and evidence_var2 == 0 else {{0: 0.9, 1: 0.1}}
5. **No Escaped Newlines/Characters:**  
   - Keep the lambda on one line; avoid using `\n` or other escape sequences within the lambda.
6. **Example:**  
   - **Without parents:**  
     lambda: {{0: 0.1, 1: 0.9}}
   - **With parents and multiple conditions:**  
     lambda evidence_var1, evidence_var2: {{0: 0.1, 1: 0.9}} if evidence_var1 == 1 and evidence_var2 == 0 else {{0: 0.9, 1: 0.1}}

Some issues occurred in the original CPD:
{ori_contents}

Please address the following issues and provide the corrected CPD:
{issues}
"""

# summarizer
PROMPTS["summarize_variable"] = """
For the following text, extract and summarize all the conditional probability distributions information related to the variable `{variable_name}`.

Text:

{text}

Instructions:
1. Focus only on conditional probability distributions information directly related to `{variable_name}`
2. If no relevant information is found, return an empty string
3. Be concise but include all details about the variable

Summarized text:
"""

# llm-as-a-judge
# nodeEdgeClassification
PROMPTS["node_list_common_nodes"] = """
Your goal is to find the target nodes that are present in the generated node list. Node names may not match exactly, but if they have the same meaning or refer to the same concept, they should be considered the same node.

Based on a target node list and a generated node list, output a list of tuples, each containing a target node name and a generated node name. You should carefully analyze both lists to identify semantic matches even when the wording differs significantly.

Target node list:

{target_nodes}

Generated node list:

{generated_nodes}

Requirements:
1. Output pairs of node names (target node, generated node) that represent the same concept or meaning.
2. Use exact node names as they appear in their respective lists - do not modify them.
3. Two nodes match if they have the same meaning or refer to the same concept.
4. Be strict in your matching criteria - avoid false positives.
5. Each target node and generated node can appear **at most once** in your output.
6. Do not include target nodes that have no match in the generated list.
7. Output using the following format:

{format_instructions}
"""

PROMPTS["edge_list_common_edges"] = """
Your goal is to find the target edges that are present in the generated edge list. Each edge is a string of the form 'cause->effect'. Two edges match only if both their cause and effect nodes refer to the same concept semantically.

Based on the target edge list and generated edge list below, identify only the true semantic matches. Be conservative in your matching - when in doubt, do not consider edges as matching.

Target edge list:
{target_edges}

Generated edge list:
{generated_edges}

Requirements:
1. Output only pairs of edges (target_edge, generated_edge) that definitely represent the same relationship.
2. Use the exact edge strings as they appear in their respective lists - do not modify them.
3. An edge matches only if both its cause and effect match semantically with the other edge.
4. Be strict in your matching criteria - avoid false positives.
5. Each target edge and generated edge can appear at most once in your output.
6. Do not include target edges that have no match in the generated list.
7. Output using the following format:

{format_instructions}
"""
PROMPTS["fix_node_pairs"] = """
Your goal is to fix issues in matching nodes between a target node list and a generated node list. You are provided with a list of node pairs (target node, generated node) that potentially have issues, and a list of identified issues.

Target node list:
{target_nodes}

Generated node list:
{generated_nodes}

Current node pairs with issues:
{node_pairs}

Issues:
{issues}

Requirements:
1. Fix the issues by finding correct node names that exist in their respective lists.
2. For nodes that don't exist, find the closest matching node name in the appropriate list.
3. If a node pair cannot be fixed, exclude it from your output.
4. Output only pairs of nodes (target_node, generated_node) that definitely represent the same node.
5. Output using the following format:

{format_instructions}
"""

PROMPTS["node_relevance_evaluation"] = """
You are an expert decision analyst specializing in decision-making under uncertainty and influence diagram modelling.

Your task is to evaluate whether each node in the generated list is relevant to the decision-making context described in the text.

Decision context:
{text}

Nodes to evaluate:
{nodes}

Requirements:
1. Evaluate if each node should be considered in the decision problem described.
2. A relevant node should meet one of the following criteria:
   - be explicitly mentioned in the text, or be a standard variable that would reasonably affect this type of decision
   - provide insight into different stakeholders' preferences
   - correspond to a decision opportunity under the context
   - correspond to an uncertain event whose outcome affects the consequence of the decision
3. For each node, determine if it is relevant (true) or not relevant (false).
4. Output using the following format:

{format_instructions}
"""

PROMPTS["graph_comparison"] = """
You are an expert decision analyst specializing in decision-making under uncertainty and influence diagram modelling.

Your task is to compare two influence diagrams and determine which one better captures the decision-making problem described in the text. The text provides essential context, though it may not cover all critical aspects of the decision problem. Your evaluation should not be solely based on the text, but should also incorporate deep thinking about the decision problem itself, drawing on your expertise in decision analysis. Evaluate the graphs based on their completeness, accuracy, and parsimony.

Decision context:
{text}

Graph A:
Nodes: {nodes_a}
Edges: {edges_a}

Graph B:
Nodes: {nodes_b}
Edges: {edges_b}

Evaluation criteria (all equally important):
1. Completeness: Does the graph comprehensively represent the decision problem by including:
   - All relevant stakeholders
   - All relevant goals and objectives of the decision-maker/stakeholders
   - All available actions and decision points
   - All key uncertainties and dependencies that influence outcomes
   
2. Accuracy: Are the variables (nodes) and relationships (edges) correctly identified and represented?
   - Do nodes accurately reflect concepts from the decision context?
   - Are the causal and informational relationships properly captured?
   
3. Parsimony: Is the graph appropriately simplified without losing essential elements?
   - Avoids redundant or overlapping nodes
   - Excludes irrelevant factors that don't meaningfully impact the decision
   - Maintains a clear structure without unnecessary complexity

Requirements:
1. Select either "A" or "B" as your preferred graph and provide a concise, well-reasoned explanation for your choice.
2. Output using the following format:

{format_instructions}
"""

PROMPTS["graph_choice"] = """
You are an expert decision analyst specializing in decision-making under uncertainty and influence diagram modelling.

Your task is to compare two influence diagrams and determine which one better captures the real-world decision problem. The text provides essential context, though it may not cover all critical aspects of the decision problem. Your evaluation should not be solely based on the text, but should also incorporate deep thinking about the decision problem itself, drawing on your expertise in decision analysis. Evaluate the graphs based on their completeness, accuracy, and parsimony.

Decision context:
{text}

Graph A:
Nodes: {nodes_a}
Edges: {edges_a}

Graph B:
Nodes: {nodes_b}
Edges: {edges_b}


Requirements:
1. The selected graph should be more realistic and relevant to the decision problem.
2. Select either "A" or "B" as your preferred graph and provide a concise, well-reasoned explanation for your choice.
3. Output using the following format:

{format_instructions}
"""

# decision
PROMPTS["decision_vanilla"] = """
Make a decision for the following text problem.

Text:

{text}

Requirements:
1. The selected decision variable must come from the alternatives:

{decision_alternatives}

2. Output using the following format:

{format_instructions}
"""

PROMPTS["decision_cot"] = """

Make a decision for the following text problem.

Text:

{text}

When making a decision, please follow these steps:

1. SITUATION ANALYSIS
   - Clearly define the decision being made and its context
   - Identify all relevant stakeholders and their perspectives
   - List the available options/alternatives

2. UNCERTAINTY ASSESSMENT
   - Identify key uncertain variables
   - Consider how these factors might interact and influence each other
   - Estimate probability distributions: What's most likely? What's the worst case?
   - Explicitly quantify confidence levels where possible (e.g., "I'm 70 percent confident that...")
   - Identify areas where knowledge is limited

3. VALUE ESTIMATION
   - Define the criteria for evaluating options (what matters in this decision?)
   - Assign utility values to different outcomes
   - Identify any value trade-offs requiring judgment calls

4. DECISION RECOMMENDATION
   - Calculate expected utility for each option (probability times utility)
   - Present final decision recommendation with clear reasoning and correct format
   - Acknowledge remaining uncertainties and how they might affect the decision if possible

Requirements:
1. The selected decision variable must come from the alternatives:

{decision_alternatives}

2. Output using the following format:

{format_instructions}
"""

PROMPTS["decision_sc"] = """ """

# Evaluator prompts
PROMPTS["evaluator_text_consistency"] = """
You are an expert decision analyst specializing in evaluating influence diagrams. Your task is to evaluate whether an influence diagram accurately and comprehensively represents the decision problem described in the source text.

The influence diagram consists of:
1. A list of nodes (variables) with their types and values
2. A list of edges (relationships) between nodes
3. Conditional probability distributions (CPDs) for nodes conditioned on their parents

Node list:
{node_list}

Edge list:
{edge_list}

CPD list:
{cpd_list}

Source text:
{source_text}

Please conduct a thorough evaluation focusing on the following aspects:

1. Edges:
   Focus on qualitative descriptions. For issues related with edges, you need to quote the qualitative descriptions from the source text.
   - Do the edges correctly capture all important relationships and dependencies mentioned in the source text?
   - Are there any missing or incorrect edges that would significantly impact decision analysis?
   - Is the direction of causality correctly represented?

2. CPDs:
   Focus on quantitative descriptions. For issues related with CPDs, you need to quote the quantitative descriptions from the source text.
   - For each variable, evaluate the probablity of its values given EACH possible combination of its parents variables. Compare the result of the CPD and the description in the source text. Show your reasoning.
   - NOTE: Don't flag issues for differences in presentation format as long as they are mathematically equivalent. For example, a discrete mapping may be a result of pre-calculation. Only report CPD issues when the probability distribution is wrong.
      For example, the value of a variable is the multiplication of two factors. It CAN be represented as a discrete mapping, as long as the calculation is correct.

3. Consistency between edges and CPDs.
   - Does the argument of the CPD match the edge? 
      For example, if an edge indicates a dependency between nodes, but the CPD shows independence (or vice versa), this is an issue that should be fixed by modifying either the edge or the CPD based on the source text.
   - Are the values of variables consistent across CPDs?
      For example, if the variable 'boo' takes value '1' and '0' with certain probabilities, but another variable 'faa' takes some value IF 'boo' takes value '-1', this can be an inconsistency issue.
   For these issues, you should provide a clear diagnosis of the failures in the diagram, whether edge or CPD should be fixed.

IMPORTANT: All issues MUST be directly supported by evidence from the source text. Do not suggest additions or changes based on your own opinion or domain knowledge unless explicitly stated in the text.

For each issue found, use EXACTLY one of these formats:
- For edge issues: "Edge from 'SourceNode' to 'TargetNode' issue description" or "Edge from 'NodeA' to 'NodeB' is missing but..."
- For CPD issues: "CPD for 'NodeName' issue description"

Format your response as a JSON object with an "issues" array. Each issue should be a string in the array.

If you identify no issues, return: {{"issues": []}}

Example response:
{{
  "issues": [
    "CPD for 'Profit' shows 80% probability of high profit with low sales, contradicting the text which states profits are directly proportional to sales, which means the probability of high profit with low sales should be lower",
    "Edge from 'Marketing' to 'Customer Satisfaction' is missing, but the text indicates marketing affects customer satisfaction"
  ]
}}
"""

PROMPTS["evaluator_improvement_suggestions"] = """
You are an expert decision analyst specializing in improving influence diagrams. You have been provided with:
1. An influence diagram with issues
2. The source text describing the decision problem

Your task is to provide specific, actionable diagnosis, fixes, and explanations to improve the diagram and address the identified issues.

Source text:
{source_text}

Node list:
{node_list}

Edge list:
{edge_list}

CPD list:
{cpd_list}

Issues identified:
{issues}

Please provide detailed improvement suggestions for each issue. Your suggestions should:
- Provide a clear diagnosis of the failures in the diagram
- Be specific about necessary modifications with explanations
- Be concise and to the point
"""

# Evaluator prompt for determining next agent
PROMPTS["evaluator_next_agent"] = """
You are an expert decision analyst helping to diagnose and fix issues in an influence diagram.
You need to determine whether the GraphAgent or ProbabilityAgent should be called to address the issues.

The GraphAgent handles issues related to:
- Node structure (missing nodes, duplicate nodes, invalid node types)
- Edge structure (missing edges, invalid edges, cycles)
- Consistency of factors and qualitative relationships with the source text

The ProbabilityAgent handles issues related to:
- Conditional Probability Distributions (CPDs)
- Stochastic function implementations
- Probability values and distributions
- Consistency of CPDs with the source text

Issues identified in the diagram:
{issues}

Error traceback (if any):
{error_traceback}

Additional comments and improvement suggestions:
{comments}

Based on the above information, determine which agent should be called next.
If both agents could potentially help but one is more critical, prioritize the more critical one.

Return ONLY one of these two options without any explanation: "GraphAgent" or "ProbabilityAgent".
"""