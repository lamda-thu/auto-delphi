DELPHI_PROMPTS = {}

DELPHI_PROMPTS["cpd_generation_from_description"] = """
You are an expert decision analyst specializing in decision-making under uncertainty and influence diagram modeling.

You will be provided with:
1. A target variable: {variable}
2. Evidence variables (parents): {parent_variable_list}
3. A user description of how the variable depends on its parents: {description}

Your task is to:
1. **Generate the conditional probability distribution (CPD)** for {variable} based on the user's description.
2. Assign probabilities **for every possible value** of {variable} given **each possible combination** of the evidence values.
3. The user description should guide your probability assignments.
4. **Output** must strictly follow the lambda string format.

**Required Output**
Please produce the **conditional probability distribution (CPD)** using a lambda function. All variables should be in lowercase.
Please provide the following Python lambda function in a code block, and return only the code (no additional text or explanation).
The function is: ```python lambda ...: ... ```

1. **Function Signature**
   - If the variable has no parents (unconditional), write a lambda with no arguments.
   - If the variable has one or more parents (conditional), include them as parameters.
2. **Dictionary Keys:**
   - Must be all valid values that the variable can take (e.g., 0, 1, 'rain', etc.).
   - If the values are numerical, they should be integers or floats, not strings.
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

DELPHI_PROMPTS["cpd_improve_from_feedback"] = """
You are an expert decision analyst specializing in decision-making under uncertainty and influence diagram modeling.

You will be provided with:
1. A target variable: {variable}
2. Evidence variables (parents): {parent_variable_list}
3. A user description of how the variable depends on its parents: {description}
4. Issues with the previous CPD attempt: {feedback}

Your task is to:
1. **Improve the conditional probability distribution (CPD)** for {variable} based on the user's description and the feedback.
2. Address all issues mentioned in the feedback.
3. Assign probabilities **for every possible value** of {variable} given **each possible combination** of the evidence values.
4. **Output** must strictly follow the lambda string format.

**Required Output**
Please produce the **conditional probability distribution (CPD)** using a lambda function. All variables should be in lowercase.
Please provide the following Python lambda function in a code block, and return only the code (no additional text or explanation).
The function is: ```python lambda ...: ... ```

1. **Function Signature**
   - If the variable has no parents (unconditional), write a lambda with no arguments.
   - If the variable has one or more parents (conditional), include them as parameters.
2. **Dictionary Keys:**
   - Must be all valid values that the variable can take (e.g., 0, 1, 'rain', etc.).
   - If the values are numerical, they should be integers or floats, not strings.
3. **Dictionary Values:**
   - Must be probabilities (floats) in the range **[0, 1]**.
   - For each parent condition, these probabilities **must sum to 1**.
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

Previous issues to address:
{feedback}
"""
