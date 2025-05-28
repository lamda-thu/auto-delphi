from typing import List, Dict, Any, Union, Callable, Mapping, Sequence, Optional
from pydantic import BaseModel, Field, PrivateAttr
from pycid.core.cpd import TabularCPD
from graph import InfluenceDiagram

Outcome = Any
Relationship = Union[TabularCPD, Dict[Outcome, float], Callable[..., Union[Outcome, Dict[Outcome, float]]]]


class ConditionalProbabilityDistributionFunction(BaseModel):
    variable: str = Field(
        "The name of the variable of interest."
    )
    condition: Dict[str, Outcome] = Field(
        "The condition is a Dictionary, the keys are variable_names and the values are the variable values. For example, {'D1': 'withdraw'} stands for the condition of decision D1==withdraw"
    )
    conditional_probability_distribution: str = Field(
        "The conditional probability distribution (CPD) of the variable given the condition. The CPD should be expressed as a code snippet containing a `python` function. The input arguments of the function are the values of the condition variables. The function returns a dictionary specifying the conditional distribution of the variable. For example, {0: 0.1, 1: 0.9} describes a Bernoulli(0.9) distribution." # Type: Relationship
    ) #TODO: add example

    def __init__(self, variable, condition, conditional_probability_distribution):
        super().__init__(variable=variable, condition=condition, conditional_probability_distribution=conditional_probability_distribution)
    
    def toTabularCPD(self):
        raise NotImplementedError
    
class StochasticFunctionCPD(BaseModel):
    """Defines the conditional probability distribution
    """

    stochastic_function: str = Field(
#         """
# A lambda function that maps parent outcomes to a distribution over outcomes for the variable. 
# The function should concisely represent the probabilistic relations between the variaibles.  Notice: 
# 1) keys of the dictionary should be the *valid values of the variable*,
# 2) the value of evidence/condition should be the *valid values of the evidence variables*, and be within 0-1
# 3) sum over the values of the dictionary **must be** *equal* to 1,
# 4) when dealing with qualitative varaibles, consider using 'if' to choose between multiple dictionaries, 
# 5) don't use line break \\n and other escape character \ in the lambda function.
# 6) if no parent, the lambda function should be like lambda: {0: 0.1, 1: 0.9}
# e.g. "lambda evidence_variable_name_1, evidence_variable_name_2: {0: 0.1, 1: 0.9} if evidence_variable_name_1 == 1 and evidence_variable_name_2 == 0 else {0: 0.9, 1: 0.1}"
# """
"""
1. **Function Form**  
   - **No parents:**  
     lambda: {var_value1: prob1, var_value2: prob2}
   - **With parents:**  
     lambda parent1, parent2: ...

2. **Dictionary Keys**  
   - Must be **all valid outcomes** of the variable (e.g., 0, 1, 'rain').

3. **Dictionary Values**  
   - Must be **probabilities** in [0, 1].  
   - **Sum to 1** for each if/else condition.

4. **Conditional Logic**  
   - Use **if/else** statements **outside** the dictionaries to **choose which dictionary** to return.  
   - **Do not** place `if condition: value` *inside* a single dictionary for partial entries. Instead, return entire dictionaries via `if... else...`.  
   - **Example:**  
     lambda evidence_variable_name_1, evidence_variable_name_2: {0: 0.1, 1: 0.9} if evidence_variable_name_1 == 1 and evidence_variable_name_2 == 0 else {0: 0.9, 1: 0.1}

5. **No Escape Characters**  
   - Keep it to **one line**; **no** `\n`.

**Example**  
- **Unconditional:**  
  lambda: {0: 0.4, 1: 0.6}
"""
    )
# class StochasticFunctionCPD(BaseModel):
#     """Defines the conditional probability distribution
#     """

#     variable: str = Field(
#         "The variable whose conditional probability distribution is defined. The variable name should *exactly match* a node in the influence diagram."
#         )
#     variable_card: int = Field(
#         "Cardinality of states of `variable`."
#     )
#     evidence: Optional[List[str]] = Field(
#         "List of variables in evidences (if any) w.r.t. which CPD is defined. The name of each evidence should *exactly match* a node in the influence diagram."
#     )
#     evidence_card: Optional[List[int]] = Field(
#         "Cardinality of states of variables in `evidence`."
#     )
#     state_names: Optional[Dict] = Field(
#         "A dict of the form {var_name: list of states} to specify the state names for the variable in the CPD. The value of the variable should come from  the *valid values of the condition variables*. The value of the evidence should come from the *valid values of the condition variables*." 
#     )
#     stochastic_function: str = Field(
#         """
# A lambda function that maps parent outcomes to a distribution over outcomes for the variable. The function should concisely represent the probabilistic relations between the variaibles.  Notice: 
# 1) keys of the dictionary should be the *valid values of the variable*,
# 2) the value of evidence should be the *valid values of the condition variables*,
# 3) sum over the values of the dictionary must be *equal* to 1 (available probability mass is spread evenly on unspecified outcomes),
# 4) when dealing with qualitative varaibles, consider using 'if' to choose between multiple dictionaries, like lambda {evidence_name}: {0: 0.1, 1: 0.9} if {evidence_name} == 1 else {0: 0.9, 1: 0.1}.
# 5) don't use line break "\\n" in the lambda function.
# 6) if no parent, the lambda function should be like lambda: {0: 0.1, 1: 0.9}
# """
#     )
    
    

#     def __init__(self, variable, stochastic_function):
#         super().__init__(variable=variable, stochastic_function=stochastic_function)


class TabularCPD(BaseModel):
    """
    Defines the conditional probability distribution table (CPD table)

    Parameters
    ----------
    variable: int, string (any hashable python object)
        The variable whose CPD is defined.

    variable_card: integer
        Cardinality/no. of states of `variable`

    values: 2D array, 2D list or 2D tuple
        Values for the CPD table. Must be of shape (variable_card, \prod evidence_card). Please refer the example for the exact format needed.

    evidence: array-like
        List of variables in evidences(if any) w.r.t. which CPD is defined.

    evidence_card: array-like
        cardinality/no. of states of variables in `evidence`(if any)

    Examples
    --------
    For a distribution of P(grade|diff, intel)

    +---------+-------------------------+------------------------+
    |diff:    |          easy           |         hard           |
    +---------+------+--------+---------+------+--------+--------+
    |aptitude:| low  | medium |  high   | low  | medium |  high  |
    +---------+------+--------+---------+------+--------+--------+
    |gradeA   | 0.1  | 0.1    |   0.1   |  0.1 |  0.1   |   0.1  |
    +---------+------+--------+---------+------+--------+--------+
    |gradeB   | 0.1  | 0.1    |   0.1   |  0.1 |  0.1   |   0.1  |
    +---------+------+--------+---------+------+--------+--------+
    |gradeC   | 0.8  | 0.8    |   0.8   |  0.8 |  0.8   |   0.8  |
    +---------+------+--------+---------+------+--------+--------+

    values should be
    [[0.1,0.1,0.1,0.1,0.1,0.1],
     [0.1,0.1,0.1,0.1,0.1,0.1],
     [0.8,0.8,0.8,0.8,0.8,0.8]]

    >>> cpd = TabularCPD('grade',3,[[0.1,0.1,0.1,0.1,0.1,0.1],
    ...                             [0.1,0.1,0.1,0.1,0.1,0.1],
    ...                             [0.8,0.8,0.8,0.8,0.8,0.8]],
    ...                             evidence=['diff', 'intel'], evidence_card=[2,3])
    """

    variable: str = Field(
        "The variable whose conditional probability distribution is defined. The variable name should *exactly match* a node in the influence diagram."
    )
    variable_card: int = Field(
        "Cardinality of states of `variable`."
    )
    values: Sequence[Sequence[float]] = Field(
        """Values for the CPD table. Must be of shape (variable_card, \prod evidence_card) if Evidence is not None, and (variable_card, 1) otherwise. Please refer the example for the exact format needed.
    Examples
    --------
    For a distribution of P(grade|diff, intel)

    +---------+-------------------------+------------------------+
    |diff:    |          easy           |         hard           |
    +---------+------+--------+---------+------+--------+--------+
    |aptitude:| low  | medium |  high   | low  | medium |  high  |
    +---------+------+--------+---------+------+--------+--------+
    |gradeA   | 0.1  | 0.1    |   0.1   |  0.1 |  0.1   |   0.1  |
    +---------+------+--------+---------+------+--------+--------+
    |gradeB   | 0.1  | 0.1    |   0.1   |  0.1 |  0.1   |   0.1  |
    +---------+------+--------+---------+------+--------+--------+
    |gradeC   | 0.8  | 0.8    |   0.8   |  0.8 |  0.8   |   0.8  |
    +---------+------+--------+---------+------+--------+--------+

    values should be
    [[0.1,0.1,0.1,0.1,0.1,0.1],
     [0.1,0.1,0.1,0.1,0.1,0.1],
     [0.8,0.8,0.8,0.8,0.8,0.8]]

    >>> cpd = TabularCPD('grade',3,[[0.1,0.1,0.1,0.1,0.1,0.1],
    ...                             [0.1,0.1,0.1,0.1,0.1,0.1],
    ...                             [0.8,0.8,0.8,0.8,0.8,0.8]],
    ...                             evidence=['diff', 'intel'], evidence_card=[2,3])      
        """
    )
    evidence: Optional[List[str]] = Field(
        "List of variables in evidences (if any) w.r.t. which CPD is defined. The name of each evidence should *exactly match* a node in the influence diagram."
    )
    evidence_card: Optional[List[int]] = Field(
        "Cardinality of states of variables in `evidence`."
    )
    state_names: Optional[Dict] = Field(
        "A dict of the form {var_name: list of states} to specify the state names for the variables in the CPD."
    )


class RuleCPD(BaseModel):
    variable: str = Field("The variable whose conditional probability distribution is defined. The variable name should *exactly match* a node in the influence diagram.")
    variable_value: str = Field("The value the variable takes (with certain probability).")
    context: str = Field("The subset of evidence variables that the distribution depends on.")
    context_value: str = Field("The values the context variables take.") #BUG
    probability: float = Field("The probability of variable taking variable_value, given that context takes context_value. The value of probability should be within the range (0, 1).")
