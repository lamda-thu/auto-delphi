# Delphi Module

This module implements a Delphi-style process for eliciting conditional probability distributions (CPDs) for influence diagram nodes.

## Overview

The Delphi process consists of two main steps:
1. **Elicitation**: Ask a human user to describe how a variable's values depend on its parent variables
2. **Generation**: Use an LLM to generate a stochastic function CPD based on the user's description

## Classes

### `ProbabilityDelphi`

The main class that implements the Delphi process at the variable level, independent of any influence diagram.

#### Usage

```python
from delphi import ProbabilityDelphi
from lamda.models import Gpt4o

# Initialize with a language model
delphi = ProbabilityDelphi(
    language_model=Gpt4o(),
    max_retries=5  # Number of retry attempts for validation
)

# Run the Delphi process for a variable
variable = "Traffic"
parent_variables = ["Weather", "Time_of_Day"]

cpd = delphi.run(variable, parent_variables)
```

#### Methods

- `run(variable: str, parent_variables: List[str])`: Main method that orchestrates the entire Delphi process
- `_elicit_cpd_description(variable: str, parent_variables: List[str])`: Prompts the user for a description
- `_generate_cpd_params(variable: str, parent_variables: List[str], description: str, feedback: str)`: Uses LLM to generate CPD parameters
- `_validate_cpd_params(cpd_params: str)`: Validates the generated CPD parameters
- `_construct_cpd(cpd_params: str)`: Constructs the final CPD object

## Example

See `examples/delphi_example.py` for a complete working example.

## Features

- **Interactive Elicitation**: Prompts human experts to describe probabilistic relationships in natural language
- **LLM-Powered Generation**: Converts natural language descriptions to executable lambda functions
- **Automatic Validation**: Validates generated CPDs and provides feedback for corrections
- **Retry Mechanism**: Automatically retries generation with feedback if validation fails
- **Post-Processing**: Automatically fixes common issues in generated lambda functions (e.g., string numbers to numeric values)

## Extending

To create a version that works with influence diagrams, subclass `ProbabilityDelphi` and override the `_construct_cpd` method to properly construct `StochasticFunctionCPD` objects with the diagram context.

Example:

```python
class InfluenceDiagramProbabilityDelphi(ProbabilityDelphi):
    def __init__(self, language_model, diagram, max_retries=5):
        super().__init__(language_model, max_retries)
        self.diagram = diagram
    
    def _construct_cpd(self, cpd_params: str) -> StochasticFunctionCPD:
        # Use diagram context to construct proper CPD
        # Similar to how lamda.agent.ProbabilityAgent does it
        pass
```

