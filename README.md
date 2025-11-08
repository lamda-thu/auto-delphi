# AutoDelphi

AutoDelphi employs large language models (LLMs) to empower Delphi-style construction of influence diagrams.
It builds upon the `lamda` library (https://github.com/lamda-thu/LAMDA) to provide a more flexible and powerful way to construct influence diagrams, especially for the quantitative components.

This repository contains
- the implementation of LAMDA (in `lamda/`)
- the implementation of AutoDelphi (in `delphi/`)
- the data used in the LAMDA experiments (in `data/`)

## Setup the environment

```bash
pip install -r requirements.txt
```

Notice:
- To properly install `pycid`, Python version 3.9 is suggested.
- `pycid==0.8.2` depends on an earlier version of `pgmpy`. You can install `pgmpy==0.1.26` after `pycid` is installed.

## Run the experiments

To run the experiments, you need to set the API keys in `APIKEYS/apiKeys.py`. For example, you can set the OpenAI API key by replacing `yourOpenaiApiKey` with your actual API key.

```python
openaiKey = "yourOpenaiApiKey"
```
## Delphi Module

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

## Cite this work

If you find this work useful, please cite:

```bibtex

```