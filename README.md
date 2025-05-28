# LAMDA: Large Language Model as Decision Analyst

LAMDA employs large language models (LLMs) to generate influence diagrams.

This repository contains
- the implementation of LAMDA (in `src/`)
- the data used in the experiments (in `data/`)
- the code for the experiments (in `experiments/`) and analysis of results (in `analysis/`)
- the scripts of minimal examples (in `scripts/`)

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

Then you can run the experiments by running the following scripts:
- `experiment/graphExtractExperiment.py`: the influence diagram graph geration experiment
- `experiment/probabilityExtractExperiment.py`: the conditional probability distribution assignment experiment
- `experiment/decisionExperiment.py`: the decision-making experiment

## Run the minimal examples

To get familiar with the functionality of LAMDA, you can run the following scripts:
- `createInfluenceDiagram.py`: create an influence diagram from JSON files
- `extractGraph.py`: extract the graph from the text
- `extractProbability.py`: load existing graph, extract the probability from the text, and solve the influence diagram for optimal policy
- `testDecisionMaker.py`: test the decision makers used in the experiments (`Vanilla`, `Cot`, `Sc`, `Dellma`, `Aid`)

## Cite this work

If you find this work useful, please cite:

```bibtex

```