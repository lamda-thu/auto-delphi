import os
import sys
if __name__=='__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    sys.path.append(project_root)

import json
import itertools
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from pycid.core.cpd import StochasticFunctionCPD

from src.graph import InfluenceDiagram

class BatchInstanceAnalyzer(object):
    """
    Analyze a batch of instances.
    Implement visualization for instance comparison.
    """
    def __init__(self, pathToInstances: list[str]):
        """ Initialize the batch instance analyzer with a list of paths to instances."""
        self.setInstanceRootPath(pathToInstances)
        self.instance_name = pathToInstances.split("/")[-1]

    def setInstanceRootPath(self, pathToInstances: str):
        """ Set the path to the instance file."""
        if not os.path.exists(pathToInstances):
            raise ValueError(f"Path {pathToInstances} does not exist.")
        if len(os.listdir(pathToInstances)) == 0:
            raise ValueError(f"Path {pathToInstances} is empty.")

        self.__pathToInstances = pathToInstances
        self.__results = None
        self.__contexts = None

    def analyze(self, first_decision: bool=False) -> list:
        """
        Analyze a batch of instances.
        This should be called multiple times to evaluate and compare different instances.
        
        Returns
        -------
        list
            The list of results.
            The list of possible parent values.
        """
        results = []
        instance_ids = []
        for instance_path in tqdm(os.listdir(self.__pathToInstances)):
            instance_id = instance_path.split("-")[-1]
            if instance_id == "base":
                continue

            instance_analyzer = InstanceAnalyzer(os.path.join(self.__pathToInstances, instance_path))
            instance_analyzer.loadInstance()
            eu = instance_analyzer.analyze(first_decision=first_decision)
            results.append(eu)
            instance_ids.append(instance_id) #TODO store instance id
        if len(results) > 0:
            contexts = list(results[0].keys())
            print(contexts)
            self.__results = results
            self.__contexts = contexts
            self.__instance_ids = instance_ids
            return results, contexts
        else:
            raise ValueError("No instance evaluated.")

    def getInstanceContextEuTable(self):
        #TODO
        if self.__results is None:
            raise ValueError("No instance evaluated.")
        rows = []
        header = ["instance", "context"]
        for context_eu_dict in self.__results:
            for cond_eu_dict in context_eu_dict.values():
                header.extend(list(cond_eu_dict.keys()))
                break
            break

        for idx, context_eu_dict in enumerate(self.__results):
            for context, cond_eu_dict in context_eu_dict.items():
                row = [
                    self.__instance_ids[idx],
                    str(context).strip().replace("(", "").replace(")", "")\
                        .replace(",", "_").replace(" ", "")
                ]
                row.extend(list(cond_eu_dict.values()))
                rows.append(row)

        eu_table = pd.DataFrame(rows, columns=header)
        return eu_table


    def getResults(self):
        return self.__results

    def heatmapInstanceEuTable(self, plot=True):
        """
        For each possible value of the context variables, draw a heatmap of the expected utility table.
        Each row corresponds to an instance, each column corresponds to a decision value.
        """
        if self.__results is None:
            raise ValueError("No instance evaluated.")
        if self.__contexts is None:
            raise ValueError("No context values.")

        figures = []

        if len(self.__contexts) == 1:
            # No context variables
            # in this case, there is only one figure
            # each row corresponds to the eu of each decision under that instance
            data = {}
            for i, result in enumerate(self.__results):
                data[f"Instance {self.__instance_ids[i]}"] = list(result.values())[0]
            df = pd.DataFrame(data).T
            # order the rows by the instance ids
            df = df.sort_index(key=lambda x: [int(idx.split()[-1].replace('cpd', '')) for idx in x])
            fig, ax = plt.subplots(figsize=(12, 12))
            sns.heatmap(df, 
                       cmap="RdYlBu",
                       annot=True,
                       fmt='.3f',
                       cbar_kws={'label': 'Expected Utility'},
                       ax=ax)
            
            #ax.set_title('Expected Utility Heatmap Across Instances')
            ax.set_xlabel('Decisions')
            ax.set_ylabel('Instances')
            figures.append(fig)
            if plot:
                plt.tight_layout()
                plt.show()
                plt.savefig(f"./output/instance_analyze/{self.instance_name}.pdf")
        else:
            for context in self.__contexts:
                # in this case, there are multiple figures, each corresponding to a context
                # for each figure, the table includes eu of each decision under that context\
                # each row corresponds to a context
                data = {}
                for i, result in enumerate(self.__results):
                    data[f"Instance {self.__instance_ids[i]}"] = result[context]
                
                df = pd.DataFrame(data).T
                df = df.sort_index(key=lambda x: [int(idx.split()[-1].replace('cpd', '')) for idx in x])
                fig, ax = plt.subplots(figsize=(6,6))
                sns.heatmap(df,
                            cmap="RdYlBu",
                            annot=True,
                            fmt='.3f',
                            cbar_kws={'label': 'Expected Utility'},
                            ax=ax)
                
                # ax.set_title(f'Expected Utility Heatmap for Context: {context}')
                ax.set_xlabel('Decisions')
                ax.set_ylabel('Instances')
                figures.append(fig)
                if plot:
                    plt.tight_layout()
                    plt.show()
                    plt.savefig(f"./output/instance_analyze/{self.instance_name}_{context}.pdf")

        return figures

class InstanceAnalyzer(object):
    """ Analyzer for a single instance. This should be called multiple times to evaluate and compare different instances."""
    def __init__(self, pathToInstance: str):
        self.node_list = None
        self.edge_list = None
        self.cpd_list = None # list of cpd params
        self.setInstancePath(pathToInstance)
    
    # instance loading
    def setInstancePath(self, pathToInstance: str):
        """ Set the path to the instance file."""
        if not os.path.exists(pathToInstance):
            raise ValueError(f"Instance folder {pathToInstance} does not exist.")
        self.__pathToInstance = pathToInstance

    def loadInstance(self):
        """ Main method for loading instance from `self.__pathToInstance`.
        
        Returns
        -------
        InfluenceDiagram
            The loaded Influence Diagram.
        """
        node_file_name = os.path.join(self.__pathToInstance, "nodes.json")
        edge_file_name = os.path.join(self.__pathToInstance, "edges.json")
        cpd_file_name = os.path.join(self.__pathToInstance, "cpd.json")

        if not os.path.exists(node_file_name) or not os.path.isfile(node_file_name):
            raise ValueError(f"Nodes file {node_file_name} does not exist.")
        if not os.path.exists(edge_file_name) or not os.path.isfile(edge_file_name):
            raise ValueError(f"Edges file {edge_file_name} does not exist.")
        if not os.path.exists(cpd_file_name) or not os.path.isfile(cpd_file_name):
            raise ValueError(f"CPD file {cpd_file_name} does not exist.")
        
        self.loadNodeList(node_file_name)
        self.loadEdgeList(edge_file_name)
        self.constructGraph()

        self.loadCpdList(cpd_file_name)
        self.constructDiagram()

        return self.getDiagram()

    def loadNodeList(self, node_file_name: str):
        with open(node_file_name, 'r', encoding='utf-8') as f:
            self.node_list = json.load(f)

    def loadEdgeList(self, edge_file_name: str):
        with open(edge_file_name, 'r', encoding='utf-8') as f:
            self.edge_list = json.load(f)
    
    def loadCpdList(self, cpd_file_name: str) -> list[dict[str, str]]:
        """ 
        Load the cpd list to `self.cpd_list`.

        Parameters
        ----------
        cpd_file_name : str
            The path to the cpd file.

        Returns
        -------
        list[dict[str, str]]
            The loaded list of cpd params.
        """
        with open(cpd_file_name, 'r', encoding='utf-8') as f:
            cpd_param_list = json.load(f) # load args for initializing a cpd

        self.cpd_list = cpd_param_list
        return cpd_param_list

    def constructGraph(self):
        """
        Construct an Influence Diagram graph with no cpd.

        Returns
        -------
        InfluenceDiagram
            The constructed Influence Diagram graph.

        Raises
        ------
        ValueError
            If the node list or edge list is not loaded.
        """
        if self.node_list is None:
            raise ValueError("Node list is not loaded.")
        if self.edge_list is None:
            raise ValueError("Edge list is not loaded.")
        self.__diagram = InfluenceDiagram(self.node_list, self.edge_list)
        return self.__diagram

    def constructDiagram(self):
        if self.__diagram is None:
            raise ValueError("Graph is not constructed.")
        if self.cpd_list is None:
            raise ValueError("CPD list is not loaded.")

        self.__diagram.setCpds(self.cpd_list)
        return self.__diagram


    # getter methods
    def getDiagram(self):
        """
        Returns
        -------
        InfluenceDiagram
            The constructed Influence Diagram graph.
        """
        return self.__diagram

    # instance analyzation
    def getOptimalPolicy(self) -> dict:
        """ Get the optimal policy."""
        return self.__diagram.solve()

    def evaluateExpectedUtilities(self, decision: str) -> dict:
        """ 
        Evaluate the expected utility of a **single** decision node.
        For decisions that are based on contexts, the expected utility is computed for all combinations of context values.

        Parameters
        ----------
        decision : str
            The decision node to be evaluated. For example, "d0".

        Returns
        -------
        dict
            A nested dictionary of [context value : [decision value : expected utility]].
        """
        domain = self.__diagram.model.domain[decision]
        utility_nodes = self.__diagram.agent_utilities[self.__diagram.decision_agent[decision]]
        descendant_utility_nodes = list(set(utility_nodes).intersection(nx.descendants(self.__diagram, decision)))
        decision_nodes = self.__diagram.get_valid_order()
        descendant_decision_nodes = [node for node in decision_nodes if node in nx.descendants(self.__diagram, decision)]
        # if there exists a descendent decision node, impute the optimal policy for the descendent decision node
        if len(descendant_decision_nodes) > 0:
            for node in descendant_decision_nodes:
                self.__diagram.impute_optimal_decision(node)

        # Generate all combinations of parent values
        parent_variables = self.__diagram.getDecompositionDict()[self.__diagram._get_label(decision)]
        parents = [self.__diagram.getNodeDict()[variable] for variable in parent_variables]
        all_parent_combinations = list(itertools.product(*[self.__diagram.model.domain[parent] for parent in parents]))

        # compute eu for a given combination of context values
        def compute_expected_utility(parent_values):
            # Use the copy of the model to avoid side effects
            copy = self.__diagram.copy()
            # Compute eu for all decision values in the domain
            eu = {}
            for d in domain:
                parent_values[decision] = d
                eu[d] = sum(copy.expected_value(descendant_utility_nodes, parent_values))
            return eu

        # Evaluate the expected utility
        expected_utilities = {}
        for parent_values in all_parent_combinations:
            parent_values_dict = dict(zip(parents, parent_values))
            expected_utilities[tuple(parent_values)] = compute_expected_utility(parent_values_dict)

        return expected_utilities#, [self.__diagram._get_label(parent)  for parent in parents]
     
    def analyze(self, first_decision: bool):
        """ 
        Main method for instance analyzation.

        Evaluates the eu of all policies, that is,\
            all possible product of decision variable values with context variable values.
        
        Warnings
        --------
        The current implementation considers a single decision node.
        """

        if first_decision:
            decision_node = self.__diagram.get_valid_order()[0]
        else:
            decision_node = self.__diagram.get_valid_order()[-1]
        eu = self.evaluateExpectedUtilities(decision_node)

        return eu

    #visualization
    def heatmapContextEuTable(self, eu: dict):
        """
        Draw the heatmap of the eu table.
        Each row correspond to a context value, each column correspond to a decision value.
        The value of the heatmap is the expected utility.

        Parameters
        ----------
        eu : dict
            A nested dictionary of [context value : [decision value : expected utility]].
        parents : list
            A list of context variable names. This should have one-to-one correspondence with the keys of eu.
        """
        # Convert the nested dictionary to a DataFrame
        # get all unique decision values
        decision_values = set()
        for context_dict in eu.values():
            decision_values.update(context_dict.keys())
        decision_values = sorted(list(decision_values))
        
        # Create a matrix of expected utilities
        context_values = sorted(eu.keys())
        matrix = np.zeros((len(context_values), len(decision_values)))
        
        # Fill the matrix with expected utilities
        for i, context in enumerate(context_values):
            for j, decision in enumerate(decision_values):
                matrix[i][j] = eu[context].get(decision, 0)
        
        # Create DataFrame for better visualization
        df = pd.DataFrame(
            matrix,
            index=context_values,
            columns=decision_values
        )
        
        # Create figure and axis
        plt.figure(figsize=(8, 8))
        
        # Create heatmap
        sns.heatmap(
            df,
            annot=True,  # Show values in cells
            fmt='.2f',   # Format numbers to 2 decimal places
            cmap='RdYlBu',  # Red-Yellow-Blue colormap
            center=0,    # Center the colormap at 0
            cbar_kws={'label': 'Expected Utility'}
        )
        
        plt.title('Expected Utility Heatmap')
        plt.xlabel('Decision Values')
        plt.ylabel('Context Values')
        
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()

if __name__=='__main__':
    #analyzer = InstanceAnalyzer("./data/da-01-longTermCareInsurance/longTermCareInsurance-cpd1")
    #analyzer = InstanceAnalyzer("./data/da-06-stayAtHome/stayAtHome-cpd2")
    #analyzer = InstanceAnalyzer("./data/da-05-chemicalUsage/chemicalUsage-cpd1")
    #analyzer = #InstanceAnalyzer("./data/da-02-carryUmbrella/carryUmbrella-cpd4")
    #analyzer.loadInstance()

    #eu = analyzer.analyze()
    #analyzer.heatmapContextEuTable(eu)
    import json
    with open("./data/dataInfo.json", "r") as f:
        data_info = json.load(f)
    for data in data_info:
        instance_name = "-".join([data["id"], data["name"]])
        analyzer = BatchInstanceAnalyzer(os.path.join("./data", instance_name))
        #analyzer = BatchInstanceAnalyzer("./data/da-02-carryUmbrella")
        analyzer.analyze(first_decision=True)
        #eu_df = analyzer.getInstanceContextEuTable()
        #eu_df.to_csv("./output/eu/da-05-chemicalUsage.csv", index=False, encoding='utf-8')
        analyzer.heatmapInstanceEuTable(plot=True)
