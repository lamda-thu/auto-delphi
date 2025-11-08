import os
import sys
import numpy as np

import pycid.core
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from typing import List, Dict, Optional, Tuple
import json
import pycid
from pycid.core import CID
from pycid.core.cpd import StochasticFunctionCPD, discrete_uniform, TabularCPD
from graph import Node, Edge, NodeList, EdgeList


class InfluenceDiagram(CID):
    """
    InfluenceDiagram class

    Attributes
    ----------
    node_list : NodeList
        List of nodes
    edge_list : EdgeList
        List of edges
    decision_node_dict : Dict[str, str]
        Dict [variable_name: variable_id] for decision nodes
    utility_node_dict : Dict[str, str]
        Dict [variable_name: variable_id] for utility nodes
    chance_node_dict : Dict[str, str]
        Dict [variable_name: variable_id] for chance nodes
    node_dict : Dict[str, str]
        Dict [variable_name: variable_id]
    _edges : List[Tuple[str, str]]
        List of edges
    """
    def __init__(self, node_list: List[dict], edge_list: List[dict], cpd_list: Optional[List[dict]]=None):
        """
        Initialization of an Influence Diagram.
        If `cpd_list` is None, initilaize an Influence Diagram Graph (without CPDs).
        Otherwise, initilaize an Influence Diagram.

        Parameters
        ----------
        node_list : List[dict]
            List of nodes, in string format.
        edge_list : List[dict]
            List of edges, in string format.
        cpd_list : Optional[List[dict]], optional
            List of cpd params, ordered by the valid order of the nodes.
            by default None.
        """
        edge_list_rm_duplicate = self._rmListDuplicate(edge_list)
        node_list_rm_duplicate = self._rmListDuplicate(node_list)
        self.edge_list = self.transformEdgeList(edge_list_rm_duplicate)
        _used_node_name_set = set([cause_name for cause_name, effect_name in self.edge_list.getEdgeNameList()] + [effect_name for cause_name, effect_name in self.edge_list.getEdgeNameList()])
        self.node_list = self.transformNodeList([node for node in node_list_rm_duplicate if node['variable_name'] in _used_node_name_set])

        # self.node_dict [nodeID -> variable_name]
        self.decision_node_dict = {
            decision_node : f"d{i}"
            for i, decision_node in enumerate(self.node_list.getDecisionVariableList())
        }
        self.utility_node_dict = {
           utility_node : f"u{i}" 
            for i, utility_node in enumerate(self.node_list.getUtilityVariableList())
        }
        self.chance_node_dict = {
            chance_node : f"c{i}"
            for i, chance_node in enumerate(self.node_list.getChanceVariableList())
        }
        self.node_dict = {
            **self.decision_node_dict, **self.utility_node_dict, **self.chance_node_dict
        } 

        self._edges = [
            (self.node_dict[cause_name], self.node_dict[effect_name]) 
            for cause_name, effect_name in self.edge_list.getEdgeNameList()
        ]

        # initialize cid with variable_id
        super().__init__(
            edges=self._edges, 
            decisions=[decision for decision in self.decision_node_dict.values()], 
            utilities=[utility for utility in self.utility_node_dict.values()]
        )
        # construct joint distribution decomposition
        self.decomposition_dict = self.constructDecompositionDict(self.edge_list)

        # instantiate domains
        # Node: set cpd according to a valid order of the nodes
        for node in self.get_valid_order([node for node in self.node_dict.values()]):
            # set domain for the decision variables
            variable_name = self._get_label(node)
            if node in self.decision_node_dict.values():
                self.model.domain[node] = self.node_list.getNode(variable_name).getVariableValues()
                if len(self.model.domain[node]) > 0:
                    self.model[node] = discrete_uniform(self.model.domain[node])
                else:
                    self.model[node] = discrete_uniform(["dummy_value"])
            else:
                try:
                    self.model.domain[node] = self.node_list.getNode(variable_name).getVariableValues()
                    if len(self.model.domain[node]) > 0:
                        self.model[node] = discrete_uniform(self.model.domain[node])
                    else:
                        self.model[node] = discrete_uniform(["dummy_value"])
                except:
                    self.model[node] = discrete_uniform(["dummy_value"])

        if cpd_list is not None:
            # check if each cpd_list element variable name is ordered
            self.setCpds(cpd_list)

    def _rmListDuplicate(self, input: list) -> list:
        processed_list = []
        for element in input:
            if element not in processed_list:
                processed_list.append(element)
        return processed_list

    def __str__(self) -> str:
        """ A string representation of the object. 
        
        Warnings
        --------
        The string only includes descriptions about the **graph**.
        CPDs are not included.
        """
        string = f"{super().__str__()}\nedges: {self.edge_list.getEdgeNameList()},\ndecision nodes: {[decision for decision in self.decision_node_dict.keys()]},\nutility nodes: {[utility for utility in self.utility_node_dict.keys()]},\nchance nodes: {[chance for chance in self.chance_node_dict.keys()]}"
        return string#super().__str__()
    
    # graph construction methpds
    # transform from Dict/JSON format to NodeList/EdgeList class instances     
    def transformNodeList(self, node_list: List[Dict[str, str]]):
        transformed_node_list = []
        for node in node_list:
            # if variable_values is not provided, set it to an empty list
            if node.get("variable_values", None) is None:
                node["variable_values"] = []
            elif not isinstance(node.get("variable_values"), list):
                node["variable_values"] = [node.get("variable_values")]
            if node.get("variable_type", None) is None:
                node["variable_type"] = "chance"
            transformed_node_list.append(
                Node.model_validate(node)
            )
        return NodeList(transformed_node_list)
    
    def transformEdgeList(self, edge_list: List[Dict[str, str]]):
        transformed_edge_list = []
        for edge in edge_list:
            transformed_edge_list.append(
                Edge.model_validate(edge)
            )
        return EdgeList(transformed_edge_list)
    
    def constructDecompositionDict(self, edge_list: EdgeList) -> Dict[str, List[str]]:
        """ return the list of variables and their parent variables. Preparation for CPD assignment
        """
        decomposition_dict = dict()
        for cause_name, effect_name in edge_list.getEdgeNameList():
            if effect_name not in decomposition_dict.keys():
                decomposition_dict[effect_name] = [cause_name]
            else:
                decomposition_dict[effect_name].append(cause_name)

            if cause_name not in decomposition_dict.keys():
                decomposition_dict[cause_name] = []
        return decomposition_dict


    # cpd construction methods
    def checkVariableValidOrder(self, variable_name_list: List[str]):
        """ 
        Check if the input `variable_name_list` is in a valid order.

        Raises
        ------
        ValueError
            If the input `variable_name_list` is not in a valid order.
        """
        valid_variable_name_list = [self._get_label(node) for node in self.get_valid_order([node for node in self.node_dict.values()])]
        print(valid_variable_name_list)
        # variable_name_list could contain fewer nodes
        # but the order has to be preserved, valid.
        valid_variable_name_list = [variable_name for variable_name in valid_variable_name_list if variable_name in variable_name_list]
        print(valid_variable_name_list)
        if valid_variable_name_list != variable_name_list:
            raise ValueError(f"invalid variable order:\n{variable_name_list}")

    def setCpdParamListValidOrder(self, cpd_param_list: List[dict]):
        """
        Order the elements according to (1) the ``variable`` name (2) and the valid order
            according to the diagram.

        Returns
        -------
        cpd_param_list : List[dict]
            The ordered list of cpd params.

        Warnings
        --------
        The cpd params are presented in ordered dictionaries.
        We require the first key to be "variable".
        """
        valid_variable_name_list = [self._get_label(node) for node in self.get_valid_order([node for node in self.node_dict.values()])]
        order_map = {
            variable_name: idx
            for idx, variable_name in enumerate(valid_variable_name_list)
        }
        # cpd_param_list = sorted(cpd_param_list, key=lambda cpd_param: order_map[cpd_param.values().__iter__().__next__()])
        cpd_param_list = sorted(cpd_param_list, key=lambda cpd_param: order_map[cpd_param.get("variable", None)])
        return cpd_param_list

    def constructStochasticFunctionCPD(self, stochasticFunctionCPD_params: Dict[str, str]) -> StochasticFunctionCPD:
        """ 
        Instantiate a StochasticFunctionCPD object.

        Parameters
        ----------
        stochasticFunctionCPD_params : dict
            A dict of cpd params.
            The first key has to be 'variable'.
        """
        stochastic_function = stochasticFunctionCPD_params['stochastic_function']
        
        if stochasticFunctionCPD_params.get('evidence', None) is not None:
            evidence = stochasticFunctionCPD_params['evidence']
            for evidence_variable in sorted(evidence, key=len, reverse=True):
                # BUG: if some variable name is a sub string in another name, the sub string will be replaced by id
                stochastic_function = stochastic_function.replace(evidence_variable.replace(" ", "_"), self.node_dict.get(evidence_variable.replace("_", " "),self.node_dict.get(evidence_variable,"")))

        cpd = StochasticFunctionCPD(
            variable = self.node_dict[stochasticFunctionCPD_params['variable']],
            stochastic_function = eval(stochastic_function),
            cbn = self
        )
        return cpd

    def addStochasticFunctionCPD(self, stochasticFunctionCPD_params: Dict[str, str]):
        """ 
        Add a cpd to the diagram specified by `stochasticFunctionCPD_params`.

        Parameters
        ----------
        stochasticFunctionCPD_params : dict
            A dict of cpd params.
            The first key has to be 'variable'.

        Returns
        -------
        cpd : StochasticFunctionCPD
            The added cpd object.
        """        
        cpd = self.constructStochasticFunctionCPD(stochasticFunctionCPD_params)
        self.add_cpds(cpd)
        return cpd

    def setCpds(self, cpd_param_list: List[dict]):
        cpd_param_list = self.setCpdParamListValidOrder(cpd_param_list)

        for cpd_param in cpd_param_list:
            self.addStochasticFunctionCPD(cpd_param)

    # getter methods
    def getDecompositionDict(self) -> Dict[str, List[str]]:
        return self.decomposition_dict
    
    def getVariableID(self, variable_name: str)->str:
        return self.node_dict[variable_name]
    
    def getDecisionNodeDict(self)->Dict[str, str]:
        return self.decision_node_dict

    def getNodeDict(self)->Dict[str, str]:
        return self.node_dict
    
    def _get_label(self, node: str):
        for variable_name in self.node_dict.keys():
            if self.node_dict[variable_name] == node:
                return variable_name
        return ""

    def getNodes(self)->List[dict]:
        __nodes = self.node_list.getNodeList()
        assert len(__nodes) > 0
        return __nodes
    
    def getEdgeNameList(self)->List[dict]:
        __edges = self.edge_list.getEdgeNameList()
        assert len(__edges) > 0
        return __edges

    def getDecisions(self)->List[dict]:
        """ 
        Get variable names of decision variables.
        The variables are in the valid order.
        """
        #__decisions = self.node_list.getDecisionNodeList() #HYF, 20241229: fix variable order
        # Notice: by default `get_valid_order` will return decision nodes only.
        decision_variable_names = [self._get_label(node) for node in self.get_valid_order()]
        assert len(decision_variable_names) > 0
        return [self.node_list.getNode(variable).getNode() for variable in decision_variable_names]

    def getVariableType(self, variable_name: str)->str:
        if variable_name in self.utility_node_dict.keys():
            return "utility"
        elif variable_name in self.decision_node_dict.keys():
            return "decision"
        else:
            return "chance"
        
    def get_random_cpds(self, n_states=None, inplace=False, seed=None):
        """
        Given a `model`, generates and adds random `TabularCPD` for each node resulting in a fully parameterized network.

        Parameters
        ----------
        n_states: int or dict (default: None)
            The number of states of each variable in the `model`. If None, randomly
            generates the number of states.

        inplace: bool (default: False)
            If inplace=True, adds the generated TabularCPDs to `model` itself, else creates
            a copy of the model.

        seed: int (default: None)
            The seed value for random number generators.

        """
        if isinstance(n_states, int):
            n_states = {var: n_states for var in self.nodes()}
        elif isinstance(n_states, dict):
            if set(n_states.keys()) != set(self.nodes()):
                raise ValueError("Number of states not specified for each variable")
        elif n_states is None:
            gen = np.random.default_rng(seed=seed)
            n_states = {
                var: gen.integers(low=1, high=5, size=1)[0] for var in self.nodes()
            }

        cpds = []
        for node in self.nodes():
            parents = list(self.predecessors(node))
            cpds.append(
                TabularCPD.get_random(
                    variable=node, evidence=parents, cardinality=self.get_cardinality(), state_names=n_states
                )
            )

        if inplace:
            self.add_cpds(*cpds)
        else:
            return cpds

    def drawCID(self, save_path: str):
        """ 
        Draw the Influence Diagram with node names instead of variable IDs and save the plot to a specified path.
        
        Parameters
        ----------
        save_path : str
            The path where the plot should be saved.
        """
        import matplotlib.pyplot as plt
        import networkx as nx
        
        # Define custom node label function that uses node_dict to get original variable names
        def custom_node_label(node):
            # Look up the variable name in node_dict that corresponds to the node ID
            for variable_name, variable_id in self.node_dict.items():
                if variable_id == node:
                    return variable_name
            return ""  # Fallback if not found
        
        # Use the built-in color function from CID
        layout = nx.kamada_kawai_layout(self)
        
        # Create label dictionary using our custom function
        label_dict = {node: custom_node_label(node) for node in self.nodes}
        
        # Draw the network
        plt.figure(figsize=(10, 8))
        
        # Draw edges
        nx.draw_networkx_edges(self, pos=layout, arrowsize=20)
        
        # Draw nodes with appropriate colors based on type but without labels on nodes
        for node in self.nodes:
            nx.draw_networkx(
                self.to_directed().subgraph([node]),
                pos=layout,
                node_size=800,
                arrowsize=20,
                node_color=self._get_color(node),
                node_shape=self._get_shape(node),
                with_labels=False  # Don't draw labels on nodes
            )
        
        # Draw the labels separately, positioned adjacent to nodes
        pos_labels = {}
        for k, v in layout.items():
            pos_labels[k] = (v[0], v[1] + 0.15)  # Position labels above nodes
            
        nx.draw_networkx_labels(self, pos_labels, label_dict, font_size=10)
        
        # Save the figure to the specified path with high resolution
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    def copy_ID_without_cpds(self):
        return InfluenceDiagram(self.node_list.getNodeList(), self.edge_list.getEdgeList(), cpd_list=None)


if __name__=='__main__':
    test = [{'variable_name': 'Fire_Separation_Measures',
         'variable_type': 'decision',
        'values': ['Implement', 'Not_Implement']},
        {'variable_name': 'Fire_Spread',
        'variable_type': 'chance',
         'values': ['Rapid_Upward', 'Contained']},
        {'variable_name': 'Fire_Integrity_Compromise',
        'variable_type': 'chance',
         'values': ['Compromised', 'Intact']},
        {'variable_name': 'Utility',
         'variable_type': 'utility',
         'values': ['Prevent_Rapid_Upward_Spread']}]
    a = Node(test[0]["variable_name"], test[0]["variable_type"], test[0]["values"])    
    b = Node(test[1]["variable_name"], test[1]["variable_type"], test[1]["values"])
    ab = Edge(a.getVariableName(), b.getVariableName())

    #cid = InfluenceDiagram(NodeList([a, b]), EdgeList([ab]))
    #cid.draw()
    cid = pycid.CID([("dummy variable", "X"), ("X", "D"), ("X", "U"), ("D", "U")], decisions=["D"], utilities=["U"])
    #cid = pycid.CID([('D0', 'C0')], decisions=['D0'],utilities=[])
    cid.model.update(
        X=pycid.discrete_uniform([0, 1]),  # A uniform random CPD over its domain [0,1]
        U=lambda X, D: int(X == D),  # specifies how 'U''s CPD functionally depends on its parents ('D' and 'S').
        D=[0, 1],
    ) 
    cid.model["dummy variable"] = pycid.discrete_uniform(["a","b"])
    #print(cid.solve())
    cid.model["X"] = cid.model["dummy variable"]
    cid.remove_node("dummy variable")
    cid.draw()    
    #cid.impute_optimal_policy()
    #print(
    #    cid.expected_utility(
    #        context={}, intervention={}
    #    )
    #)