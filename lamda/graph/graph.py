import uuid
import os
import json
from enum import Enum
from typing import List, Dict
from pydantic import BaseModel, Field, PrivateAttr


class VariableType(Enum):
    decision = "decision"
    chance = "chance"
    utility = "utility"


class Node(BaseModel):
    variable_name: str = Field(
        description="The name for the variable (node) relevant for decision-making. It should be a word or phrase that is concise, understandable and unambiguous."
        )
    variable_type: VariableType = Field(
        description="The type of the variable in the influence diagram. There are three types of nodes: utility, decision, chance. Variable of type 'utility' reflects the prefernce or goals of the decision-maker. Variable of type 'decision' represent a decision to be made. Variable of type 'chance' are random and cannot be directly intervened on."
        )
    variable_values: list = Field(
        description="The possible values of the variable, at least two values are needed. Complete with common knowledge if necessary. The values should reflect absolute quantity instead of the aplitude of change. For example, 'more severe' is not a valid value, while 'severe' and 'extremely severe' will be more suitable. The cardinality of the list of values should be not less than two."
    )
    __id: uuid.UUID = PrivateAttr(default_factory=uuid.uuid1)

    def __init__(self, variable_name, variable_type, variable_values):
        super().__init__(variable_name=variable_name,
                         variable_type=variable_type, variable_values=variable_values)
    
    def __hash__(self):
        return self.__id

    def get_id(self):
        return str(self.__id)
    
    def getNode(self):
        return {
            "variable_name": self.variable_name,
            "variable_type": self.variable_type.name,
            "variable_values": self.variable_values
        }

    def getVariableName(self):
        return self.variable_name
    
    def getVariableType(self):
        return self.variable_type.name
    
    def getVariableValues(self):
        return self.variable_values
    

class NodeList(BaseModel):
    node_list: List[Node] = Field(
        "A python list containing Node class objects."
    )
    def __init__(self, node_list):
        super().__init__(node_list=node_list)

    def __len__(self):
        return len(self.node_list)
    
    def __getitem__(self, index):
        return self.node_list[index]

    def getNode(self, variable_name):
        """ find a node in the node list with exact match of variable_name. """
        for node in self.node_list:
            if node.getVariableName() == variable_name:
                return node
        return None
    
    def getNodeList(self):
        return [node.getNode() for node in self.node_list]
    
    def getDecisionNodeList(self):
        """ Get complete dict of decision variables."""
        return [node.getNode() for node in self.node_list if node.getVariableType() == "decision"]

    def getDecisionVariableList(self):
        """ Get variable names of decision variables."""
        return [node.getVariableName() for node in self.node_list if node.getVariableType() == "decision"]
    
    def getUtilityVariableList(self):
        return [node.getVariableName() for node in self.node_list if node.getVariableType() == "utility"]   
     
    def getChanceVariableList(self):
        return [node.getVariableName() for node in self.node_list if node.getVariableType() == "chance"]

    def save(self, output_dir):
        node_file_name = os.path.join(output_dir, "nodes.json")
        with open(node_file_name, 'w', encoding='utf-8') as f:
            json.dump(self.getNodeList(), f, indent=4, ensure_ascii=False)

    def load(self, input_dir):
        node_file_name = os.path.join(input_dir, "nodes.json")
        with open(node_file_name, 'r', encoding='utf-8') as f:
            json_node_list = json.load(f)
            self.node_list = []
            for node in json_node_list:
                try:
                    # Check if variable_values is a string and convert it to a list if needed
                    if isinstance(node["variable_values"], str):
                        # Use a default list if the value is just a string description
                        print(f"Warning: Converting string variable_values to list for {node['variable_name']}: {node['variable_values']}")

                        node["variable_values"] = []
                        
                    self.node_list.append(Node(node.get("variable_name"), node.get("variable_type"), node.get("variable_values")))
                except Exception as e:
                    print(f"Error with node {node.get('variable_name', 'unknown')}: {str(e)}")
                    print("Skipping this node.")


class Edge(BaseModel): #TODO: check consistency with node names
    cause: str = Field(
        description="The name (variable_name) of the cause, which is the variable that influences the variable of interest, either through information (to decision nodes) or probabilistic influence (to chance nodes or utility nodes)."
    )
    effect: str = Field(
        description="The name (variable_name) of the effect, which is the variable of interest. If the effect variable is of type 'chance' or 'utility', the distribution of the effect is (probabilisticly) dependent on the cause, which may be conceptually expressed as P(effect | cause). If the effect variable is of type 'decision', it means the cause variable is known when the corresponding decision is made."
    )
    __id: uuid.UUID = PrivateAttr(default_factory=uuid.uuid1)

    def __init__(self, cause, effect):
        super().__init__(cause=cause, effect=effect)

    def __hash__(self):
        return self.__id
    
    def get_id(self):
        return str(self.__id)
    
    def getCause(self):
        return self.cause
    
    def getEffect(self):
        return self.effect
    
    def getEdge(self):
        return {
            "cause": self.cause,
            "effect": self.effect
        }

class EdgeList(BaseModel):
    edge_list: List[Edge] = Field(
        "A python list containing Edge class objects."
    )

    def __init__(self, edge_list):
        super().__init__(edge_list=edge_list)

    def __len__(self):
        return len(self.edge_list)
    
    def __getitem__(self, index):
        return self.edge_list[index]

    def findEdge(self, cause, effect):
        for edge in self.edge_list:
            if edge.getCause() == cause and edge.getEffect() == effect:
                return edge
        return None

    def getEdgeNameList(self):
        """
        return a list of tuples, each tuple contains the cause and effect of an edge.
        
        Warning
        -------
        This will be deprecated in the future. 
        Use getEdgeList instead.        
        """
        return [(edge.getCause(), edge.getEffect()) for edge in self.edge_list]
    
    def getEdgeList(self):
        return [edge.getEdge() for edge in self.edge_list]
    
    def save(self, output_dir):
        edge_file_name = os.path.join(output_dir, "edges.json")
        with open(edge_file_name, 'w', encoding='utf-8') as f:
            json.dump(self.getEdgeList(), f, indent=4, ensure_ascii=False)
    
    def load(self, input_dir):
        edge_file_name = os.path.join(input_dir, "edges.json")
        with open(edge_file_name, 'r', encoding='utf-8') as f:
            json_edge_list = json.load(f)
            self.edge_list = []
            for edge in json_edge_list:
                if edge.get("cause") is None or edge.get("effect") is None:
                    print(f"Skipping edge {edge} due to missing cause or effect")
                    continue
                self.edge_list.append(Edge(edge["cause"], edge["effect"]))

class NodeAndEdgeList(BaseModel):
    node_list: List[Node] = Field(description="List of nodes with variable_name, variable_type, and variable_values")
    edge_list: List[Edge] = Field(description="List of edges with cause and effect relationships")

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
    print(NodeList([a, b]).getNode(a.getVariableName()).getVariableType())