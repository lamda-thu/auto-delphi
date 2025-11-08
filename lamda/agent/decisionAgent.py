import os
import sys
from collections import Counter

from typing import List, Dict, Optional

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from prompt import PROMPTS

from models import Gpt4Turbo

class DecisionResponse(BaseModel):
    decision: str = Field(description="The alternative chosen for the decision. The alternative value should strictly follow the list of alternatives.")
    explanation: str = Field(description="A string that describes, in detail, the reasoning behind your decision.")

class DecisionAgent:
    def __init__(
        self,
        language_model,
        max_retries: int = 3
    ):
        self.language_model = language_model
        self.max_retries = max_retries
    
    def getLanguageModel(self):
        return self.language_model

    def _majorityVode(self, decisions: List[str]) -> str:
        """
        Return the majority vote decision.
        """
        vote_counts = Counter(decisions)
        return max(vote_counts, key=vote_counts.get)

    def generateSingleDecisionVanilla(self, text: str, decision_alternatives: Dict[str, List[str]]) -> Dict[str, str]:
        print(f"AWAITING vanilla-{self.language_model.__class__.__name__} DECISION...")

        decision_prompt_template = PROMPTS["decision_vanilla"]

        parser = JsonOutputParser(pydantic_object=DecisionResponse)

        prompt = PromptTemplate(
            template=decision_prompt_template,
            input_variables=["text", "information", "decision_alternatives"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        chain = prompt | self.language_model | parser
        
        for retry in range(self.max_retries):
            try:
                contents = chain.invoke({"text": text, "decision_alternatives": decision_alternatives})
                return {
                    decision_variable: contents['decision']\
                    for decision_variable in decision_alternatives.keys()
                }
            except Exception as e:
                print(f"Error on attempt {retry+1}/{self.max_retries}: {e}")
                if retry == self.max_retries - 1:
                    return {
                        decision_variable: None \
                        for decision_variable in decision_alternatives.keys()
                    }
    
    def generateSingleDecisionCot(self, text: str, decision_alternatives: Dict[str, List[str]], verbose: bool) -> Dict[str, str]:
        print(f"AWAITING COT-{self.language_model.__class__.__name__} DECISION...")

        decision_prompt_template = PROMPTS["decision_cot"]

        parser = JsonOutputParser(pydantic_object=DecisionResponse)

        prompt = PromptTemplate(
            template=decision_prompt_template,
            input_variables=["text", "information", "decision_alternatives"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        chain = prompt | self.language_model | parser
        
        for retry in range(self.max_retries):
            try:
                contents = chain.invoke({"text": text, "decision_alternatives": decision_alternatives})
                if verbose:
                    print(contents)
                return {
                    decision_variable: contents['decision']\
                    for decision_variable in decision_alternatives.keys()
                }
            except Exception as e:
                print(f"Error on attempt {retry+1}/{self.max_retries}: {e}")
                if retry == self.max_retries - 1:
                    return {
                        decision_variable: None \
                        for decision_variable in decision_alternatives.keys()
                    }

    def generateSingleDecisionSc(self, text: str, decision_alternatives: Dict[str, List[str]], K: int) -> Dict[str, str]:
        """
        Sample `K` decisions and make the final decision based on the majority vote.

        Warnings
        --------
        The temperature of the language model should be set to a higher value.

        Warnings
        --------
        The current check only works for single-choice decisions.
        """
        print(f"AWAITING SC-{self.language_model.__class__.__name__} DECISION...")

        decision_votes = []
        decision_prompt_template = PROMPTS["decision_vanilla"]
        parser = JsonOutputParser(pydantic_object=DecisionResponse)
        prompt = PromptTemplate(
            template=decision_prompt_template,
            input_variables=["text", "decision_alternatives"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        chain = prompt | self.language_model | parser
        
        for attempt in range(K):
            for retry in range(self.max_retries):
                try:
                    contents = chain.invoke({"text": text, "decision_alternatives": decision_alternatives})
                    decision_votes.append(contents['decision'])
                    break
                except Exception as e:
                    print(f"Error on attempt {retry+1}/{self.max_retries} for sample {attempt+1}/{K}: {e}")
                    if retry == self.max_retries - 1:
                        print(f"Failed to get decision for sample {attempt+1}/{K} after {self.max_retries} retries")
        
        if not decision_votes:
            return {
                decision_variable: None \
                for decision_variable in decision_alternatives.keys()
            }
        
        return {
            decision_variable: self._majorityVode(decision_votes)\
            for decision_variable in decision_alternatives.keys()
        }
