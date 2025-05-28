from langchain_core.language_models import LLM

from prompt import PROMPTS

class SummarizerAgent:
    def __init__(self, language_model: LLM):
        self.language_model = language_model

    def summarize(self, text: str, variable_name: str) -> str:
        """
        Summarize the text, focusing on the information related with variable `variable_name`.

        Parameters
        ----------
        text: str
            The text to summarize.
        variable_name: str
            The name of the variable to focus on.

        Returns
        -------
        str
            The summarized text.
        """
        
        summary_prompt = PROMPTS["summarize_variable"].format(text=text, variable_name=variable_name)

        response = self.language_model.invoke(summary_prompt)

        return response
