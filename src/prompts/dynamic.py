from typing import Any, Dict
from langchain.prompts import PromptTemplate


class DynamicPromptTemplate(PromptTemplate):
    """
    Reimplementation of the base prompt template that will pass the context to the partial variables
    """

    def _merge_partial_and_user_variables(self, **kwargs: Any) -> Dict[str, Any]:
        # Get partial params:
        partial_kwargs = {k: v if isinstance(v, str) else v(**kwargs)
                          for k, v in self.partial_variables.items()}
        return {**partial_kwargs, **kwargs}
