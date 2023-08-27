from typing import Dict, List, Tuple
from langchain.prompts.example_selector.semantic_similarity import (SemanticSimilarityExampleSelector as Selector,
                                                                    sorted_values)


class SemanticSimilarityExampleSelector(Selector):
    def select_with_score(self, input_variables: Dict[str, str]) -> List[Tuple[dict, float]]:
        """Select which examples to use based on semantic similarity."""
        # Get the docs with the highest similarity.
        if self.input_keys:
            input_variables = {key: input_variables[key] for key in self.input_keys}
        query = " ".join(sorted_values(input_variables))
        example_docs = self.vectorstore.similarity_search_with_score(query, k=self.k)
        # Get the examples from the metadata.
        # This assumes that examples are stored in metadata.
        examples = [(dict(e[0].metadata), e[1]) for e in example_docs]
        # If example keys are provided, filter examples to those keys.
        if self.example_keys:
            examples = [({k: eg[0][k] for k in self.example_keys}, eg[1]) for eg in examples]
        return examples
