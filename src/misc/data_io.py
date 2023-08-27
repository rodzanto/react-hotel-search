from typing import Any, Dict
from prompts.example_selector.semantic_similarity import SemanticSimilarityExampleSelector


def few_shot_examples(example_selector: SemanticSimilarityExampleSelector,
                      embedding_distance_threshold: float,
                      context: Dict[str, Any]) -> str:
    """
    Few-shot example selector based on cosine distance between embeddings

    Parameters
    ----------
    example_selector : Configured selector object to use when searching for relevant examples
    embedding_distance_threshold : Distance threshold. Only examples whose distance to the given query is
                                   below the given threshold will be returned.
    context : User's query context

    Returns
    -------
    Concatenated examples
    """
    similar_examples = [e[0] for e in example_selector.select_with_score({'input': context.get('input')})
                        if e[1] < embedding_distance_threshold]
    return '\n\n'.join([f'User input: {e["input"]}\nGenerated SQL query: {e["sql_cmd"]}' for e in similar_examples])
