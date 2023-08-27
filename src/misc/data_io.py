from app import EMBEDDING_DISTANCE_THRESHOLD


def few_shot_examples(example_selector, embedding_distance_threshold, context) -> str:
    similar_examples = [e[0] for e in example_selector.select_with_score({'input': context.get('input')})
                        if e[1] < embedding_distance_threshold]
    return '\n\n'.join([f'User input: {e["input"]}\nGenerated SQL query: {e["sql_cmd"]}' for e in similar_examples])
