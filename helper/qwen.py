from ninept import qwen

def get_cross_validation_technique(prompt):
    qwen_res = qwen(prompt,role='you are a cross validation expert')
    print(prompt)
    return qwen_res