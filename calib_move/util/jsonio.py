import json


def str_2_json(file_path: str, data: str):
    with open(file_path, mode="w", encoding="utf-8") as file:
        file.write(data)

def json_2_dict(file_path: str):
    with open(file_path, mode="r", encoding="utf-8") as file:
        data = json.load(file)
    return data