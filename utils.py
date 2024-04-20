import json


def write_dict_data_to_file(file_path, data, indent=2):
    with open(file_path, "w", encoding='utf-8') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=indent)