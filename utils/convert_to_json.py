import csv
import json
import xml.etree.ElementTree as ET

def convert_to_json(file_path):
    file_extension = file_path.split(".")[-1]
    with open(file_path, "r", encoding="utf-8") as file:
        if file_extension == "csv":
            data = []
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
            return json.dumps(data)
        elif file_extension == "xml":
            tree = ET.parse(file)
            root = tree.getroot()
            data = []
            for item in root:
                item_data = {}
                for child in item:
                    item_data[child.tag] = child.text
                data.append(item_data)
            return json.dumps(data)
        elif file_extension == "txt":
            data = []
            for line in file:
                data.append(line.strip())
            return json.dumps(data)
        else:
            raise Exception("Unsupported file type")
