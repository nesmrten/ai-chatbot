import json


class DataFeeder:
    def __init__(self, file_path="data/data.json"):
        self.file_path = file_path

    def add_data(self, intent, response):
        data = self.read_file()
        if intent not in data:
            data[intent] = []
        data[intent].append(response)
        self.write_file(data)

    def get_data(self, intent):
        data = self.read_file()
        if intent in data:
            return data[intent]
        else:
            return []

    def read_file(self):
        with open(self.file_path, "r") as f:
            return json.load(f)

    def write_file(self, data):
        with open(self.file_path, "w") as f:
            json.dump(data, f, indent=4)
