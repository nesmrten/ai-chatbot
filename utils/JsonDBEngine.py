import json
import os
import gzip
import hashlib
import threading


class JsonDbEngine:
    def __init__(self, db_path):
        self.db_path = db_path
        self.indexes = {}
        self.cache = {}
        self.locks = {}

    def insert(self, collection_name, document):
        collection_path = os.path.join(
            self.db_path, f"{collection_name}.json.gz")
        if not os.path.exists(collection_path):
            with gzip.open(collection_path, "wt") as f:
                f.write(json.dumps([]))

        with gzip.open(collection_path, "rt") as f:
            collection = json.load(f)

        collection.append(document)

        with gzip.open(collection_path, "wt") as f:
            f.write(json.dumps(collection))

        self.invalidate_cache(collection_name)

    def find(self, collection_name, query=None, sort_by=None):
        if query is None:
            query = {}

        if sort_by is None:
            sort_by = []

        collection_path = os.path.join(
            self.db_path, f"{collection_name}.json.gz")
        if not os.path.exists(collection_path):
            return []

        cache_key = f"{collection_name}:{json.dumps(query)}:{json.dumps(sort_by)}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        with gzip.open(collection_path, "rt") as f:
            collection = json.load(f)

        # apply indexes
        for field, index in self.indexes.get(collection_name, {}).items():
            if field in query:
                if query[field] in index:
                    collection = [
                        doc for doc in collection if doc[field] == query[field]]
                else:
                    return []

        # apply query
        for field, value in query.items():
            if field not in self.indexes.get(collection_name, {}):
                collection = [
                    doc for doc in collection if doc.get(field) == value]

        # apply sorting
        for sort_field in reversed(sort_by):
            collection = sorted(collection, key=lambda x: x.get(
                sort_field.lstrip("-"), ""))

        self.cache[cache_key] = collection

        return collection

    def create_index(self, collection_name, field):
        if collection_name not in self.indexes:
            self.indexes[collection_name] = {}

        if field not in self.indexes[collection_name]:
            self.indexes[collection_name][field] = set()

            collection_path = os.path.join(
                self.db_path, f"{collection_name}.json.gz")
            if os.path.exists(collection_path):
                with gzip.open(collection_path, "rt") as f:
                    collection = json.load(f)

                for doc in collection:
                    value = doc.get(field)
                    if value is None:
                        # add value to index
                        self.indexes[collection_name][field].add(value)

    def invalidate_cache(self, collection_name):
        keys_to_delete = []
        for key in self.cache.keys():
            if key.startswith(collection_name):
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del self.cache[key]

    def create_lock(self, collection_name):
        if collection_name not in self.locks:
            self.locks[collection_name] = threading.Lock()

    def acquire_lock(self, collection_name):
        if collection_name in self.locks:
            self.locks[collection_name].acquire()

    def release_lock(self, collection_name):
        if collection_name in self.locks:
            self.locks[collection_name].release()

    def get_collection_size(self, collection_name):
        collection_path = os.path.join(
            self.db_path, f"{collection_name}.json.gz")
        if not os.path.exists(collection_path):
            return 0
        with gzip.open(collection_path, "rt") as f:
            collection = json.load(f)
        return len(collection)

    def compress_collection(self, collection_name):
        collection_path = os.path.join(
            self.db_path, f"{collection_name}.json.gz")
        uncompressed_path = os.path.join(
            self.db_path, f"{collection_name}.json")
        if os.path.exists(collection_path):
            with open(uncompressed_path, "wb") as f_uncompressed:
                with gzip.open(collection_path, "rb") as f_compressed:
                    f_uncompressed.write(f_compressed.read())
            os.remove(collection_path)
        with open(uncompressed_path, "rb") as f_uncompressed:
            with gzip.open(collection_path, "wb") as f_compressed:
                f_compressed.write(f_uncompressed.read())
        os.remove(uncompressed_path)

    def shard_collection(self, collection_name, num_shards):
        collection_path = os.path.join(
            self.db_path, f"{collection_name}.json.gz")
        if not os.path.exists(collection_path):
            return

        shards = [os.path.join(
            self.db_path, f"{collection_name}_shard_{i}.json.gz") for i in range(num_shards)]
        for shard in shards:
            with gzip.open(shard, "wt") as f:
                f.write(json.dumps([]))

        with gzip.open(collection_path, "rt") as f:
            collection = json.load(f)

        for doc in collection:
            shard_index = int(hashlib.md5(json.dumps(
                doc).encode()).hexdigest(), 16) % num_shards
            with gzip.open(shards[shard_index], "rt") as f:
                shard_collection = json.load(f)
            shard_collection.append(doc)
            with gzip.open(shards[shard_index], "wt") as f:
                f.write(json.dumps(shard_collection))

        os.remove(collection_path)
        self.indexes[collection_name] = {}

    def run_transaction(self, collection_name, transaction):
        self.create_lock(collection_name)
        self.acquire_lock(collection_name)

        try:
            collection_path = os.path.join(
                self.db_path, f"{collection_name}.json.gz")
            with gzip.open(collection_path, "rt") as f:
                collection = json.load(f)

            transaction(collection)

            with gzip.open(collection_path, "wt") as f:
                f.write(json.dumps(collection))
        finally:
            self.release_lock(collection_name)

    def update(self, collection_name, query=None, update=None):
        if query is None:
            query = {}

        if update is None:
            update = {}

        collection_path = os.path.join(
            self.db_path, f"{collection_name}.json.gz")
        if not os.path.exists(collection_path):
            return

        with gzip.open(collection_path, "rt") as f:
            collection = json.load(f)

        updated_docs = []

        # apply query
        for doc in collection:
            matches_query = True
            for field, value in query.items():
                if doc.get(field) != value:
                    matches_query = False
                    break
            if matches_query:
                updated_doc = doc.copy()
                for field, value in update.items():
                    updated_doc[field] = value
                updated_docs.append(updated_doc)

        # update collection
        for i, doc in enumerate(collection):
            for updated_doc in updated_docs:
                if doc == updated_doc:
                    collection[i] = updated_doc
                    break

        with gzip.open(collection_path, "wt") as f:
            f.write(json.dumps(collection))

        self.invalidate_cache(collection_name)

    def compress_db(self):
        for filename in os.listdir(self.db_path):
            if filename.endswith(".json"):
                collection_name = filename[:-5]
                self.compress_collection(collection_name)

    def shard_db(self, num_shards):
        for filename in os.listdir(self.db_path):
            if filename.endswith(".json.gz"):
                collection_name = filename[:-8]
                self.shard_collection(collection_name, num_shards)
