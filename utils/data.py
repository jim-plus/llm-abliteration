import json
import pandas


def load_data(path: str) -> list[str]:
    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            return f.readlines()
    elif path.endswith(".parquet"):
        df = pandas.read_parquet(path)
        data = df.get("text")
        if data is None:
            raise ValueError("No 'text' column found in parquet file")
        return data.tolist()
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON data should be a list")
        return data
    elif path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line.strip())["text"] for line in f if line.strip()]
    else:
        raise ValueError("Unsupported file format")
