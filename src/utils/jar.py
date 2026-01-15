from datetime import datetime
import os
import pickle
import time
from typing import Any, ClassVar


class Jar:

    base_path: ClassVar[str] = ".jar"

    def __init__(self, prefix: str = "") -> None:
        self.prefix: str = prefix

    def _get_dir_path(self, name: str) -> str:
        name_dir = os.path.dirname(name)
        return os.path.join(self.base_path, self.prefix, name_dir)

    def _get_full_path(self, name: str) -> str:
        dir_path = self._get_dir_path(name)
        timestamp = int(time.time())

        filename = f"{os.path.basename(name)}-{timestamp}.pkl"
        return os.path.join(dir_path, filename)

    def _find_files(self, name: str) -> dict[str, datetime]:
        dir_path = self._get_dir_path(name)
        if not os.path.exists(dir_path):
            return {}

        files = [
            f
            for f in os.listdir(dir_path)
            if f.startswith(os.path.basename(name) + "-") and f.endswith(".pkl")
        ]

        return {
            os.path.join(dir_path, f): datetime.fromtimestamp(int(f.split("-")[-1].split(".")[0]))
            for f in files
        }

    def _get_latest_file(self, name: str) -> str:
        files = self._find_files(name)
        if not files:
            raise KeyError(f"No object named '{name}' found")

        latest_file = max(files.keys(), key=files.get)
        return latest_file

    def add(self, name: str, obj: Any) -> None:
        full_path = self._get_full_path(name)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        with open(full_path, "wb") as f:
            pickle.dump(obj, f)

    def get(self, name: str) -> Any:
        if name.endswith(".pkl"):
            name = name[:-4]
        full_path = self._get_latest_file(name)
        with open(full_path, "rb") as f:
            return pickle.load(f)


    def get_all(self, name: str) -> dict[datetime, Any]:
        files = self._find_files(name)
        results: dict[datetime, Any] = {}

        for f, timestamp in files.items():
            with open(f, "rb") as file:
                results[timestamp] = pickle.load(file)

        return results

    def remove_latest(self, name: str) -> None:
        full_path = self._get_latest_file(name)
        os.remove(full_path)

    def remove_all_but_latest(self, name: str) -> None:
        latest = self._get_latest_file(name)
        files = self._find_files(name)
        for f in files.keys():
            if f != latest:
                os.remove(f)

    def remove_all(self, name: str) -> None:
        files = self._find_files(name)
        for f in files.keys():
            os.remove(f)

    def __contains__(self, name: str) -> bool:
        return len(self._find_files(name)) > 0

    def object_names(self) -> set[str]:
        object_names: set[str] = set()
        base_dir = os.path.join(self.base_path, self.prefix)
        for root, _, files in os.walk(base_dir):
            for file in files:
                if not file.endswith(".pkl"):
                    continue

                relative_path = os.path.relpath(root, base_dir)
                name = "-".join(file.split("-")[:-1])
                object_name = os.path.join(relative_path, name).replace("\\", "/")
                object_names.add(object_name)

        return object_names
