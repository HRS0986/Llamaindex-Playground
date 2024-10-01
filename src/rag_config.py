import os
import re
import yaml


class RagConfig:
    def __init__(self, config_file="./src/config.yaml", config_file_type="yaml"):
        self.config = self.__load_config(config_file, config_file_type)

    def __load_config(self, config_file: str, config_file_type: str) -> dict:
        if config_file_type == "yaml":
            yaml.add_implicit_resolver("!env", re.compile(r"\$[A-Z_]+"))
            yaml.add_constructor("!env", self.__env_constructor)
            with open(config_file, "r") as f:
                return yaml.load(f, Loader=yaml.FullLoader)
        else:
            raise ValueError("Invalid config file type")

    @property
    def vector_store(self) -> dict:
        return self.config["vector_store"]

    @staticmethod
    def __env_constructor(loader, node):
        value = loader.construct_scalar(node)
        return os.getenv(value[1:], value)
