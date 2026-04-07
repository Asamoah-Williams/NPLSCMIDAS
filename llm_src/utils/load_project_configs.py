import os
import yaml
from dotenv import load_dotenv
from pyprojroot import here


class LoadProjectConfigs:
    def __init__(self) -> None:
        with open(here("llm_src/configs/project_config.yml"), encoding="utf-8") as cfg:
            app_cfg = yaml.load(cfg, Loader=yaml.FullLoader)

            # Load langsmith config
        os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
        os.environ["LANGSMITH_TRACING_V2"] = app_cfg["langsmith"]["tracing"]
        os.environ["LANGSMITH_PROJECT"] = app_cfg["langsmith"]["project_name"]
