import yaml
from pyprojroot import here
import os
from dotenv import load_dotenv

class LoadToolsConfig:
    def __init__(self):
        load_dotenv()

        with open(here("llm_src/configs/tools_config.yml"), encoding="utf-8") as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)

        # set environment variables
        os.environ['OPENAI_API_KEY'] = os.getenv("OPEN_AI_API_KEY")
        os.environ["CHROMA_API_KEY"] = os.getenv("CHROMA_API_KEY")
        os.environ["CHROMA_TENANT"] = os.getenv("CHROMA_TENANT")
        os.environ["CHROMA_DATABASE"] = os.getenv("CHROMA_DATABASE")
        os.environ["SQLDB_UID"] = os.getenv("SQLDB_UID")
        os.environ["SQLDB_PASSWORD"] = os.getenv("SQLDB_PASSWORD")

        #Postgres
        self.postgres_host = app_config["postgres_db_config"]["host"]
        self.postgres_database = app_config["postgres_db_config"]["database"]
        self.postgres_user = app_config["postgres_db_config"]["user"]
        self.postgres_password = app_config["postgres_db_config"]["password"]
        self.postgres_port = app_config["postgres_db_config"]["port"]
        self.postgres_autocommit = app_config["postgres_db_config"]["autocommit"]
        self.postgres_prepare_threshold = app_config["postgres_db_config"]["prepare_threshold"]

        # Primary agent
        self.primary_agent_llm = app_config["primary_agent"]["llm"]
        self.primary_agent_llm_temperature = app_config["primary_agent"]["llm_temperature"]
        
        # SQL Agent configs 
        self.sqlagent_llm = app_config["sqlagent_configs"]["llm"]
        self.sqlagent_llm_temperature = float(
            app_config["sqlagent_configs"]["llm_temperature"])
        
        # GuideRAGTool configs 
        self.guiderag_collection_name = app_config["guiderag_configs"]["collection_name"]
        self.guiderag_llm = app_config["guiderag_configs"]["llm"]
        self.guiderag_llm_temperature = app_config["guiderag_configs"]["llm_temperature"]
        self.guiderag_embedding_model = app_config["guiderag_configs"]["embedding_model"]
        self.guiderag_chunk_size = app_config["guiderag_configs"]["chunk_size"]
        self.guiderag_chunk_overlap = app_config["guiderag_configs"]["chunk_overlap"]
        self.guiderag_k = app_config["guiderag_configs"]["k"]

        # TavilySearchTool configs
        self.tavily_search_max_results = app_config["tavily_configs"]["max_results"]