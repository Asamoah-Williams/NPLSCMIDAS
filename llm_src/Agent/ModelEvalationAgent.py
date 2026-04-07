from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
import os
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
# from pydantic import BaseModel, Field
from ..utils.load_tools_config import LoadToolsConfig
import urllib
from langchain_core.tools import tool, InjectedToolCallId
from typing import Annotated
from langgraph.types import Command
# from langchain_core.messages import ToolMessage
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from math import inf
from .State import State
# from openai import OpenAI
# import base64
import re
# , base64
# from datetime import datetime


TOOLS_CFG = LoadToolsConfig()

# class Table(BaseModel):
#     """
#     Represents a table in the SQL database.

#     Attributes:
#         name (str): The name of the table in the SQL database.
#     """
#     name: str = Field(description="Name of table in SQL database.")

class EvaluationTool:
    """
    A specialized SQL agent that interacts with the MSSQL database using a Large Language Model (LLM).

    The agent handles SQL queries by mapping user questiond to relevant SQL tables based on categories like "NPL" and "GDP". It uses an extraction chain to determine the relevant tabes based on the question and then execute the queries against the appropriate tables.

    Attributes:
        db (SQLDatabase): The SQL database agent that represents the MSSQL database.

        sql_llm (ChatOpenAI): The large language model for interpreting questions to SQL.

        table_llm (ChatOpenAI): The large language model for identifying relevant tables from queries.

    Methods:
        __init__: Initializes the agent by setting up the MSSQL database and the large language models.

        table_details: Formats table names and their descriptions for prompting.

        query_to_sql: Converts user query to SQL by creating chain of operations that maps user questions to SQL tables and executes queries.

        execute_sql: Executes SQL query using the MSSQL database agent.
    """

    def __init__(self, llm: str, llm_temperature: float) -> None:
        """
        Initializes the MSSQL database and the large language models.

        
        Args:
            llm (str): The name of the OpenAI model to use.
            llm_temperature (int): The temperature for controlling the randomness of the responses.
        """


        uid = os.getenv("SQLDB_UID")
        password = urllib.parse.quote_plus(os.getenv("SQLDB_PASSWORD")) 
        connectionString=f"mssql+pyodbc://{uid}:{password}@192.168.10.204/NPL?driver=ODBC+Driver+17+for+SQL+Server&TrustServerCertificate=yes"
        
        db_engine = create_engine(connectionString)
        self.db = SQLDatabase(db_engine, view_support=True, schema="dbo", max_string_length=inf)
        
        self.sql_llm = init_chat_model(llm, temperature=llm_temperature)
        self.llm = ChatOpenAI(model=TOOLS_CFG.primary_agent_llm, 
                    temperature=TOOLS_CFG.primary_agent_llm_temperature)
        toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        self.tools = toolkit.get_tools()

    #     self.dt_call_re = re.compile(
    #     r"datetime\.datetime\(\s*(\d{4})\s*,\s*(\d{1,2})\s*,\s*(\d{1,2})\s*,\s*(\d{1,2})\s*,\s*(\d{1,2})\s*,\s*(\d{1,2})(?:\s*,\s*(\d{1,6}))?\s*\)"
    # )


    
    def evaluate(self, user_question: str):
        """
        Evaluates the user question by converting it to SQL, querying the MSSQL database, and returning the answer.
        Args:
            user_question (str): The user's question to be evaluated.

        Returns:
            str: The answer to the user's question based on the MSSQL database query results.
        """
        sql_system_prompt = """
        You are an agent designed to interact with the forecast_path_metrics_latest table of the MSSQL database.
        Given an input question, create a syntactically correct {dialect} query to run,
        then look at the results of the query and return the answer. Unless the user
        specifies a specific number of examples they wish to obtain, always limit your
        query to at most {top_k} results.

        IMPORTANT: You must ALWAYS execute the SQL query you generate. Do not stop after just getting the table schema.

        You have access to the following tools:

        {tools}

        CRITICAL RULES:
            1. After getting table information, you MUST execute a SQL query to get actual data
            2. Do not stop after just looking at table schemas - always query for data
            3. Generate and run SQL queries to answer the user's question
            4. If you get an error, fix the query and try again

            Given an input question:
            1. Look at table structure if needed
            2. Write a SQL query
            3. Execute the SQL query
            4. Use the results to answer the question

            RULES FOR PICKING TABLES AND RUNNING QUERIES:
            - ONLY use the table forecast_path_metrics_latest
            - only use this table (model_comparison ) WHEN THE DATA FROM forecast_path_metrics_latest FAILS OR THE USER REQUESTS FOR THEM
            - When users request information about multiple variables, YOU MUST CREATE A SINGLE QUERY that include ALL requested variables in a single query. 
            - If multiple variables are mentioned your query must encompass getting data from BOTH variables and using merging strategies
            - Never create separate queries for each variable when they should be analyzed together
            - Use JOINs, UNIONs, or subqueries as needed to combine data from multiple sources
            - If variables are from different tables, find common keys (like DATE) to join them
            - DATE MUST be returned in the form YYYY-MM-DD and MUST be a string


            RULES FOR DESCRIBING IMAGES 
            - Provide a detailed analysis (5-10 sentences) on the performance of the trained model based on the data.
            - Conclude on what the data means for a Non-Performing Loans(NPL) prediction model
            - Consider the data in the context of the Ghanaian financial and banking sector 
            - For IMAGES: Describe what you actually see in the charts/graphs/heatmaps - trends, patterns, values, axes labels, etc.
            - DO NOT just say the image is 'encoded data' - analyze the actual chart/graph content
            - Correctly cite figures from the data and images provided
            - Explain the data in terms that are useful to finance and data officers

            RULES FOR DESCRIBING NUMBERS
            - Provide a detailed analysis (2-10 sentences) on the performance of the trained model based on the data.
            - Conclude on what the data means for a Non-Paying Loans(NPL) prediction model
            - Consider the data in the context of the Ghanaian financial and banking sector 
            - Correctly cite figures from the data and images provided
            - Explain the data in terms that are useful to finance and data officers

        You can order the results by a relevant column to return the most interesting
        examples in the database. Never query for all the columns from a specific table,

        You MUST double check your query before executing it. If you get an error while
        executing a query, rewrite the query and try again.

        IF MORE THAN ONE VARIABLE IS BEING REQUESTED THE 

        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
        database.

        To start you should ALWAYS look at the tables in the database to see what you
        can query. Do NOT skip this step.

        Then you should query the schema of the most relevant tables.
        """.format(
            dialect=self.db.dialect,
            top_k=5,
            tools = self.tools
        )

        agent = create_react_agent(
            self.sql_llm,
            self.tools,
            prompt=sql_system_prompt,
        )

        input = {"messages": [{"role": "human", "content": user_question}]}

        ans = agent.invoke(input)
        conclusion = ans["messages"][-1].content

        return conclusion



@tool
def model_evaluation_tool(state:State, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """
    Converts user question to SQL, queries the MSSQL Database, and answers general explanatory questions on the evaluation metrics for the non-paying loans prediction model. The evaluation metrics which are holdout root mean square error,holdout mean absolute error,holdout mean absolute percentage error and holdout r-squared. The other evualation metrics are the model comparison figure, feature importance figure, residual density figure, backtest heatmap figure, and scenario diagnostics figure. 

    Args:
        state (State): Current state of the graph (does not need to be passed in manually)
        tool_call_id (str): The unique identifier for the tool call. This is injected automatically and should not be provided manually.

    Returns:
        Command: A Command object containing the results of the model evaluation or an error message if the evaluation fails.
    """

    try:
        user_question = state.get("user_question")

        agent = EvaluationTool(
            llm=TOOLS_CFG.sqlagent_llm,
            llm_temperature=TOOLS_CFG.sqlagent_llm_temperature
        )
        
        results = agent.evaluate(user_question)

        return Command(update={
            "results": [results],
            "messages": [ToolMessage("Model Evaluation Tool successful", tool_call_id=tool_call_id)]
        })

    except Exception as e:
        return Command(update={
            "results": [f"Model evaluation failed: {str(e)}"],
            "messages": [ToolMessage(content=f"Model Evaluation Tool failed: {str(e)}", tool_call_id=tool_call_id)]
        })

 