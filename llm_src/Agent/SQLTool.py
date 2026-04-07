from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langgraph.prebuilt import create_react_agent
import ast
import pandas as pd
from pydantic import BaseModel, Field
# from ..utils.load_tools_config import LoadToolsConfig
from ..utils.load_tools_config import LoadToolsConfig
import urllib
# from langchain_core.tools import tool, InjectedToolCallId
from typing import Annotated
from langgraph.types import Command
# from langchain_core.messages import ToolMessage
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, ToolMessage
from .State import State
import json
from langchain_core.tools import tool, InjectedToolCallId


TOOLS_CFG = LoadToolsConfig()

class Table(BaseModel):
    """
    Represents a table in the SQL database.

    Attributes:
        name (str): The name of the table in the SQL database.
    """
    name: str = Field(description="Name of table in SQL database.")

class SQLTool:
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

        uid = "awc_sql"
        password = urllib.parse.quote_plus("AwC@2023")

        connectionString = f"mssql+pyodbc://{uid}:{password}@192.168.10.204/NPL?driver=ODBC Driver 17 for SQL Server&TrustServerCertificate=yes"

        db_engine = create_engine(connectionString)
        self.db = SQLDatabase(db_engine, view_support=True, schema="dbo")
        
        self.llm = init_chat_model(llm, temperature = llm_temperature)
        toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        self.tools = toolkit.get_tools()


    def table_details(self, path: str) -> str:
        """
            Formats table names and descriptions for prompting

            Args:
                path: the path to a csv file where each row is a table and its decription

            Returns:
                table_details: a string with table names and descriptions
        """

        table_description = pd.read_csv(path)
        table_details = ""

        for index, row in table_description.iterrows():
            table_details = table_details + "Table Name:" + str(row['Table']) + "\n" + "Table Description:" + str(row['Description']) + "\n\n"

        self.tnames_descrip = table_details

        return table_details
        

        
@tool
def sql_tool(state:State, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Converts user question to SQL and queries the MSSQL Database on the following tables/topics;
      't_npl_raw_CBLR', 't_npl_raw_DEGU', 't_npl_raw_GDP', 't_npl_raw_GLA',
    't_npl_raw_NPL', 't_npl_transformed_CBLR', 't_npl_transformed_DEGU', 't_npl_transformed_GDP', 't_npl_transformed_GLA', 't_npl_transformed_NPL'
    
    Args:
        state (State): The current state containing the user's question. 
        tool_call_id (str): The unique identifier for the tool call, injected automatically.
        
    Returns:
        Command: A command object containing the results and messages from the tool execution.
    """

    try:
        agent = SQLTool(
            llm=TOOLS_CFG.sqlagent_llm,
            llm_temperature=TOOLS_CFG.sqlagent_llm_temperature
        )
        table_details = agent.table_details("llm_src/database_table_descriptions.csv")
        system_prompt = """
            You are an agent designed to interact with a SQL database.
            Given an input question, create a syntactically correct {dialect} query to run,
            then look at the results of the query and return the answer. Unless the user
            specifies a specific number of examples they wish to obtain, always limit your
            query to at most {top_k} results. These are the names from the table description only choose the tables from {table_details}

            CRITICAL RULES:
            - When users request information about multiple variables, YOU MUST CREATE A SINGLE QUERY that include ALL requested variables in a single query. 
            - If multiple variables are mentioned (e.g., "NPL_diff and DEGU_diff"), your query must encompass getting data from BOTH variables and using merging strategies
            - Never create separate queries for each variable when they should be analyzed together
            - Use JOINs, UNIONs, or subqueries as needed to combine data from multiple sources
            - If variables are from different tables, find common keys (like DATE) to join them
            - DATE should not be returned as datetype 

            Example: If asked for "NPL_diff and DEGU_diff trends", create a query that shows both metrics together, not separate queries for each.

            You can order the results by a relevant column to return the most interesting
            examples in the database. Never query for all the columns from a specific table,
            only ask for the relevant columns given the question.

            IMPORTANT: You must ALWAYS execute the SQL query you generate. Do not stop after just getting the table schema.

            You have access to the following tools:

            {tools}

            You MUST double check your query before executing it. If you get an error while
            executing a query, rewrite the query and try again.

            IF MORE THAN ONE VARIABLE IS BEING REQUESTED THE 

            DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
            database.

            To start you should ALWAYS look at the tables in the database to see what you
            can query. Do NOT skip this step.

            THE RESULTS SHOULD ONLY BE IN THE FOLLOWING FORMAT, SO MAKE SURE THE SQL QUERY ACCOMMODATES THIS:
            [[x, y]]
            or 
            [[label, x, y]]

            If the question does not include specific years then use ONLY THE TWO YEARS OF DATA IN DESCENDING ORDER.

            Then you should query the schema of the most relevant tables.
            """.format(
                dialect=agent.db.dialect,
                top_k=5,
                tools = agent.tools,
                table_details = table_details
            )

        agent = create_react_agent(
            agent.llm,
            agent.tools,
            prompt=system_prompt,
        )

        user_question = state.get("user_question")
        input = {"messages": [{"role": "user", "content": user_question}]}


        ans_dict = {
            "data":[],
            "sql_query": None
        }

        # try up to 3 times to get data
        for i in range(3):
            ans = agent.invoke(input)
            
            for message in ans["messages"]:
                if type(message) == ToolMessage: 
                    if message.name  == 'sql_db_query':
                        try:
                            data = message.content 
                            # data literal eval
                            try:
                                data = ast.literal_eval(data)
                                if type(data) == str:
                                    data = []
                                ans_dict["data"] = data
                            # json.dumps
                            except:
                                try:
                                    data = json.dumps(data)
                                    if type(data) == str:
                                        data = []
                                    ans_dict["data"] = data
                                except:
                                    print("data conversion not working")
                        except TypeError as e:
                            print("DATA", data, e)
                            ans_dict["data"] = []
                elif type(message) == AIMessage:
                    if message.tool_calls and message.tool_calls[0].get("name") == 'sql_db_query':
                            query = message.tool_calls[0].get("args")["query"]
                            ans_dict["sql_query"] = query

            if len(ans_dict["data"]) > 0:
                break

        
        return Command(update={
            "sql_results": ans_dict["data"],
            "sql_query": ans_dict["sql_query"],
            "messages": [ToolMessage("SQL tool successful", tool_call_id=tool_call_id)]
        })
        
    except Exception as e:
        return Command(update={
            "results": [f"Model evaluation failed: {str(e)}"],
            "messages": [ToolMessage(content=f"Main SQL Tool failed: {str(e)}", tool_call_id=tool_call_id)]
        })