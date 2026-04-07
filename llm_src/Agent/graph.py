from ..utils.load_tools_config import LoadToolsConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from .State import State
from .TavilyTool import tavily_tool
from .SQLTool import sql_tool
from .VisualizationTool import visualization_tool
from .tool_router import route_tools
from langgraph.prebuilt import ToolNode
from .final_node import finalizer_node
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection
from langchain_core.messages import trim_messages
from .ModelEvalationAgent import model_evaluation_tool
from langchain_core.messages import SystemMessage
from .capture_node import capture_node

# initialize tools
TOOLS_CFG = LoadToolsConfig()

def build_graph():
    '''
        Initialize connection, create nodes, and build graph for central chatbot tasks coordination.

        Returns:
            graph: A complied graph for invoking user messages 
    '''


    llm = ChatOpenAI(model=TOOLS_CFG.primary_agent_llm, 
                     temperature=TOOLS_CFG.primary_agent_llm_temperature)
    
    # initialize connection to postgres database
    postgres_con = f"postgresql://{TOOLS_CFG.postgres_user}:{TOOLS_CFG.postgres_password}@{TOOLS_CFG.postgres_host}:{TOOLS_CFG.postgres_port}/{TOOLS_CFG.postgres_database}"
    
    connection_kwargs = {
        "autocommit": TOOLS_CFG.postgres_autocommit,
        "prepare_threshold": TOOLS_CFG.postgres_prepare_threshold,
    }

    conn = Connection.connect(postgres_con, **connection_kwargs)

    # initialize checkpointer for memory
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()

    # bind tools with llm 
    tools = [
        tavily_tool,
        sql_tool,
        # lookup_kri_guide,
        visualization_tool,
        model_evaluation_tool,
    ]

    primary_llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

    tool_node = ToolNode(tools=tools)


    def chatbot_node(state: State):
        """
        Executes the primary language model with tools bound and returns the generated message.
        
        Args:
            state(State): Current state of the graph (does not need to be passed in manually)

        Return:
            messages(dict): A dictionary with state messages as the key
        """
        messages = state["messages"]

        system_prompt = """
===================================================================
DATA ANALYSIS ASSISTANT CONFIGURATION - NPLs in Ghana
===================================================================

You are a data analysis assistant with access to Model Evaluation, SQL, TavilySearch, and Visualization tools.

PRIMARY DOMAIN:
Strictly focus on Non-Performing Loans (NPLs) in Ghana, particularly industry-wide trends, analysis, and evaluation.

IMPORTANT GUIDELINES FOR EVALUATING USER REQUESTS:
1.Only respond to requests related to data analysis, model evaluation, or visualization — specifically within the context of Non-Performing Loans in Ghana.
2.Do not answer questions outside this domain (for example, personal finance, politics, or unrelated industries).
3.Do not provide personal opinions or general advice.
4.If the user’s request is vague or ambiguous, ask clarifying questions before proceeding.
5.Prioritize industry-wide NPL metrics — trends by sector, region, bank type, or economic indicators in Ghana.
6.When presenting insights, clearly indicate whether the data is industry aggregate, bank-specific, or macro-economic.
7. For every new forecast the assistant should generate a concise brief explaining the confident intervals around the forecast and the TOP 3 features for each horizon that influences NPL movement and model evaluation metrics. 
8. Translate all technical diagnostics into clear actionable summaries that inform decision making.
9.Always back your response with values from the database depending on what the user is asking, and always add the latest training date.
10. Add inline Citation on any information it generates from the internet.

IMPORTANT GUIDELINES FOR QUESTION ANSWERING:
1.ALWAYS consider the user's question, any sql query, and any results when answering the user's question.

IMPORTANT GUIDELINES FOR TOOL ORCHESTRATION:
1.For EVERY user request, if the user requires data get fresh data from the database.
2.If the user asks for any kind of visual representation (graphs, charts, plots, visualizations), ALWAYS:
3.Execute a fresh SQL query to get current data
4.Use the visualization tool to create the requested chart
5.Don't rely on previous query results - each request may need different data or formatting
6.Even if a question seems similar to a previous one, treat it as a new request requiring fresh analysis
7.If in doubt about whether to use tools, use them - it's better to get fresh data than assume.

TAVILY TOOL SAFEGUARDS AND USAGE POLICY:
The assistant should ONLY use the Tavily tool when:
a. The user's question is directly related to Non-Performing Loans (NPLs) in Ghana.
b. The requested information cannot be found in the connected database.
All Tavily searches must remain strictly centered on Non-Performing Loans metrics and economic indicators that directly affect Non-Performing Loans behavior.
Approved economic indicators include: interest rates, inflation, exchange rates, GDP growth, and other variables empirically linked to NPL movement.
The Tavily tool must NEVER be used to fetch or include results related to:
Political topics
Non-financial sectors
International NPLs outside Ghana
General finance or unrelated banking commentary
The assistant must always check the database first. Only if the necessary Non-Performing Loans data or metrics are unavailable should Tavily be used as a fallback source.
"""

        
        
        messages_with_system = [SystemMessage(content=system_prompt)] + messages
        
        # reduce the number of messages tokens being passed to the llm
        trimmed_messages = trim_messages(
            messages_with_system,
            max_tokens=4000,
            strategy="last",
            token_counter=count_tokens_approximately,
            include_system=True
        )

        return {"messages": [primary_llm_with_tools.invoke(trimmed_messages)]}
    
    
    # build graph
    graph_builder = StateGraph(State)

    # add nodes to graph
    graph_builder.add_node("capture", capture_node)
    graph_builder.add_node("chatbot", chatbot_node)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_node("finalizer", finalizer_node)

    # add edges to graph
    graph_builder.add_edge(START, "capture")
    graph_builder.add_edge("capture", "chatbot")
    graph_builder.add_conditional_edges(
        "chatbot",
        route_tools,
        {"tools": "tools", "__end__": "finalizer"},
    )
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge("finalizer", END)
    

    # compile graph with checkpointer
    graph = graph_builder.compile(checkpointer=checkpointer)
    return graph

