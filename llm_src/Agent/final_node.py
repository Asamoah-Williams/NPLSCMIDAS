from langgraph.types import Command
from langchain_core.prompts import ChatPromptTemplate
from ..utils.load_tools_config import LoadToolsConfig
from langchain_openai import ChatOpenAI
from .State import State
from langchain_core.messages import trim_messages
from langchain_core.messages.utils import count_tokens_approximately


# loads config file 
TOOLS_CFG = LoadToolsConfig()


def finalizer_node(state: State) -> Command:
    """
    Creates a concise final answer based on the graph state and messages

    Args: 
        state(State): The current state of the graph (graph passes this to the node)

    Returns:
        Command: Updates the current state of the graph with values
    
    """
    llm = ChatOpenAI(model=TOOLS_CFG.primary_agent_llm, 
                    temperature=TOOLS_CFG.primary_agent_llm_temperature, streaming=True)
    
    # get information from state
    question = state.get("user_question")
    visualization = state.get("visualization", None)
    visualization_reason = state.get("visualization_reason", None)
    messages = state.get("messages", None)
    sql_results = state.get("sql_results", None)
    response = messages[-1].content
    results = state.get("results", None)

    # trim message to prevent context window errors
    trimmed_messages = trim_messages(
        messages,
        max_tokens=1000,
        strategy="last",
        token_counter=count_tokens_approximately,
        include_system=True
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
            """
            Your name is CHAT DBG, a friendly finance expert finalizing the answer for the user. Write a concise, grounded response considering ONLY the user question {question}, the results {results}, the sql query results {sql_results}, and the formulated response {response}.
            IF THE VISUALIZATION VALUE: {visualization} IS NOT NONE OR ANY OTHER VALUE INDICATING NO VISUALIZATION, include a 50 word concise and accurate summary of the visualization using the visualization name {visualization}. If the visualization value {visualization} is empty do not include any summary or mention the lack of a visualization. The summary of the visualization should always come after the answer to the user's prompt. If something in the user prompt is unclear, state it briefly. If the answer is not sufficient for the user's question state it briefly. Do not include any links to images.
        """),
        ("human",
            """
        Messages: {messages} \n
        Visualization: {visualization} \n
        Visualization Reason: {visualization_reason} \n
        Question: {question} \n
        Response: {response} \n
        Answer Results: {results} \n
        SQL Query Results: {sql_results}
        Provide a concise, grounded answer to my question.
        """ )
    ])

    # format prompt with values
    prompt = prompt.format_messages(
        messages = trimmed_messages,
        visualization = visualization,
        visualization_reason = visualization_reason,
        question = question,
        response = response,
        sql_results = sql_results,
        results = results
    )

    final_answer = llm.invoke(prompt)

    return Command(update={
        "answer": final_answer.content,
        "answer_done": True
    })

