from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import ast
from langchain_core.tools import tool, InjectedToolCallId
from ..utils.load_tools_config import LoadToolsConfig
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from typing import Annotated
from .State import State
from langgraph.prebuilt import InjectedState
from ..utils.load_project_configs import LoadProjectConfigs
from ..utils.graph_instructions import graph_instructions
import json
from langchain_core.messages import trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import tool, InjectedToolCallId


TOOLS_CFG = LoadToolsConfig()
PROJECTS_CFG = LoadProjectConfigs()

class VisualizationTool:
    def __init__(self, llm, llm_temperature):
        self.llm = ChatOpenAI(model=llm, temperature=llm_temperature)

    def visualization_type(self, question, query, results, messages):
        """
        Determines the most suitable type of visualization based on the user's question, SQL query, query results, and message history.
        The function uses a language model to analyze the inputs and recommend a visualization type along with a brief explanation.
        Args:
            question (str): The user's question that needs to be visualized.
            query (str): The SQL query that was executed to retrieve the data.
            results (list): The results obtained from executing the SQL query.
            messages (list): The message history that may provide additional context.
        Returns:
            tuple: A tuple containing the recommended visualization type (str) and the reason for the recommendation (str).
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", '''
                You are an AI assistant that recommends appropriate data visualizations based on the user's question {question}. 
             
                If the user does not provide a visualization in their question, recommend one based on the user's question, message history, SQL query, and query results, suggest the most suitable type of graph or chart to visualize the data. If no visualization is appropriate, indicate that.

                Available chart types and their use cases:
                - Bar Graphs: Best for comparing categorical data or showing changes over time when categories are discrete and the number of categories is more than 2. Use for questions like "What are the sales figures for each product?" or "How does the population of cities compare? or "What percentage of each city is male?"
                - Horizontal Bar Graphs: Best for comparing categorical data or showing changes over time when the number of categories is small or the disparity between categories is large. Use for questions like "Show the revenue of A and B?" or "How does the population of 2 cities compare?" or "How many men and women got promoted?" or "What percentage of men and what percentage of women got promoted?" when the disparity between categories is large.
                - Scatter Plots: Useful for identifying relationships or correlations between two numerical variables or plotting distributions of data. Best used when both x axis and y axis are continuous. Use for questions like "Plot a distribution of the fares (where the x axis is the fare and the y axis is the count of people who paid that fare)" or "Is there a relationship between advertising spend and sales?" or "How do height and weight correlate in the dataset? Do not use it for questions that do not have a continuous x axis."
                - Pie Charts: Ideal for showing proportions or percentages within a whole. Use for questions like "What is the market share distribution among different companies?" or "What percentage of the total revenue comes from each product?"
                - Line Graphs: Best for showing trends and distributions over time. Best used when both x axis and y axis are continuous. Used for questions like "How have website visits changed over the year?" or "What is the trend in temperature over the past decade?". Do not use it for questions that do not have a continuous x axis or a time based x axis.

    
                Visualization recommendation MUST be base on question {question}.
                Consider these types of questions when recommending a visualization:
                1. Aggregations and Summarizations (e.g., "What is the average revenue by month?" - Line Graph)
                2. Comparisons (e.g., "Compare the sales figures of Product A and Product B over the last year." - Line or Column Graph)
                3. Plotting Distributions (e.g., "Plot a distribution of the age of users" - Scatter Plot)
                4. Trends Over Time (e.g., "What is the trend in the number of active users over the past year?" - Line Graph)
                5. Proportions (e.g., "What is the market share of the products?" - Pie Chart)
                6. Correlations (e.g., "Is there a correlation between marketing spend and revenue?" - Scatter Plot)
                
                Tables to use for Visualisation
                When the user asks to generate a visualization, always retrieve data only from the following tables:
                1. actual_vs_pred
                2. backtest_heatmap_abs_error_pp
                3. champion_by_horizon
                4. feature_importance_h0
                5. feature_importance_h1
                6. feature_importance_h2
                7. forecast_path
                8. model_comparison
                
                Behavior Rules
                1. Treat any visualization request (e.g., plots, charts, heatmaps, comparisons, feature importance, forecasts) as a command to pull data exclusively from the tables listed above.
                2. If the user does not specify a table, infer the most relevant table(s) based on context.
                3. Always mention which table(s) were used to generate the visualization.
                4. Do not use or query any other tables unless explicitly instructed.
                5. Maintain visualization consistency with standard charting conventions (e.g., labeled axes, clear legends, and titles).        
                
                Provide your response in the following format:
                Recommended Visualization: [Chart type or "None"]. ONLY USE THE FOLLOWING CHART TYPES: bar, horizontal_bar, line, pie, scatter, none
                Reason: [Brief explanation for your recommendation]
             
                The recommended visualization and reasoning MUST always match.
             
                DO NOT include any kind of descriptive noun such as plot or graph.
             
                Prioritize question {question} when recommending the visualization.
             
                If the recommended visualization is anything other than bar, horizontal_bar, line, pie, scatter, or none, reevaluate your answer and choose one of them.
                '''),
                            ("human", '''
                User question: {question}
                SQL query: {sql_query}
                Query results: {results}
                Message History: {messages}
                Recommend a visualization:'''
            ),
        ])

        vis_prompt = prompt.invoke({
            "question": question,
            "sql_query": query,
            "results": results,
            "messages": messages
        })

        response = self.llm.invoke(vis_prompt)

        lines = response.content.split('\n')
        visualization = lines[0].split(': ')[1]
        reason = lines[1].split(': ')[1]

        return visualization, reason
    
    def format_data(self, visualization, results, sql_query, question):
        """
        Formats the data for visualization based on the specified visualization type.
        Args:
            visualization (str): The type of visualization (e.g., "bar", "line", "scatter", "none").
            results (list): The results obtained from executing the SQL query.
            sql_query (str): The SQL query that was executed to retrieve the data.
            question (str): The user's question that needs to be visualized.
        Returns:
            dict: A dictionary containing the formatted data for visualization.
        """


        if visualization == "none":
            return {"data_for_visualization": None}
        
        if visualization == "scatter":
            try:
                return self.format_scatter_data(results, question, sql_query, visualization)
            except Exception as e:
                return self.format_other_visualizations(question=question, visualization=visualization, results=results, sql_query=sql_query)
        
        if visualization == "bar" or visualization == "horizontal_bar":
            try:
                return self.format_bar_data(results, question)
            except Exception as e:
                return self.format_other_visualizations(question=question, visualization=visualization, results=results, sql_query=sql_query)
        
        if visualization == "line":
            try:
                return self.format_line_data(results, question, sql_query)
            except Exception as e:
                return self.format_other_visualizations(question=question, visualization=visualization, results=results, sql_query=sql_query)
            
        return self.format_other_visualizations(question=question, visualization=visualization, results=results, sql_query=sql_query)

    def format_line_data(self, results, question, query):
        """
        Formats the data for a line graph visualization.

        Args:
            results (list): The results obtained from executing the SQL query.
            question (str): The user's question that needs to be visualized.
            query (str): The SQL query that was executed to retrieve the data.
        
        Returns:
            dict: A dictionary containing the formatted data for the line graph visualization.
        """

        formatted_data = {}

        if len(results[0]) == 2:
            x_values = [str(row[0]) for row in results]
            y_values = [float(row[1]) for row in results]

        # Use LLM to get a relevant label
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a data labeling expert. Given a question, some data, and the SQL Query, provide a concise and relevant label for the data series."),
                ("human", "Question: {question}\n Data (first few rows): {data}\n SQL Query {query}\n\nProvide a concise label for this y axis. For example, if the data is the GDP figures over time, the label could be 'GDP'. If the data is the non paying loans growth, the label could be 'NPL'. If the data is the community bank leverage ratio trend, the label could be 'Community bank leverage ratio'."),
            ])
            prompt = prompt.invoke({
                "question": question,
                "data": str(results[:2]),
                "query": query
            })
            label = self.llm.invoke(prompt).content

            formatted_data= {
                "xValues": x_values,
                "yValues": [
                    {
                        "data": y_values,
                        "label": label.strip()
                    }
                ]
            }
            
        elif len(results[0]) == 3:
            x_values = []
            y_values = [{"data": [],
                 "label": ""},
                 {"data": [],
                 "label": ""}
                ]

            for row in results:
                # Determine which item is x (string not convertible to float and not containing "/")
                if type(row[0]) != float:
                    x1, y1, y2 = row[0], row[1], row[2]
                else:
                    y1, x1, y2 = row[0], row[1], row[2]

                
                x_values.append(x1)
                y_values[0].get("data").append(y1)
                y_values[1].get("data").append(y2)

            formatted_data = {
                "xValues": x_values,
                "y_values": y_values,
                "yAxisLabel": ""
            }

            # get a relevant lable for the y-axis 
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a data labeling expert. Given a question and some data, provide a concise and relevant label for the numeric values and a label for the y axis."),
                ("human", "Question: {question}\n Data (first few rows): {data}\n SQL Query {query}\n\nProvide a concise label for the second and third values as well as the y axis. For example, if the data is the NPL and GDP figures over time, the first label could be 'NPL', second label 'GDP' and third label 'NPL and GDP over time'. Return the data in the form [value1Label, value2Label, yAxisLabel]"),
            ])
            prompt = prompt.invoke({
                "question": question,
                "data": str(results[:2]),
                "query": query
            })
            label = self.llm.invoke(prompt)
            try:
                label = ast.literal_eval(label.content)
                y_values[0]["label"] = label[0]
                y_values[1]["label"] = label[1]
                formatted_data["yAxisLabel"] = label[2]
            except:
                print("error with vis")
       
        return {"formatted_data": formatted_data}

    def format_scatter_data(self, results, question, sql_query, visualization):
        """
        Formats the data for a scatter plot visualization.
        Args:
            results (list): The results obtained from executing the SQL query.
            question (str): The user's question that needs to be visualized.
            sql_query (str): The SQL query that was executed to retrieve the data.
            visualization (str): The type of visualization (should be "scatter").
            
        Returns:
            dict: A dictionary containing the formatted data for the scatter plot visualization.
        """
        if isinstance(results, str):
            results = eval(results)

        formatted_data = {"series": [], "xLabel": "", "yLabel": ""}

        if len(results[0]) == 2:
            formatted_data["series"].append({
                "data": [
                    {"x": float(x), "y": float(y), "id": i+1}
                    for i, (x, y) in enumerate(results)
                ]
            })
            
            # get a relevant lable for the y-axis 
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a data labeling expert. Given a question, a query, some data, and the visualization type, provide a concise and relevant label for the x axis and the y axis."),
                ("human", "Question: {question}\n Query:{query} \n Data (first few rows): {data}\n Visualization type:{visualization}\n\nProvide a concise label for the x axis and the y axis. For example, if the data is the NPL and GDP, the x axis label could be 'NPL' and the y axis label could be 'GDP'. Return the data in the form '[xLabel, yLabel]'")
            ])
            prompt_val = prompt.invoke({
                "question": question,
                "data": results,
                "query": sql_query,
                "visualization": visualization
            })
            label = self.llm.invoke(prompt_val).content
            label = ast.literal_eval(label)

            formatted_data["xLabel"] = label[0]
            formatted_data["yLabel"] = label[1]

        elif len(results[0]) == 3:
            entities = {}
            for row in results:
                # Determine which item is the label (string not convertible to float and not containing "/")
                if isinstance(row[0], str) and type(row[0]) != float and "-" not in row[0]:
                    label, x, y = row[0], row[1], row[2]
                else:
                    x, label, y = row[0], row[1], row[2]

                if label not in entities:
                    entities[label] = []
                entities[label].append({"x": x, "y": y, "id": len(entities[label])+1})
            
            for label, data in entities.items():
                formatted_data["series"].append({
                    "data": data,
                    "label": label
                })

            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a data labeling expert. Given a question, a query, some data, and the visualization type, provide a concise and relevant label for the x axis and the y axis."),
                ("human", "Question: {question}\n Query:{query} \n Data (first few rows): {data}\n Visualization type:{visualization}\n\nProvide a concise label for the x axis and the y axis. For example, if the data is the NPL and GDP, the x axis label could be 'NPL' and the y axis label could be 'GDP'. Return the data in the form '[xLabel, yLabel]'"),
            ])
            prompt = prompt.invoke({
                "question": question,
                "data": results,
                "query": sql_query,
                "visualization": visualization
            })

            label = self.llm.invoke(prompt).content
            label = ast.literal_eval(label)
            formatted_data["xLabel"] = label[0]
            formatted_data["yLabel"] = label[1]

        else:
            raise ValueError("Unexpected data format in results")                

        return {"formatted_data": formatted_data}

    def format_bar_data(self, results, question):
        """
        Formats the data for a bar graph visualization.

        Args:
            results (list): The results obtained from executing the SQL query.
            question (str): The user's question that needs to be visualized.
        
        Returns:
            dict: A dictionary containing the formatted data for the bar graph visualization.
        """


        if isinstance(results, str):
            results = eval(results)

        if len(results[0]) == 2:
            # Simple bar chart with one series
            xValues = [str(row[0]) for row in results]
            data = [float(row[1]) for row in results]
            
            # Use LLM to get a relevant label
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a data labeling expert. Given a question and some data, provide a concise and relevant label for the data series."),
                ("human", "Question: {question}\nData (first few rows): {data}\n\nProvide a concise label for this y axis. For example, if the data is the sales figures for products, the label could be 'Sales'. If the data is the population of cities, the label could be 'Population'. If the data is the revenue by region, the label could be 'Revenue'."),
            ])
            prompt = prompt.invoke({
                "question": question,
                "data": data
            })
            yLabel = self.llm.invoke(prompt).content
            
            yValues = [{"data": data, "label": yLabel}]

        elif len(results[0]) == 3:
            # Grouped bar chart with multiple series
            categories = set(row[1] for row in results) #eg. date
            xValues = list(categories) #eg. date
            entities = set(row[0] for row in results) #eg. gdp, npl
            yValues = [] #eg. more numerical like um the amount of gdp
            # for each entity get all the data for that entity
            for entity in entities:
                entity_data = [float(row[2]) for row in results if row[0] == entity]
                yValues.append({"data": entity_data, "label": str(entity)})
        else:
            raise ValueError("Unexpected data format in results")

        formatted_data = {
            "xValues": xValues,
            "yValues": yValues
        }

        return {"formatted_data": formatted_data}

    def format_other_visualizations(self, visualization, question, results, sql_query):
        """
        Formats the data for visualizations other than bar, line, or scatter plots using a language model.
        
        Args:
            visualization (str): The type of visualization (e.g., "pie", etc.).
            question (str): The user's question that needs to be visualized.
            results (list): The results obtained from executing the SQL query.
            sql_query (str): The SQL query that was executed to retrieve the data.
        
        Returns:
            dict: A dictionary containing the formatted data for the specified visualization.
        q"""

        instructions = graph_instructions[visualization]
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Data expert who formats data according to the required needs. You are given the question asked by the user, it's sql query, the result of the query and the format you need to format it in."),
            ("human", 'For the given question: {question}\n\nSQL query: {sql_query}\nResult: {results}\n\nUse the following example to structure the data: {instructions}. Just give the json string. Do not format it')
        ])
        response = self.llm.invoke(prompt, question=question, sql_query=sql_query, results=results, instructions=instructions)
            
        try:
            formatted_data_for_visualization = json.loads(response)
            return {"formatted_data_for_visualization": formatted_data_for_visualization}
        except json.JSONDecodeError:
            return {"error": "Failed to format data for visualization", "raw_response": response}


@tool
def visualization_tool(state: Annotated[State, InjectedState],
     tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:

    """
    Chooses the type of visualization that the question requires and formats data for visualization. 
    This tool should be called ONLY after the sql_tool.

    Args: 
        results: state["results"] as stored in the state WITH NO MUTATIONS REORDERING FOR THE INTENDED VISUALIZATION.  
        sql_query: state["sql_query"] as stored in the state with no mutations 
        question: the user's question 

    Returns:
        Command: A command object containing the visualization type, reason, formatted data for visualization, and messages from the tool execution.
    """

    try:
        question=state.get("user_question", None),
        sql_query = state.get("sql_query", None),
        sql_results=state.get("sql_results", None)
        messages = state.get("messages", None)

        if not sql_results:
            return Command(update={
                "messages": [ToolMessage("No data available for visualization", tool_call_id=tool_call_id)],
                "visualization": "none",
                "visualization_reason": "none",
                "formatted_data_for_visualization": {}
            })
        
        trimmed_messages = trim_messages(
                messages,
                max_tokens=200,
                strategy="last",
                token_counter=count_tokens_approximately,
                include_system=True
            )

        vis = VisualizationTool("gpt-3.5-turbo", 0)
        visualization, reason = vis.visualization_type(
            question=question,
            query = sql_query,
            results=sql_results,
            messages = trimmed_messages 
        )
        visualization = visualization.strip().lower()

        formatted_data = vis.format_data(visualization=visualization,
                                        results=sql_results,sql_query=sql_query,
                                        question=question)
        
        return Command(update={
            "messages": [ToolMessage("Success", tool_call_id=tool_call_id)],
            "visualization": visualization,
            "visualization_reason": reason,
            "formatted_data_for_visualization": formatted_data
        })
    except Exception as e:
        return Command(update={
            "results": f"Model evaluation failed: {str(e)}",
            "messages": [ToolMessage(content=f"Visualization Tool failed: {str(e)}", tool_call_id=tool_call_id)]
        })

