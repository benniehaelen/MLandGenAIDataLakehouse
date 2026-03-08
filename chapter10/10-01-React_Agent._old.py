# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC <div style="display:flex; align-items:flex-start; margin-bottom:1rem;">
# MAGIC   <!-- Left: Book cover -->
# MAGIC   <img
# MAGIC     src="https://i.imgur.com/ITL8dZE.jpeg"
# MAGIC     style="width:35%; margin-right:1rem; border-radius:4px; box-shadow:0 2px 6px rgba(0,0,0,0.1);"
# MAGIC     alt="Book Cover"/>
# MAGIC   <!-- Right: Metadata -->
# MAGIC   <div style="flex:1;">
# MAGIC     <!-- O'Reilly logo above title -->
# MAGIC     <div style="display:flex; flex-direction:column; align-items:flex-start; margin-bottom:0.75rem;">
# MAGIC       <img
# MAGIC         src="https://cdn.oreillystatic.com/images/sitewide-headers/oreilly_logo_mark_red.svg"
# MAGIC         style="height:2rem; margin-bottom:0.25rem;"
# MAGIC         alt="O'Reilly"/>
# MAGIC       <span style="font-size:1.75rem; font-weight:bold; line-height:1.2;">
# MAGIC         AI, ML and GenAI in the Lakehouse
# MAGIC       </span>
# MAGIC     </div>
# MAGIC     <!-- Details -->
# MAGIC     <div style="font-size:0.9rem; color:#555; margin-bottom:1rem; line-height:1.4;">
# MAGIC       <div><strong>Name:</strong> 10-01-ReAct Agent: Autonomous Lakehouse Intelligence</div>
# MAGIC       <div><strong>Author:</strong> Bennie Haelen</div>
# MAGIC       <div><strong>Date:</strong> 7-26-2025</div>
# MAGIC     </div>
# MAGIC     <!-- Purpose -->
# MAGIC     <div style="font-weight:600; margin-bottom:0.75rem;">
# MAGIC       Purpose: Implements a ReAct agent that diagnoses a drop in premium customer
# MAGIC       engagement by autonomously querying the Medallion Architecture layers,
# MAGIC       forming hypotheses, and producing a structured diagnostic report.
# MAGIC     </div>
# MAGIC     <!-- Table of Contents -->
# MAGIC     <div style="margin-top:0;">
# MAGIC       <h3 style="margin:0 0 0.25rem;">Table of Contents</h3>
# MAGIC       <ol style="padding-left:1.25rem; margin:0; color:#333;">
# MAGIC         <li>Notebook Initialization (install libraries, imports, MLflow)</li>
# MAGIC         <li>Set Up the Lakehouse Environment (schemas and sample data)</li>
# MAGIC         <li>Define the Agent Tools (SQL, Schema, Trend Calculator)</li>
# MAGIC         <li>Implement the ReAct Agent and System Prompt</li>
# MAGIC         <li>Simulated Agent for Offline Demonstration</li>
# MAGIC         <li>Agent Invocation and ReAct Trace Display</li>
# MAGIC       </ol>
# MAGIC     </div>
# MAGIC   </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Notebook Initialization
# MAGIC
# MAGIC Initializes the notebook in three steps:
# MAGIC 1. Install and pin required libraries
# MAGIC 2. Import all dependencies
# MAGIC 3. Enable MLflow autologging for agent observability

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Required Libraries

# COMMAND ----------

# ─────────────────────────────────────────────────────────────────────────────
# Cell 1: Install Dependencies
# ─────────────────────────────────────────────────────────────────────────────
# Pinning langchain to 0.2.x ensures create_react_agent remains available via
# langchain.agents. LangChain 0.3.x removed the legacy AgentExecutor pattern
# in favor of LangGraph. This notebook uses the 0.2.x API for clarity and
# compatibility with the patterns described in this chapter.
#
# Required packages:
#   langchain==0.2.16          - Agent orchestration and ReAct loop
#   langchain-openai==0.1.23   - OpenAI model integration
#   langchain-community==0.2.16 - ConversationBufferMemory and community tools
#   pandas                      - Tabular formatting of query results
# ─────────────────────────────────────────────────────────────────────────────

%pip install --upgrade \
    "langchain==0.2.16" \
    "langchain-openai==0.1.23" \
    "langchain-community==0.2.16" \
    pandas

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Statements

# COMMAND ----------

# ─────────────────────────────────────────────────────────────────────────────
# Cell 2: Imports and Configuration
# ─────────────────────────────────────────────────────────────────────────────

# ── Suppress TensorFlow environment warnings ─────────────────────────────────
# The Databricks cluster runtime includes TensorFlow, which emits informational
# messages about CUDA drivers and oneDNN on CPU-only clusters. These do not
# affect this notebook. Setting these env vars before any imports ensures TF
# reads them at initialization time.
#   TF_CPP_MIN_LOG_LEVEL=3   suppresses all TF logs below ERROR level
#   TF_ENABLE_ONEDNN_OPTS=0  disables the oneDNN floating-point warning
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ── Standard library ─────────────────────────────────────────────────────────
import json
import re
import warnings
from typing import Any, Dict, Optional

# ── Third-party ──────────────────────────────────────────────────────────────
import pandas as pd

# Suppress Pydantic v1/v2 mixing warnings emitted by langchain_community
# internals. These arise because some community utilities still use the
# pydantic_v1 compatibility shim. Safe to suppress: does not affect runtime
# behavior for the components used in this notebook.
warnings.filterwarnings("ignore", message=".*pydantic_v1.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Mixing V1 models and V2 models.*", category=UserWarning)

# ── LangChain: agent orchestration and tool abstractions ─────────────────────
# create_react_agent and AgentExecutor are available in langchain==0.2.x.
# LangChain 0.3.x migrated these to LangGraph; the LangGraph-based approach
# is covered in a later chapter of this book.
from langchain_openai import ChatOpenAI                          # OpenAI reasoning engine
from langchain_core.tools import BaseTool                        # Base class for custom tools
from langchain_core.prompts import PromptTemplate                # System prompt construction
from langchain_core.callbacks import CallbackManagerForToolRun  # Tool callback hooks
from langchain.agents import create_react_agent, AgentExecutor  # ReAct loop wiring
from langchain.memory import ConversationBufferMemory            # Short-term conversation memory

# ── Databricks Connect: Lakehouse query interface ────────────────────────────
# DatabricksSession provides a Spark session connected to the active cluster.
# Config supplies workspace credentials for SDK-based authentication.
from databricks.connect import DatabricksSession
from databricks.sdk.core import Config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Enable MLflow Autologging

# COMMAND ----------

# ── MLflow: experiment tracking and agent observability ──────────────────────
# autolog() instruments LangChain automatically, capturing the following
# artifacts to the active MLflow experiment for every agent run:
#
#   - Full prompt and response for each LLM call
#   - Tool invocations: name, input, and output for each step
#   - Token usage: prompt tokens, completion tokens, and total cost
#   - Latency per reasoning step and end-to-end run duration
#   - The final synthesized response
#
# This produces a complete, replayable audit trail in the Databricks
# Experiments UI, which is valuable for debugging multi-step runs and
# monitoring token costs across iterations. No additional instrumentation
# code is required; autolog hooks into LangChain's callback system
# transparently and activates for all agent runs in this session.
# ─────────────────────────────────────────────────────────────────────────────
import mlflow
mlflow.langchain.autolog()

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Set Up the Lakehouse Environment
# MAGIC
# MAGIC The following function creates a self-contained three-tier Medallion
# MAGIC Architecture in the `book_ai_ml_lakehouse` catalog, populated with
# MAGIC realistic sample data that the agent will query during its investigation.

# COMMAND ----------

def setup_lakehouse_environment():
    """
    Creates Bronze, Silver, and Gold schemas and populates them with sample
    data. Running this function makes the notebook fully self-contained:
    no external data sources are required.

    Schema layout:
        bronze.clickstream_raw      Raw user event stream (page views, clicks)
        silver.customer_profiles    Customer demographics and tier classification
        silver.session_summaries    Per-session duration and conversion data
        gold.monthly_engagement     Monthly aggregated engagement by customer tier

    Notes:
        overwriteSchema=true is set on all writes so the function can be
        re-run safely after any column-level schema changes without a
        Delta schema mismatch error.
    """
    print("Setting up the Databricks Lakehouse environment...")

    # Set the target catalog for the session
    current_catalog = "book_ai_ml_lakehouse"
    spark.catalog.setCurrentCatalog(current_catalog)
    print(f"Using catalog: '{spark.catalog.currentCatalog()}'")

    # Create Bronze, Silver, and Gold schemas if they do not already exist
    for layer in ["bronze", "silver", "gold"]:
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {current_catalog}.{layer}")
    print("Schemas created successfully.")

    # ── Gold layer: monthly aggregated engagement metrics ────────────────────
    # This is the first table the agent queries to confirm the engagement drop.
    gold_data = {
        "month":                    pd.to_datetime(["2025-07-31", "2025-06-30", "2025-05-31"]),
        "avg_engagement_premium":   [71.0, 83.0, 85.0],
        "avg_engagement_standard":  [60.0, 62.0, 64.0],
        "total_revenue":            [250000.0, 245000.0, 240000.0],
        "premium_customer_count":   [1050, 1045, 1040],
    }
    (spark.createDataFrame(pd.DataFrame(gold_data))
         .write
         .mode("overwrite")
         .option("overwriteSchema", "true")
         .saveAsTable(f"{current_catalog}.gold.monthly_engagement"))

    # ── Silver layer: customer profiles ──────────────────────────────────────
    # Contains the tier field the agent uses to filter premium customers.
    # engagement_score allows the agent to identify the lowest-engagement
    # customer by name. lifetime_value supports value-based follow-up questions.
    silver_customers_data = {
        "customer_id":      list(range(101, 111)),
        "tier":             ["premium", "standard"] * 5,
        "signup_date":      pd.to_datetime(pd.date_range("2024-01-01", periods=10)),
        "lifetime_value":   [500 + i * 100 for i in range(10)],
        "engagement_score": [71, 45, 83, 50, 68, 42, 79, 55, 62, 48],
    }
    (spark.createDataFrame(pd.DataFrame(silver_customers_data))
         .write
         .mode("overwrite")
         .option("overwriteSchema", "true")
         .saveAsTable(f"{current_catalog}.silver.customer_profiles"))

    # ── Silver layer: session summaries ──────────────────────────────────────
    # Derived from the bronze clickstream. The agent drills into this table
    # to test the hypothesis that session duration declined alongside engagement.
    silver_sessions_data = {
        "session_id":       [f"s{i}" for i in range(20)],
        "customer_id":      [101 + (i % 10) for i in range(20)],
        "session_date":     pd.to_datetime(pd.date_range("2025-07-15", periods=20)),
        "duration_minutes": [10 + (i % 15) for i in range(20)],
        "converted":        [i % 3 == 0 for i in range(20)],
    }
    (spark.createDataFrame(pd.DataFrame(silver_sessions_data))
         .write
         .mode("overwrite")
         .option("overwriteSchema", "true")
         .saveAsTable(f"{current_catalog}.silver.session_summaries"))

    # ── Bronze layer: raw clickstream events ─────────────────────────────────
    # Raw, unprocessed events sourced from a streaming system such as Kafka
    # or Azure Event Hubs. Serves as the upstream source for session_summaries.
    # freq="min" is used instead of the deprecated "T" alias (removed in pandas 2.2).
    bronze_data = {
        "user_id":    [101 + (i % 10) for i in range(50)],
        "event_type": ["page_view", "click", "page_view", "add_to_cart", "purchase"] * 10,
        "timestamp":  pd.to_datetime(pd.date_range("2025-07-31 10:00", periods=50, freq="min")),
    }
    (spark.createDataFrame(pd.DataFrame(bronze_data))
         .write
         .mode("overwrite")
         .option("overwriteSchema", "true")
         .saveAsTable(f"{current_catalog}.bronze.clickstream_raw"))

    print("All tables created and populated successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Define the Agent Tools
# MAGIC
# MAGIC The agent is equipped with three purpose-built tools that map to the
# MAGIC Action step of the ReAct loop. Each tool returns a string that the agent
# MAGIC incorporates as its Observation before deciding the next step.

# COMMAND ----------

# MAGIC %md
# MAGIC ## DatabricksQueryTool

# COMMAND ----------

class DatabricksQueryTool(BaseTool):
    """
    The agent's primary interface to the Lakehouse.

    Accepts a SQL query string, executes it against the Databricks cluster
    via the notebook-global Spark session, and returns the result as a JSON
    string. The tool handles LLM-generated markdown code fences gracefully
    by stripping them before execution.
    """
    name: str = "databricks_sql_query"
    description: str = (
        "Executes a SQL query against the Databricks Lakehouse and returns "
        "the results as a JSON string. Input must be a valid Databricks SQL "
        "query string. Example: "
        "\"SELECT * FROM book_ai_ml_lakehouse.silver.customer_profiles "
        "WHERE tier = 'premium' LIMIT 10\""
    )

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute a SQL query and return results as a JSON string."""
        print(f"\n[DatabricksQueryTool] Raw input:\n{query}\n")
        try:
            # Strip markdown code fences that the LLM sometimes wraps around SQL
            match = re.search(r"```(?:sql)?\s*(.*?)\s*```", query, re.DOTALL)
            clean_query = match.group(1).strip() if match else query.strip()
            print(f"[DatabricksQueryTool] Executing: {clean_query}")

            result_df = spark.sql(clean_query).toPandas()

            if result_df.empty:
                return "Query returned no results."

            return result_df.to_json(orient="records")

        except Exception as e:
            return (
                f"Error executing query: {str(e)}. "
                "Check SQL syntax and verify table and column names against the schema."
            )

# COMMAND ----------

# MAGIC %md
# MAGIC ## SchemaInfoTool

# COMMAND ----------

class SchemaInfoTool(BaseTool):
    """
    The agent's map of the Lakehouse.

    Returns a structured description of every table and column available
    in the Bronze, Silver, and Gold layers. The agent consults this tool
    at the start of every investigation to understand what data is available
    before formulating its first SQL query.
    """
    name: str = "get_lakehouse_schema"
    description: str = (
        "Returns the schema for all tables in the Bronze, Silver, and Gold "
        "layers of the Lakehouse, including column names, types, and descriptions. "
        "Always call this tool first before writing any SQL query."
    )

    def _run(
        self,
        query: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Return the Lakehouse schema as a structured string."""
        # Uses the notebook-global spark session and the catalog set at startup.
        # No new DatabricksSession is created to avoid redundant connections.
        current_catalog = spark.catalog.currentCatalog()

        return f"""
DATABRICKS LAKEHOUSE SCHEMA (CATALOG: {current_catalog})

GOLD LAYER — Business-level monthly aggregates:
  {current_catalog}.gold.monthly_engagement
    month                   (date)   Month of the aggregated data
    avg_engagement_premium  (float)  Average engagement score for premium customers
    avg_engagement_standard (float)  Average engagement score for standard customers
    total_revenue           (float)  Total revenue for the month
    premium_customer_count  (int)    Number of active premium customers

SILVER LAYER — Cleaned and enriched data:
  {current_catalog}.silver.customer_profiles
    customer_id     (int)    Unique customer identifier
    tier            (string) Customer tier: 'premium' or 'standard'
    signup_date     (date)   Date the customer signed up
    lifetime_value  (float)  Total amount spent by the customer to date
    engagement_score (int)   Current engagement score for this customer (0-100)

  {current_catalog}.silver.session_summaries
    session_id       (string)  Unique session identifier
    customer_id      (int)     Customer who initiated the session
    session_date     (date)    Date of the session
    duration_minutes (int)     Session duration in minutes
    converted        (boolean) Whether the session resulted in a purchase

BRONZE LAYER — Raw event data:
  {current_catalog}.bronze.clickstream_raw
    user_id     (int)       User who generated the event
    event_type  (string)    Event type: 'page_view', 'click', 'add_to_cart', 'purchase'
    timestamp   (timestamp) Exact time the event was recorded
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ## TrendCalculatorTool

# COMMAND ----------

class TrendCalculatorTool(BaseTool):
    """
    Gives the agent the ability to reason numerically about change.

    Accepts a comma-separated string of metric_name, current_value, and
    previous_value, and returns a structured JSON object containing the
    percentage change, direction, and raw values. Returning JSON rather
    than a sentence forces the agent to process the result in its next
    Thought step rather than treating it as a final answer.
    """
    name: str = "calculate_trend"
    description: str = (
        "Calculates the percentage change between two numeric values and returns "
        "a structured JSON result. Input must be a comma-separated string in the "
        "format: 'metric_name,current_value,previous_value'. "
        "Example: 'Premium Engagement Score,71.0,83.0'"
    )

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Calculate percentage change and return a descriptive JSON string."""
        try:
            clean_query = query.strip().strip("'\"")
            parts = clean_query.split(",")

            if len(parts) != 3:
                return "Error: Input must be 'metric_name,current_value,previous_value'."

            metric_name, current_str, previous_str = parts
            current_value  = float(current_str.strip().strip("'\""))
            previous_value = float(previous_str.strip().strip("'\""))

            if previous_value == 0:
                pct_change = float("inf") if current_value > 0 else 0.0
            else:
                pct_change = ((current_value - previous_value) / previous_value) * 100

            direction = "increased" if pct_change > 0 else "decreased"

            result = {
                "metric_name":      metric_name.strip(),
                "trend":            direction,
                "percentage_change": round(abs(pct_change), 1),
                "from_value":       previous_value,
                "to_value":         current_value,
            }
            return json.dumps(result)

        except ValueError as e:
            return f"Error: Could not parse numeric values. Details: {e}"
        except Exception as e:
            return f"Unexpected error during trend calculation: {str(e)}"

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. ReAct Agent Implementation
# MAGIC
# MAGIC The `LakehouseReactAgent` class wires together the LLM, the three tools,
# MAGIC conversation memory, and the system prompt into a runnable AgentExecutor.
# MAGIC The system prompt defines the agent's reasoning strategy, output format,
# MAGIC and memory-first behavior for follow-up questions.

# COMMAND ----------

class LakehouseReactAgent:
    """
    A ReAct agent for intelligent, autonomous exploration of a Databricks Lakehouse.

    The agent uses a Thought/Action/Observation loop to answer business questions
    by querying the Medallion Architecture layers, forming hypotheses, and
    synthesizing structured diagnostic reports. Conversation memory allows it
    to answer follow-up questions from prior context without re-querying.
    """

    # Catalog used throughout this notebook
    CATALOG = "book_ai_ml_lakehouse"

    def __init__(self, openai_api_key: str):
        """
        Initializes the LLM, tool registry, system prompt, memory, and
        AgentExecutor.

        Args:
            openai_api_key: OpenAI API key retrieved from Databricks Secrets.
        """
        # ── LLM ──────────────────────────────────────────────────────────────
        # temperature=0 ensures deterministic, reproducible reasoning steps.
        # gpt-4o provides the multi-step reasoning quality this agent requires.
        self.llm = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0,
            openai_api_key=openai_api_key,
        )

        # ── Tools ─────────────────────────────────────────────────────────────
        self.tools = [
            DatabricksQueryTool(),
            SchemaInfoTool(),
            TrendCalculatorTool(),
        ]

        # ── System prompt ─────────────────────────────────────────────────────
        # Defines the agent's role, reasoning strategy, output format, and
        # critical SQL and memory rules. The {tools}, {tool_names},
        # {chat_history}, {input}, and {agent_scratchpad} placeholders are
        # required by the LangChain ReAct prompt contract.
        prompt_template = PromptTemplate.from_template("""
You are an expert data analyst specializing in the Databricks Lakehouse.
Your mission is to help users understand their business data by forming
hypotheses, querying the Lakehouse, and providing clear, actionable insights.

---
MEMORY-FIRST RULE:
Before performing any analysis, always review the chat history below.
If the user's question references previous results ("that customer",
"the drop you mentioned", etc.) or can be answered from recent conversation
history, answer directly from memory WITHOUT re-querying the database.
---

YOUR REASONING STRATEGY (when memory does not contain the answer):
1. Consult the schema first using get_lakehouse_schema to understand available tables
2. Start with Gold layer data to confirm the primary metric
3. Drill into Silver layer data to test hypotheses about the root cause
4. Use calculate_trend to quantify any changes you observe
5. Synthesize findings into a structured final answer

---
FINAL ANSWER FORMAT:
Your final answer must contain these three sections in markdown:

**Key Finding:** A single sentence summarizing the primary metric change.

**Business Impact:** A short paragraph explaining why this change matters
to the business, including risk of churn or revenue impact where relevant.

**Recommended Next Steps and Hypotheses:** A numbered list of concrete
investigative or remediation actions the business should take.

---
CRITICAL RULES:
- Always call get_lakehouse_schema before writing your first SQL query
- Use single quotes for string literals in SQL: WHERE tier = 'premium'
- Use CAST('2025-07-01' AS DATE) for reliable date comparisons
- Action Input must be a clean raw string with no markdown formatting
- Never skip directly to Final Answer without at least one tool invocation
  unless the answer is clearly present in the chat history
- Always include all queried identifier fields (customer_id, session_id, etc.)
  in your Final Answer so they are available for follow-up questions.
---
TOOLS AVAILABLE:
{tools}

---
INTERACTION FORMAT:

Question: The user's question.
Thought: Your reasoning. Always check memory first. Then state your plan.
Action: The tool to invoke, chosen from [{tool_names}].
Action Input: The exact input string for the tool.
Observation: The result returned by the tool.
... (this Thought/Action/Observation cycle repeats as needed)
Thought: I now have sufficient information to answer the question.
Final Answer: Your structured response using the three-section format above.

Your final message MUST begin with the exact literal text:
Final Answer:
Do not write anything after an Observation unless you are ready to produce
that Final Answer.

---
PREVIOUS CONVERSATION:
{chat_history}

---
Question: {input}

Thought: {agent_scratchpad}
""")

        # Inject tool names into the prompt at construction time
        tool_names = ", ".join([t.name for t in self.tools])
        self.prompt = prompt_template.partial(tool_names=tool_names)

        # ── ReAct agent ───────────────────────────────────────────────────────
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt,
        )

        # ── Conversation memory ───────────────────────────────────────────────
        # Retains the last 10 conversation turns so the agent can answer
        # follow-up questions without re-querying the Lakehouse.
        self.memory = ConversationBufferMemory(
            k=10,
            memory_key="chat_history",
            return_messages=True,
        )

        # ── AgentExecutor ─────────────────────────────────────────────────────
        # verbose=True prints the live Thought/Action/Observation trace to stdout.
        # return_intermediate_steps=True captures the full trace in the result
        # dict so display_react_trace() can render it in structured form.
        # max_iterations=12 prevents runaway loops on ambiguous questions.
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=12,
            handle_parsing_errors="Check your output format and try again.",
            return_intermediate_steps=True,
            output_key="output", 
        )

    def analyze(self, question: str) -> Dict[str, Any]:
        """
        Submits a business question to the ReAct agent and returns a result
        dictionary containing the final answer, the full intermediate step
        trace, and a success flag.

        Args:
            question: Natural-language business question to investigate.

        Returns:
            dict with keys: answer (str), intermediate_steps (list), success (bool)
        """
        try:
            result = self.agent_executor.invoke({"input": question})
            return {
                "answer":             result.get("output", "No output generated."),
                "intermediate_steps": result.get("intermediate_steps", []),
                "success":            True,
            }
        except Exception as e:
            return {
                "answer":             f"Agent execution error: {str(e)}",
                "intermediate_steps": [],
                "success":            False,
            }

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Simulated Agent for Offline Demonstration
# MAGIC
# MAGIC The `DemoReactAgent` class replays a hardcoded Thought/Action/Observation
# MAGIC sequence that mirrors the live agent's reasoning path. It requires no API
# MAGIC key or live Databricks connection, making it useful for classroom
# MAGIC demonstrations, CI environments, and verifying the tool implementations
# MAGIC in isolation before running the full live agent.

# COMMAND ----------

class DemoReactAgent:
    """
    A simulated ReAct agent that replays a representative reasoning sequence.

    Instantiates the real SchemaInfoTool and TrendCalculatorTool so that the
    schema and trend calculation outputs are genuine. The SQL query observation
    is hardcoded to avoid requiring a live Spark session.
    """

    def __init__(self):
        # Use real tool instances so schema and trend outputs are authentic
        self.schema_tool = SchemaInfoTool()
        self.trend_tool   = TrendCalculatorTool()

    def analyze(self, question: str) -> Dict[str, Any]:
        """
        Replays a four-step ReAct trace for the premium engagement question.

        Steps:
            1. Consult the schema (real SchemaInfoTool output)
            2. Query gold.monthly_engagement (hardcoded observation)
            3. Calculate the percentage change (real TrendCalculatorTool output)
            4. Synthesize the final diagnostic report
        """
        print(f"\nQuestion: {question}\n")
        print("=" * 70)
        print("DEMO AGENT REASONING TRACE")
        print("=" * 70)

        steps = []

        # ── Step 1: Consult the schema ────────────────────────────────────────
        thought1 = (
            "I need to understand what data is available before writing any SQL. "
            "I will call get_lakehouse_schema first."
        )
        action1       = "get_lakehouse_schema"
        action_input1 = ""
        observation1  = self.schema_tool._run(action_input1)

        print(f"\nThought       : {thought1}")
        print(f"Action        : {action1}")
        print(f"Action Input  : (none)")
        print(f"Observation   :\n{observation1}")
        steps.append((thought1, action1, action_input1, observation1))

        # ── Step 2: Query the Gold layer ──────────────────────────────────────
        thought2 = (
            "The gold.monthly_engagement table contains avg_engagement_premium "
            "by month. I will retrieve the two most recent months to identify "
            "the direction and magnitude of the change."
        )
        action2 = "databricks_sql_query"
        action_input2 = (
            "SELECT month, avg_engagement_premium "
            "FROM book_ai_ml_lakehouse.gold.monthly_engagement "
            "ORDER BY month DESC LIMIT 2"
        )
        observation2 = (
            '[{"month":"2025-07-31","avg_engagement_premium":71.0},'
            '{"month":"2025-06-30","avg_engagement_premium":83.0}]'
        )

        print(f"\nThought       : {thought2}")
        print(f"Action        : {action2}")
        print(f"Action Input  : {action_input2}")
        print(f"Observation   : {observation2}")
        steps.append((thought2, action2, action_input2, observation2))

        # ── Step 3: Quantify the change ───────────────────────────────────────
        thought3 = (
            "Engagement dropped from 83.0 in June to 71.0 in July. "
            "I will use calculate_trend to get the exact percentage change "
            "before drawing conclusions."
        )
        action3       = "calculate_trend"
        action_input3 = "Premium Engagement Score,71.0,83.0"
        observation3  = self.trend_tool._run(action_input3)

        print(f"\nThought       : {thought3}")
        print(f"Action        : {action3}")
        print(f"Action Input  : {action_input3}")
        print(f"Observation   : {observation3}")
        steps.append((thought3, action3, action_input3, observation3))

        # ── Step 4: Synthesize the final answer ───────────────────────────────
        thought4 = (
            "I have confirmed the drop, quantified it at 14.5%, and have enough "
            "context to produce a structured diagnostic report."
        )
        print(f"\nThought       : {thought4}")

        final_answer = """
**Key Finding:**
Premium customer average engagement fell 14.5%, from 83.0 in June 2025
to 71.0 in July 2025.

**Business Impact:**
A drop of this magnitude in the premium segment is a leading indicator of
declining satisfaction among the highest-value customers. If unaddressed,
this level of disengagement typically precedes measurable churn within one
to two billing cycles, with direct revenue and lifetime value implications.

**Recommended Next Steps and Hypotheses:**
1. Drill-Down Analysis: Query silver.session_summaries to determine whether
   average session duration also declined in July, which would suggest a
   product experience issue rather than an external factor.
2. Conversion Rate Check: Compare the converted flag in session_summaries
   across June and July for premium customers to see if purchase intent dropped.
3. Product Change Review: Identify any features deprecated or modified in
   late June or early July that could have degraded the premium experience.
4. Competitive Intelligence: Assess whether a competitor launched a new
   offering in the same period that may be attracting premium customers away.
"""
        print(f"\nFinal Answer  :{final_answer}")

        return {
            "answer":             final_answer,
            "intermediate_steps": steps,
            "success":            True,
        }

# COMMAND ----------

# MAGIC %md
# MAGIC # 6. Agent Invocation
# MAGIC
# MAGIC The following cells set up the environment, retrieve credentials,
# MAGIC instantiate the live agent, and run three questions that demonstrate
# MAGIC the ReAct reasoning loop in action.
# MAGIC
# MAGIC The `display_react_trace()` helper renders the full
# MAGIC Thought/Action/Observation trace from the result dictionary so the
# MAGIC agent's reasoning is visible in the cell output.

# COMMAND ----------

def display_react_trace(result: Dict[str, Any]) -> None:
    """
    Renders the full ReAct reasoning trace captured by AgentExecutor.

    Iterates over the intermediate_steps list and prints each step's tool
    name, tool input, and observation in a structured format. This makes
    the agent's Thought/Action/Observation loop visible in the notebook
    output, which verbose=True alone does not guarantee in all Databricks
    display contexts.

    Args:
        result: The dict returned by LakehouseReactAgent.analyze(), containing
                keys 'answer', 'intermediate_steps', and 'success'.
    """
    steps = result.get("intermediate_steps", [])

    print("\n" + "=" * 70)
    print("REACT REASONING TRACE")
    print("=" * 70)

    if not steps:
        print("No intermediate steps recorded.")
    else:
        for i, (action, observation) in enumerate(steps, start=1):
            print(f"\n--- Step {i} ---")
            print(f"Action        : {action.tool}")
            print(f"Action Input  : {action.tool_input}")
            print(f"Observation   : {observation}")

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(result.get("answer", "No answer returned."))

# COMMAND ----------

# ── Retrieve OpenAI API key from Databricks Secrets ──────────────────────────
# The key must be stored under scope="book", key="OPENAI_API_KEY".
# To create it: databricks secrets put-secret book OPENAI_API_KEY
print("=" * 70)
print("ReAct Agent for Databricks Lakehouse Intelligence")
print("=" * 70)

try:
    openai_api_key = dbutils.secrets.get(scope="book", key="OPENAI_API_KEY")
    print("OpenAI API key retrieved successfully.")
except Exception as e:
    print("ERROR: Could not retrieve the OpenAI API key.")
    print("Ensure that secret scope 'book' and key 'OPENAI_API_KEY' exist.")
    print(f"Details: {e}")
    dbutils.notebook.exit("Halting: OpenAI API key not found in Databricks Secrets.")

if not openai_api_key:
    print("ERROR: API key value is empty.")
    dbutils.notebook.exit("Halting: OPENAI_API_KEY secret is present but empty.")

# COMMAND ----------

# ── Set up the Lakehouse environment ─────────────────────────────────────────
try:
    setup_lakehouse_environment()
except Exception as e:
    raise RuntimeError(
        f"Lakehouse environment setup failed: {type(e).__name__}: {e}"
    ) from e

# COMMAND ----------

# ── Instantiate the agent ─────────────────────────────────────────────────────
agent = LakehouseReactAgent(openai_api_key=openai_api_key)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Question 1: Diagnose the Engagement Drop
# MAGIC
# MAGIC This is the primary scenario. The agent must confirm the drop from the
# MAGIC Gold layer, quantify it, and drill into the Silver layer to form
# MAGIC hypotheses about the root cause.

# COMMAND ----------

question_1 = "Why did our premium customers' engagement drop last month?"
print(f"Question: {question_1}\n")
result_1 = agent.analyze(question_1)
display_react_trace(result_1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Question 2: Identify the Lowest-Engagement Premium Customer
# MAGIC
# MAGIC Tests the agent's ability to drill into the Silver layer and filter by
# MAGIC tier and engagement score to identify a specific customer record.

# COMMAND ----------

question_2 = (
    "Which premium customer has the lowest engagement score? "
    "Include their customer_id, engagement score, and lifetime value in your answer."
)
print(f"Question: {question_2}\n")
result_2 = agent.analyze(question_2)
display_react_trace(result_2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Question 3: Memory Follow-Up
# MAGIC
# MAGIC Tests the agent's memory. It should answer from the prior conversation
# MAGIC context without re-querying the Lakehouse, demonstrating that
# MAGIC ConversationBufferMemory is working correctly across turns.

# COMMAND ----------

question_3 = "What was the customer ID and lifetime value you just found?"
print(f"Question: {question_3}\n")
result_3 = agent.analyze(question_3)
display_react_trace(result_3)
