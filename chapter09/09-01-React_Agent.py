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
# MAGIC         <li>Notebook Initialization
# MAGIC           <ul style="margin-top:0.2rem;">
# MAGIC             <li>Install Required Libraries</li>
# MAGIC             <li>Import Statements</li>
# MAGIC             <li>Enable MLflow Autologging</li>
# MAGIC           </ul>
# MAGIC         </li>
# MAGIC         <li>Set Up the Lakehouse Environment</li>
# MAGIC         <li>Define the Agent Tools
# MAGIC           <ul style="margin-top:0.2rem;">
# MAGIC             <li>DatabricksQueryTool</li>
# MAGIC             <li>SchemaInfoTool</li>
# MAGIC             <li>TrendCalculatorTool</li>
# MAGIC           </ul>
# MAGIC         </li>
# MAGIC         <li>ReAct Agent Implementation</li>
# MAGIC         <li>Agent Invocation
# MAGIC           <ul style="margin-top:0.2rem;">
# MAGIC             <li>ReAct Trace Helper</li>
# MAGIC             <li>Retrieve API Key and Configure Environment</li>
# MAGIC             <li>Question 1: Diagnose the Engagement Drop</li>
# MAGIC             <li>Question 2: Identify the Lowest-Engagement Premium Customer</li>
# MAGIC             <li>Question 3: Memory Follow-Up</li>
# MAGIC           </ul>
# MAGIC         </li>
# MAGIC       </ol>
# MAGIC     </div>
# MAGIC   </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Notebook Initialization
# MAGIC
# MAGIC Initializes the notebook in three steps:
# MAGIC
# MAGIC 1. Install and pin required libraries
# MAGIC 2. Import all dependencies
# MAGIC 3. Enable MLflow autologging for agent observability

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1 Install Required Libraries
# MAGIC
# MAGIC LangChain is pinned to version 0.2.x because `create_react_agent` and
# MAGIC `AgentExecutor` were removed from `langchain.agents` in LangChain 0.3.x,
# MAGIC which migrated to LangGraph as the primary agent framework. This notebook
# MAGIC uses the 0.2.x API to illustrate the ReAct pattern clearly. The LangGraph
# MAGIC approach is covered in a later chapter.

# COMMAND ----------

# ─────────────────────────────────────────────────────────────────────────────
# Required packages:
#   langchain==0.2.16           - Agent orchestration and ReAct loop
#   langchain-openai==0.1.23    - OpenAI model integration
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
# MAGIC ## 1.2 Import Statements
# MAGIC
# MAGIC All imports are consolidated in this single cell so the notebook can be
# MAGIC re-run cell by cell without encountering `NameError` from missing imports.
# MAGIC TensorFlow environment warnings are suppressed here before any imports
# MAGIC that could trigger TF initialization.

# COMMAND ----------

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

# ── MLflow: imported here so autolog is available after restartPython ─────────
import mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.3 Enable MLflow Autologging
# MAGIC
# MAGIC `mlflow.langchain.autolog()` instruments LangChain automatically, capturing
# MAGIC the following artifacts to the active MLflow experiment for every agent run:
# MAGIC
# MAGIC - Full prompt and response for each LLM call
# MAGIC - Tool invocations: name, input, and output for each step
# MAGIC - Token usage: prompt tokens, completion tokens, and total cost
# MAGIC - Latency per reasoning step and end-to-end run duration
# MAGIC - The final synthesized response
# MAGIC
# MAGIC This produces a complete, replayable audit trail in the Databricks
# MAGIC Experiments UI. No additional instrumentation code is required; autolog
# MAGIC hooks into LangChain's callback system transparently and activates for
# MAGIC all agent runs in this session.

# COMMAND ----------

mlflow.langchain.autolog()

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Set Up the Lakehouse Environment
# MAGIC
# MAGIC The `setup_lakehouse_environment()` function creates a self-contained
# MAGIC three-tier Medallion Architecture in the `book_ai_ml_lakehouse` catalog
# MAGIC and populates it with realistic sample data. No external data sources
# MAGIC are required; the notebook is fully runnable after this cell completes.
# MAGIC
# MAGIC The tables created are:
# MAGIC
# MAGIC | Layer  | Table                        | Purpose                                      |
# MAGIC |--------|------------------------------|----------------------------------------------|
# MAGIC | Bronze | `bronze.clickstream_raw`     | Raw user events from a streaming source      |
# MAGIC | Silver | `silver.customer_profiles`   | Customer demographics and engagement scores  |
# MAGIC | Silver | `silver.session_summaries`   | Per-session duration and conversion data     |
# MAGIC | Gold   | `gold.monthly_engagement`    | Monthly aggregated engagement by tier        |
# MAGIC
# MAGIC `overwriteSchema=true` is set on all writes so this function can be
# MAGIC re-run safely after any column-level schema changes.

# COMMAND ----------

def setup_lakehouse_environment():
    """
    Creates Bronze, Silver, and Gold schemas and populates them with deterministic
    sample data designed to support a real diagnostic agent workflow.

    Design goals:
        - Gold layer confirms that premium engagement dropped from June to July 2025
        - Silver layer contains enough detail for the agent to test why it dropped
        - Premium customers deteriorate in July (shorter sessions, lower conversion)
        - Standard customers remain relatively stable for contrast
        - Bronze layer contains representative raw clickstream events derived from
          the Silver session summaries so the Medallion narrative remains coherent

    Notes:
        overwriteSchema=true is set on all writes so the function can be re-run
        safely after schema changes without Delta schema mismatch errors.

        The data is deterministic on purpose so the notebook produces stable,
        reproducible outputs across runs.
    """
    print("Setting up the Databricks Lakehouse environment...")

    current_catalog = "book_ai_ml_lakehouse"
    spark.catalog.setCurrentCatalog(current_catalog)
    print(f"Using catalog: '{spark.catalog.currentCatalog()}'")

    # Create Bronze, Silver, and Gold schemas if they do not already exist
    for layer in ["bronze", "silver", "gold"]:
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {current_catalog}.{layer}")
    print("Schemas created successfully.")

    def write_table(pdf, table_name):
        (spark.createDataFrame(pdf)
             .write
             .mode("overwrite")
             .option("overwriteSchema", "true")
             .saveAsTable(f"{current_catalog}.{table_name}"))

    # -------------------------------------------------------------------------
    # Gold layer: monthly aggregated engagement metrics
    # -------------------------------------------------------------------------
    # The agent should confirm the premium engagement drop here first.
    # Standard engagement stays broadly stable so the drop appears premium-specific.
    gold_data = pd.DataFrame({
        "month": pd.to_datetime(["2025-07-31", "2025-06-30", "2025-05-31"]),
        "avg_engagement_premium": [71.0, 83.0, 85.0],
        "avg_engagement_standard": [61.8, 62.0, 62.4],
        "total_revenue": [250000.0, 251500.0, 249000.0],
        "premium_customer_count": [1050, 1048, 1045],
    })
    write_table(gold_data, "gold.monthly_engagement")

    # -------------------------------------------------------------------------
    # Silver layer: customer profiles
    # -------------------------------------------------------------------------
    # Keep the same IDs and the same premium/standard alternation so downstream
    # questions remain stable. Customer 109 remains the lowest-engagement premium
    # customer for the memory follow-up example later in the notebook.
    silver_customers_data = pd.DataFrame({
        "customer_id": list(range(101, 111)),
        "tier": ["premium", "standard"] * 5,
        "signup_date": pd.to_datetime(pd.date_range("2024-01-01", periods=10)),
        "lifetime_value": [500 + i * 100 for i in range(10)],
        "engagement_score": [71, 45, 83, 50, 68, 42, 79, 55, 62, 48],
    })
    write_table(silver_customers_data, "silver.customer_profiles")

    premium_ids = [101, 103, 105, 107, 109]
    standard_ids = [102, 104, 106, 108, 110]

    # -------------------------------------------------------------------------
    # Silver layer: session summaries
    # -------------------------------------------------------------------------
    # We deliberately create a strong diagnostic signal:
    #   - Premium customers have healthy June sessions
    #   - Premium customers deteriorate in July
    #   - Standard customers stay relatively stable across both months
    #
    # This gives the agent something real to test when asked:
    # "Why did premium customer engagement drop last month?"
    session_rows = []

    def add_sessions(
        month_start,
        customer_ids,
        cohort_label,
        n_sessions,
        duration_pattern,
        conversion_every
    ):
        base_date = pd.Timestamp(month_start)
        month_tag = base_date.strftime("%Y%m")

        for i in range(n_sessions):
            session_rows.append({
                "session_id": f"{cohort_label}_{month_tag}_{i+1}",
                "customer_id": customer_ids[i % len(customer_ids)],
                "session_date": base_date + pd.Timedelta(days=i % 28),
                "duration_minutes": int(duration_pattern[i % len(duration_pattern)]),
                "converted": (i % conversion_every == 0),
            })

    # June premium: longer sessions, higher conversion
    add_sessions(
        month_start="2025-06-01",
        customer_ids=premium_ids,
        cohort_label="premium",
        n_sessions=60,
        duration_pattern=[20, 21, 22, 23, 24, 25, 26, 27, 28],
        conversion_every=3,   # ~33%
    )

    # July premium: shorter sessions, lower conversion
    add_sessions(
        month_start="2025-07-01",
        customer_ids=premium_ids,
        cohort_label="premium",
        n_sessions=60,
        duration_pattern=[10, 11, 12, 13, 14, 15, 16],
        conversion_every=6,   # ~17%
    )

    # June standard: stable baseline
    add_sessions(
        month_start="2025-06-01",
        customer_ids=standard_ids,
        cohort_label="standard",
        n_sessions=60,
        duration_pattern=[14, 15, 16, 17, 18, 19],
        conversion_every=5,   # ~20%
    )

    # July standard: still stable
    add_sessions(
        month_start="2025-07-01",
        customer_ids=standard_ids,
        cohort_label="standard",
        n_sessions=60,
        duration_pattern=[14, 15, 16, 17, 18, 19],
        conversion_every=5,   # ~20%
    )

    silver_sessions_data = pd.DataFrame(session_rows)
    write_table(silver_sessions_data, "silver.session_summaries")

    # -------------------------------------------------------------------------
    # Bronze layer: raw clickstream events
    # -------------------------------------------------------------------------
    # Generate representative raw events from the session summaries so the Bronze
    # layer is consistent with the Silver layer. This is still synthetic data,
    # but it preserves the Medallion storyline more cleanly than an unrelated
    # Bronze table.
    bronze_rows = []

    for idx, row in silver_sessions_data.iterrows():
        session_start = (
            row["session_date"]
            + pd.Timedelta(hours=9 + (idx % 8))
            + pd.Timedelta(minutes=(idx * 7) % 60)
        )

        duration = int(row["duration_minutes"])
        third = max(1, duration // 3)
        half = max(2, duration // 2)

        # Every session starts with a page view and a click
        bronze_rows.append({
            "user_id": row["customer_id"],
            "event_type": "page_view",
            "timestamp": session_start,
        })
        bronze_rows.append({
            "user_id": row["customer_id"],
            "event_type": "click",
            "timestamp": session_start + pd.Timedelta(minutes=third),
        })

        # Add a downstream action that reflects whether the session converted
        if bool(row["converted"]):
            bronze_rows.append({
                "user_id": row["customer_id"],
                "event_type": "add_to_cart",
                "timestamp": session_start + pd.Timedelta(minutes=half),
            })
            bronze_rows.append({
                "user_id": row["customer_id"],
                "event_type": "purchase",
                "timestamp": session_start + pd.Timedelta(minutes=duration),
            })
        else:
            bronze_rows.append({
                "user_id": row["customer_id"],
                "event_type": "page_view",
                "timestamp": session_start + pd.Timedelta(minutes=half),
            })

    bronze_data = pd.DataFrame(bronze_rows).sort_values("timestamp").reset_index(drop=True)
    write_table(bronze_data, "bronze.clickstream_raw")

    print("All tables created and populated successfully.")

    # Optional sanity check for notebook users
    print("\nSanity check: premium session behavior by month")
    display(
        spark.sql(f"""
            SELECT
                date_trunc('month', s.session_date) AS month,
                AVG(s.duration_minutes) AS avg_duration,
                AVG(CASE WHEN s.converted THEN 1.0 ELSE 0.0 END) AS conversion_rate,
                COUNT(*) AS sessions
            FROM {current_catalog}.silver.session_summaries s
            JOIN {current_catalog}.silver.customer_profiles c
              ON s.customer_id = c.customer_id
            WHERE c.tier = 'premium'
            GROUP BY 1
            ORDER BY 1
        """)
    )

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Define the Agent Tools
# MAGIC
# MAGIC The agent is equipped with three purpose-built tools that map to the
# MAGIC **Action** step of the ReAct loop. Each tool receives a string input from
# MAGIC the agent, performs its operation, and returns a string result that becomes
# MAGIC the agent's **Observation** for the next reasoning step.
# MAGIC
# MAGIC | Tool | Role | Returns |
# MAGIC |---|---|---|
# MAGIC | `DatabricksQueryTool` | Execute SQL against the Lakehouse | JSON rows |
# MAGIC | `SchemaInfoTool` | Describe available tables and columns | Structured text |
# MAGIC | `TrendCalculatorTool` | Quantify change between two values | JSON object |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.1 DatabricksQueryTool
# MAGIC
# MAGIC The agent's primary interface to the Lakehouse. It executes any SQL query
# MAGIC against the active Spark session and returns the results as a JSON string.
# MAGIC The tool strips LLM-generated markdown code fences from the input before
# MAGIC execution, which prevents parsing errors when the model wraps SQL in
# MAGIC triple-backtick blocks. Date columns are converted to ISO strings to
# MAGIC produce human-readable output rather than epoch milliseconds.

# COMMAND ----------

class DatabricksQueryTool(BaseTool):
    """
    Executes SQL queries against the Databricks Lakehouse.

    Accepts a SQL query string, cleans it of any LLM-generated markdown
    formatting, executes it via the notebook-global Spark session, and
    returns the result as a JSON string for the agent to reason over.
    Date columns are serialized as ISO strings (YYYY-MM-DD) rather than
    epoch milliseconds to keep observations human-readable.
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
            if match:
                clean_query = match.group(1).strip()
            else:
                # Strip whitespace first, then strip any surrounding quotes the LLM
                # may have added. The LLM frequently passes Action Input as a quoted
                # string (e.g. "SELECT ..."), and LangChain's parser sometimes strips
                # only the leading quote, leaving a trailing " that breaks SQL parsing.
                clean_query = query.strip().strip('"').strip("'").strip()

            print(f"[DatabricksQueryTool] Executing: {clean_query}")

            result_df = spark.sql(clean_query).toPandas()

            if result_df.empty:
                return "Query returned no results."

            # Convert all datetime columns to ISO date strings.
            # "datetime" catches all datetime64 resolutions (ns, us, ms) without
            # requiring explicit dtype strings, avoiding the pandas specificity error.
            for col in result_df.select_dtypes(include=["datetime"]).columns:
                result_df[col] = result_df[col].dt.strftime("%Y-%m-%d")

            return result_df.to_json(orient="records")

        except Exception as e:
            return (
                f"Error executing query: {str(e)}. "
                "Check SQL syntax and verify table and column names against the schema."
            )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2 SchemaInfoTool
# MAGIC
# MAGIC The agent's map of the Lakehouse. It returns a structured description of
# MAGIC every table and column across all three Medallion Architecture layers.
# MAGIC The system prompt instructs the agent to call this tool before writing
# MAGIC any SQL query, ensuring it never references non-existent columns.

# COMMAND ----------

class SchemaInfoTool(BaseTool):
    """
    Returns the Lakehouse schema for agent planning.

    Provides a structured description of all tables in the Bronze, Silver,
    and Gold layers, including column names, types, and descriptions. The
    agent uses this as its starting point before formulating any SQL query.
    Uses the notebook-global spark session rather than creating a new
    DatabricksSession to avoid redundant connections.
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
    customer_id      (int)    Unique customer identifier
    tier             (string) Customer tier: 'premium' or 'standard'
    signup_date      (date)   Date the customer signed up
    lifetime_value   (float)  Total amount spent by the customer to date
    engagement_score (int)    Current engagement score for this customer (0-100)

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
# MAGIC ## 3.3 TrendCalculatorTool
# MAGIC
# MAGIC Gives the agent the ability to reason numerically about change. Rather than
# MAGIC inferring a percentage from raw values, the agent explicitly calls this tool
# MAGIC and receives a structured JSON result. Returning JSON forces the agent to
# MAGIC process the result in its next Thought step rather than treating the
# MAGIC observation as a ready-made final answer, which keeps the reasoning loop
# MAGIC honest and visible.

# COMMAND ----------

class TrendCalculatorTool(BaseTool):
    """
    Calculates percentage change between two numeric values.

    Accepts a comma-separated string of metric_name, current_value, and
    previous_value, and returns a structured JSON object containing the
    percentage change, direction, and raw from/to values. Returning JSON
    rather than a prose sentence forces the agent to process the result
    explicitly in its next Thought step.
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
                "metric_name":       metric_name.strip(),
                "trend":             direction,
                "percentage_change": round(abs(pct_change), 1),
                "from_value":        previous_value,
                "to_value":          current_value,
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
# MAGIC The `LakehouseReactAgent` class wires together four components into a
# MAGIC runnable agent:
# MAGIC
# MAGIC - **LLM** — GPT-4o at temperature 0 for deterministic reasoning
# MAGIC - **Tools** — the three tools defined in Section 3
# MAGIC - **System prompt** — defines the reasoning strategy, output format, SQL
# MAGIC   rules, and memory-first behavior for follow-up questions
# MAGIC - **Memory** — `ConversationBufferMemory` retaining the last 10 turns so
# MAGIC   the agent can answer follow-up questions without re-querying the Lakehouse
# MAGIC
# MAGIC The `AgentExecutor` runs the ReAct loop, enforcing a maximum of 12
# MAGIC iterations to prevent runaway execution on ambiguous questions.

# COMMAND ----------

class LakehouseReactAgent:
    """
    A ReAct agent for intelligent, autonomous exploration of a Databricks Lakehouse.

    The agent uses a Thought/Action/Observation loop to answer business questions
    by querying the Medallion Architecture layers, forming hypotheses, and
    synthesizing structured diagnostic reports. Conversation memory allows it
    to answer follow-up questions from prior context without re-querying.
    """

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
Your mission is to answer business questions by selecting the right Lakehouse
layer, forming testable hypotheses, querying governed data, and producing
clear, evidence-based conclusions.

---
MEMORY-FIRST RULE:
Before performing any analysis, always review the chat history below.
If the user's question references previous results ("that customer",
"the drop you mentioned", etc.) or can be answered from recent conversation
history, answer directly from memory WITHOUT re-querying the database.
---

TASK ROUTING RULES:
Choose the most appropriate Lakehouse layer for the question.

1. For change-over-time or diagnostic questions such as:
   - "Why did X drop?"
   - "Why did X increase?"
   - "What caused this change?"
   you must:
   a. confirm the change in the Gold layer,
   b. form at least one explicit causal hypothesis,
   c. test that hypothesis with at least one Silver-layer query,
   d. use calculate_trend to quantify the relevant change,
   e. only then produce the Final Answer.

2. For customer-level lookups, rankings, filters, or record retrieval:
   query the most relevant Silver table directly after consulting the schema.
   Do NOT start with the Gold layer unless the question is explicitly about
   an aggregate metric over time.

3. Use the Bronze layer only when raw event-level evidence is required and
   the question cannot be answered from Gold or Silver.

4. Never claim a root cause unless it is supported by query results.
   If the available data supports only a plausible hypothesis, say so clearly.

---
YOUR REASONING STRATEGY (when memory does not contain the answer):
1. Call get_lakehouse_schema before your first SQL query.
2. Decide which Lakehouse layer is most relevant for the question.
3. Query only the columns needed to answer the question.
4. If diagnosing a change over time, compare the relevant periods explicitly
   (for example, July 2025 versus June 2025).
5. Use calculate_trend for percentage change calculations instead of mental math.
6. If the evidence is insufficient to prove a cause, state that clearly and
   recommend the next best query or business action.

---
FINAL ANSWER FORMAT:
For diagnostic "why did it change?" questions, your final answer must contain
these four sections in markdown:

**Observed Change:** A concise statement of what changed in the Gold layer.
Include any specific customer_id, session_id, or numeric identifiers
as labeled values on their own line when relevant, for example:
  customer_id: 109
  lifetime_value: 1300
  engagement_score: 62

**Tested Hypothesis:** State the specific hypothesis you tested in the
Silver layer.

**Supporting Evidence:** Present the query results or computed metrics that
support or weaken the hypothesis. Be explicit about the comparison period
and the direction of change.

**Business Impact and Next Steps:** Explain why the finding matters to the
business and recommend the next investigative or remediation actions.

For non-diagnostic lookup or ranking questions, adapt the structure to:

**Answer:** The direct answer to the question.
**Method:** The table(s) and logic used to find it.
**Supporting Data:** The key result rows or identifiers.
**Business Impact and Next Steps:** Include only when it adds value.

---
CRITICAL RULES:
- Always call get_lakehouse_schema before your first SQL query.
- Use single quotes for string literals in SQL: WHERE tier = 'premium'
- Use CAST('2025-07-01' AS DATE) for reliable date comparisons
- Action Input must be a clean raw string with no markdown formatting
- Never skip directly to Final Answer without at least one tool invocation
  unless the answer is clearly present in the chat history
- For any "why did it change?" question, do NOT stop after only a Gold query
  and a trend calculation; you must test at least one Silver-layer hypothesis
  before producing the Final Answer
- Always include all queried identifier fields (customer_id, session_id, etc.)
  in your Final Answer so they are available for follow-up questions
- Do not invent facts, causes, or metrics that are not present in the observations

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
Final Answer: Your structured response using the required format above.

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
        # output_key="output" resolves the multiple-output-keys warning that
        # occurs when return_intermediate_steps and memory are used together.
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
# MAGIC # 5. Agent Invocation
# MAGIC
# MAGIC The following cells retrieve credentials, set up the environment,
# MAGIC instantiate the live agent, and run three questions that demonstrate
# MAGIC the ReAct reasoning loop in action.
# MAGIC
# MAGIC The three questions are designed to exercise different agent capabilities:
# MAGIC
# MAGIC | Question | Capability demonstrated |
# MAGIC |---|---|
# MAGIC | Why did premium engagement drop? | Multi-step reasoning across Gold and Silver layers |
# MAGIC | Which customer has the lowest engagement? | Targeted Silver layer query with filtering |
# MAGIC | What was the customer ID you just found? | Memory recall with no tool invocations |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.1 ReAct Trace Helper
# MAGIC
# MAGIC `display_react_trace()` formats the full Thought/Action/Observation trace
# MAGIC captured by `AgentExecutor` into a clean, labeled structure. Although
# MAGIC `verbose=True` prints a live trace during execution, this helper produces
# MAGIC the consistent output format used throughout the book's printed examples.
# MAGIC
# MAGIC When Question 3 is answered entirely from memory, the step list will be
# MAGIC empty and the output will confirm that no tool calls were made.

# COMMAND ----------

def display_react_trace(result: Dict[str, Any]) -> None:
    """
    Renders the full ReAct reasoning trace captured by AgentExecutor.

    Iterates over the intermediate_steps list and prints each step's tool
    name, tool input, and observation in a structured, labeled format.

    Args:
        result: The dict returned by LakehouseReactAgent.analyze(), containing
                keys 'answer', 'intermediate_steps', and 'success'.
    """
    steps = result.get("intermediate_steps", [])

    print("\n" + "=" * 70)
    print("REACT REASONING TRACE")
    print("=" * 70)

    if not steps:
        print("No intermediate steps recorded (answer came from memory).")
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

# MAGIC %md
# MAGIC ## 5.2 Retrieve API Key and Configure Environment
# MAGIC
# MAGIC The OpenAI API key is retrieved from Databricks Secrets using scope `book`
# MAGIC and key `OPENAI_API_KEY`. To create the secret if it does not yet exist,
# MAGIC run the following from the Databricks CLI:
# MAGIC
# MAGIC ```
# MAGIC databricks secrets create-scope book
# MAGIC databricks secrets put-secret book OPENAI_API_KEY
# MAGIC ```
# MAGIC
# MAGIC Re-run this cell any time you want to reset the agent and clear the
# MAGIC conversation memory buffer before starting a new question sequence.

# COMMAND ----------

print("=" * 70)
print("ReAct Agent for Databricks Lakehouse Intelligence")
print("=" * 70)

# ── Retrieve OpenAI API key from Databricks Secrets ──────────────────────────
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

# ── Set up the Lakehouse environment ─────────────────────────────────────────
try:
    setup_lakehouse_environment()
except Exception as e:
    raise RuntimeError(
        f"Lakehouse environment setup failed: {type(e).__name__}: {e}"
    ) from e

# ── Instantiate the agent ─────────────────────────────────────────────────────
# A fresh instance clears the memory buffer. Re-run this cell any time you
# want to reset conversation history before a new question sequence.
agent = LakehouseReactAgent(openai_api_key=openai_api_key)
print("\nAgent instantiated and ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.3 Question 1: Diagnose the Engagement Drop
# MAGIC
# MAGIC The primary scenario from the chapter. The agent must:
# MAGIC
# MAGIC 1. Consult the schema to orient itself
# MAGIC 2. Query the Gold layer to confirm the engagement drop
# MAGIC 3. Use the TrendCalculatorTool to quantify the percentage change
# MAGIC 4. Synthesize a structured diagnostic report
# MAGIC
# MAGIC This question exercises the full Thought/Action/Observation loop across
# MAGIC three reasoning steps before producing a final answer.

# COMMAND ----------

question_1 = """
Why did our premium customers' engagement drop last month?

You must:
1. confirm the drop using the Gold layer,
2. test at least one causal hypothesis using the Silver layer,
3. compare July 2025 with June 2025,
4. support your conclusion with data before giving the final answer.
"""

print(f"Question: {question_1}\n")
result_1 = agent.analyze(question_1)
display_react_trace(result_1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.4 Question 2: Identify the Lowest-Engagement Premium Customer
# MAGIC
# MAGIC Tests the agent's ability to filter by tier, sort by engagement score,
# MAGIC and return a specific customer record from the Silver layer. The question
# MAGIC explicitly requests `customer_id` and `lifetime_value` so those values
# MAGIC are stored in conversation memory and available for Question 3.

# COMMAND ----------

question_2 = (
    "Which premium customer has the lowest engagement score? "
    "Your final answer MUST include the exact customer_id integer, "
    "the engagement score, and the lifetime value as separate labeled values."
)
print(f"Question: {question_2}\n")
result_2 = agent.analyze(question_2)
display_react_trace(result_2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.5 Question 3: Memory Follow-Up
# MAGIC
# MAGIC Tests the agent's `ConversationBufferMemory`. Because `customer_id` and
# MAGIC `lifetime_value` were explicitly included in the Question 2 answer, the
# MAGIC agent should resolve this question entirely from memory with zero tool
# MAGIC invocations. The trace output will show an empty step list and the message
# MAGIC "answer came from memory" confirming that `ConversationBufferMemory`
# MAGIC is working correctly across turns.

# COMMAND ----------

question_3 = "What was the customer ID and lifetime value you just found?"
print(f"Question: {question_3}\n")
result_3 = agent.analyze(question_3)
display_react_trace(result_3)

# COMMAND ----------

# MAGIC %md
# MAGIC # Follow-up Question

# COMMAND ----------

question_4 = (
    "The engagement dropped 14.5% in July. "
    "Query the session summaries for premium customers in July to test whether "
    "average session duration also declined, and tell me if this supports the "
    "hypothesis that the drop was caused by a product experience issue."
)
print(f"Question: {question_4}\n")
result_4 = agent.analyze(question_4)
display_react_trace(result_4)
