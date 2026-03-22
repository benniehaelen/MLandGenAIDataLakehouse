# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC <div style="display:flex; align-items:flex-start; margin-bottom:1rem;">
# MAGIC   <!-- Left: Book cover -->
# MAGIC   <img
# MAGIC     src="https://adb-1376134742576436.16.azuredatabricks.net/files/Images/book_cover.JPG"
# MAGIC     style="width:35%; margin-right:1rem; border-radius:4px; box-shadow:0 2px 6px rgba(0,0,0,0.1);"
# MAGIC     alt="Book Cover"/>
# MAGIC   <!-- Right: Metadata -->
# MAGIC   <div style="flex:1;">
# MAGIC     <!-- O'Reilly logo above title -->
# MAGIC     <div style="display:flex; flex-direction:column; align-items:flex-start; margin-bottom:0.75rem;">
# MAGIC       <img
# MAGIC         src="https://cdn.oreillystatic.com/images/sitewide-headers/oreilly_logo_mark_red.svg"
# MAGIC         style="height:2rem; margin-bottom:0.25rem;"
# MAGIC         alt="O‘Reilly"/>
# MAGIC       <span style="font-size:1.75rem; font-weight:bold; line-height:1.2;">
# MAGIC         AI, ML and GenAI in the Lakehouse
# MAGIC       </span>
# MAGIC     </div>
# MAGIC     <!-- Details, now each on its own line -->
# MAGIC     <div style="font-size:0.9rem; color:#555; margin-bottom:1rem; line-height:1.4;">
# MAGIC       <div><strong>Name:</strong> 10-01-databricks_mcp_notebook_client</div>
# MAGIC       <div><strong>Author:</strong> Bennie Haelen</div>
# MAGIC       <div><strong>Date:</strong> 3‑15‑2026</div>
# MAGIC     </div>
# MAGIC     <!-- Purpose -->
# MAGIC     <div style="font-weight:600; margin-bottom:0.75rem;">
# MAGIC       Purpose: This notebook implements an MCP Client Tool
# MAGIC     </div>
# MAGIC     <!-- Outline -->
# MAGIC     <div style="margin-top:0;">
# MAGIC       <h3 style="margin:0 0 0.25rem;">Table of Contents</h3>
# MAGIC       <ol style="padding-left:1.25rem; margin:0; color:#333;">
# MAGIC         <li>Create the UC Catalog if it does not yet exist</li>
# MAGIC         <li>Create the UC Schema if it does not yet exist</li>
# MAGIC         <li>Print out the current Catalog and Schema</li>
# MAGIC       </ol>
# MAGIC     </div>
# MAGIC   </div>
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Databricks MCP Client for Notebooks
# MAGIC
# MAGIC This notebook provides a client interface to interact with the Databricks MCP Server
# MAGIC from within Databricks notebooks. It allows you to:
# MAGIC - Query Unity Catalog metadata
# MAGIC - Execute SQL queries
# MAGIC - Use natural language to generate and execute queries
# MAGIC - Create visualizations with Plotly
# MAGIC
# MAGIC ## Setup
# MAGIC 1. Install required packages
# MAGIC 2. Configure environment variables
# MAGIC 3. Use the NotebookMCPClient class

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC # Setup: Databricks CLI Installation & Secrets Configuration
# MAGIC
# MAGIC Run these steps **once from your local machine** before executing the MCP client notebook.
# MAGIC This notebook documents the full setup process and can be kept as a reference header.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Step 1: Install the Databricks CLI
# MAGIC
# MAGIC Run the appropriate commands for your operating system from a terminal or PowerShell window.
# MAGIC
# MAGIC **macOS (Homebrew)**
# MAGIC ```bash
# MAGIC brew tap databricks/tap
# MAGIC brew install databricks
# MAGIC ```
# MAGIC
# MAGIC **Windows (winget — recommended)**
# MAGIC ```powershell
# MAGIC winget search databricks
# MAGIC winget install Databricks.DatabricksCLI
# MAGIC ```
# MAGIC
# MAGIC **Windows (Chocolatey — experimental)**
# MAGIC ```powershell
# MAGIC choco install databricks-cli
# MAGIC ```
# MAGIC
# MAGIC **Linux (curl)**
# MAGIC ```bash
# MAGIC curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh
# MAGIC ```
# MAGIC
# MAGIC **Verify the installation (all platforms)**
# MAGIC ```bash
# MAGIC databricks -v
# MAGIC ```
# MAGIC Version 0.205.0 or higher should be displayed.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Step 2: Authenticate the CLI
# MAGIC
# MAGIC Configure the CLI to point at your Databricks workspace using OAuth (opens a browser):
# MAGIC ```bash
# MAGIC databricks auth login --host https://<your-workspace-url>
# MAGIC ```
# MAGIC
# MAGIC Or use a Personal Access Token (PAT):
# MAGIC ```bash
# MAGIC databricks configure --token
# MAGIC ```
# MAGIC You will be prompted for your workspace URL and token.
# MAGIC
# MAGIC **Verify authentication is working:**
# MAGIC ```bash
# MAGIC databricks current-user me
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Step 3: Generate a Personal Access Token (PAT)
# MAGIC
# MAGIC 1. In your Databricks workspace, click your username (top right)
# MAGIC 2. Go to **Settings > Developer > Access Tokens**
# MAGIC 3. Click **Generate new token**
# MAGIC 4. Give it a description (e.g. `databricks-mcp`) and set an expiration
# MAGIC 5. Copy the token value — you will not be able to see it again
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Step 4: Find Your SQL Warehouse ID
# MAGIC
# MAGIC 1. In your Databricks workspace, go to **SQL Warehouses** in the left sidebar
# MAGIC 2. Click your warehouse, then open the **Connection Details** tab
# MAGIC 3. Copy the **HTTP Path** value — it will look like:
# MAGIC    `/sql/1.0/warehouses/abc1234def567890`
# MAGIC 4. The warehouse ID is the last segment only: `abc1234def567890`
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Step 5: Create the Secret Scope
# MAGIC
# MAGIC Run this once to create the `databricks-mcp` scope:
# MAGIC ```bash
# MAGIC databricks secrets create-scope databricks-mcp
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Step 6: Store the Secrets
# MAGIC
# MAGIC Use `--string-value` to avoid the interactive prompt issue on Windows PowerShell.
# MAGIC Replace each placeholder with your actual value.
# MAGIC
# MAGIC **macOS / Linux**
# MAGIC ```bash
# MAGIC databricks secrets put-secret databricks-mcp host --string-value "https://<your-workspace-url>"
# MAGIC databricks secrets put-secret databricks-mcp token --string-value "dapi<your-pat-token>"
# MAGIC databricks secrets put-secret databricks-mcp warehouse-id --string-value "<your-warehouse-id>"
# MAGIC databricks secrets put-secret databricks-mcp anthropic-api-key --string-value "sk-ant-<your-key>"
# MAGIC ```
# MAGIC
# MAGIC **Windows PowerShell**
# MAGIC ```powershell
# MAGIC databricks secrets put-secret databricks-mcp host --string-value "https://<your-workspace-url>"
# MAGIC databricks secrets put-secret databricks-mcp token --string-value "dapi<your-pat-token>"
# MAGIC databricks secrets put-secret databricks-mcp warehouse-id --string-value "<your-warehouse-id>"
# MAGIC databricks secrets put-secret databricks-mcp anthropic-api-key --string-value "sk-ant-<your-key>"
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Step 7: Verify All Secrets Are Present
# MAGIC
# MAGIC ```bash
# MAGIC databricks secrets list-secrets databricks-mcp
# MAGIC ```
# MAGIC
# MAGIC You should see all four keys listed. Values will display as `REDACTED` — that is expected.
# MAGIC
# MAGIC ```
# MAGIC Key               Last Updated (UTC)
# MAGIC ----------------  -------------------
# MAGIC anthropic-api-key 2025-01-01 00:00:00
# MAGIC host              2025-01-01 00:00:00
# MAGIC token             2025-01-01 00:00:00
# MAGIC warehouse-id      2025-01-01 00:00:00
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC ## Step 8: Validate Secrets from the Notebook
# MAGIC
# MAGIC Run the cell below to confirm the notebook can read all four secrets.
# MAGIC Values are intentionally not printed in full — only their lengths are shown
# MAGIC to verify they were stored correctly without exposing sensitive data.

# COMMAND ----------

# MAGIC %pip install mcp anthropic plotly pandas kaleido --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import json
import os
from typing import Dict, Any, Optional, List
import asyncio
from dataclasses import dataclass
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from anthropic import Anthropic

# COMMAND ----------

def validate_secrets():
    """Validate that all required secrets are present and non-empty."""
    required = {
        "host":              ("databricks-mcp", "host"),
        "token":             ("databricks-mcp", "token"),
        "warehouse-id":      ("databricks-mcp", "warehouse-id"),
        "anthropic-api-key": ("databricks-mcp", "anthropic-api-key"),
    }

    all_ok = True
    print("=== Secret Validation ===\n")

    for label, (scope, key) in required.items():
        try:
            value = dbutils.secrets.get(scope=scope, key=key)
            if not value or not value.strip():
                print(f"  [WARN]  {label}: found but empty")
                all_ok = False
            else:
                print(f"  [OK]    {label}: present ({len(value)} chars)")
        except Exception as e:
            print(f"  [ERROR] {label}: not found — {e}")
            all_ok = False

    print()
    if all_ok:
        print("All secrets validated successfully. You are ready to run the MCP client notebook.")
    else:
        print("One or more secrets are missing or empty. Re-run the CLI setup steps above.")

validate_secrets()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Load Configuration
# MAGIC
# MAGIC Once validation passes, run this cell to load all secrets into variables
# MAGIC used by the rest of the notebook.

# COMMAND ----------

# Configuration - Update these or use Databricks secrets
DATABRICKS_HOST = dbutils.secrets.get(scope="databricks-mcp", key="host") if dbutils.secrets.listScopes() else os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = dbutils.secrets.get(scope="databricks-mcp", key="token") if dbutils.secrets.listScopes() else os.getenv("DATABRICKS_TOKEN")
DATABRICKS_WAREHOUSE_ID = dbutils.secrets.get(scope="databricks-mcp", key="warehouse-id") if dbutils.secrets.listScopes() else os.getenv("DATABRICKS_WAREHOUSE_ID")
ANTHROPIC_API_KEY = dbutils.secrets.get(scope="databricks-mcp", key="anthropic-api-key") if dbutils.secrets.listScopes() else os.getenv("ANTHROPIC_API_KEY")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notebook MCP Client
# MAGIC
# MAGIC This is a simplified client that works directly in Databricks notebooks without needing
# MAGIC to run a separate MCP server process.

# COMMAND ----------

class NotebookMCPClient:
    """
    Simplified MCP client for Databricks notebooks.
    Uses the Databricks SDK directly within the notebook environment.
    """
    
    def __init__(self, warehouse_id: Optional[str] = None, anthropic_api_key: Optional[str] = None):
        """
        Initialize the notebook client
        
        Args:
            warehouse_id: SQL warehouse ID for query execution
            anthropic_api_key: Anthropic API key for NL queries
        """
        self.warehouse_id = warehouse_id or DATABRICKS_WAREHOUSE_ID
        self.anthropic_client = Anthropic(api_key=anthropic_api_key) if anthropic_api_key else None
        
        # Import Databricks SDK
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.core import Config
        
        # Initialize workspace client
        config = Config(
            host=DATABRICKS_HOST or dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get(),
            token=DATABRICKS_TOKEN or dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
        )
        self.workspace_client = WorkspaceClient(config=config)
    
    def list_catalogs(self) -> pd.DataFrame:
        """List all Unity Catalogs"""
        catalogs = list(self.workspace_client.catalogs.list())
        return pd.DataFrame([
            {
                "name": c.name,
                "comment": c.comment,
                "owner": c.owner,
                "created_at": str(c.created_at) if c.created_at else None
            }
            for c in catalogs
        ])
    
    def list_schemas(self, catalog: str) -> pd.DataFrame:
        """List all schemas in a catalog"""
        schemas = list(self.workspace_client.schemas.list(catalog_name=catalog))
        return pd.DataFrame([
            {
                "catalog": catalog,
                "name": s.name,
                "comment": s.comment,
                "owner": s.owner,
                "created_at": str(s.created_at) if s.created_at else None
            }
            for s in schemas
        ])
    
    def list_tables(self, catalog: str, schema: str) -> pd.DataFrame:
        """List all tables in a schema"""
        tables = list(self.workspace_client.tables.list(
            catalog_name=catalog,
            schema_name=schema
        ))
        return pd.DataFrame([
            {
                "catalog": catalog,
                "schema": schema,
                "name": t.name,
                "table_type": t.table_type.value if t.table_type else None,
                "comment": t.comment,
                "owner": t.owner
            }
            for t in tables
        ])
    
    def get_table_info(self, catalog: str, schema: str, table: str) -> Dict[str, Any]:
        """Get detailed information about a table"""
        table_info = self.workspace_client.tables.get(
            full_name=f"{catalog}.{schema}.{table}"
        )
        
        return {
            "name": table_info.name,
            "catalog_name": table_info.catalog_name,
            "schema_name": table_info.schema_name,
            "table_type": table_info.table_type.value if table_info.table_type else None,
            "data_source_format": table_info.data_source_format.value if table_info.data_source_format else None,
            "columns": [
                {
                    "name": col.name,
                    "type_name": col.type_name.value if col.type_name else None,
                    "type_text": col.type_text,
                    "comment": col.comment,
                    "nullable": col.nullable,
                    "position": col.position
                }
                for col in (table_info.columns or [])
            ],
            "owner": table_info.owner,
            "comment": table_info.comment
        }
    
    def execute_sql(self, query: str, warehouse_id: Optional[str] = None) -> pd.DataFrame:
        """
        Execute a SQL query and return results as DataFrame
        
        Args:
            query: SQL query to execute
            warehouse_id: Optional warehouse ID (uses default if not provided)
        
        Returns:
            DataFrame with query results
        """
        wh_id = warehouse_id or self.warehouse_id
        
        if not wh_id:
            raise ValueError("No warehouse_id provided and DATABRICKS_WAREHOUSE_ID not set")
        
        # Execute query
        response = self.workspace_client.statement_execution.execute_statement(
            warehouse_id=wh_id,
            statement=query,
            wait_timeout="30s"
        )
        
        # Convert to DataFrame
        if response.result and response.result.data_array:
            columns = [col.name for col in (response.manifest.schema.columns or [])]
            df = pd.DataFrame(response.result.data_array, columns=columns)
            return df
        else:
            return pd.DataFrame()
    
    def query_natural_language(
        self,
        question: str,
        catalog: str,
        schema: str,
        table: str,
        warehouse_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Convert natural language question to SQL and execute
        
        Args:
            question: Natural language question
            catalog: Catalog name
            schema: Schema name
            table: Table name
            warehouse_id: Optional warehouse ID
        
        Returns:
            Dictionary with 'sql', 'data' (DataFrame), and 'explanation'
        """
        if not self.anthropic_client:
            raise ValueError("Anthropic client not initialized. Provide anthropic_api_key.")
        
        # Get table schema
        table_info = self.get_table_info(catalog, schema, table)
        
        schema_text = "\n".join([
            f"- {col['name']} ({col['type_text'] or col['type_name']}): {col['comment'] or 'No description'}"
            for col in table_info['columns']
        ])
        
        # Generate SQL using Claude
        prompt = f"""Convert this natural language question to a SQL query for Databricks Delta Lake.

Table: {catalog}.{schema}.{table}
Description: {table_info['comment'] or 'No description'}

Schema:
{schema_text}

Question: {question}

Provide only the SQL query without any explanation or markdown formatting."""
        
        message = self.anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        
        sql_query = message.content[0].text.strip()
        
        # Remove markdown code blocks if present
        if sql_query.startswith("```"):
            lines = sql_query.split("\n")
            sql_query = "\n".join(lines[1:-1]) if len(lines) > 2 else sql_query
        
        # Execute query
        df = self.execute_sql(sql_query, warehouse_id)
        
        return {
            "sql": sql_query,
            "data": df,
            "question": question
        }
    
    def create_chart(
        self,
        query: str = None,
        data: pd.DataFrame = None,
        chart_type: str = "bar",
        x: Optional[str] = None,
        y: Optional[str] = None,
        title: str = "Chart",
        **kwargs
    ) -> go.Figure:
        """
        Create a Plotly chart from query results or DataFrame
        
        Args:
            query: SQL query (if data not provided)
            data: DataFrame (if query not provided)
            chart_type: Type of chart (bar, line, scatter, pie, histogram, box)
            x: X-axis column name
            y: Y-axis column name
            title: Chart title
            **kwargs: Additional arguments passed to Plotly
        
        Returns:
            Plotly Figure object
        """
        # Get data
        if data is None and query:
            data = self.execute_sql(query)
        elif data is None:
            raise ValueError("Must provide either 'query' or 'data'")
        
        if data.empty:
            raise ValueError("No data to chart")
        
        # Auto-detect columns if not specified
        if not x and len(data.columns) > 0:
            x = data.columns[0]
        if not y and len(data.columns) > 1:
            y = data.columns[1]
        
        # Create chart
        if chart_type == "bar":
            fig = px.bar(data, x=x, y=y, title=title, **kwargs)
        elif chart_type == "line":
            fig = px.line(data, x=x, y=y, title=title, **kwargs)
        elif chart_type == "scatter":
            fig = px.scatter(data, x=x, y=y, title=title, **kwargs)
        elif chart_type == "pie":
            fig = px.pie(data, names=x, values=y, title=title, **kwargs)
        elif chart_type == "histogram":
            fig = px.histogram(data, x=x, title=title, **kwargs)
        elif chart_type == "box":
            fig = px.box(data, y=y, title=title, **kwargs)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
        
        return fig
    
    def explore_catalog(self, catalog: str = None) -> Dict[str, Any]:
        """
        Get a complete overview of a catalog or all catalogs
        
        Args:
            catalog: Specific catalog to explore (None for all)
        
        Returns:
            Dictionary with catalog structure
        """
        if catalog:
            schemas = self.list_schemas(catalog)
            structure = {
                "catalog": catalog,
                "schemas": {}
            }
            
            for _, schema_row in schemas.iterrows():
                schema_name = schema_row['name']
                tables = self.list_tables(catalog, schema_name)
                structure["schemas"][schema_name] = {
                    "comment": schema_row['comment'],
                    "tables": tables['name'].tolist()
                }
            
            return structure
        else:
            catalogs = self.list_catalogs()
            return {
                "catalogs": [
                    {
                        "name": row['name'],
                        "comment": row['comment'],
                        "owner": row['owner']
                    }
                    for _, row in catalogs.iterrows()
                ]
            }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Usage Examples

# COMMAND ----------

# Initialize the client
client = NotebookMCPClient(
    warehouse_id=DATABRICKS_WAREHOUSE_ID,
    anthropic_api_key=ANTHROPIC_API_KEY
)

# COMMAND ----------

# Example 1: List all catalogs
print("=== Unity Catalogs ===")
catalogs_df = client.list_catalogs()
display(catalogs_df)

# COMMAND ----------

# Example 2: List schemas in a catalog
catalog_name = "book_ai_ml_lakehouse"
print(f"=== Schemas in {catalog_name} ===")
schemas_df = client.list_schemas(catalog_name)
display(schemas_df)

# COMMAND ----------

# Example 3: List tables in a schema
# Replace with your catalog and schema names
catalog_name = "book_ai_ml_lakehouse"
schema_name = "gold"

tables_df = client.list_tables(catalog_name, schema_name)

if tables_df.empty:
    print(f"No tables found in {catalog_name}.{schema_name}")
else:
    print(f"=== Tables in {catalog_name}.{schema_name} ===")
    display(tables_df)

# COMMAND ----------

# Example 4: Get detailed table information
# Replace with your table details
catalog_name = "book_ai_ml_lakehouse"
schema_name = "gold"
table_name = "monthly_engagement"

try:
    table_info = client.get_table_info(catalog_name, schema_name, table_name)
    print(f"=== Table Info: {catalog_name}.{schema_name}.{table_name} ===")
    print(json.dumps(table_info, indent=2))
except Exception as e:
    print(f"Table not found or error: {e}")

# COMMAND ----------

# Example 5: Execute SQL query
query = """
SELECT pickup_zip, COUNT(*) as trip_count
FROM samples.nyctaxi.trips
GROUP BY pickup_zip
ORDER BY trip_count DESC
LIMIT 10
"""

try:
    result_df = client.execute_sql(query)
    print(f"Query returned {len(result_df)} rows")
    display(result_df)
except Exception as e:
    print(f"Error executing query: {e}")

# COMMAND ----------

# Example 6: Natural language query
# Requires ANTHROPIC_API_KEY to be set
try:
    result = client.query_natural_language(
        question="What are the top 10 pickup zip codes by number of trips?",
        catalog="samples",
        schema="nyctaxi",
        table="trips"
    )
    
    print("=== Generated SQL ===")
    print(result['sql'])
    print("\n=== Results ===")
    display(result['data'])
except Exception as e:
    print(f"Error with NL query: {e}")

# COMMAND ----------

# Example 7: Create a chart
query = """
SELECT category, COUNT(*) as count
FROM main.default.your_table
GROUP BY category
ORDER BY count DESC
LIMIT 10
"""

try:
    fig = client.create_chart(
        query=query,
        chart_type="bar",
        x="category",
        y="count",
        title="Top 10 Categories",
        color="category"
    )
    fig.show()
except Exception as e:
    print(f"Error creating chart: {e}")

# COMMAND ----------

# Bar Chart — Top 10 Pickup Zip Codes by Trip Count
fig = client.create_chart(
    query="""
        SELECT pickup_zip, COUNT(*) as trip_count
        FROM samples.nyctaxi.trips
        WHERE pickup_zip IS NOT NULL
        GROUP BY pickup_zip
        ORDER BY trip_count DESC
        LIMIT 10
    """,
    chart_type="bar",
    x="pickup_zip",
    y="trip_count",
    title="Top 10 Pickup Zip Codes by Trip Count",
    color="pickup_zip"
)
fig.show()

# COMMAND ----------

# Line Chart — Daily Trip Count Over Time
fig = client.create_chart(
    query="""
        SELECT DATE(tpep_pickup_datetime) as trip_date, COUNT(*) as trips
        FROM samples.nyctaxi.trips
        GROUP BY DATE(tpep_pickup_datetime)
        ORDER BY trip_date
        LIMIT 30
    """,
    chart_type="line",
    x="trip_date",
    y="trips",
    title="Daily Trip Count (First 30 Days)"
)
fig.show()

# COMMAND ----------

# Pie Chart — Trip Distribution by Top 5 Pickup Zip Codes
fig = client.create_chart(
    query="""
        SELECT CAST(pickup_zip AS STRING) as zip, COUNT(*) as trips
        FROM samples.nyctaxi.trips
        WHERE pickup_zip IS NOT NULL
        GROUP BY pickup_zip
        ORDER BY trips DESC
        LIMIT 5
    """,
    chart_type="pie",
    x="zip",
    y="trips",
    title="Trip Distribution by Top 5 Pickup Zips"
)
fig.show()

# COMMAND ----------

# Scatter Plot — Trip Distance vs Fare Amount
fig = client.create_chart(
    query="""
        SELECT trip_distance, fare_amount
        FROM samples.nyctaxi.trips
        WHERE trip_distance > 0 AND fare_amount > 0
        LIMIT 500
    """,
    chart_type="scatter",
    x="trip_distance",
    y="fare_amount",
    title="Trip Distance vs Fare Amount"
)
fig.show()

# COMMAND ----------

# Example 9: Explore entire catalog structure
catalog_to_explore = "samples"

try:
    structure = client.explore_catalog(catalog_to_explore)
    print(f"=== Catalog Structure: {catalog_to_explore} ===")
    print(json.dumps(structure, indent=2))
except Exception as e:
    print(f"Error exploring catalog: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced Usage: Interactive Querying

# COMMAND ----------

def interactive_query(client: NotebookMCPClient, catalog: str, schema: str, table: str):
    """
    Interactive helper function for querying tables with natural language
    """
    
    # Get table info
    table_info = client.get_table_info(catalog, schema, table)
    
    print(f"📊 Table: {catalog}.{schema}.{table}")
    print(f"📝 Description: {table_info.get('comment', 'No description')}")
    print(f"\n📋 Columns ({len(table_info['columns'])}):")
    
    for col in table_info['columns']:
        print(f"  • {col['name']} ({col['type_text'] or col['type_name']})")
        if col['comment']:
            print(f"    └─ {col['comment']}")
    
    print("\n" + "="*80)
    print("💡 You can now ask questions about this table using natural language!")
    print("="*80)

# Usage
interactive_query(client, "samples", "accuweather", "forecast_daily_calendar_imperial")

# COMMAND ----------

# MAGIC %md
# MAGIC ## End of Notebook
# MAGIC
# MAGIC You now have a complete toolkit for:
# MAGIC - Exploring Unity Catalog
# MAGIC - Executing SQL and natural language queries  
# MAGIC - Creating visualizations
# MAGIC - Analyzing data
# MAGIC
# MAGIC Feel free to customize and extend these examples for your specific use cases!
