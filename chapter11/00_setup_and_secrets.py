# Databricks notebook source
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
# MAGIC ## Step 8: Validate Secrets from the Notebook
# MAGIC
# MAGIC Run the cell below to confirm the notebook can read all four secrets.
# MAGIC Values are intentionally not printed in full — only their lengths are shown
# MAGIC to verify they were stored correctly without exposing sensitive data.

# COMMAND ----------

import os

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

# Load secrets — falls back to environment variables for local development
def _get_secret(scope: str, key: str, env_var: str) -> str:
    """Read from Databricks secrets with environment variable fallback."""
    try:
        return dbutils.secrets.get(scope=scope, key=key)
    except Exception:
        return os.getenv(env_var, "")

DATABRICKS_HOST         = _get_secret("databricks-mcp", "host",              "DATABRICKS_HOST")
DATABRICKS_TOKEN        = _get_secret("databricks-mcp", "token",             "DATABRICKS_TOKEN")
DATABRICKS_WAREHOUSE_ID = _get_secret("databricks-mcp", "warehouse-id",      "DATABRICKS_WAREHOUSE_ID")
ANTHROPIC_API_KEY       = _get_secret("databricks-mcp", "anthropic-api-key", "ANTHROPIC_API_KEY")

print("Configuration loaded. Ready to initialize NotebookMCPClient.")
