import sqlite3
import os
from typing import List, Tuple, Any, Union

# Resolve absolute project root path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))            # agent/tools
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))        # your_project/


class SQLiteTool:
    def __init__(self, db_path: str, debug: bool = False):
        """
        Initialize connection to SQLite database using safe absolute path.
        """
        self.db_path = os.path.join(PROJECT_ROOT, db_path)
        self.debug = debug

        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database not found at: {self.db_path}")

        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def execute_query(self, query: str) -> Union[List[dict], dict]:
        """
        Execute SQL query and return list of rows as dicts.
        Returns {'error': str} if the query fails.
        Automatic fallback for missing 'CostOfGoods' column.
        """
        try:
            # Clean the query first
            query = query.strip()
            
            # Remove any accidental wrapping
            if query.startswith('[') and query.endswith(']'):
                query = query[1:-1].strip()
            if query.startswith('"') and query.endswith('"'):
                query = query[1:-1].strip()
            if query.startswith("'") and query.endswith("'"):
                query = query[1:-1].strip()
            
            cursor = self.conn.cursor()
            if self.debug:
                print(f"[SQLiteTool] Executing SQL:\n{query}\n")
            
            cursor.execute(query)
            rows = [dict(row) for row in cursor.fetchall()]

            # Fallback for missing CostOfGoods
            if rows and len(rows) > 0:
                first_row_keys = rows[0].keys()
                if "CostOfGoods" not in first_row_keys:
                    if self.debug:
                        print("[SQLiteTool] 'CostOfGoods' missing, applying fallback: 0.7*UnitPrice")
                    for row in rows:
                        if "UnitPrice" in row:
                            row["CostOfGoods"] = row.get("UnitPrice", 0) * 0.7

            cursor.close()
            return rows
            
        except sqlite3.Error as e:
            error_msg = f"SQLite error: {e}"
            if self.debug:
                print(error_msg)
                print(f"Query was: {query}")

            # Try to detect if CostOfGoods column is missing and emulate fallback
            if "no such column: CostOfGoods" in str(e):
                if self.debug:
                    print("[SQLiteTool] Detected missing 'CostOfGoods', trying fallback query...")
                # naive fallback: replace CostOfGoods with 0.7*UnitPrice in query
                fallback_query = query.replace("CostOfGoods", "(UnitPrice*0.7) AS CostOfGoods")
                return self.execute_query(fallback_query)

            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

    def get_tables(self) -> List[str]:
        """
        Return list of table names in the database.
        """
        tables = self.execute_query("SELECT name FROM sqlite_master WHERE type='table';")
        if isinstance(tables, dict) and "error" in tables:
            return []
        return [row["name"] for row in tables]

    def get_schema(self, table_name: str) -> List[Tuple[str, str]]:
        """
        Return list of (column_name, column_type) tuples for a given table.
        """
        try:
            cursor = self.conn.cursor()
            # Handle table names with spaces
            if ' ' in table_name:
                cursor.execute(f'PRAGMA table_info("{table_name}");')
            else:
                cursor.execute(f"PRAGMA table_info({table_name});")
            columns = [(row["name"], row["type"]) for row in cursor.fetchall()]
            cursor.close()
            return columns
        except sqlite3.Error as e:
            if self.debug:
                print(f"SQLite schema error for table {table_name}: {e}")
            return []

    def column_exists(self, table_name: str, column_name: str) -> bool:
        """
        Check if a specific column exists in the given table.
        """
        schema = self.get_schema(table_name)
        return any(col == column_name for col, _ in schema)

    def close(self):
        """
        Close the database connection.
        """
        if self.conn:
            self.conn.close()
            if self.debug:
                print("[SQLiteTool] Connection closed.")