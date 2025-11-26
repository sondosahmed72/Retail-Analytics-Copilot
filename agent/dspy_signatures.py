import dspy
from typing import Optional, Any

# -----------------------------------------------------
# Router Signature
# -----------------------------------------------------
class RouterSignature(dspy.Signature):
    """Classify if question needs: rag (docs only), sql (DB only), or hybrid (both)."""
    question: str = dspy.InputField(desc="User's retail analytics question")
    route: str = dspy.OutputField(desc="One of: rag, sql, hybrid")


# -----------------------------------------------------
# NL → SQL Signature
# -----------------------------------------------------
class NL2SQLSignature(dspy.Signature):
    """Generate SQLite query from natural language question and schema."""
    question: str = dspy.InputField(desc="User's question requiring SQL")
    db_schema: str = dspy.InputField(desc="Database schema information")
    sql_query: str = dspy.OutputField(desc="Valid SQLite query")


# -----------------------------------------------------
# SQL Analysis Signature
# -----------------------------------------------------
class SQLAnalysisSignature(dspy.Signature):
    """Analyze SQL results and answer the question."""
    question: str = dspy.InputField(desc="Original question")
    sql_result: str = dspy.InputField(desc="SQL query results")
    answer: str = dspy.OutputField(desc="Extracted answer from SQL results")


# -----------------------------------------------------
# RAG Answer Signature
# -----------------------------------------------------
class RAGAnswerSignature(dspy.Signature):
    """Answer question using retrieved document context."""
    question: str = dspy.InputField(desc="User's question")
    context: str = dspy.InputField(desc="Retrieved document chunks")
    answer: str = dspy.OutputField(desc="Answer based on context")


# -----------------------------------------------------
# Hybrid Synthesizer Signature
# -----------------------------------------------------
class HybridSynthesizerSignature(dspy.Signature):
    """Synthesize final answer from both SQL and RAG results, matching format_hint."""
    question: str = dspy.InputField(desc="Original question")
    sql_answer: str = dspy.InputField(desc="Answer from SQL analysis")
    rag_answer: str = dspy.InputField(desc="Answer from RAG")
    format_hint: str = dspy.InputField(desc="Expected output format (int, float, dict, list)")
    answer: str = dspy.OutputField(desc="Final answer in requested format")


# -----------------------------------------------------
# Helper function for router
# -----------------------------------------------------
def router_fn(question: str, lm) -> str:
    """
    Route question to: rag, sql, or hybrid.
    Returns one of: 'rag', 'sql', 'hybrid'
    Uses simple text-based approach to avoid structured output issues.
    """
    # Simple heuristic-based router for reliability
    question_lower = question.lower()
    
    # RAG indicators
    if any(word in question_lower for word in ['policy', 'return window', 'days', 'according to']):
        if 'during' not in question_lower and 'summer' not in question_lower and 'winter' not in question_lower:
            return 'rag'
    
    # SQL indicators
    if any(word in question_lower for word in ['top', 'revenue', 'all-time', 'products by']):
        if 'during' not in question_lower and 'summer' not in question_lower and 'winter' not in question_lower:
            return 'sql'
    
    # Hybrid indicators (needs both docs and DB)
    if any(word in question_lower for word in ['during', 'summer', 'winter', 'kpi', 'margin', 'aov']):
        return 'hybrid'
    
    # Default to hybrid for safety
    return 'hybrid'