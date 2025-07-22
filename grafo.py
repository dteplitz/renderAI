import logging
from langgraph.graph import StateGraph, MessagesState, START, END
from config import get_sqlite_checkpointer
from node_groq import node_groq

# Configure logging for this module
logger = logging.getLogger(__name__)

logger.info("Building LangGraph...")

# Build graph
builder = StateGraph(state_schema=MessagesState)
builder.add_node("chat", node_groq)
builder.set_entry_point("chat")
builder.set_finish_point("chat")  # End after one execution

logger.info("Compiling graph with checkpointing...")

# Compile with checkpointing
try:
    checkpointer = get_sqlite_checkpointer()
    graph_app = builder.compile(checkpointer=checkpointer)
    logger.info("Graph compiled successfully with checkpointing")
except Exception as e:
    logger.error(f"Error compiling graph: {str(e)}", exc_info=True)
    raise
