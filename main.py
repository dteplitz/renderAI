from fastapi import FastAPI
import logging
from pydantic import BaseModel
from fastapi import HTTPException
from grafo import graph_app
from fastapi import Request

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Orquestador IA API",
    description="API para gestionar consultas a modelos de IA usando MCP y LangGraph",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

class Message(BaseModel):
    thread_id: str
    content: str

class PreguntaUsuario(BaseModel):
    pregunta: str

@app.get("/test")
def read_root():
    return {"mensaje": "Hola mundo"}

@app.get("/", summary="Health Check", tags=["Health"])
async def health_check():
    """
    **Endpoint de verificación de salud de la API.**
    
    - **Devuelve:** Estado de la API
    """
    logger.info("Health check endpoint called")
    return {"status": "OK", "message": "Orquestador IA API is running"}

@app.post("/chat", summary="Chat con IA", tags=["Chat"])
async def chat_endpoint(msg: Message):
    """
    **Endpoint para chat con el modelo de IA.**
    
    - **Recibe:** Un JSON con thread_id y content
    - **Procesa:** La consulta en el flujo de LangGraph
    - **Devuelve:** La respuesta generada por el modelo de IA
    
    **Ejemplo de JSON de entrada:**
    ```json
    {
        "thread_id": "123",
        "content": "¿Qué es la inteligencia artificial?"
    }
    ```
    """
    logger.info(f"Chat endpoint called with thread_id: {msg.thread_id}, content: {msg.content[:100]}...")
    
    try:
        user_msg = {"role": "user", "content": msg.content}
        config = {"configurable": {"thread_id": msg.thread_id}}
        
        logger.info(f"Invoking graph_app with messages: {user_msg}")
        logger.info(f"Config: {config}")
        
        result = graph_app.invoke({"messages": [user_msg]}, config=config)
        
        logger.info(f"Graph result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        logger.info(f"Result type: {type(result)}")
        
        if isinstance(result, dict) and "messages" in result and len(result["messages"]) > 0:
            last_message = result["messages"][-1]
            logger.info(f"Last message type: {type(last_message)}")
            logger.info(f"Last message: {last_message}")
            
            # Extract content from AIMessage object
            if hasattr(last_message, 'content'):
                response_content = last_message.content
            elif isinstance(last_message, dict) and "content" in last_message:
                response_content = last_message["content"]
            else:
                logger.error(f"Cannot extract content from message: {last_message}")
                raise HTTPException(status_code=500, detail="Cannot extract content from response")
            
            logger.info(f"Response content: {response_content[:100]}...")
            return {"response": response_content}
        else:
            logger.error(f"Unexpected result structure: {result}")
            raise HTTPException(status_code=500, detail="Unexpected response structure from graph")
            
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error en el chat: {str(e)}")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log all requests"""
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code}")
    return response