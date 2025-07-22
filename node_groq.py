import logging
from groq import Groq
from langchain.prompts import ChatPromptTemplate
from config import GROQ_API_KEY

# Configure logging for this module
logger = logging.getLogger(__name__)

# Configurable LLM setup
def get_node_groq_llm(model="llama3-8b-8192", temperature=0.4, max_tokens=200):
    logger.info(f"Creating Groq client with model: {model}")
    return Groq(api_key=GROQ_API_KEY)

# Prompt to define the role of the node
#prompt = ChatPromptTemplate.from_template(
#    """
#    Sos un asesor financiero que esta ayudando a un usuario a planificar su retiro.
#    El usuario tiene que decidir cual es el objetivo monetario final de su retiro.
#    Cuando decida la cantidad de plata, lo ingresara en el formulario que tiene en pantalla.
#    ---
#    {input}
#    """
#)
prompt = ChatPromptTemplate.from_template(
    "{input}"
)

def node_groq(messages):
    logger.info(f"node_groq called with messages type: {type(messages)}")
    logger.info(f"Messages content: {messages}")
    
    try:
        # Handle both dict and list formats
        if isinstance(messages, dict) and "messages" in messages:
            # If it's a dict with "messages" key, extract the list
            messages_list = messages["messages"]
            logger.info(f"Extracted messages list: {messages_list}")
        elif isinstance(messages, list):
            # If it's already a list
            messages_list = messages
            logger.info(f"Messages is already a list: {messages_list}")
        else:
            logger.error(f"Unexpected messages format: {type(messages)} - {messages}")
            raise ValueError(f"Unexpected messages format: {type(messages)}")
        
        if not messages_list:
            logger.error("No messages in list")
            raise ValueError("No messages provided")
        
        last_message = messages_list[-1]
        logger.info(f"Last message: {last_message}")
        
        client = get_node_groq_llm()
        logger.info("Groq client created successfully")
        
        # Extraer el historial completo como texto para el prompt
        historial = "\n".join(
            f"{msg['role'] if isinstance(msg, dict) else getattr(msg, 'role', 'user')}: {msg['content'] if isinstance(msg, dict) else getattr(msg, 'content', '')}"
            for msg in messages_list
        )
        formatted_prompt = prompt.format_messages(input=historial)
        logger.info(f"Historial para el prompt: {historial}")
        logger.info(f"Formatted prompt: {formatted_prompt}")
        
        # Convert LangChain messages to Groq format
        groq_messages = []
        for msg in formatted_prompt:
            # Map LangChain message types to Groq roles
            if msg.type == "system":
                role = "system"
            elif msg.type == "human":
                role = "user"
            elif msg.type == "ai":
                role = "assistant"
            else:
                role = "user"  # Default to user
            
            groq_messages.append({
                "role": role,
                "content": msg.content
            })
        
        logger.info(f"Groq messages: {groq_messages}")
        
        logger.info("Calling Groq API...")
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=groq_messages,
            temperature=0.4,
            max_tokens=200
        )
        
        logger.info(f"Groq response received: {response.choices[0].message.content[:100]}...")
        
        # Convert response back to LangChain format
        from langchain.schema import AIMessage
        ai_message = AIMessage(content=response.choices[0].message.content)
        
        result = {"messages": messages_list + [ai_message]}
        logger.info(f"Returning result with {len(result['messages'])} messages")
        return result
        
    except Exception as e:
        logger.error(f"Error in node_groq: {str(e)}", exc_info=True)
        raise

