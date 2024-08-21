import os
from datetime import datetime, timedelta
from typing import Annotated, Literal, TypedDict, List
from dotenv import load_dotenv
from mem0 import Memory
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
import json

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables")

MEMORY_CONFIG = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "my_collection",
            "path": "chroma_data",
        }
    }
}

# Initialize memory
mem0 = Memory.from_config(MEMORY_CONFIG)

# Helper functions
def parse_date(date_string: str) -> datetime:
    """Parse a date string into a datetime object."""
    try:
        return datetime.strptime(date_string, "%Y-%m-%d %H:%M")
    except ValueError:
        return datetime.strptime(date_string, "%Y-%m-%d")

# Tool definitions
@tool
def get_current_date() -> str:
    """Get the current date."""
    return datetime.now().strftime("%Y-%m-%d")

@tool
def get_day_of_week(date_string: str) -> str:
    """Determine the day of the week for a given date."""
    try:
        date = datetime.strptime(date_string, "%Y-%m-%d")
        return date.strftime("%A")
    except ValueError:
        return "Invalid date format. Please use YYYY-MM-DD."

@tool
def search_memories(query: str, email: str) -> str:
    """Search for past memories and schedule items based on a query and email."""
    memories = mem0.search(query, user_id=email)
    return str(memories)

@tool
def add_schedule_item(email: str, date_time: str, duration: str, description: str) -> str:
    """Add a new item to the user's schedule."""
    try:
        parsed_date_time = parse_date(date_time)
        parsed_duration = timedelta(minutes=int(duration))
        
        schedule_item = {
            "date_time": parsed_date_time.isoformat(),
            "duration": str(parsed_duration),
            "description": description
        }
        
        # Store the schedule item in mem0
        mem0.add(f"Schedule: {json.dumps(schedule_item)}", user_id=email)
        
        return f"Added to schedule: {description} on {parsed_date_time.strftime('%Y-%m-%d %H:%M')} for {duration} minutes"
    except ValueError:
        return "Invalid date, time, or duration format. Please try again."

@tool
def get_schedule(email: str, start_date: str, end_date: str) -> str:
    """Retrieve schedule items for a given date range."""
    try:
        start = parse_date(start_date)
        end = parse_date(end_date)
        
        # Search for schedule items in mem0
        query = f"Schedule: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
        schedule_items = mem0.search(query, user_id=email)
        
        if not schedule_items:
            return "No schedule items found for the given date range."
        
        return str(schedule_items)
    except ValueError:
        return "Invalid date format. Please use YYYY-MM-DD or YYYY-MM-DD HH:MM."

# Set up tools and model
tools = [get_current_date, get_day_of_week, search_memories, add_schedule_item, get_schedule]
tool_node = ToolNode(tools)
model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

# State definition
class State(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], add_messages]
    email: str

# Workflow functions
def should_continue(state: State) -> Literal["tools", END]:
    last_message = state['messages'][-1]
    return "tools" if last_message.tool_calls else END

def call_model(state: State):
    messages = state['messages']
    email = state['email']
    
    system_message_content = f"""You are a helpful assistant with access to the user's past interactions, schedule, and awareness of the current date. 
    The user's email is {email}. Before responding to the user, always follow these steps in order:

    1. Use the get_current_date tool to get today's date.
    2. Use the search_memories tool to retrieve relevant past interactions and schedule items. The search_memories tool takes two arguments: 
       the query (which should be the user's latest message) and the user's email.
    3. If the user asks about a specific date or day of the week, use the get_day_of_week tool to determine it.
    4. If the user wants to add a schedule item, use the add_schedule_item tool. It requires email, date_time, duration (in minutes), and description.
    5. If the user asks about their schedule for a specific period, use the get_schedule tool. It requires email, start_date, and end_date.

    Always use these tools when appropriate, even if you think there might not be relevant information. If no relevant data is found, 
    acknowledge this in your response and proceed based on your general knowledge.

    Incorporate the retrieved information, date awareness, schedule details, and any relevant memories into your responses to provide 
    personalized and context-aware answers. Be sure to reference past interactions and schedule items when appropriate, and use the 
    current date information to make your responses more relevant and timely.

    Remember to always use these tools in the order specified above before formulating your response."""

    full_messages = [SystemMessage(content=system_message_content)] + messages
    response = model.invoke(full_messages)
    
    # Store the interaction in Mem0
    mem0.add(f"User: {messages[-1].content}\nAssistant: {response.content}", user_id=email)
    
    return {"messages": [response]}

# Set up workflow
workflow = StateGraph(State)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", 'agent')

# Compile app
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# Conversation runner
def run_conversation(user_input: str, email: str):
    config = {"configurable": {"thread_id": email}}
    state = {"messages": [HumanMessage(content=user_input)], "email": email}
    final_state = app.invoke(state, config=config)
    return final_state["messages"][-1].content

# Gradio interface
def chat_interface(message, history, email):
    if not email:
        return "Please enter your email address before starting the chat."
    return run_conversation(message, email)

# Gradio app setup
with gr.Blocks() as demo:
    with gr.Column(scale=1):
        gr.Markdown("# Personal Assistant")
        email_input = gr.Textbox(label="Enter Your Email", placeholder="e.g., user@example.com")
    
    with gr.Column(scale=4):
        chatbot = gr.ChatInterface(
            fn=chat_interface,
            additional_inputs=[email_input],
            chatbot=gr.Chatbot(height=400),
            title="",
            description="",
        )

if __name__ == "__main__":
    demo.launch()