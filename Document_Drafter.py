from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import os

load_dotenv("D:\PYTHON\Agentic AI\.ENV FILES\.env")

document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update(content: str) -> str:
    """This function will update the current document as per the user input"""
    global document_content
    document_content = content
    return f"Document updated: {document_content}"

@tool
def save(filename: str) -> str:
    """This function will save the updated document as per the user input"""
    global document_content
    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"
    try:
        with open(filename, 'w') as file:
            file.write(document_content)
        print(f"\n💾 Saved to: {filename}")
        return f"Saved to '{filename}'."
    except Exception as e:
        return f"Error saving: {str(e)}"

tools = [update, save]

model = ChatGroq(model="mixtral-8x7b-32768", api_key=os.getenv("GROQ_API_KEY")).bind_tools(tools)

def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a writing assistant. Help the user create and modify documents.
    - Use 'update' tool to modify content.
    - Use 'save' tool to save and finish.
    - Always show current document content after changes.
    Current content: {document_content}
    """)
    if not state["messages"]:
        user_input = "What would you like to create?"
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\nWhat would you like to do? ")
        print(f"\n👤 User: {user_input}")
        user_message = HumanMessage(content=user_input)
    all_messages = [system_prompt] + list(state["messages"]) + [user_message]
    response = model.invoke(all_messages)
    print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 Tools: {[tc['name'] for tc in response.tool_calls]}")
    return {"messages": list(state["messages"]) + [user_message, response]}

def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    if not messages:
        return "continue"
    for message in reversed(messages):
        if (isinstance(message, ToolMessage) and 
            "saved" in message.content.lower() and
            "document" in message.content.lower()):
            return "end"
    return "continue"

def print_messages(messages):
    if not messages:
        return
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ Tool result: {message.content}")

graph = StateGraph(AgentState)
graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("agent")
graph.add_edge("agent", "tools")
graph.add_conditional_edges(
    "tools",
    should_continue,
    {"continue": "agent", "end": END}
)
app = graph.compile()

def run_document_agent():
    print("\n===== DRAFTER =====")
    state = {"messages": []}
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    print("\n===== DRAFTER FINISHED =====")

if __name__ == "__main__":
    run_document_agent()