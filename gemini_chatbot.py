import os
import chainlit as cl
import google.generativeai as genai
from langgraph.graph import StateGraph, MessagesState
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain.tools import tool
import datetime
import asyncio

# Load API Key from Environment Variables
load_dotenv()
gemini_api_key = os.getenv("GEMNI")

if not gemini_api_key:
    raise ValueError("⚠️ Gemini API Key not found! Please check your environment variables.")

# Configure Google Gemini AI
genai.configure(api_key=gemini_api_key)

# Define a tool to execute Python code
@tool
def execute_python(code: str) -> str:
    """Executes Python code and returns the result."""
    try:
        exec_globals = {}
        exec(code, exec_globals)
        output = exec_globals.get("result", "✅ Code executed successfully, but no output was returned.")
        return f"🛠️ **Executed Code Output:**\n```\n{output}\n```"
    except Exception as e:
        return f"❌ **Execution Error:**\n```\n{e}\n```"

# Function to handle AI-generated code
def generate_code(prompt: str) -> str:
    """Generates high-quality Python code using Gemini AI."""
    model = genai.GenerativeModel("gemini-1.5-ultra")  # Use Ultra for high intelligence
    system_prompt = """You are an expert Python developer. Generate optimized, error-free, well-commented Python code."""

    try:
        response = model.generate_content([system_prompt, prompt])
        return response.text if response.text else "❌ **Error:** AI couldn't generate code."
    except Exception as e:
        return f"❌ **Error in AI Code Generation:** {e}"

# Function to handle AI conversation
def gemini_response(state: MessagesState):
    """Handles user queries intelligently, including code generation and execution."""
    conversation = state["messages"]
    last_message = conversation[-1]

    if isinstance(last_message, HumanMessage):
        user_input = last_message.content
    else:
        raise ValueError("Expected the last message to be from a HumanMessage")

    try:
        # Code generation request
        if "generate code" in user_input.lower() or "write python" in user_input.lower():
            generated_code = generate_code(user_input)
            return {"messages": [AIMessage(content=f"📝 **Generated Python Code:**\n```python\n{generated_code}\n```")]}

        # AI Assistant Response
        system_prompt = """
        You are the most advanced AI assistant. Your capabilities include:
        - Answering **all types of questions** accurately.
        - **Generating & executing Python code** with explanations.
        - Providing **real-time factual information**.
        - Engaging in **detailed conversations**.
        - Thinking critically and giving **practical examples**.
        Always **analyze** user queries before responding. Be **concise** if needed, but provide **detailed** responses when required.
        """
        
        model = genai.GenerativeModel("gemini-1.5-flash")  # Use Ultra for high accuracy
        response = model.generate_content([system_prompt, user_input])

        if not response or not response.text:
            return {"messages": [AIMessage(content="❌ **Error:** AI couldn't generate a response. Please try again!")]}

        return {"messages": [AIMessage(content=response.text)]}

    except Exception as e:
        return {"messages": [AIMessage(content=f"❌ **Unexpected Error:** {e}")]}

# Create LangGraph chatbot state graph
graph = StateGraph(MessagesState)
graph.add_node("gemini", gemini_response)
graph.set_entry_point("gemini")
compiled_graph = graph.compile()

# Chainlit UI Setup
@cl.on_chat_start
async def start_chat():
    """Triggered when a user starts a new chat session."""
    welcome_message = (
        "**🤖 Welcome to the Ultimate AI Chatbot!**\n\n"
        "💡 **Ask me anything!** I can:\n"
        "✅ Answer **complex questions**\n"
        "✅ Generate & **execute Python code**\n"
        "✅ Provide **real-time knowledge**\n"
        "✅ Engage in **detailed conversations**\n\n"
        "**Just type your query below!** 🚀"
    )
    
    await cl.Message(content=welcome_message).send()

@cl.on_message
async def chat_with_gemini(message):
    """Handles user messages and AI responses."""
    
    try:
        # Handle real-time requests (e.g., asking for time)
        if "time" in message.content.lower():
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            await cl.Message(content=f"🕒 **Current Time:** {current_time}").send()
            return

        # Show "Thinking..." message
        thinking_message = await cl.Message(content="🤖 *Thinking...*").send()
        await asyncio.sleep(1.5)  # Short delay for natural response timing

        # Process user input and ensure AI always responds
        response = compiled_graph.invoke({"messages": [HumanMessage(content=message.content)]})

        if response and "messages" in response and response["messages"]:
            await cl.Message(content=f"💬 {response['messages'][-1].content}").send()
        else:
            await cl.Message(content="❌ **Error:** AI failed to respond. Please try again!").send()

    except Exception as e:
        await cl.Message(content=f"❌ **Critical Error:** {e}").send()

# Run the Chainlit app
if __name__ == "__main__":
    cl.run()
