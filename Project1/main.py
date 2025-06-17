from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq # allows us to use Groq within langchain and laggraph.
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent # complex framework that allows us three to build AI agents.
from dotenv import load_dotenv # allows us to load our .env file within python script.

load_dotenv()

def main():
    model = ChatGroq(model="llama3-8b-8192", temperature=0)


    tools = []
    agent_executor = create_react_agent(model, tools)

    print("Welcome! I'm your assistant.")
    print("How can I help you today?")
    print("Type 'quit' to exit.")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if user_input.lower() == "quit":
            break

        print("\nAssistant: ", end="")
        for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content=user_input)]}
        ):
            if "agent" in chunk and "messages" in chunk["agent"]:
                for message in chunk["agent"]["messages"]:
                    print(message.content, end="")
        print()

if __name__ == "__main__":
    main()
