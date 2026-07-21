from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessageChunk
from dental_agent.workflows.graph import dental_graph

BANNER = """
        DENTAL APPOINTMENT MANAGEMENT SYSTEM
"""

def run():
    print(BANNER)
    history = []
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("/nGoodbye!")
            break
        
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "bye"}:
            print("Goodbye!")
            break
        
        history.append(HumanMessage(content=user_input))
        
        print("\nAgent: ", end="" , flush= True)
        final_message = None
        
        try:
            for event_type, data in dental_graph.stream(
                {"messages": history},
                stream_mode=["messages", "values"],
            ):
                if event_type == "messages":
                    chunk , meta = data
                    if(
                        isinstance(chunk , AIMessageChunk)
                        and chunk.content 
                        and not getattr(chunk, "tool_calls", None)
                    ):
                        print(chunk.content, end="", flush=True)
                elif event_type == "values":
                    final_messages = data.get("messages", [])
        except Exception as e:
            print(f"/nError: {e}")
            history.pop()
            continue
        
        print()
        if final_messages:
            history = final_messages
            
if __name__ == "__main__":
    run()
    
    
