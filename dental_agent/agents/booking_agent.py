from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode
from dental_agent.config.settings import GROQ_API_KEY, MODEL_NAME, TEMPERATURE
from dental_agent.models.state import AppointmentState
from dental_agent.tools.csv_reader import get_available_slots, check_slot_availability
from dental_agent.tools.csv_writer import book_appointment
from dental_agent.utils import guard_tool_response, log_transition, sanitize_messages

BOOKING_TOOLS = [get_available_slots, check_slot_availability, book_appointment]

BOOKING_SYSTEM = """Booking agent.
Book only when patient_id, doctor_name, and date_slot are known.
Ask one concise question for missing details.
Use at most one tool this turn: check_slot_availability before booking, or book_appointment after availability is already confirmed.
After a tool result, answer from that result and do not call tools again.
Date format: M/D/YYYY H:MM."""

BOOKING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", BOOKING_SYSTEM),
    ("placeholder", "{messages}"),
])

booking_tool_node = ToolNode(tools=BOOKING_TOOLS)


def booking_agent_node(state: AppointmentState) -> dict:
    after_tool = bool(state["messages"]) and isinstance(state["messages"][-1], ToolMessage)
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=MODEL_NAME,
        temperature=TEMPERATURE,
    )
    if not after_tool:
        llm = llm.bind_tools(BOOKING_TOOLS)

    chain = BOOKING_PROMPT | llm
    response = chain.invoke({"messages": sanitize_messages(state["messages"])})
    if not after_tool:
        response = guard_tool_response(response, state["messages"], "booking_agent")
    else:
        log_transition("booking_agent", "END")
    return {
        "messages": [response],
        "final_response": response.content if not response.tool_calls else None,
    }
