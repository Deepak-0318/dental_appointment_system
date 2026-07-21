from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode
from dental_agent.config.settings import GROQ_API_KEY, MODEL_NAME, TEMPERATURE
from dental_agent.models.state import AppointmentState
from dental_agent.tools.csv_reader import get_patient_appointments, get_available_slots
from dental_agent.tools.csv_writer import reschedule_appointment
from dental_agent.utils import guard_tool_response, log_transition, sanitize_messages

RESCHEDULE_TOOLS = [get_patient_appointments, get_available_slots, reschedule_appointment]

RESCHEDULE_SYSTEM = """Rescheduling agent.
Reschedule only when patient_id, current_date_slot, new_date_slot, and doctor_name are known.
Ask one concise question for missing details.
Use at most one lookup tool when patient_id or doctor_name is known.
After a tool result, answer from that result and do not call tools again.
Date format: M/D/YYYY H:MM."""

RESCHEDULE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RESCHEDULE_SYSTEM),
    ("placeholder", "{messages}"),
])

rescheduling_tool_node = ToolNode(tools=RESCHEDULE_TOOLS)


def rescheduling_agent_node(state: AppointmentState) -> dict:
    after_tool = bool(state["messages"]) and isinstance(state["messages"][-1], ToolMessage)
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=MODEL_NAME,
        temperature=TEMPERATURE,
    )
    if not after_tool:
        llm = llm.bind_tools(RESCHEDULE_TOOLS)

    chain = RESCHEDULE_PROMPT | llm
    response = chain.invoke({"messages": sanitize_messages(state["messages"])})
    if not after_tool:
        response = guard_tool_response(response, state["messages"], "rescheduling_agent")
    else:
        log_transition("rescheduling_agent", "END")
    return {
        "messages": [response],
        "final_response": response.content if not response.tool_calls else None,
    }
