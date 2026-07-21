from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode
from dental_agent.config.settings import GROQ_API_KEY, MODEL_NAME, TEMPERATURE
from dental_agent.models.state import AppointmentState
from dental_agent.tools.csv_reader import (
    get_available_slots,
    get_patient_appointments,
    check_slot_availability,
    list_doctors_by_specialization,
)
from dental_agent.utils import guard_tool_response, log_transition, sanitize_messages

INFO_TOOLS = [
    get_available_slots,
    get_patient_appointments,
    check_slot_availability,
    list_doctors_by_specialization,
]

INFO_SYSTEM = """Information agent.
Use one tool only when real schedule data is needed. If parameters are missing, ask one concise question.
After a tool result, answer from that result and do not call tools again.
Valid specializations: general_dentist, oral_surgeon, orthodontist, cosmetic_dentist, prosthodontist, pediatric_dentist, emergency_dentist.
Date format: M/D/YYYY H:MM."""

INFO_PROMPT = ChatPromptTemplate.from_messages([
    ("system", INFO_SYSTEM),
    ("placeholder", "{messages}"),
])

info_tool_node = ToolNode(tools=INFO_TOOLS)


def info_agent_node(state: AppointmentState) -> dict:
    after_tool = bool(state["messages"]) and isinstance(state["messages"][-1], ToolMessage)
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=MODEL_NAME,
        temperature=TEMPERATURE,
    )
    if not after_tool:
        llm = llm.bind_tools(INFO_TOOLS)

    chain = INFO_PROMPT | llm
    response = chain.invoke({"messages": sanitize_messages(state["messages"])})
    if not after_tool:
        response = guard_tool_response(response, state["messages"], "info_agent")
    else:
        log_transition("info_agent", "END")
    return {
        "messages": [response],
        "final_response": response.content if not response.tool_calls else None,
    }
