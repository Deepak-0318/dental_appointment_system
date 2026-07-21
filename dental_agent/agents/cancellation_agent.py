from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode
from dental_agent.config.settings import GROQ_API_KEY, MODEL_NAME, TEMPERATURE
from dental_agent.models.state import AppointmentState
from dental_agent.tools.csv_reader import get_patient_appointments
from dental_agent.tools.csv_writer import cancel_appointment
from dental_agent.utils import guard_tool_response, log_transition, sanitize_messages

CANCEL_TOOLS = [get_patient_appointments, cancel_appointment]

CANCEL_SYSTEM = """Cancellation agent.
Cancel only after patient_id, date_slot, and user confirmation are known.
Ask one concise question for missing details.
Use get_patient_appointments only when patient_id is known and the slot is unknown.
Use cancel_appointment only after confirmation.
After a tool result, answer from that result and do not call tools again.
Date format: M/D/YYYY H:MM."""

CANCEL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", CANCEL_SYSTEM),
    ("placeholder", "{messages}"),
])

cancellation_tool_node = ToolNode(tools=CANCEL_TOOLS)


def cancellation_agent_node(state: AppointmentState) -> dict:
    after_tool = bool(state["messages"]) and isinstance(state["messages"][-1], ToolMessage)
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=MODEL_NAME,
        temperature=TEMPERATURE,
    )
    if not after_tool:
        llm = llm.bind_tools(CANCEL_TOOLS)

    chain = CANCEL_PROMPT | llm
    response = chain.invoke({"messages": sanitize_messages(state["messages"])})
    if not after_tool:
        response = guard_tool_response(response, state["messages"], "cancellation_agent")
    else:
        log_transition("cancellation_agent", "END")
    return {
        "messages": [response],
        "final_response": response.content if not response.tool_calls else None,
    }
