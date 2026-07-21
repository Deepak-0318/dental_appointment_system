from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage
from langgraph.prebuilt import tools_condition

from dental_agent.models.state import AppointmentState
from dental_agent.agents.supervisor import supervisor_node
from dental_agent.agents.info_agent import info_agent_node, info_tool_node
from dental_agent.agents.booking_agent import booking_agent_node, booking_tool_node
from dental_agent.agents.cancellation_agent import cancellation_agent_node, cancellation_tool_node
from dental_agent.agents.rescheduling_agent import rescheduling_agent_node, rescheduling_tool_node
from dental_agent.utils import log_transition


def route_from_supervisor(state: AppointmentState) -> str:
    """Read next_agent from state and return the corresponding node name."""
    target = state.get("next_agent", "info_agent")
    valid = {"info_agent", "booking_agent", "cancellation_agent", "rescheduling_agent", "end"}
    next_node = target if target in valid else "info_agent"
    log_transition("supervisor", "END" if next_node == "end" else next_node)
    return next_node


def _route_tools(state: AppointmentState) -> str:
    """Use LangGraph's built-in tool routing, mapping no-tool responses to END."""
    return "tools" if tools_condition(state) == "tools" else "end"


def _logged_tool_node(tool_node, current_node: str, next_node: str):
    """Wrap ToolNode so debug logs include selected tool args and the next node."""
    def run(state: AppointmentState) -> dict:
        selected_tool = None
        tool_args = {}
        messages = state.get("messages", [])
        if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
            tool_call = messages[-1].tool_calls[0]
            selected_tool = tool_call.get("name")
            tool_args = tool_call.get("args") or {}
        log_transition(current_node, next_node, selected_tool, tool_args)
        return tool_node.invoke(state)

    return run


def build_graph():
    graph = StateGraph(AppointmentState)

    # Register nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("info_agent", info_agent_node)
    graph.add_node("info_tools", _logged_tool_node(info_tool_node, "info_tools", "info_agent"))
    graph.add_node("booking_agent", booking_agent_node)
    graph.add_node("booking_tools", _logged_tool_node(booking_tool_node, "booking_tools", "booking_agent"))
    graph.add_node("cancellation_agent", cancellation_agent_node)
    graph.add_node("cancellation_tools", _logged_tool_node(cancellation_tool_node, "cancellation_tools", "cancellation_agent"))
    graph.add_node("rescheduling_agent", rescheduling_agent_node)
    graph.add_node("rescheduling_tools", _logged_tool_node(rescheduling_tool_node, "rescheduling_tools", "rescheduling_agent"))

    # Entry point
    graph.add_edge(START, "supervisor")

    # Supervisor routes to sub-agents
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "info_agent": "info_agent",
            "booking_agent": "booking_agent",
            "cancellation_agent": "cancellation_agent",
            "rescheduling_agent": "rescheduling_agent",
            "end": END,
        },
    )

    # Info agent loop: agent → tools → agent → END
    graph.add_conditional_edges(
        "info_agent",
        _route_tools,
        {"tools": "info_tools", "end": END},
    )
    graph.add_edge("info_tools", "info_agent")

    # Booking agent loop
    graph.add_conditional_edges(
        "booking_agent",
        _route_tools,
        {"tools": "booking_tools", "end": END},
    )
    graph.add_edge("booking_tools", "booking_agent")

    # Cancellation agent loop
    graph.add_conditional_edges(
        "cancellation_agent",
        _route_tools,
        {"tools": "cancellation_tools", "end": END},
    )
    graph.add_edge("cancellation_tools", "cancellation_agent")

    # Rescheduling agent loop
    graph.add_conditional_edges(
        "rescheduling_agent",
        _route_tools,
        {"tools": "rescheduling_tools", "end": END},
    )
    graph.add_edge("rescheduling_tools", "rescheduling_agent")

    return graph.compile()


dental_graph = build_graph()
