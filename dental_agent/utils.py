import json
import logging
from typing import List, Optional
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("dental_agent")

REQUIRED_TOOL_ARGS = {
    "get_patient_appointments": ["patient_id"],
    "check_slot_availability": ["doctor_name", "date_slot"],
    "list_doctors_by_specialization": ["specialization"],
    "book_appointment": ["patient_id", "doctor_name", "date_slot"],
    "cancel_appointment": ["patient_id", "date_slot"],
    "reschedule_appointment": [
        "patient_id",
        "current_date_slot",
        "new_date_slot",
        "doctor_name",
    ],
}


def sanitize_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    xAI (grok) API rejects any message with empty/null content.
    Replace empty content (None, "", or []) with a single space so the API
    accepts the message while preserving all other metadata.
    """
    result = []
    for msg in messages:
        content = msg.content
        is_empty = content is None or content == "" or content == []
        if is_empty:
            if isinstance(msg, AIMessage):
                result.append(
                    AIMessage(
                        content=" ",
                        tool_calls=getattr(msg, "tool_calls", None),
                        id=getattr(msg, "id", None),
                        response_metadata=getattr(msg, "response_metadata", {}),
                        usage_metadata=getattr(msg, "usage_metadata", None),
                    )
                )
            elif isinstance(msg, HumanMessage):
                result.append(
                    HumanMessage(
                        content=" ",
                        id=getattr(msg, "id", None),
                        response_metadata=getattr(msg, "response_metadata", {}),
                        usage_metadata=getattr(msg, "usage_metadata", None),
                    )
                )
            elif isinstance(msg, SystemMessage):
                result.append(
                    SystemMessage(
                        content=" ",
                        id=getattr(msg, "id", None),
                        response_metadata=getattr(msg, "response_metadata", {}),
                        usage_metadata=getattr(msg, "usage_metadata", None),
                    )
                )
            elif isinstance(msg, ToolMessage):
                result.append(
                    ToolMessage(
                        content=" ",
                        tool_call_id=getattr(msg, "tool_call_id", None),
                        id=getattr(msg, "id", None),
                        response_metadata=getattr(msg, "response_metadata", {}),
                        usage_metadata=getattr(msg, "usage_metadata", None),
                    )
                )
            else:
                msg_type = type(msg)
                result.append(
                    msg_type(
                        content=" ",
                        **{k: v for k, v in msg.__dict__.items() if k != "content"},
                    )
                )
        else:
            result.append(msg)
    return result


def log_transition(
    current_node: str,
    next_node: str,
    selected_tool: Optional[str] = None,
    tool_args: Optional[dict] = None,
) -> None:
    """Lightweight graph trace for node, tool, args, and next-node debugging."""
    logger.info(
        "[debug] current_node=%s selected_tool=%s tool_args=%s next_node=%s",
        current_node,
        selected_tool or "",
        tool_args or {},
        next_node,
    )


def _tool_signature(tool_call: dict) -> str:
    args = tool_call.get("args") or {}
    return json.dumps(
        {"name": tool_call.get("name"), "args": args},
        sort_keys=True,
        default=str,
    )


def _has_seen_tool_call(messages: List[BaseMessage], tool_call: dict) -> bool:
    target = _tool_signature(tool_call)
    latest_human_index = max(
        (index for index, message in enumerate(messages) if isinstance(message, HumanMessage)),
        default=-1,
    )
    return any(
        isinstance(message, AIMessage)
        and any(_tool_signature(existing) == target for existing in message.tool_calls)
        for message in messages[latest_human_index + 1 :]
    )


def _missing_required_args(tool_call: dict) -> List[str]:
    args = tool_call.get("args") or {}
    required = REQUIRED_TOOL_ARGS.get(tool_call.get("name"), [])
    return [name for name in required if not args.get(name)]


def guard_tool_response(
    response: AIMessage,
    history: List[BaseMessage],
    current_node: str,
) -> AIMessage:
    """Allow one well-formed, non-repeated tool call; otherwise answer/clarify."""
    if not response.tool_calls:
        log_transition(current_node, "END")
        return response

    tool_call = response.tool_calls[0]
    tool_name = tool_call.get("name")
    tool_args = tool_call.get("args") or {}

    if _has_seen_tool_call(history, tool_call):
        log_transition(current_node, "END", tool_name, tool_args)
        return AIMessage(
            content=(
                "I already checked with those same details. "
                "Please provide new details or clarify what you want to do next."
            )
        )

    missing_args = _missing_required_args(tool_call)
    if missing_args:
        log_transition(current_node, "END", tool_name, tool_args)
        return AIMessage(content=f"Please provide {missing_args[0]} so I can help.")

    if len(response.tool_calls) > 1:
        # Keep each turn to one tool unless another pass is truly required.
        response.tool_calls = [tool_call]

    log_transition(current_node, "tools", tool_name, tool_args)
    return response
