from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from dental_agent.workflows.graph import dental_graph

load_dotenv()

st.set_page_config(
    page_title="Dental Appointment Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main .block-container {
            max-width: 1080px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .app-header {
            border-bottom: 1px solid #e5e7eb;
            padding-bottom: 1rem;
            margin-bottom: 1.25rem;
        }
        .app-title {
            color: #0f172a;
            font-size: 2rem;
            font-weight: 700;
            margin: 0;
        }
        .app-subtitle {
            color: #475569;
            font-size: 1rem;
            margin-top: 0.35rem;
        }
        .status-card {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 0.9rem;
            background: #ffffff;
            margin-bottom: 0.75rem;
        }
        .status-label {
            color: #64748b;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0;
        }
        .status-value {
            color: #0f172a;
            font-size: 1.05rem;
            font-weight: 650;
            margin-top: 0.15rem;
        }
        div[data-testid="stChatInput"] {
            border-top: 1px solid #e5e7eb;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def _last_assistant_text(messages: list) -> str:
    """Return the latest assistant text while ignoring tool-call placeholders."""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            return str(message.content)
    return "I could not generate a response. Please try again."


def _reset_chat() -> None:
    st.session_state.backend_messages = []
    st.session_state.chat_messages = []


if "backend_messages" not in st.session_state:
    st.session_state.backend_messages = []
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

with st.sidebar:
    st.markdown("### Console")
    st.markdown(
        """
        <div class="status-card">
            <div class="status-label">Backend</div>
            <div class="status-value">LangGraph connected</div>
        </div>
        <div class="status-card">
            <div class="status-label">Workflow</div>
            <div class="status-value">Supervisor -> Agent -> Tool</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.button("New conversation", use_container_width=True, on_click=_reset_chat)
    st.caption("Ask about doctors, free slots, bookings, cancellations, or rescheduling.")

st.markdown(
    """
    <div class="app-header">
        <h1 class="app-title">Dental Appointment Assistant</h1>
        <div class="app-subtitle">
            Manage availability, bookings, cancellations, and appointment lookups from one clean workspace.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.chat_messages:
    with st.chat_message("assistant"):
        st.write("Hello. How can I help with your dental appointment today?")

for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("Type your appointment request")
if prompt:
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    st.session_state.backend_messages.append(HumanMessage(content=prompt))

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Checking the appointment system..."):
            try:
                result = dental_graph.invoke({"messages": st.session_state.backend_messages})
                st.session_state.backend_messages = result.get(
                    "messages",
                    st.session_state.backend_messages,
                )
                response = _last_assistant_text(st.session_state.backend_messages)
            except Exception as exc:
                response = (
                    "I could not reach the appointment backend. "
                    f"Please check the configuration and try again. Error: {exc}"
                )
        st.write(response)

    st.session_state.chat_messages.append({"role": "assistant", "content": response})
