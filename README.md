# Dental Appointment Management Assistant

A LangGraph-powered dental appointment assistant with a professional Streamlit frontend. The assistant routes patient requests through a supervisor and specialist agents for appointment information, booking, cancellation, and rescheduling.

## Features

- Conversational appointment assistant
- Supervisor-driven routing to specialist agents
- Doctor and slot availability lookup from CSV data
- Appointment booking, cancellation, and rescheduling tools
- Streamlit web frontend connected to the LangGraph backend
- CLI entrypoint for terminal-based usage

## Architecture

The backend keeps the existing workflow:

```text
Supervisor -> Agent -> Tool -> Agent -> END
```

Main components:

- `dental_agent/workflows/graph.py` builds the LangGraph workflow.
- `dental_agent/agents/` contains the supervisor and specialist agents.
- `dental_agent/tools/` contains CSV reader and writer tools.
- `doctor_availability.csv` stores appointment availability and patient bookings.
- `frontend.py` provides the Streamlit interface.
- `main.py` provides the CLI interface.

## Requirements

- Python 3.9+
- Groq API key

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
MODEL_NAME=llama-3.3-70b-versatile
TEMPERATURE=0
```

`MODEL_NAME` and `TEMPERATURE` are optional. The defaults are already configured in `dental_agent/config/settings.py`.

## Run The Frontend

```bash
streamlit run frontend.py
```

The web app opens a chat workspace where users can ask questions such as:

- Who are the doctors available?
- How many appointments are free now?
- Show my appointments.
- Book an appointment.
- Cancel my appointment.

## Run The CLI

```bash
python main.py
```

Type `quit`, `exit`, or `bye` to end the CLI session.

## Data

Appointment data is stored in:

```text
doctor_availability.csv
```

The CSV includes:

- `date_slot`
- `specialization`
- `doctor_name`
- `is_available`
- `patient_to_attend`

Tool functions read and update this file directly, so changes made by bookings, cancellations, and rescheduling are persisted.

## Project Structure

```text
.
+-- dental_agent/
|   +-- agents/
|   +-- config/
|   +-- models/
|   +-- tools/
|   +-- workflows/
|   +-- agent.py
|   +-- utils.py
+-- doctor_availability.csv
+-- frontend.py
+-- main.py
+-- README.md
+-- requirements.txt
```

## Notes

- The frontend is connected directly to the LangGraph backend.
- The frontend was not run or browser-tested here, per request.
- Backend execution requires a valid Groq API key and network access.
