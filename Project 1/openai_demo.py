"""
OpenAI API Demo
Demonstrates: basic completion, streaming, and function calling.
Requires: OPENAI_API_KEY in .env
"""

import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

QUESTION = "What is an IVR system?"


# ── 1. Basic Chat Completion ──────────────────────────────────────────────────

def basic_completion():
    print("=" * 60)
    print("1. BASIC CHAT COMPLETION")
    print("=" * 60)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": QUESTION}],
    )

    print(response.choices[0].message.content)
    print()


# ── 2. Streaming Completion ───────────────────────────────────────────────────

def streaming_completion():
    print("=" * 60)
    print("2. STREAMING COMPLETION (tokens printed as they arrive)")
    print("=" * 60)

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": QUESTION}],
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)

    print("\n")


# ── 3. Function Calling ───────────────────────────────────────────────────────

def get_call_logs(limit: int = 5) -> list[dict]:
    """Returns fake IVR call log data."""
    logs = [
        {
            "call_id": "c-001",
            "caller": "+1-555-0101",
            "timestamp": "2026-02-24T08:12:34Z",
            "duration_sec": 142,
            "outcome": "resolved",
            "ivr_path": ["main_menu", "billing", "pay_balance"],
        },
        {
            "call_id": "c-002",
            "caller": "+1-555-0202",
            "timestamp": "2026-02-24T09:05:11Z",
            "duration_sec": 67,
            "outcome": "transferred_to_agent",
            "ivr_path": ["main_menu", "support"],
        },
        {
            "call_id": "c-003",
            "caller": "+1-555-0303",
            "timestamp": "2026-02-24T10:22:58Z",
            "duration_sec": 203,
            "outcome": "resolved",
            "ivr_path": ["main_menu", "appointments", "reschedule"],
        },
        {
            "call_id": "c-004",
            "caller": "+1-555-0404",
            "timestamp": "2026-02-24T11:44:02Z",
            "duration_sec": 31,
            "outcome": "abandoned",
            "ivr_path": ["main_menu"],
        },
        {
            "call_id": "c-005",
            "caller": "+1-555-0505",
            "timestamp": "2026-02-24T13:01:19Z",
            "duration_sec": 178,
            "outcome": "resolved",
            "ivr_path": ["main_menu", "billing", "dispute_charge"],
        },
    ]
    return logs[:limit]


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_call_logs",
            "description": (
                "Retrieves recent IVR call logs. "
                "Returns a list of call records including caller, timestamp, "
                "duration, outcome, and the IVR menu path taken."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of call logs to return (default 5).",
                    }
                },
                "required": [],
            },
        },
    }
]


def function_calling():
    print("=" * 60)
    print("3. FUNCTION CALLING")
    print("=" * 60)

    messages = [
        {
            "role": "user",
            "content": "Show me a summary of the most recent IVR call logs.",
        }
    ]

    # First API call — model decides to call the function
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )

    assistant_message = response.choices[0].message

    # Check whether the model wants to call a tool
    if assistant_message.tool_calls:
        tool_call = assistant_message.tool_calls[0]
        fn_name = tool_call.function.name
        fn_args = json.loads(tool_call.function.arguments)

        print(f"Model requested function: {fn_name}({fn_args})")

        # Execute the local function
        if fn_name == "get_call_logs":
            result = get_call_logs(**fn_args)
        else:
            result = {"error": f"Unknown function: {fn_name}"}

        print(f"Function returned {len(result)} records.")
        print()

        # Append assistant + tool result to the conversation
        messages.append(assistant_message)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            }
        )

        # Second API call — model generates the final answer
        final_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )

        print(final_response.choices[0].message.content)
    else:
        # Model answered without calling a function
        print(assistant_message.content)

    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    basic_completion()
    streaming_completion()
    function_calling()
