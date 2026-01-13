# llm_client.py
from typing import List, Dict
from openai import OpenAI

MAX_HISTORY_TURNS = 5  # only keep last N turns to prevent unbounded growth

def generate_response(
    openai_key: str,
    user_message: str,
    context: str,
    conversation_history: List[Dict],
    model: str = "gpt-3.5-turbo"
) -> str:
    """
    Generate a NASA mission expert response using OpenAI, grounded in retrieved context.
    Maintains conversation history and prunes it to last MAX_HISTORY_TURNS.
    """

    # === SYSTEM PROMPT ===
    system_prompt = (
        "You are a NASA mission expert specializing in space missions, "
        "spacecraft, astronomy, and planetary science.\n\n"
        "Rules:\n"
        "- Use ONLY the provided context to answer the question.\n"
        "- Cite sources using the format [DOC_ID] after each factual claim.\n"
        "- If the answer is not in the context, say 'I don't know based on the provided documents.'\n"
        "- Do NOT use outside knowledge.\n"
        "- Keep answers clear, concise, and educational."
    )

    # === CREATE OPENAI CLIENT ===
    client = OpenAI(api_key=openai_key)

    # === BUILD MESSAGE HISTORY ===
    messages: List[Dict] = [{"role": "system", "content": system_prompt}]

    # Add context as system message
    if context:
        messages.append({
            "role": "system",
            "content": f"Context to use for answering the question:\n{context}"
        })

    # Prune conversation history to last N turns
    history_to_use = conversation_history[-MAX_HISTORY_TURNS*2:]  # each turn = user + assistant

    for msg in history_to_use:
        if "role" in msg and "content" in msg:
            messages.append(msg)

    # Add current user message
    messages.append({"role": "user", "content": user_message})

    # === CALL OPENAI ===
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=600
    )

    return response.choices[0].message.content
