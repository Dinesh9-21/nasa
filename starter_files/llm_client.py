# llm_client.py
from typing import Dict, List
from openai import OpenAI

MAX_HISTORY_TURNS = 5  # Keep last N user+assistant pairs


def generate_response(
    openai_key: str,
    user_message: str,
    context: str,
    conversation_history: List[Dict],
    model: str = "gpt-3.5-turbo"
) -> str:
    """
    Generate a NASA mission expert response using OpenAI, grounded in retrieved context.
    Maintains pruned conversation history and cites sources per turn.
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

    client = OpenAI(api_key=openai_key)

    messages: List[Dict] = []
    messages.append({"role": "system", "content": system_prompt})

    # Include context as a system message
    if context:
        messages.append({
            "role": "system",
            "content": f"Context to use for answering the question:\n{context}"
        })

    # === PRUNE conversation history ===
    # Keep only last MAX_HISTORY_TURNS user+assistant pairs
    pruned_history = conversation_history[-MAX_HISTORY_TURNS*2:]  # each turn = user+assistant
    for msg in pruned_history:
        if "role" in msg and "content" in msg:
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Add current user question
    messages.append({"role": "user", "content": user_message})

    # === CALL OPENAI ===
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=600
    )

    return response.choices[0].message.content
