from typing import Dict, List
from openai import OpenAI

def generate_response(
    openai_key: str,
    user_message: str,
    context: str,
    conversation_history: List[Dict],
    model: str = "gpt-3.5-turbo"
) -> str:
    """
    Generate a NASA mission expert response using OpenAI, grounded in retrieved context.

    Args:
        openai_key: OpenAI API key
        user_message: The user's question
        context: Retrieved context from ChromaDB
        conversation_history: List of previous messages (role + content)
        model: OpenAI model to use

    Returns:
        Assistant response as a string
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
    messages: List[Dict] = []

    # Add system prompt
    messages.append({"role": "system", "content": system_prompt})

    # Add context if available
    if context:
        messages.append({
            "role": "system",
            "content": f"Context to use for answering the question:\n{context}"
        })

    # Append previous conversation history
    for msg in conversation_history:
        if "role" in msg and "content" in msg:
            messages.append(msg)

    # Add the user question
    messages.append({
        "role": "user",
        "content": user_message
    })

    # === CALL OPENAI ===
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,  # Low creativity for factual answers
        max_tokens=600
    )

    # Return assistant content
    return response.choices[0].message.content
