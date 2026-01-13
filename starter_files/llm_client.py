from typing import Dict, List
from openai import OpenAI

def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""

    # TODO: Define system prompt
    # TODO: Set context in messages
    # TODO: Add chat history
    # TODO: Creaet OpenAI Client
    # TODO: Send request to OpenAI
    # TODO: Return response

   
    system_prompt = (
        """You are a NASA domain expert specializing in space missions, spacecraft,
astronomy, and planetary science. Your answers must be:
- Factually accurate
- Grounded in the provided context
- Clear and educational
If the context does not contain the answer, say so explicitly.
"""
    )

    client = OpenAI(api_key=openai_key)

   
    messages = []

    # Add system prompt
    messages.append({
        "role": "system",
        "content": system_prompt
    })

    
    if context:
        messages.append({
            "role": "system",
            "content": f"Use the following context to answer the question:\n{context}"
        })

    
    for msg in conversation_history:
        messages.append(msg)

   
    messages.append({
        "role": "user",
        "content": user_message
    })

    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3  # low creativity for factual answers
    )

    
    return response.choices[0].message.content