import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import asyncio
import websockets
import json

# Use CPU for inference.
device = "cpu"
print("Using device:", device.upper())

# Load the fine-tuned model and tokenizer from the specified directory.
model_path = r"C:\FYP\000 Raad\Finetuning\SmolLM2-360m-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Setup a text-generation pipeline.
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def create_prompt(npc_role: str, player_input: str, emotion: str) -> str:
    """
    Build a prompt from the NPC role, player input, and emotion.
    """
    return (
        f"NPC Role: {npc_role}\n"
        f"Player Input: {player_input}\n"
        f"Emotion: {emotion}\n"
        "Response:"
    )

def truncate_on_ngram_repetition(text: str, n: int = 3) -> str:
    """
    Truncate the generated text to avoid n-gram repetition.
    """
    words = text.split()
    seen_ngrams = {}
    for i in range(len(words) - (n - 1)):
        ngram = tuple(words[i:i+n])
        if ngram in seen_ngrams:
            truncated_text = " ".join(words[:i])
            last_period = truncated_text.rfind('.')
            return truncated_text[:last_period+1] if last_period != -1 else truncated_text
        seen_ngrams[ngram] = i
    return text

def generate_npc_response(npc_role: str, player_input: str, emotion: str, max_length: int = 100):
    """
    Generate a response using the fine-tuned model, and return the generated text.
    """
    prompt = create_prompt(npc_role, player_input, emotion)
    output = generator(prompt, max_length=max_length, num_return_sequences=1, truncation=True)

    # Remove the prompt from the generated text.
    generated_text = output[0]['generated_text'][len(prompt):].strip()

    # Optionally truncate on double newlines.
    if "\n\n" in generated_text:
        generated_text = generated_text.split("\n\n")[0]

    generated_text = truncate_on_ngram_repetition(generated_text)
    return generated_text.strip()

async def handler(websocket, path=None):
    """
    Handles incoming WebSocket connections.
    Receives JSON messages containing "npc_role", "emotion", and "player_input",
    generates a response using the model, and sends back a JSON response.
    """
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                npc_role = data.get("npc_role", "Default NPC")
                emotion = data.get("emotion", "Neutral")
                player_input = data.get("player_input", "")
                
                print(f"Received data: npc_role={npc_role}, emotion={emotion}, player_input={player_input}")
                
                response_text = generate_npc_response(npc_role, player_input, emotion)
                
                result = {
                    "response": response_text
                }
                await websocket.send(json.dumps(result))
            except Exception as e:
                error_msg = {"error": str(e)}
                await websocket.send(json.dumps(error_msg))
    except websockets.exceptions.ConnectionClosed as e:
        # Log a message when the connection is closed gracefully or abruptly.
        print("Connection closed:", e)
    except Exception as e:
        print("Handler exception:", e)

async def main():
    async with websockets.serve(handler, "localhost", 8765):
        print("WebSocket server started on ws://localhost:8765")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
