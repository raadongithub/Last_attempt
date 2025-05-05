import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import asyncio
import websockets
import json
import concurrent.futures # Added for running blocking code

# --- Optimizations ---
# 1. Correctly determine device
if torch.cuda.is_available():
    device = "cuda"
    # 2. Use half-precision (FP16) for faster inference on compatible GPUs
    # Ensure your model fine-tuning is compatible with FP16.
    # BF16 (torch.bfloat16) might be another option if your GPU/model supports it well.
    model_dtype = torch.float16
else:
    device = "cpu"
    model_dtype = torch.float32 # FP16 is usually beneficial only on GPU

print(f"Using device: {device.upper()}")
print(f"Using dtype: {model_dtype}")

# Load the fine-tuned model and tokenizer from the specified directory.
# Load model with specified dtype
model_path = r"finetunedModel"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=model_dtype, # Apply half-precision here
    # low_cpu_mem_usage=True, # Might help if loading large models on limited RAM before moving to GPU
)

# --- Optional: torch.compile (Requires PyTorch 2.0+) ---
# Uncomment the following line to enable. Can significantly speed up inference
# after an initial compilation cost on the first run. May not work with all models/operations.
# try:
#     model = torch.compile(model, mode="reduce-overhead") # Or other modes like "max-autotune"
#     print("Model compiled successfully.")
# except Exception as e:
#     print(f"torch.compile failed: {e}. Proceeding without compilation.")
# --- End Optional: torch.compile ---


# Setup a text-generation pipeline explicitly on the target device
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device # Explicitly set the device for the pipeline
)

# Create a thread pool executor for running blocking tasks
executor = concurrent.futures.ThreadPoolExecutor()


def create_prompt(npc_role: str, player_input: str, emotion: str) -> str:
    """
    Build a prompt from the NPC role, player input, and emotion.
    (Core functionality unchanged)
    """
    # Using f-string formatting directly for consistency
    return f"NPC Role: {npc_role}\nPlayer Input: {player_input}\nEmotion: {emotion}\nResponse:"

def truncate_on_ngram_repetition(text: str, n: int = 3) -> str:
    """
    Truncate the generated text to avoid n-gram repetition.
    (Core functionality unchanged)
    """
    words = text.split()
    if len(words) < n: # Avoid index errors for short texts
        return text
        
    seen_ngrams = {}
    for i in range(len(words) - (n - 1)):
        ngram = tuple(words[i:i+n])
        if ngram in seen_ngrams:
            # Truncate before the repeated n-gram starts
            truncated_text = " ".join(words[:i])
            # Find the last sentence boundary for cleaner cuts
            last_period = truncated_text.rfind('.')
            last_question = truncated_text.rfind('?')
            last_exclamation = truncated_text.rfind('!')
            last_boundary = max(last_period, last_question, last_exclamation)
            
            return truncated_text[:last_boundary+1].strip() if last_boundary != -1 else truncated_text.strip()
        seen_ngrams[ngram] = i
    return text

# This function is now synchronous as it will be run in an executor thread
def generate_npc_response(npc_role: str, player_input: str, emotion: str, max_length: int = 100):
    """
    Generate a response using the fine-tuned model, and return the generated text.
    (Core functionality unchanged, but generation parameters added)
    """
    prompt = create_prompt(npc_role, player_input, emotion)
    
    # --- Optimization: Generation Parameters ---
    # Explicitly set parameters. Greedy search (do_sample=False) is often fastest.
    # If you need creative/varied responses, set do_sample=True and tune
    # temperature, top_k, top_p.
    # pad_token_id is important to suppress warnings when not sampling.
    output = generator(
        prompt,
        max_length=max_length, # Be mindful: max_length includes prompt length
        num_return_sequences=1,
        truncation=True,
        do_sample=False, # Set to True if you need sampling
        # --- Parameters for sampling (if do_sample=True) ---
        # temperature=0.7,
        # top_k=50,
        # top_p=0.95,
        # --- End Sampling Parameters ---
        pad_token_id=tokenizer.eos_token_id # Suppress warning when not sampling
    )

    # Remove the prompt from the generated text.
    # Check if output exists and has the expected structure
    generated_text = ""
    if output and isinstance(output, list) and output[0] and 'generated_text' in output[0]:
         # Ensure prompt removal doesn't fail if generation is shorter than prompt somehow
         full_text = output[0]['generated_text']
         if full_text.startswith(prompt):
              generated_text = full_text[len(prompt):].strip()
         else:
              # Fallback or warning if prompt isn't at the start (might indicate unexpected model behavior)
              print("Warning: Generated text doesn't start with the prompt.")
              generated_text = full_text # Or decide how to handle this case
    else:
         print("Warning: Unexpected output format from generator.")
         # Handle error or return empty response

    # Optionally truncate on double newlines.
    if "\n\n" in generated_text:
        generated_text = generated_text.split("\n\n", 1)[0] # Split only once

    generated_text = truncate_on_ngram_repetition(generated_text)
    return generated_text.strip()

async def handler(websocket, path=None):
    """
    Handles incoming WebSocket connections using an executor for model inference.
    """
    loop = asyncio.get_running_loop()
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                npc_role = data.get("npc_role", "Default NPC")
                emotion = data.get("emotion", "Neutral")
                player_input = data.get("player_input", "")

                print(f"Received data: npc_role={npc_role}, emotion={emotion}, player_input={player_input}")

                # --- Optimization: Run blocking generation in executor ---
                response_text = await loop.run_in_executor(
                    executor, # The thread pool
                    generate_npc_response, # The function to run
                    npc_role, # Arguments for the function
                    player_input,
                    emotion
                    # max_length can be passed here if needed, e.g.:
                    # npc_role, player_input, emotion, 150 # for max_length=150
                )
                # --- End Optimization ---

                result = {"response": response_text}
                await websocket.send(json.dumps(result))

            except json.JSONDecodeError:
                print("Received invalid JSON")
                error_msg = {"error": "Invalid JSON format"}
                await websocket.send(json.dumps(error_msg))
            except Exception as e:
                print(f"Error processing message: {e}") # Log specific error
                # Consider logging traceback for debugging: import traceback; traceback.print_exc()
                error_msg = {"error": f"An error occurred: {e}"}
                await websocket.send(json.dumps(error_msg))

    except websockets.exceptions.ConnectionClosed as e:
        print(f"Connection closed: {e}")
    except Exception as e:
        print(f"Handler exception: {e}")
        # Consider logging traceback here too


async def main():
    # Ensure the server closes gracefully and shuts down the executor
    try:
        async with websockets.serve(handler, "localhost", 8765):
            print("WebSocket server started on ws://localhost:8765")
            print("Using model:", model_path)
            await asyncio.Future()  # run forever
    finally:
        print("Shutting down executor...")
        executor.shutdown(wait=True) # Allow pending tasks to complete
        print("Executor shut down.")


if _name_ == "_main_":
    # Set start method for multiprocessing if needed (might be relevant for torch.compile or complex setups)
    # import multiprocessing
    # multiprocessing.set_start_method('spawn', force=True) # or 'forkserver'
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped manually.")