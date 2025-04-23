import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
from safetensors.torch import load_file

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Truncation.*")

# Define model directory
model_dir = r"\SmolLM2-Model-Safetensor"

# Load tokenizer
model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    x=4,
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# Load fine-tuned weights
state_dict = load_file(os.path.join(model_dir, "model.safetensors"))
model.load_state_dict(state_dict, strict=False)

# Move model to appropriate device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print(f"Model loaded successfully on {device}.")

# Create pipeline with truncation explicitly set to True
pipe_device = 0 if torch.cuda.is_available() else -1
generator = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer,
    truncation=True  # Explicitly enable truncation
)

# Utility functions
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

def create_prompt(npc_role: str, player_input: str, emotion: str) -> str:
    """
    Create a formatted prompt string based on the NPC role, player input, and emotion.
    """
    return (
        f"NPC Role: {npc_role}\n"
        f"Player Input: {player_input}\n"
        f"Emotion: {emotion}\n"
        f"Response:"
    )

def generate_npc_response(npc_role: str, player_input: str, emotion: str, max_length: int = 100) -> str:
    """
    Generate a response using the fine-tuned model, and return the generated text.
    """
    prompt = create_prompt(npc_role, player_input, emotion)
    output = generator(
        prompt,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
    )

    # Remove the prompt from the generated text
    generated_text = output[0]['generated_text'][len(prompt):].strip()

    # Optionally truncate on first newline
    if "\n" in generated_text:
        generated_text = generated_text.split("\n")[0]

    generated_text = truncate_on_ngram_repetition(generated_text)
    return generated_text.strip()

# Example usage
if __name__ == "__main__":
    response = generate_npc_response(
        npc_role="Mechanic",
        player_input="My car's engine is making weird sounds; can you help?",
        emotion="Angry"
    )
    print("Generated Response:")
    print(response)
