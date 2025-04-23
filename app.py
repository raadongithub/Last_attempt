import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from safetensors.torch import load_file
import json
import asyncio
import threading
import warnings
from flask import Flask, render_template, request
from flask_socketio import SocketIO

warnings.filterwarnings("ignore", category=UserWarning, message=".*Truncation.*")

# Initialize Flask and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Model loading and initialization
class NPCModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        self.load_model()

    def load_model(self):
        try:
            # Define model directory
            model_dir = r"SmolLM2-Model-Safetensor"
            
            # Load tokenizer
            model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            # Load base model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
            )
            
            # Load fine-tuned weights
            state_dict = load_file(os.path.join(model_dir, "model.safetensors"))
            self.model.load_state_dict(state_dict, strict=False)
            
            # Move model to the appropriate device
            self.model.to(self.device)
            self.tokenizer.model_max_length = self.model.config.max_position_embeddings
            
            self.model_loaded = True
            print(f"Model loaded successfully on {self.device}.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False

    def truncate_on_ngram_repetition(self, text: str, n: int = 3) -> str:
        """Truncate the generated text to avoid n-gram repetition."""
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

    def create_prompt(self, npc_role: str, player_input: str, emotion: str) -> str:
        """Create a formatted prompt string."""
        return (
            f"NPC Role: {npc_role}\n"
            f"Player Input: {player_input}\n"
            f"Emotion: {emotion}\n"
            f"Response:"
        )

    def generate_response(self, npc_role: str, player_input: str, emotion: str, max_length: int = 100) -> str:
        """Generate a response using the fine-tuned model."""
        if not self.model_loaded:
            return "Error: Model not loaded"
        
        try:
            prompt = self.create_prompt(npc_role, player_input, emotion)
            
            # Tokenize and move to correct device
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Set to inference mode for optimal performance
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.9,
                )
            
            # Decode the generated tokens
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Remove the prompt from the generated text
            generated_text = generated_text[len(prompt):].strip()
            
            # Truncate on first newline
            if "\n" in generated_text:
                generated_text = generated_text.split("\n")[0]
            
            generated_text = self.truncate_on_ngram_repetition(generated_text)
            return generated_text.strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"

# Initialize the model in a background thread
model = NPCModel()

# Flask routes
@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Early Retirement Model</title>
    </head>
    <body>
        <h1>Model is live</h1>
    </body>
    </html>
    """

@app.route('/status')
def status():
    return json.dumps({"status": "online", "model_loaded": model.model_loaded, "device": model.device})

# WebSocket handlers
@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('generate_response')
def handle_generate_response(data):
    try:
        npc_role = data.get('npc_role', '')
        player_input = data.get('player_input', '')
        emotion = data.get('emotion', '')
        max_length = data.get('max_length', 100)

        response = model.generate_response(npc_role, player_input, emotion, max_length)
        
        return {"response": response, "status": "success"}
    except Exception as e:
        print(f"WebSocket error: {e}")
        return {"response": "", "status": "error", "message": str(e)}

# Optional REST API endpoint for non-WebSocket clients
@app.route('/api/generate', methods=['POST'])
def api_generate():
    try:
        data = request.json
        npc_role = data.get('npc_role', '')
        player_input = data.get('player_input', '')
        emotion = data.get('emotion', '')
        max_length = data.get('max_length', 100)

        response = model.generate_response(npc_role, player_input, emotion, max_length)
        
        return json.dumps({"response": response, "status": "success"})
    except Exception as e:
        return json.dumps({"response": "", "status": "error", "message": str(e)})

if __name__ == "__main__":
    # Run the Flask app with SocketIO
    # Use threaded=True for better handling of multiple connections
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)