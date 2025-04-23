import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from safetensors.torch import load_file
import json
import warnings
from flask import Flask, request
from flask_socketio import SocketIO

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize Flask and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', ping_timeout=60, ping_interval=1)

# Model loading and initialization
class NPCModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        # Set a shorter generation length for faster responses
        self.default_max_length = 150

    def load_model(self):
        try:
            # Define model directory
            model_dir = r"SmolLM2-Model-Safetensor"
            
            # Load tokenizer
            model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            # Load base model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,  # Reduce memory usage during loading
            )
            
            # Load fine-tuned weights
            state_dict = load_file(os.path.join(model_dir, "model.safetensors"))
            self.model.load_state_dict(state_dict, strict=False)
            
            # Move model to device
            self.model.to(self.device)
            self.tokenizer.model_max_length = self.model.config.max_position_embeddings
            
            # Set model to evaluation mode for faster inference
            self.model.eval()
            
            # Pre-compile model with TorchScript for faster inference if using CUDA
            if self.device == "cuda":
                try:
                    # Dummy input for tracing
                    dummy_input = self.tokenizer("Hello", return_tensors="pt").to(self.device)
                    self.model = torch.jit.trace(self.model, (dummy_input.input_ids,), check_trace=False)
                    print("Model optimized with TorchScript")
                except Exception as e:
                    print(f"Could not optimize with TorchScript: {e}")
            
            self.model_loaded = True
            print(f"Model loaded successfully on {self.device}.")
            
            # Warm up the model with a test input (reduces initial latency)
            self.warm_up()
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False
    
    def warm_up(self):
        """Warm up the model to reduce initial latency"""
        try:
            test_input = "Hello"
            inputs = self.tokenizer(test_input, return_tensors="pt").to(self.device)
            with torch.no_grad():
                self.model.generate(**inputs, max_length=20)
            print("Model warmed up successfully")
        except Exception as e:
            print(f"Model warm-up failed: {e}")

    def create_prompt(self, npc_role: str, player_input: str, emotion: str) -> str:
        """Create a formatted prompt string."""
        return (
            f"NPC Role: {npc_role}\n"
            f"Player Input: {player_input}\n"
            f"Emotion: {emotion}\n"
            f"Response:"
        )

    def generate_response(self, npc_role: str, player_input: str, emotion: str, max_length: int = None) -> str:
        """Generate a response using the fine-tuned model."""
        if not self.model_loaded:
            return "Error: Model not loaded"
        
        # Use default if not specified
        if max_length is None or max_length > 500:
            max_length = self.default_max_length
            
        try:
            prompt = self.create_prompt(npc_role, player_input, emotion)
            
            # Tokenize input with truncation enabled
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
            
            # Generate with optimized parameters
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,  # Slightly reduced for faster decisions
                    top_k=40,
                    top_p=0.9,
                    repetition_penalty=1.2,  # Add penalty to avoid repetition
                    pad_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3,  # Built-in n-gram repetition prevention
                )
            
            # Decode the generated tokens
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Remove the prompt from the generated text
            response = generated_text[len(prompt):].strip()
            
            # Truncate on first newline
            if "\n" in response:
                response = response.split("\n")[0]
            
            return response.strip()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Sorry, I'm having trouble processing your request."

# Initialize the model
model = NPCModel()

# Load the model in the main thread before server starts
print("Loading model...")
model.load_model()

# Flask routes
@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>NPC Model Service</title>
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
        # Extract data with defaults
        npc_role = data.get('npc_role', 'Shopkeeper')
        player_input = data.get('player_input', '')
        emotion = data.get('emotion', 'Neutral')
        max_length = min(data.get('max_length', 150), 250)  # Cap max length
        
        # Generate the response
        response = model.generate_response(npc_role, player_input, emotion, max_length)
        
        # Return a simple response format
        return {"response": response, "status": "success"}
    except Exception as e:
        print(f"WebSocket error: {e}")
        return {"response": "Sorry, I couldn't understand that.", "status": "error"}

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)