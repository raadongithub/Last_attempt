import sys
import socketio
import signal
import threading
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

# Initialize Socket.IO client with timeouts
sio = socketio.Client(reconnection=True, reconnection_attempts=5, reconnection_delay=1)
connected = False
response_event = threading.Event()
response_timeout = 30  # Response timeout in seconds

# Fixed values for development
FIXED_NPC_ROLE = "Shopkeeper"
FIXED_EMOTION = "Neutral"
SERVER_URL = "http://192.168.18.128:5000/"

@sio.event
def connect():
    global connected
    print("\nConnected to server!")
    connected = True

@sio.event
def disconnect():
    global connected
    print("\nDisconnected from server!")
    connected = False

@sio.on('generate_response')
def on_response(data):
    # Handle the response
    try:
        if isinstance(data, str):
            # Try to parse as JSON if it's a string
            try:
                import json
                data = json.loads(data)
            except:
                # If not valid JSON, create a response object
                data = {'response': data, 'status': 'success'}
        
        print("\nNPC says:", data.get('response', 'No response'))
    except Exception as e:
        print(f"\nError processing response: {e}")
    
    # Notify waiting thread
    response_event.set()

# Handle SocketIO errors
@sio.event
def connect_error(error):
    print(f"\nConnection error: {error}")

@sio.event
def connect_timeout():
    print("\nConnection timeout")

def signal_handler(sig, frame):
    print("\nDisconnecting and exiting...")
    if sio.connected:
        sio.disconnect()
    sys.exit(0)

# Function to wait for response with timeout
def wait_for_response():
    response_received = response_event.wait(timeout=response_timeout)
    if not response_received:
        print("\nResponse timeout. Server might be busy or not responding.")
        return False
    return True

async def main():
    global connected, response_event
    
    # Register signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)
    
    # Connect to the server
    try:
        print(f"Connecting to {SERVER_URL}...")
        # Set shorter timeouts
        sio.connect(SERVER_URL, transports=['websocket'], wait_timeout=10)
    except Exception as e:
        print(f"Failed to connect: {e}")
        return
    
    session = PromptSession()
    
    # Print instructions
    print("\n=== Interactive NPC Chat ===")
    print(f"NPC Role: {FIXED_NPC_ROLE}")
    print(f"Emotion: {FIXED_EMOTION}")
    print("Type 'quit' or 'exit' to disconnect and exit")
    
    with patch_stdout():
        while connected:
            try:
                # Get player input
                player_input = await session.prompt_async("\nYou say: ")
                
                if player_input.lower() in ['quit', 'exit']:
                    break
                
                # Skip empty inputs
                if not player_input.strip():
                    continue
                
                # Reset the response event
                response_event.clear()
                
                # Prepare the request data - using shorter max_length
                request_data = {
                    'npc_role': FIXED_NPC_ROLE,
                    'player_input': player_input,
                    'emotion': FIXED_EMOTION,
                    'max_length': 150  # Reduced max length for faster response
                }
                
                # Send the request
                print("Sending request...")
                sio.emit('generate_response', request_data)
                
                # Wait for the response with timeout
                print("Waiting for NPC response...")
                wait_for_response()
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
    
    # Disconnect when done
    if sio.connected:
        sio.disconnect()
    print("Disconnected. Goodbye!")

if __name__ == "__main__":
    # Import asyncio here to avoid potential conflicts
    import asyncio
    
    # Setup required packages
    print("Checking for required packages...")
    try:
        import prompt_toolkit
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                            "python-socketio[client]", "prompt_toolkit", "requests"])
        print("Packages installed successfully.")
    
    # Run the main function
    asyncio.run(main())