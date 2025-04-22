import asyncio
import websockets
import json

async def send_and_receive_message(websocket, npc_role, emotion):
    while True:
        # Get player input from the user
        player_input = input("Player Input: ").strip()
        
        # Break the loop if the user types 'exit'
        if player_input.lower() == 'exit':
            print("Exiting the conversation.")
            break
        
        # Create the message payload
        message = {
            "npc_role": npc_role,
            "emotion": emotion,
            "player_input": player_input
        }
        
        # Send the message
        await websocket.send(json.dumps(message))
        print(f"Sent data: npc_role={npc_role}, emotion={emotion}, player_input={player_input}")

        # Receive the response
        response = await websocket.recv()
        response_data = json.loads(response)
        
        if "response" in response_data:
            print(f"NPC Response: {response_data['response']}\n")
        elif "error" in response_data:
            print(f"Error: {response_data['error']}\n")

async def main():
    uri = "ws://localhost:8765"
    npc_role = "Mechanic"
    emotion = "Envy"

    async with websockets.connect(uri) as websocket:
        await send_and_receive_message(websocket, npc_role, emotion)

if __name__ == "__main__":
    asyncio.run(main())