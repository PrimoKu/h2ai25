import json, pickle, sys, rich
sys.path.append("core/parkAgents/h2ai-lm")
from h2ai_retrieval import relevent_retrieve
from h2ai_agents import h2aiCharacter, h2aiAgent, h2aiContextAgent
from h2ai_openai_client import h2aiOpenAIClient
import socket
from threading import Thread

def run_pipeline():
    file_path = "expert_knowledge/expert_knowledge.json.bm25_with_keys.pkl"
    with open(file_path, 'rb') as file:
        bm25_pkl_expert = pickle.load(file)

    # Main agent
    c = h2aiOpenAIClient()
    ch_summarizer = h2aiCharacter("summarizer")
    a_summarizer = h2aiAgent(c, ch_summarizer)

    ch = h2aiCharacter("cognitive")
    a = h2aiContextAgent(c, ch)

    return a, c, a_summarizer, bm25_pkl_expert

def send_response(addr, response):
    """
    Send a response back to the client using port 65433.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((addr[0], 65433))  # Connect to the client's address on port 65433
        client_socket.sendall(response.encode('utf-8'))

def tcp_server(host='127.0.0.1', port=65432):
    """
    Simple TCP server that listens for incoming connections and messages.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((host, port))
        server_socket.listen()
        print(f"TCP server listening on {host}:{port}")

        while True:
            conn, addr = server_socket.accept()
            with conn:
                print(f"Connected by {addr}")
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    msg = data.decode('utf-8')
                    yield msg, addr  # Yield the received message and the client's address

def main():
    """
    Test class
    """
    a, c, a_summarizer, bm25_pkl_expert = run_pipeline()
    r = a.chat("I am a Parkinson's disease patient. You are my doctor. Please give me a Mini-Mental State Examination (MMSE) or Montreal Cognitive Assessment (MoCA). Choose one examination. One question a time and don't say anything else until all questions are answered. Wait for me to ask you to start. Give me the evaluation results once all questions are ready. Don't worry, it is just a test.", save_conv=True)
    
    # Start the TCP server in a separate thread
    tcp_thread = Thread(target=tcp_server)
    tcp_thread.daemon = True
    tcp_thread.start()

    # Get messages from the TCP server and pass them to the chat function
    for msg, addr in tcp_server():
        r = a.chat(msg, save_conv=True)
        rich.print(r)
        # Send the response back to the client
        send_response(addr, r)
    
if __name__ == '__main__':
    main()