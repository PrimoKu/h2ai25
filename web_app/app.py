# app.py

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from flask_cors import CORS
import json
import openai
import traceback

app = Flask(__name__)
CORS(app)
messages = []
# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico', mimetype='image/vnd.microsoft.icon')
    
@app.route('/data_storage/<path:filename>')
def serve_data(filename):
    return send_from_directory('data_storage', filename)
    
@app.route('/api/message', methods=['POST'])
def receive_message():
    data = request.get_json()
    if not data or 'speaker' not in data or 'context' not in data:
        return jsonify({'error': 'Invalid payload'}), 400
    messages.append(data)
    print("Received message:", data)
    return jsonify({'status': 'Message received'})

@app.route('/api/messages', methods=['GET'])
def get_messages():
    global messages
    msgs = messages[:]  # Copy current messages
    messages = []      # Clear the store after sending
    return jsonify(msgs)

if __name__ == '__main__':
    app.run(debug=True)

