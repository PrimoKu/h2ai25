# app.py

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json
import openai
import traceback

app = Flask(__name__)

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
    
if __name__ == '__main__':
    app.run(debug=True)

