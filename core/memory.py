import json
import os
from langchain.memory import ConversationBufferMemory

MEMORY_FILE = "memory_storage.json"

class MemoryManager:
    def __init__(self):
        if not os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "w") as f:
                json.dump({}, f)

    def _load_memory(self):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)

    def _save_memory(self, memory_data):
        with open(MEMORY_FILE, "w") as f:
            json.dump(memory_data, f, indent=4)

    def save_memory(self, patient_id, memory):
        """Save memory data as JSON."""
        memory_data = self._load_memory()
        memory_data[patient_id] = memory.load_memory_variables({})
        self._save_memory(memory_data)

    def load_memory(self, patient_id):
        """Load memory from JSON file."""
        memory_data = self._load_memory()
        memory = ConversationBufferMemory()
        if patient_id in memory_data:
            for item in memory_data[patient_id]["history"]:
                memory.save_context({"input": item["input"]}, {"output": item["output"]})
        return memory

    def clear_memory(self, patient_id):
        """Clear memory for a specific patient."""
        memory_data = self._load_memory()
        if patient_id in memory_data:
            del memory_data[patient_id]
            self._save_memory(memory_data)

memory_manager = MemoryManager()
