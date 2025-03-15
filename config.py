import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Paths for sensor data storage
DATA_STORAGE_PATH = "data_storage/"
BLOOD_PRESSURE_CSV = os.path.join(DATA_STORAGE_PATH, "blood_pressure.csv")
HEART_RATE_CSV = os.path.join(DATA_STORAGE_PATH, "heart_rate.csv")
MOTOR_SKILLS_CSV = os.path.join(DATA_STORAGE_PATH, "motor_skill.csv")

# Paths for analyzed data storage
ANALYZED_BLOOD_PRESSURE_JSON = os.path.join(DATA_STORAGE_PATH, "analyzed_blood_pressure.json")
ANALYZED_HEART_RATE_JSON = os.path.join(DATA_STORAGE_PATH, "analyzed_heart_rate.json")
ANALYZED_MOTOR_SKILLS_JSON = os.path.join(DATA_STORAGE_PATH, "analyzed_motor_skill.json")
ANALYZED_PARKINSON_JSON = os.path.join(DATA_STORAGE_PATH, "parkinson_analysis.json")

# Define time interval for analysis (in seconds)
ANALYSIS_INTERVAL = 60