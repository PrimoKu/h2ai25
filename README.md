# Parkinson's Disease Monitoring System

A comprehensive multi-agent LLM-powered system for monitoring and analyzing health metrics to detect potential Parkinson's disease indicators.

## Overview

This project implements a real-time health monitoring system using multiple specialized AI agents to analyze different health metrics and synthesize the data to identify potential signs of Parkinson's disease. The system combines physiological sensors, cognitive assessments, mood analysis, and camera-based monitoring to provide a holistic analysis of Parkinson's disease indicators.

## System Architecture

```
┌───────────────────────────────────────────────────────────┐
│                         server.py                         │
└────────────────────────────┬──────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────┐
│                      Data Collection                      │
├───────────────┬───────────────┬───────────────┬───────────┤
│  parkSensors  │    parkCam    │ parkCognitive │  parkMood │
│ (Physiology)  │  (Monitoring) │  (Cognitive)  │  (Speech) │
└───────┬───────┴───────┬───────┴───────┬───────┴───────┬───┘
        │               │               │               │
        │               │               │               │
        ▼               ▼               ▼               ▼
┌───────────────────────────────────────────────────────────┐
│                   CSV Data Storage Files                  │
└────────────────────────────┬──────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────┐
│                        parkAgents                         │
├───────────────┬───────────────┬───────────────┬───────────┤
│     Blood     │     Heart     │     Motor     │   Speech  │
│    Pressure   │      Rate     │     Skills    │  Analysis │
│    Analysis   │    Analysis   │    Analysis   │           │
└───────┬───────┴───────┬───────┴───────┬───────┴───────┬───┘
        │               │               │               │
        └───────────────┼───────────────┼───────────────┘
                        │               │
                        ▼               ▼
┌───────────────────────────────────────────────────────────┐
│                    Parkinson Analyst                      │
│               (Meta-analysis of all data)                 │
└────────────────────────────┬──────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────┐
│                    JSON Analysis Files                    │
│              (Comprehensive assessment)                   │
└───────────────────────────────────────────────────────────┘
```

## Setup and Installation

### Prerequisites

- Python 3.8+
- OpenAI API key
- Required Python packages (see `requirements.txt`)
- Camera setup for parkCam features
- Audio recording capabilities for speech analysis

### Installation

1. Clone the repository
   ```
   git clone https://github.com/yourusername/multi_agent_server.git
   cd multi_agent_server
   ```

2. Install dependencies
   ```
   pip install -r requirements.txt
   ```

3. Configure the `config.py` file with your API keys and file paths

4. Ensure data storage directory exists
   ```
   mkdir -p data_storage
   ```

## Usage

Start the system by running:

```
python server.py
```

This will:
1. Initialize all sensor modules
2. Start the camera monitoring systems
3. Launch cognitive assessment tools
4. Enable audio emotion recognition
5. Start all analysis agents
6. Begin comprehensive Parkinson's disease monitoring

## Project Structure

```
multi_agent_server/
│── core/
│   │── parkAgents/                      # Analysis agents
│   │   │── agent_manager.py             # Manages all agent threads
│   │   │── blood_pressure_analyst.py    # Blood pressure monitoring agent
│   │   │── heart_rate_analyst.py        # Heart rate monitoring agent
│   │   │── motor_skill_analyst.py       # Motor skills monitoring agent
│   │   │── parkinson_analyst.py         # Meta-analysis agent
│   │   │── speech_analyst.py            # Speech pattern analysis agent
│   │── parkCam/                         # Camera-based monitoring systems
│   │   │── assessment/                  # Task-based assessments
│   │   │   │── draw.py                  # Drawing task analysis
│   │   │   │── flip.py                  # Object manipulation assessment
│   │   │   │── tap.py                   # Finger tapping test
│   │   │   │── write.py                 # Handwriting analysis
│   │   │── monitoring/                  # Passive monitoring systems
│   │   │   │── gait.py                  # Walking pattern analysis
│   │   │   │── posture.py               # Posture assessment
│   │   │   │── tremor.py                # Tremor detection
│   │   │   │── write.py                 # Real-time writing analysis
│   │   │── parkMedicine.py              # Medical recommendations
│   │── parkCognitive/                   # Cognitive assessment tools
│   │   │── cognitive_assessment.py      # Cognitive function evaluation
│   │── parkMood/                        # Emotional state analysis
│   │   │── AudioDoctor.py               # Audio-based health assessment
│   │   │── AudioEmotionRecognizer.py    # Emotion detection from voice
│   │── parkSensors/                     # Physical sensors
│   │   │── blood_pressure_sensor.py     # Blood pressure monitoring
│   │   │── heart_rate_sensor.py         # Heart rate monitoring
│── data_storage/                        # Data storage directory
│   │── gait_data.csv                    # Walking pattern data
│   │── postural_data.csv                # Posture assessment data
│   │── tremor_data.csv                  # Tremor measurements
│   │── blood_pressure_data.csv          # Blood pressure readings
│   │── heart_rate_data.csv              # Heart rate measurements
│   │── speech_data.csv                  # Speech pattern data
│   │── analyzed_blood_pressure.json     # Blood pressure analysis results
│   │── analyzed_heart_rate.json         # Heart rate analysis results
│   │── analyzed_motor_skill.json        # Motor skill analysis results
│   │── parkinson_analysis.json          # Comprehensive Parkinson's assessment
│── config.py                            # Configuration settings
│── server.py                            # Main entry point
```

## System Components

### parkSensors
Physical sensors for monitoring physiological metrics including:
- Blood pressure monitoring
- Heart rate tracking

### parkCam
Camera-based monitoring and assessment tools:
- **Assessment**: Active tasks like drawing, object manipulation, finger tapping and handwriting
- **Monitoring**: Passive observation of gait, posture, tremor and writing

### parkCognitive
Tools for assessing cognitive function, which can be affected in Parkinson's disease.

### parkMood
Audio analysis tools for:
- Speech pattern analysis for Parkinson's indicators
- Emotional state assessment through voice analysis

### parkAgents
LLM-powered analysis agents:
- Domain-specific agents for analyzing individual health metrics
- Parkinson Analyst for integrating all data into a comprehensive assessment

## How It Works

1. **Data Collection**: Multiple sensors and systems collect physiological, motor, cognitive, and emotional data
2. **Domain Analysis**: Specialized agents analyze specific health domains
3. **Visual Assessment**: Camera systems monitor movement patterns and perform specific task assessments
4. **Audio Analysis**: Voice patterns and emotional indicators are monitored
5. **Meta Analysis**: The Parkinson Analyst synthesizes all inputs to detect potential Parkinson's disease indicators
6. **Data Storage**: Raw data and analysis results are stored for historical tracking and review

## Technical Details

- **Multithreading**: Components run concurrently in separate threads
- **LangChain Integration**: Utilizes LangChain's agent framework for structured LLM interactions
- **GPT-4**: Analyses are performed using OpenAI's GPT-4 model
- **Computer Vision**: Camera-based monitoring uses computer vision techniques
- **Audio Processing**: Speech and emotion analysis leverages audio processing libraries

## Healthcare Applications

This system can assist healthcare providers by:
- Providing continuous monitoring between clinical visits
- Detecting subtle changes in patient condition
- Enabling remote assessment of Parkinson's symptoms
- Supporting early intervention through early detection
- Creating objective data-based progression tracking

## Limitations and Considerations

- **Not a Diagnostic Tool**: This system is designed to support, not replace, medical professionals
- **Data Privacy**: Health data should be handled according to HIPAA and other relevant regulations
- **API Costs**: Running multiple GPT-4 instances can be expensive for continuous monitoring
- **Technical Requirements**: Camera and audio setup may be complex for home deployment
- **Validation Required**: System outputs should be validated in clinical settings

## Future Enhancements

- Integration with wearable devices
- Mobile app for patient self-monitoring
- Medication tracking and adherence monitoring
- Machine learning models for personalized baseline establishment
- Integration with electronic health records

## License

[MIT License](LICENSE)

## Contributors

- Yihao Liu
- Yu-Chun Ku
- Tong Mu

## Acknowledgements

- OpenAI for GPT-4
- LangChain for the agent framework
