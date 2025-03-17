# Multi-Agent Parkinson's Disease Monitoring System

An advanced multi-agent system leveraging LLMs and computer vision to monitor, analyze, and detect potential Parkinson's disease indicators with an integrated web application for data visualization.

## Overview

This project implements a sophisticated health monitoring system that combines multiple specialized AI agents, computer vision analysis, and sensor data to identify potential signs of Parkinson's disease. The system includes a web application for visualizing all data assessments and analyses.

Key features:

- **Physiological metrics**: Blood pressure, heart rate monitoring
- **Motor functions**: Gait, posture, tremor, writing, grip strength, coordination
- **Cognitive assessment**: Memory, cognition, processing speed
- **Speech and mood analysis**: Speech patterns and emotional indicators
- **Web-based visualization**: Interactive dashboard for all monitoring and analysis results

Each domain is analyzed by specialized AI agents powered by OpenAI's GPT-4o language models, with a meta-analysis agent that synthesizes all inputs to provide comprehensive Parkinson's disease assessment.

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
- OpenAI API key (for GPT-4 agents)
- Required Python packages (see `requirements.txt`)
- Webcam or camera for computer vision features
- Sensors for physiological measurements

### Installation

1. Clone the repository
   ```
   git clone https://github.com/yourusername/multi-agent-parkinsons-system.git
   cd multi-agent-parkinsons-system
   ```

2. Install dependencies
   ```
   pip install -r requirements.txt
   ```

3. Configure the `config.py` file with your API keys and file paths
   ```python
   # API Keys
   OPENAI_API_KEY = "your-openai-api-key"
   
   # File Paths (already set up with correct project structure)
   BLOOD_PRESSURE_CSV = "web_app/data_storage/blood_pressure_data.csv"
   HEART_RATE_CSV = "web_app/data_storage/heart_rate_data.csv"
   # Additional file paths...
   
   ANALYZED_BLOOD_PRESSURE_JSON = "web_app/data_storage/analyzed_blood_pressure.json"
   ANALYZED_HEART_RATE_JSON = "web_app/data_storage/analyzed_heart_rate.json"
   # Additional analysis file paths...
   ```

4. Ensure you have access to required hardware (sensors, camera)

## Usage

### Starting the Monitoring System

Start the main system by running:

```
python server.py
```

This will:
1. Initialize all sensor and data collection modules
2. Start computer vision monitoring if enabled
3. Launch all agent threads for continuous analysis
4. Begin cognitive and speech assessment when triggered
5. Keep the server running until manual termination (Ctrl+C)

### Starting the Web Application

Start the web interface by running:

```
cd web_app
python app.py
```

This will launch a web server that provides:
1. Real-time visualization of all sensor data
2. Interactive dashboard for viewing analysis results
3. Historical trends and patterns
4. Comprehensive Parkinson's assessment reports

Access the web interface by navigating to `http://localhost:5000` in your browser (or the port specified in the app configuration).

## Project Structure

```
multi_agent_server/
│── core/
│   │── parkAgents/                    # LLM-based analysis agents
│   │   │── agent_manager.py           # Orchestrates all agents
│   │   │── blood_pressure_analyst.py  # Blood pressure monitoring
│   │   │── heart_rate_analyst.py      # Heart rate analysis
│   │   │── motor_skill_analyst.py     # Motor function analysis
│   │   │── parkinson_analyst.py       # Meta-analysis agent
│   │   │── speech_analyst.py          # Speech pattern analysis
│   │   │── h2ai-lm/                   # LLM integration
│   │       │── h2ai_agents.py         # Agent implementations
│   │       │── h2ai_openai_client.py  # OpenAI compatibility layer
│   │       │── h2ai_retrival.py       # Knowledge retrieval
│   │── parkCam/                       # Computer vision components
│   │   │── assessment/                # Task-based assessments
│   │   │   │── draw.py                # Drawing task analysis
│   │   │   │── flip.py                # Card flipping test
│   │   │   │── tap.py                 # Tapping test analysis
│   │   │   │── write.py               # Writing analysis
│   │   │── monitoring/                # Continuous monitoring
│   │       │── gait.py                # Walking pattern analysis
│   │       │── posture.py             # Posture analysis
│   │       │── tremor.py              # Tremor detection
│   │       │── write.py               # Handwriting analysis
│   │   │── parkMedicine.py            # Medical context integration
│   │── parkCognitive/                 # Cognitive assessment
│   │   │── cognitive_assessment.py    # Cognitive test suite
│   │── parkMood/                      # Emotional analysis
│   │   │── AudioDoctor.py             # Audio health analysis
│   │   │── AudioEmotionRecognizer.py  # Emotion detection from speech
│   │── parkSensors/                   # Physical sensor modules
│       │── blood_pressure_sensor.py   # Blood pressure monitoring
│       │── heart_rate_sensor.py       # Heart rate monitoring
│── expert_knowledge/                  # Knowledge base for the system
│── web_app/                           # Web application for visualization
│   │── data_storage/                  # Data and analysis storage
│   │   │── gait_data.csv              # Walking data
│   │   │── postural_data.csv          # Posture metrics
│   │   │── tremor_data.csv            # Tremor measurements
│   │   │── blood_pressure_data.csv    # Blood pressure readings
│   │   │── heart_rate_data.csv        # Heart rate data
│   │   │── speech_data.csv            # Speech recordings/analysis
│   │   │── analyzed_blood_pressure.json # Agent analysis results
│   │   │── analyzed_heart_rate.json   # Agent analysis results
│   │   │── analyzed_motor_skill.json  # Agent analysis results
│   │   │── parkinson_analysis.json    # Final combined assessment
│   │── app.py                         # Web application entry point
│── cognitive_assessment_stand_alone.py # Standalone cognitive test
│── cognitive_assessment.py            # Integrated cognitive assessment
│── config.py                          # Configuration settings
│── server.py                          # Main system entry point
```

## Key Components

### 1. Sensor Data Collection
The `parkSensors` module interfaces with physical sensors to collect physiological data like blood pressure and heart rate.

### 2. Computer Vision Analysis
The `parkCam` module uses computer vision to:
- Monitor motor symptoms (tremor, gait abnormalities)
- Assess performance on structured tasks (drawing, writing, card flipping)
- Track posture and movement patterns

### 3. Cognitive Assessment
The `parkCognitive` module provides tests to evaluate cognitive functions often affected by Parkinson's disease.

### 4. Mood and Speech Analysis
The `parkMood` module analyzes speech patterns and emotional indicators that may signal neurological changes.

### 5. LLM-Powered Agents
The `parkAgents` module contains specialized LLM agents that analyze domain-specific data and provide expert assessments.

### 6. Meta-Analysis
The `parkinson_analyst.py` integrates all agent outputs to provide a comprehensive Parkinson's disease assessment.

### 7. Web Application
The `web_app` directory contains a web application that:
- Visualizes all collected data in real-time
- Displays analysis results from individual agents
- Presents comprehensive assessment reports
- Tracks historical trends and changes over time
- Provides an intuitive interface for healthcare providers and researchers

## Web Application Features

The web application serves as a comprehensive dashboard for monitoring and assessment:

- **Real-time Visualization**: Live displays of sensor data and computer vision analysis
- **Interactive Dashboards**: Customizable views for different data types and analyses
- **Historical Data**: Trend analysis and progression tracking over time
- **Assessment Reports**: Detailed reports from each specialized agent
- **Integrated Analysis**: Comprehensive Parkinson's disease assessment
- **Alert System**: Notifications for significant changes or concerning patterns
- **Export Functionality**: Export data and reports for medical records or research

## Extending the System

### Adding New Sensors
1. Create a new sensor module in `core/parkSensors/`
2. Update `config.py` with new file paths
3. Modify relevant analysis agents to use the new data
4. Add visualization component to the web application

### Adding New Computer Vision Features
1. Add new monitoring or assessment scripts to the appropriate `parkCam` subdirectory
2. Integrate with existing agent frameworks or create a new agent for the data
3. Update the web application to display the new analysis

### Adding New Agent Types
1. Create a new agent in `core/parkAgents/`
2. Add the agent to `agent_manager.py`
3. Update `parkinson_analyst.py` to incorporate the new analysis
4. Add corresponding visualization to the web application

## Technical Details

- **Multi-Modal Analysis**: Combines text, audio, and visual data for comprehensive assessment
- **LLM Integration**: Uses OpenAI's GPT-4o language models for analysis
- **Multithreaded Architecture**: Parallel processing of different data streams and analyses
- **Scheduled Monitoring**: Regular assessment cycles with configurable frequencies
- **Computer Vision**: Real-time monitoring of physical symptoms and task performance
- **Web Framework**: Flask/Dash-based web application with interactive visualizations
- **Data Storage**: Structured CSV and JSON storage with optional database integration

## Limitations and Considerations

- **Not a Medical Diagnostic Tool**: This system is for research and monitoring purposes only and should not replace professional medical diagnosis
- **Privacy Concerns**: Handles sensitive health data requiring appropriate privacy measures
- **Resource Intensity**: The full system requires significant computational resources
- **API Costs**: Using multiple LLM instances can incur substantial API costs
- **Sensor Accuracy**: Results are dependent on the quality and accuracy of input sensors

## Future Development

- Integration with wearable devices for continuous monitoring
- Mobile application for remote assessment and monitoring
- Enhanced machine learning models for improved symptom detection
- Integration with electronic health records
- Personalized treatment recommendation engine
- Expanded visualization capabilities in the web application

<!-- ## License

[MIT License](LICENSE) -->

## Contributors

- Yihao Liu
- Yu-Chun Ku
- Tong Mu

## Acknowledgements

- OpenAI for GPT-4o
- LangChain for the agent framework
- Medical advisors and Parkinson's disease specialists for domain expertise