import json, pickle, sys, rich
sys.path.append("core/parkAgents/h2ai-lm")
from h2ai_retrieval import relevent_retrieve
from h2ai_agents import h2aiCharacter, h2aiAgent, h2aiContextAgent
from h2ai_openai_client import h2aiOpenAIClient


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

def main():
    """
    Test class
    """
    a, c, a_summarizer, bm25_pkl_expert = run_pipeline()
    r = a.chat("I am a Parkinson's disease patient. You are my doctor. Please give me a Mini-Mental State Examination (MMSE) or Montreal Cognitive Assessment (MoCA). Choose one examination. One question a time and don't say anything else until all questions are answered. Wait for me to ask you to start. Give me the evaluation results once all questions are ready. Don't worry, it is just a test.", save_conv=True)
    while True:
        r = a.chat(input("Please enter your prompt: "), save_conv=True)
        rich.print(r)
    
if __name__ == '__main__':
    main()
