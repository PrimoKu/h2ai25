from h2ai_openai_client import h2aiOpenAIClient
import random
import rich
import pickle
import json

h2aiNames = ["Yihao Liu", "Yu-Chun Ku", "Tong Mu"]

h2aiCharacterNames = {
    "summarizer"
        : h2aiNames[random.randint(0, len(h2aiNames)-1)],
    "cognitive"
        : h2aiNames[random.randint(0, len(h2aiNames)-1)]
}

h2aiCharacterRoles = {
    "summarizer" 
        : (
            "a medical professional that provides summary of large documents "
        )  ,
    "cognitive"
        : (
            "a cognitive function assessment tool that helps to assess cognitive "
        )
}

h2aiCharacterStyles = {
    "summarizer" 
        : (
            "You generally try to speak professionally using medical terminology "
            "and concisely."
        ),
    "cognitive"
        : (
            "You generally try to speak professionally using medical terminology "
            "and concisely."
        )
}

h2aiCharacterStarters = {
    "summarizer" 
        : "Hi, I am your summarizer. How can I help you today?",
    "cognitive"
        : "Hi, I am your cognitive function assessment tool. How can I help you today?"    
}

class h2aiCharacter():
    """
    Agent characters
    """

    def __init__(self, entity, language="English"):
        """
        Init
        """

        self.name = h2aiCharacterNames[entity]
        self.role = h2aiCharacterRoles[entity]
        self.style = h2aiCharacterStyles[entity]
        self.starter = h2aiCharacterStarters[entity]
        self.language = language

class h2aiAgent():
    """
    Base agent
    """

    def __init__(self, client : h2aiOpenAIClient, character : h2aiCharacter):
        """
        Init
        """
        self.name = character.name
        self.client = client
        self.prior = (
            f"Your name is {character.name}, and you speak {character.language}. "
            f"You are {character.role}. "
            f"{character.style}"
        )
        self.conversation = [
            {
                "role": "system", 
                "content": self.prior
            },
            {
                "role": "assistant",
                "content": character.starter
            }
        ]

    def chat(self, msg, verbose=False, memory=True, save_conv=False):
        """
        Get response
        """
        r = self.client.call_gpt_conversation(
            msg, self.conversation
        )
        if not memory:
            self.conversation.pop()
        else:
            self.conversation.append(
                {
                    "role": "assistant",
                    "content": r
                }
            )
        if verbose:
            rich.print(self.conversation)
        if save_conv:
            self.save_conversation()
        return r
    
    def save_conversation(self):
        with open("sample_conversation", 'w') as file:
            json.dump(self.conversation, file)

class h2aiContextAgent(h2aiAgent):
    """
    Agent class that has context
    """
    def __init__(self, client: h2aiOpenAIClient, character: h2aiCharacter, context=None, conversation=None):
        """
        Init
        """

        super().__init__(client, character)
        self.context = context
        if conversation:
            self.conversation = conversation
        self.conversation.append(
            {
                "role": "system",
                "content": (
                    "The following is the context of the conversion. "
                    "Use the context to answer the followed questions. "
                    "\n"
                    f"{str(self.context)}"
                    "\n"
                )
            }
        )

    def chat(self, msg, verbose=False, save_conv=False):
        """
        Get response
        """
        r = self.client.call_gpt_conversation(
            msg, self.conversation
        )
        self.conversation.append(
            {
                "role": "assistant",
                "content": r
            }
        )
        if verbose:
            rich.print(self.conversation)
        if save_conv:
            self.save_conversation()
        return r


def main():
    """
    Test class
    """
    c = h2aiOpenAIClient()
    ch = h2aiCharacter("summarizer")
    a = h2aiAgent(c, ch)
    while True:
        r = a.chat(input("Please enter your prompt: "), True, False)
        rich.print(r)

if __name__ == '__main__':
    main()