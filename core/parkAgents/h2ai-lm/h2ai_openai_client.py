import dotenv
import openai
import rich

class h2aiOpenAIClient:
    """
    Client class to use OpenAI APIs
    """

    def __init__(self):
        """
        Init
        """
        dotenv.load_dotenv(override=True)
        self.client = openai.OpenAI()

    def call_gpt(self, msg, verbose=False):
        """
        Call the API and do a prompt completion
        """
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user", 
                    "content": msg
                }
            ], 
            model="gpt-4-turbo",
            temperature=0.8,
            max_tokens=256
        )
        if verbose:
            rich.print(response)
        return response.choices[0].message.content
    
    def call_gpt_conversation(self, msg, conversation, verbose=False):
        """
        Call the API with a conversation history
        """
        conversation.append(
            {
                "role": "user", 
                "content": msg
            }
        )
        response = self.client.chat.completions.create(
            messages=conversation, 
            model="gpt-4-turbo",
            temperature=0.8,
            max_tokens=256
        )
        if verbose:
            rich.print(response)
        return response.choices[0].message.content

def main():
    """
    Test class
    """
    c = h2aiOpenAIClient()
    while True:
        r = c.call_gpt(input("Please enter your prompt: "))
        rich.print(r)

if __name__ == '__main__':
    main()