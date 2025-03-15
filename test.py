from core.parkAgents.stateful_agent import BaseLangChainAgent
import config

# For demonstration, we define a dummy retriever.
# In practice, replace this with an actual retriever from a vector store.
class DummyRetriever:
    def get_relevant_documents(self, query):
        # Returns a list of dummy documents relevant to the query.
        return [{"page_content": f"Dummy document relevant to '{query}'."}]

dummy_retriever = DummyRetriever()

# Initialize the agent with your OpenAI API key and the retriever.
agent = BaseLangChainAgent(
    openai_api_key= config.OPENAI_API_KEY,
    retriever=dummy_retriever
)

# Get a response from the agent.
user_query = "What are the benefits of retrieval augmented generation?"
reply = agent.get_response(user_query)
print("Assistant:", reply)
