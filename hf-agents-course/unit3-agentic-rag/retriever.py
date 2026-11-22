# ============================================================================
#  retriever.py: the RAG tool (course page "Creating a RAG Tool for Guest Stories")
# ============================================================================
# The R of RAG: find the right guest in a list, and give their text to Alfred.
#
#   STEP 1  Load and prepare the dataset    (the guest list, from Hugging Face)
#   STEP 2  Create the retriever tool       (BM25: search by matching words)
#
# Same idea as a Lupin tool: a description the MODEL reads (name, description,
# inputs) + a Python function that RUNS (forward). smolagents just puts both
# in one class instead of our describe() + FUNCTIONS dict.
#
# Course code, unchanged, except: it's wrapped in load_guest_dataset(),
# because app.py imports that name (as the course's app.py does).
# ============================================================================

import datasets
from langchain_core.documents import Document
from smolagents import Tool
from langchain_community.retrievers import BM25Retriever


# ----------------------------------------------------------------------------
#  STEP 2: the tool (defined first, because STEP 1 below uses it)
# ----------------------------------------------------------------------------
class GuestInfoRetrieverTool(Tool):
    # What the MODEL sees (the equivalent of Lupin's describe(...)):
    name = "guest_info_retriever"
    description = "Retrieves detailed information about gala guests based on their name or relation."
    inputs = {
        "query": {
            "type": "string",
            "description": "The name or relation of the guest you want information about."
        }
    }
    output_type = "string"

    def __init__(self, docs):
        self.is_initialized = False
        # BM25 = score each guest by how many of the query's words appear in
        # their text (rare words count more). No AI here, just word counting.
        self.retriever = BM25Retriever.from_documents(docs)

    # What really RUNS when the model calls the tool (like Lupin's read()):
    def forward(self, query: str):
        results = self.retriever.invoke(query)       # the best-matching guests, best first
        if results:
            return "\n\n".join([doc.page_content for doc in results[:3]])   # keep the top 3, as text
        else:
            return "No matching guest information found."


# ----------------------------------------------------------------------------
#  STEP 1: load the guests and build the tool
# ----------------------------------------------------------------------------
def load_guest_dataset():
    # Load the dataset (downloads it from Hugging Face the first time)
    guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

    # Convert dataset entries into Document objects: one text per guest,
    # with the 4 fields on 4 lines. This text is what the search looks in,
    # and what Alfred receives back.
    docs = [
        Document(
            page_content="\n".join([
                f"Name: {guest['name']}",
                f"Relation: {guest['relation']}",
                f"Description: {guest['description']}",
                f"Email: {guest['email']}"
            ]),
            metadata={"name": guest["name"]}
        )
        for guest in guest_dataset
    ]

    # Initialize the tool
    return GuestInfoRetrieverTool(docs)


# Run this file alone to test the tool WITHOUT any model:
#   python3 retriever.py
if __name__ == "__main__":
    guest_info_tool = load_guest_dataset()
    print(guest_info_tool("Lady Ada Lovelace"))
