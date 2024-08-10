import chromadb

def load_chroma_db(name: str) -> chromadb.Collection:
    """
    Loads a ChromaDB collection from the client.

    Parameters:
    - name (str): The name of the collection to load.

    Returns:
    - chromadb.Collection: The loaded ChromaDB collection.
    """
    # Initialize the Chroma client
    client = chromadb.Client()

    # Retrieve the collection by name
    db = client.get_collection(name=name)

    return db

def get_relevant_passage(query: str, db: chromadb.Collection, n_results: int):
    """
    Retrieves relevant passages from the ChromaDB based on the user query.

    Parameters:
    - query (str): The query text to search for in the database.
    - db (chromadb.Collection): The ChromaDB collection to query.
    - n_results (int): The number of relevant results to return.

    Returns:
    - List[str]: A list of the most relevant passages.
    """
    # Query the ChromaDB collection
    results = db.query(query_texts=[query], n_results=n_results)

    # Extract relevant passages
    passages = results['documents']
    
    return passages

# Example usage
collection_name = "rag_experiment"

# Load the database
db = load_chroma_db(name=collection_name)

# Retrieve relevant passages
relevant_passages = get_relevant_passage(query="regular expression for phone number matching", db=db, n_results=3)

# Print the relevant passages
for i, passage in enumerate(relevant_passages):
    print(f"Passage {i + 1}: {passage}")
