"""
Mini-agent: RAG + tool use + memory, all combined.

Learning project. This ties together everything from the previous three
projects into one bot that decides FOR ITSELF which capability to use for
each question:

  - Your CV (RAG / semantic search over a PDF)       -> exposed as a tool
  - A calculator                                      -> tool
  - Current date/time                                 -> tool
  - Weather for a city                                -> tool
  - Wikipedia search (NEW: general knowledge lookup)  -> tool

The key design idea: RAG isn't special anymore. It's just another tool in
the list, alongside the others. Claude picks whichever tool (if any) fits
the question -- including calling more than one for a single question.

"""

import os
import sys
import glob
import ast
import operator
from datetime import datetime

import requests
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic

MODEL_CLAUDE = "claude-sonnet-5"
MODEL_EMBEDDINGS = "paraphrase-multilingual-MiniLM-L12-v2"
DOCUMENT_FOLDER = "documents"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 3
MAX_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to several tools. Use "
    "search_cv for any question about the user's personal background, "
    "work experience, education or skills. Use the other tools when they "
    "give you real, accurate information instead of guessing. If no tool "
    "fits, just answer directly."
)

#====================================================================================
# -- RAG SETUP : same as the RAG PROJECT https://github.com/GuillermoSaSu/ChatbotRAG
#====================================================================================

# -- First step: extract text from PDF --

def extract_text_pdf(pdf_path: str) -> str:
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# -- Second step: break the text into fragments --
def chunk_text(text: str, size: int, solape: int) -> list[str]:
    fragments = []
    start = 0
    while start < len(text):
        end = start + size
        fragments.append(text[start:end].strip())
        start += size - solape
    #Delete empty fragments or ones which are too short.
    return [f for f in fragments if len(f) > 30]

# -- Third step: prepare the embedding index --
def build_index(model_embeddings: SentenceTransformer, fragments: list[str]):
    """
    Convert each text fragment into a vector (embedding).
    We return the matrix of vectors so that searches can be performed on it.
    """
    vectors = model_embeddings.encode(fragments, normalize_embeddings=True)
    return np.array(vectors)

# -- Forth step: semantic search --
def search_chunks(
        question: str,
        model_embeddings: SentenceTransformer,
        fragments: list[str],
        vector_fragments: np.ndarray,
        top_k: int,
) -> list[str]:
    """
    Given a question, we convert it into a vector and calculate its similarity
    (dot product, since the vectors are normalized = cosine similarity)
    with each chunk. We return the top_k most similar ones.
    """
    vector_query = model_embeddings.encode([question], normalize_embeddings=True)[0]
    similarities = vector_fragments @ vector_query #dot product
    better_index = np.argsort(similarities)[::-1][:top_k]
    return [fragments[i] for i in better_index]

#===================================================================================================
# -- TOOL FUNCTIONS : same as the Tool use PROJECT https://github.com/GuillermoSaSu/ChatBotToolUse
#===================================================================================================

# --- Step 1: define the actual Python function ("tools") ---

def get_current_time() -> str:
    #Tool without parameters. Returns the date.
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#Only these operators are allowed in the calculator, so we never accidentally execute arbitrary code via eval()
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsopported expression")

def calculate(expression: str) -> str:
    #A tool with a  parameter. Safely evaluates a basic math expression.
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as ex:
        return f"Error evaluating expression: {ex}"

def get_weather(city: str) -> str:
    #A tool that calls a real external API (no key needed, wttr.in).
    try:
        response = requests.get(f"https://wttr.in/{city}?format=%C+%t", timeout=5)
        response.raise_for_status()
        return response.text.strip()
    except Exception as ex:
        return f"Error fetching weather: {ex}"

def search_wikipedia(topic : str) -> str:
    """NEW tool: looks up a short summary of a topic on Wikipedia (no API key needed)."""
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(topic)}"
        headers = {
            "User-Agent" : "MiniAgentLearningProject/1.0 (educational project; contact: youremail@email.com)"
        }
        r = requests.get(url, headers=headers, timeout = 5)
        if r.status_code == 404:
            return f"No Wikipedia article found for topic {topic}."
        r.raise_for_status()
        data = r.json()
        return data.get("extract", "No summary available.")
    except Exception as ex:
        return f"Error searching Wikipedia: {ex}"

# =========================================================================
# TOOL SCHEMAS -- what Claude sees to decide when/how to call each one
# =========================================================================

TOOLS = [
    {
        "name" : "search_cv",
        "description" : "Search the user's CV for information about their background, work experience, education, or skills.",
        "input_schema" : {
            "type": "object",
            "properties" : {"query": {"type": "string", "description": "What to look for in the CV"}},
            "required": ["query"],
        },
    },
    {
        "name" : "get_current_time",
        "description" : "Get the current date and time.",
        "input_schema" : {"type" : "object", "properties" : {}},
    },
    {
        "name" : "calculate",
        "description" : "Evaluate a basic math expression (+, -, *, /, **)",
        "input_schema" : {
            "type" : "object",
            "properties" : {
                "expression" : {
                    "type" : "string",
                    "description" : "Math expression e.g. '12 * (3+4)",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name" : "get_weather",
        "description" : "Get the current weather for a city.",
        "input_schema" : {
            "type" : "object",
            "properties" : {
                "city" : {"type" : "string", "description" : "City name, e.g. 'Burgos'"},
            },
            "required": ["city"],
        },
    },
    {
        "name" : "search_wikipedia",
        "description" : "Look up a short summary of a general-knowledge topic on Wikipedia.",
        "input_schema" : {
            "type" : "object",
            "properties" : {"topic" : {"type" : "string", "description" : "Topic or article title"}},
            "required": ["topic"],
        },
    },
]

def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Please set the ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)

    pdf_paths = glob.glob(os.path.join(DOCUMENT_FOLDER, "*.pdf"))
    if not pdf_paths:
        print("No PDF files found.")
        sys.exit(1)

    print(f"📄 Processing: {pdf_paths[0]}")
    text = extract_text_pdf(pdf_paths[0])
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"   {len(chunks)} chunks generated.")

    print("🧠 Loading local embeddings model...")
    embed_model = SentenceTransformer(MODEL_EMBEDDINGS)
    vectors = build_index(embed_model, chunks)

    # Now that we have chunks/vectors in scope we can define search_cv as a
    # closure that has access to them. This is why it's defined here and not next to the other tool functions above.
    def search_cv(query: str) -> str:
        relevant = search_chunks(query, embed_model, chunks, vectors, TOP_K)
        return "\n\n---\n\n".join(relevant)

    tool_functions = {
        "search_cv": lambda **kw: search_cv(kw["query"]),
        "get_current_time": lambda **kw: get_current_time(),
        "calculate": lambda **kw: calculate(kw["expression"]),
        "get_weather": lambda **kw: get_weather(kw["city"]),
        "search_wikipedia": lambda **kw: search_wikipedia(kw["topic"]),
    }

    client = Anthropic()
    history = []

    print("\n🤖 Ready. Ask about your CV, the weather, a calculation, or any general topic.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_message = input("You: ").strip()
        except(EOFError, KeyboardInterrupt):
            print("See you!")
            break

        if user_message.lower() == "exit":
            print("See you!")
            break
        if not user_message:
            continue

        history.append({"role" : "user", "content" : user_message})

        while True:
            response = client.messages.create(
                model = MODEL_CLAUDE,
                max_tokens = MAX_TOKENS,
                system = SYSTEM_PROMPT,
                tools = TOOLS,
                messages = history,
            )

            history.append({"role" : "assistant", "content" : response.content})

            if response.stop_reason != "tool_use":
                for block in response.content:
                    if block.type == "text":
                        print(f"Claude: {block.text}\n")
                break

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"   🔧 Calling: {block.name}({block.input})")
                    result = tool_functions[block.name](**block.input)
                    tool_results.append({
                        "type" : "tool_result",
                        "tool_use_id" : block.id,
                        "content" : str(result),
                    })

            history.append({"role" : "user", "content" : tool_results})

if __name__ == "__main__":
    main()