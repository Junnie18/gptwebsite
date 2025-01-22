import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sympy import symbols, sympify, diff, integrate, limit
from fastapi import FastAPI
from pydantic import BaseModel

################################################################################
# 1. Define Our FastAPI App
################################################################################
app = FastAPI()

################################################################################
# 2. Load Model at Startup (Optional: for Performance)
################################################################################
print("Loading GPT-J model...")
model_name = "EleutherAI/gpt-j-6B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
print("Model loaded.")

################################################################################
# 3. Your Existing Utility Functions
################################################################################
def handle_calculus_request(query: str) -> str:
    try:
        query = query.lower()

        # Differentiate
        if "differentiate" in query or "derivative" in query:
            parts = query.split("with respect to")
            if len(parts) == 2:
                func = parts[0].replace("differentiate", "").replace("derivative of", "").strip()
                var = parts[1].strip()
                variable = symbols(var)
                expression = sympify(func)
                result = diff(expression, variable)
                return f"The derivative of {func} with respect to {var} is: {result}"

        # Integrate
        elif "integrate" in query:
            parts = query.split("with respect to")
            if len(parts) == 2:
                func = parts[0].replace("integrate", "").strip()
                var = parts[1].strip()
                variable = symbols(var)
                expression = sympify(func)
                result = integrate(expression, variable)
                return f"The integral of {func} with respect to {var} is: {result}"

        # Limit
        elif "limit" in query:
            match = re.search(r"limit of (.+) as (.+) approaches (.+)", query)
            if match:
                func, var, value = match.groups()
                variable = symbols(var.strip())
                expression = sympify(func.strip())
                limit_value = sympify(value.strip())
                result = limit(expression, variable, limit_value)
                return f"The limit of {func} as {var} approaches {value} is: {result}"

        # Fallback for unsupported calculus queries
        else:
            return "Unsupported calculus query. Please specify differentiation, integration, or limit."

    except Exception as e:
        return f"Error processing calculus query: {e}"

def is_calculus_query(query: str) -> bool:
    keywords = ["differentiate", "derivative", "integrate", "limit"]
    return any(keyword in query.lower() for keyword in keywords)

################################################################################
# 4. Define FastAPI Request Model
################################################################################
class QueryRequest(BaseModel):
    query: str

################################################################################
# 5. Define Our Endpoints
################################################################################
@app.post("/generate")
def generate(request: QueryRequest):
    """
    POST endpoint to handle both general GPT-J queries and calculus queries.
    Example JSON body:
    {
      "query": "differentiate x^2 with respect to x"
    }
    """
    query = request.query.strip()
    if is_calculus_query(query):
        response = handle_calculus_request(query)
    else:
        # Use GPT-J model for a general query
        inputs = tokenizer(query, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            temperature=0.2,
            top_k=50,
            do_sample=True
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"response": response}

################################################################################
# Optional: A Simple Health Check or Root Endpoint
################################################################################
@app.get("/")
def root():
    return {"message": "Hello! GPT-J with Calculus is ready to go!"}
