import re
import os
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from sympy import symbols, sympify, diff, integrate, limit

################################################################################
# 1. Initialize FastAPI App
################################################################################
app = FastAPI()

################################################################################
# 2. Load GPT-J Model at Startup
################################################################################
print("Loading GPT-J model...")
model_name = "EleutherAI/gpt-j-6B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
print("Model loaded.")

################################################################################
# 3. Helper Functions for Calculus Queries
################################################################################
def handle_calculus_request(query: str) -> str:
    """Processes calculus-related queries."""
    try:
        query = query.lower()

        # Differentiation
        if "differentiate" in query or "derivative" in query:
            parts = query.split("with respect to")
            if len(parts) == 2:
                func = parts[0].replace("differentiate", "").replace("derivative of", "").strip()
                var = parts[1].strip()
                variable = symbols(var)
                expression = sympify(func)
                result = diff(expression, variable)
                return f"The derivative of {func} with respect to {var} is: {result}"

        # Integration
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

        # Unsupported Query
        else:
            return "Unsupported calculus query. Please specify differentiation, integration, or limit."

    except Exception as e:
        return f"Error processing calculus query: {e}"

def is_calculus_query(query: str) -> bool:
    """Determines if the query is related to calculus."""
    keywords = ["differentiate", "derivative", "integrate", "limit"]
    return any(keyword in query.lower() for keyword in keywords)

################################################################################
# 4. FastAPI Request Model
################################################################################
class QueryRequest(BaseModel):
    query: str

################################################################################
# 5. API Endpoints
################################################################################
@app.post("/generate")
def generate(request: QueryRequest):
    """
    POST endpoint for handling both general GPT-J queries and calculus queries.

    Example JSON Body:
    {
      "query": "differentiate x^2 with respect to x"
    }
    """
    query = request.query.strip()

    if is_calculus_query(query):
        response = handle_calculus_request(query)
    else:
        # Process general GPT-J query
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

@app.get("/")
def root():
    """Health check endpoint."""
    return {"message": "Hello! GPT-J with Calculus is running successfully!"}

################################################################################
# 6. Run the FastAPI Server (Required for Render)
################################################################################
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render assigns a dynamic port
    uvicorn.run(app, host="0.0.0.0", port=port)
