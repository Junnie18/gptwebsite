import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sympy import symbols, sympify, diff, integrate, limit

# Load GPT-J model and tokenizer
def load_gptj_model():
    print("Loading GPT-J model...")
    model_name = "EleutherAI/gpt-j-6B"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    print("Model loaded.")
    return model, tokenizer, device

# Utility function for handling calculus queries
def handle_calculus_request(query):
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

# Detect whether the query is calculus-related
def is_calculus_query(query):
    keywords = ["differentiate", "derivative", "integrate", "limit"]
    return any(keyword in query.lower() for keyword in keywords)

# Main function
def main():
    # Load GPT-J model and tokenizer
    model, tokenizer, device = load_gptj_model()

    print("\nWelcome to GPT-J with Calculus! Type 'exit' to quit.\n")
    while True:
        # Get user input
        query = input("Enter your query: ").strip()
        if query.lower() == "exit":
            print("Exiting. Goodbye!")
            break

        # Check if the query is calculus-related
        if is_calculus_query(query):
            response = handle_calculus_request(query)
        else:
            # Process general GPT-J query
            print("Thinking...")
            inputs = tokenizer(query, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.2,
                top_k=50,
                do_sample=True
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Print the response
        print("\nResponse:")
        print(response)
        print("-" * 50)

if __name__ == "__main__":
    main()
