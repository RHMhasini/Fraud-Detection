import requests
import json
from pathlib import Path
import logging
import os
from openai import OpenAI

# -------------------
# Configuration
# -------------------
API_BASE_URL = "http://127.0.0.1:8000"  # Your running API URL
SCORES_ENDPOINT = f"{API_BASE_URL}/ml_scores"
TRANSACTIONS_ENDPOINT = f"{API_BASE_URL}/transactions"
OUTPUT_FILE = Path(__file__).resolve().parent.parent / "outputs" / "transaction_verifications.json"

# Thresholds for classification
THRESHOLD_PASS = 0.2  # Adjusted to flag more (change back to 0.5 if needed)
THRESHOLD_DENY = 0.8   # Above this: deny; between pass and deny: flag

# Groq API Configuration (using OpenAI-compatible client)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Get API key from environment variable
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set. Please set it before running the script.")
client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------
# Function to classify transaction
# -------------------
def classify_transaction(ensemble_score: float) -> str:
    if ensemble_score < THRESHOLD_PASS:
        return "pass"
    elif ensemble_score < THRESHOLD_DENY:
        return "flag"
    else:
        return "deny"

# -------------------
# Function to generate LLM explanation (only for the last flagged/denied transaction)
# -------------------
def generate_explanation(transaction_data: dict, ensemble_score: float, status: str) -> str:
    prompt = f"""
    Analyze the following transaction data and explain why it was classified as '{status}' (ensemble fraud score: {ensemble_score:.3f}).
    Provide a human-readable explanation based on the details. Focus on suspicious patterns like high device counts, IP counts, purchase value, or other anomalies.
    Transaction Data: {json.dumps(transaction_data, indent=2)}
    Keep the explanation concise (1-2 sentences) and natural.
    """
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Updated to a supported Groq model
            messages=[
                {"role": "system", "content": "You are a helpful assistant explaining fraud detection decisions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,  # Limit response length
            temperature=0.7  # Balanced creativity
        )
        explanation = response.choices[0].message.content.strip()
        logging.info(f"LLM Explanation for Transaction {transaction_data.get('transaction_id')}: {explanation}")
        return explanation
    except Exception as e:
        logging.error(f"LLM API error: {e}")
        return f"Unable to generate explanation due to API error: {str(e)}. Transaction flagged/denied based on score {ensemble_score:.3f}."

# -------------------
# Main Verifier Function
# -------------------
def verify_transactions():
    try:
        # Fetch scores from API
        response = requests.get(SCORES_ENDPOINT)
        response.raise_for_status()
        
        scores_data = response.json()
        if not scores_data:
            logging.warning("No ML scores found in API.")
            return []
        
        # Fetch transactions for details
        trans_response = requests.get(TRANSACTIONS_ENDPOINT)
        trans_response.raise_for_status()
        transactions_data = trans_response.json()
        
        # Create a dict for quick lookup by transaction_id
        trans_dict = {t["transaction_id"]: t for t in transactions_data}
        
        # Sort scores by transaction_id descending to find the last (highest ID)
        scores_data.sort(key=lambda x: x.get("transaction_id", 0), reverse=True)
        last_transaction_id = scores_data[0].get("transaction_id") if scores_data else None
        
        # Process each score
        verifications = []
        for score in scores_data:
            transaction_id = score.get("transaction_id")
            ensemble_score = score.get("ensemble_score")
            
            if transaction_id is None or ensemble_score is None:
                logging.error(f"Invalid score data: {score}")
                continue
            
            status = classify_transaction(ensemble_score)
            verification = {
                "transaction_id": transaction_id,
                "ensemble_score": ensemble_score,
                "status": status
            }
            
            # Generate explanation ONLY for the last transaction if it's flag/deny
            if transaction_id == last_transaction_id and status in ["flag", "deny"]:
                transaction_details = trans_dict.get(transaction_id, {})
                if transaction_details:
                    explanation = generate_explanation(transaction_details, ensemble_score, status)
                    verification["explanation"] = explanation
                else:
                    verification["explanation"] = "Transaction details not found."
            else:
                verification["explanation"] = None  # No explanation for others or pass
            
            verifications.append(verification)
            logging.info(f"Verified Transaction {transaction_id}: Score {ensemble_score:.3f} -> {status}")
        
        # Output results
        print("Transaction Verifications:")
        print(json.dumps(verifications, indent=2))
        
        # Save to file
        OUTPUT_FILE.parent.mkdir(exist_ok=True)
        with open(OUTPUT_FILE, "w") as f:
            json.dump(verifications, f, indent=2)
        logging.info(f"Results saved to {OUTPUT_FILE}")
        
        return verifications
    
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
        print(f"Error: Could not fetch data from API. Ensure the API is running at {API_BASE_URL}.")
        return []
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return []

# -------------------
# Run the Verifier
# -------------------
if __name__ == "__main__":
    verify_transactions()
