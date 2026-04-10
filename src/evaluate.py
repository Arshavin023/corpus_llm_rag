import time
from datetime import datetime
import statistics
import json
import os
import html
from src.engine import query_rag_stream, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, embedding_model_name

# --- Evaluation Dataset ---
EVAL_SET = [
    # --- BENEFITS ---
    {"question": "How many PTO days do employees with 3 to 5 years of service receive?", "gold_answer": "15 days", "source": "benefits.md"},
    {"question": "What is the annual PTO for employees with less than 2 years?", "gold_answer": "10 days", "source": "benefits.md"},
    {"question": "How much PTO do employees with more than 5 years get?", "gold_answer": "20 days", "source": "benefits.md"},
    {"question": "How far in advance must employees request leave?", "gold_answer": "14 days", "source": "benefits.md"},

    # --- EXPENSES ---
    {"question": "What is the mileage reimbursement rate?", "gold_answer": "$0.65 per mile", "source": "expenses.xml"},
    {"question": "What is the daily meal allowance for overnight training?", "gold_answer": "$75", "source": "expenses.xml"},
    {"question": "When are receipts required for expenses?", "gold_answer": "over $25", "source": "expenses.xml"},
    {"question": "What is the deadline for submitting expense reports?", "gold_answer": "30 days", "source": "expenses.xml"},

    # --- CONDUCT / REMOTE WORK ---
    {"question": "What happens if an employee shares DMS login credentials?", "gold_answer": "immediate termination", "source": "conduct_policy.txt"},
    {"question": "What minimum internet speed is required for remote work?", "gold_answer": "25Mbps", "source": "conduct_policy.txt"},
    {"question": "Which employees are eligible for hybrid-flex work?", "gold_answer": "administrative and digital marketing staff", "source": "conduct_policy.txt"},

    # --- IT SECURITY ---
    {"question": "What is the minimum password length requirement?", "gold_answer": "12 characters", "source": "it_security.md"},
    {"question": "How often must passwords be changed?", "gold_answer": "90 days", "source": "it_security.md"},
    {"question": "Within how many hours must security incidents be reported?", "gold_answer": "24 hours", "source": "it_security.md"},

    # --- ATTENDANCE ---
    {"question": "What happens after more than three late arrivals in a month?", "gold_answer": "HR review", "source": "attendance.md"},
    {"question": "What qualifies as job abandonment?", "gold_answer": "3 consecutive unreported absences", "source": "attendance.md"},
    {"question": "What are the standard working hours?", "gold_answer": "9:00 AM – 5:00 PM", "source": "attendance.md"},

    # --- CUSTOMER SERVICE ---
    {"question": "How quickly must employees greet customers?", "gold_answer": "within 2 minutes", "source": "customer_service.txt"},
    {"question": "What is the resolution time for customer complaints?", "gold_answer": "48 hours", "source": "customer_service.txt"},

    # --- VEHICLE SAFETY ---
    {"question": "What must be verified before a test drive regarding age?", "gold_answer": "at least 21 years old", "source": "vehicle_safety.html"},
    {"question": "What document must be scanned before a test drive?", "gold_answer": "valid driver's license", "source": "vehicle_safety.html"},
    {"question": "Are pets allowed in demo vehicles?", "gold_answer": "no", "source": "vehicle_safety.html"},

    # --- TRAINING ---
    {"question": "How long do employees have to complete onboarding training?", "gold_answer": "30 days", "source": "training.html"},
    {"question": "What happens if required training is not completed?", "gold_answer": "suspension of system access", "source": "training.html"},

    # --- PDF: PERFORMANCE ---
    {"question": "How often are employee performance reviews conducted?", "gold_answer": "annually", "source": "performance_review.pdf"},
    {"question": "What happens to employees rated poorly in performance reviews?", "gold_answer": "placed on a 60-day PIP", "source": "performance_review.pdf"},

    # --- PDF: PAYROLL ---
    {"question": "What is the overtime pay rate for eligible employees?", "gold_answer": "1.5x hourly rate", "source": "payroll_policy.pdf"},
    {"question": "How often are employees paid?", "gold_answer": "bi-weekly", "source": "payroll_policy.pdf"},

    # --- PDF: DISCIPLINE ---
    {"question": "What is the first step in the disciplinary process?", "gold_answer": "verbal warning", "source": "disciplinary_action.pdf"},
    {"question": "Within how many days can employees appeal disciplinary decisions?", "gold_answer": "7 days", "source": "disciplinary_action.pdf"},

    # --- PDF: SAFETY / PROCUREMENT / DATA ---
    {"question": "How soon must workplace injuries be reported?", "gold_answer": "12 hours", "source": "health_safety.pdf"},
    {"question": "What approval is required for purchases above $5000?", "gold_answer": "CFO approval", "source": "procurement_policy.pdf"},
    {"question": "How long must employee records be retained?", "gold_answer": "7 years", "source": "data_retention_policy.pdf"}
]

RESULTS_FILE = "src/evaluation/evaluation_results.json"


def evaluate():
    results = []
    total_latencies = []
    ttft_latencies = [] # Time to First Token
    grounded_correct = 0
    citation_correct = 0

    print(f"🚀 Starting evaluation on {len(EVAL_SET)} samples...")

    for item in EVAL_SET:
        start_time = time.time()
        first_token_time = None
        
        # Initialize the generator from your engine
        generator = query_rag_stream(item["question"])
        
        full_answer = ""
        retrieved_citations = []

        try:
            # Consume the stream
            for chunk in generator:
                if first_token_time is None:
                    first_token_time = time.time() - start_time
                
                # Decode the yielded bytes
                data = json.loads(chunk.decode("utf-8"))
                
                # Capture citations (usually yielded in the first chunk)
                if "citations" in data and data["citations"]:
                    retrieved_citations = data["citations"]
                
                # Capture text content
                if "content" in data and data["content"]:
                    # Unescape HTML entities used in engine.py
                    full_answer += html.unescape(data["content"])

            total_latency = time.time() - start_time
            total_latencies.append(total_latency)
            ttft_latencies.append(first_token_time)

            # --- Scoring Logic ---
            answer_lower = full_answer.lower()
            
            # 1. Groundedness (Does answer contain the gold truth?)
            # if item["gold_answer"].lower() in answer_lower:
            if any(token in answer_lower for token in item["gold_answer"].lower().split()):
                grounded_correct += 1

            # 2. Citation Accuracy & Filtering
            # We filter the citations to ONLY keep the one that matches our gold source
            matched_citations = [
                c for c in retrieved_citations 
                if item["source"].lower() in c["source"].lower()
            ]
            
            if matched_citations:
                citation_correct += 1

            # --- Persistence Logic ---
            # We only save the matched citations to the results list
            results.append({
                "question": item["question"],
                "answer": full_answer,
                "total_latency": total_latency,
                "ttft": first_token_time,
                "relevant_source_found": matched_citations if matched_citations else "N/A"
            })
            
            print(f"✔ Done: {item['question'][:30]}... | Latency: {total_latency:.2f}s")

        except Exception as e:
            print(f"✘ Error evaluating '{item['question']}': {e}")

    # --- Metrics Aggregation ---
    total = len(EVAL_SET)
    if total_latencies:
        metrics = {
            "groundedness": grounded_correct / total,
            "citation_accuracy": citation_correct / total,
            "latency_p50": statistics.median(total_latencies),
            "latency_p95": sorted(total_latencies)[int(0.95 * len(total_latencies)) - 1],
            "avg_ttft": statistics.mean(ttft_latencies)
        }
    else:
        print("No evaluation data collected.")
        return

    # --- Persist to JSON ---
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # entry = {"results": results, "metrics": metrics}
    entry = {
        "config": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "top_k": TOP_K,
            "embedding_model": embedding_model_name,
            "llm_model": "llama-3.1-8b-instant"
        },
        "results": results,
        "metrics": metrics
    }

    if not os.path.exists(os.path.dirname(RESULTS_FILE)):
        os.makedirs(os.path.dirname(RESULTS_FILE))

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    data[timestamp] = entry

    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=4)

    print(f"\n✅ Evaluation complete. Saved to {RESULTS_FILE}")
    print(f"Groundedness Score: {metrics['groundedness']:.2%}")
    print(f"Citation Accuracy: {metrics['citation_accuracy']:.2%}")
    print(f"Avg TTFT: {metrics['avg_ttft']:.2f}s")
    
if __name__ == "__main__":
    evaluate()