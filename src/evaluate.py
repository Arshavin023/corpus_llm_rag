import time
from datetime import datetime
import statistics
import json
import os
import html
from src.engine import query_rag_stream  

# --- Evaluation Dataset ---
EVAL_SET = [
    {"question": "How many PTO days do employees with 3-5 years get?", "gold_answer": "15 days", "source": "benefits.md"},
    {"question": "What is the PTO for new employees?", "gold_answer": "10 days", "source": "benefits.md"},
    {"question": "What is the reimbursement rate for mileage?", "gold_answer": "$0.65 per mile", "source": "expenses.xml"},
    {"question": "What is the meal per diem limit?", "gold_answer": "$75", "source": "expenses.xml"},
    {"question": "What happens if login credentials are shared?", "gold_answer": "immediate termination", "source": "conduct_policy.txt"},
    {"question": "How many days in advance must leave be requested?", "gold_answer": "14 days", "source": "benefits.md"},
    {"question": "What is the minimum internet speed for remote work?", "gold_answer": "25Mbps", "source": "remote_work.md"}, 
    {"question": "What is the reimbursement submission deadline?", "gold_answer": "30 days", "source": "expenses.xml"},
    {"question": "What is required before a test drive?", "gold_answer": "valid driver's license", "source": "vehicle_safety.html"},
    {"question": "Are pets allowed in demo vehicles?", "gold_answer": "not allowed", "source": "vehicle_safety.html"},
    {"question": "How long before passwords expire?", "gold_answer": "90 days", "source": "it_security.md"},
    {"question": "What is the minimum password length?", "gold_answer": "12 characters", "source": "it_security.md"},
    {"question": "How soon must security incidents be reported?", "gold_answer": "24 hours", "source": "it_security.md"},
    {"question": "What happens after 3 late arrivals?", "gold_answer": "HR review", "source": "attendance.md"},
    {"question": "How long to resolve customer complaints?", "gold_answer": "48 hours", "source": "customer_service.txt"}
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
            if item["gold_answer"].lower() in answer_lower:
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
    entry = {"results": results, "metrics": metrics}

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