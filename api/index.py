from fastapi import FastAPI
import numpy as np
import time

app = FastAPI()

documents = [
    {"id": i,
     "content": f"Scientific abstract about machine learning applications topic {i}.",
     "metadata": {"source": "research_paper"}}
    for i in range(121)
]

def simple_similarity(query, text):
    query_words = set(query.lower().split())
    text_words = set(text.lower().split())
    return len(query_words & text_words) / (len(query_words) + 1)

@app.get("/ping")
async def ping():
    return {"message": "working"}


@app.post("/")
async def search(body: dict):
    start = time.time()

    query = body.get("query", "")
    k = body.get("k", 8)
    rerank = body.get("rerank", True)
    rerankK = body.get("rerankK", 5)

    if not query:
        return {"results": [], "reranked": False, "metrics": {"latency": 0, "totalDocs": 121}}

    scores = [
        simple_similarity(query, doc["content"])
        for doc in documents
    ]

    top_indices = np.argsort(scores)[-k:][::-1]

    candidates = [
        {
            "id": documents[i]["id"],
            "score": float(scores[i]),
            "content": documents[i]["content"],
            "metadata": documents[i]["metadata"]
        }
        for i in top_indices
    ]

    # Normalize scores
    max_score = max(c["score"] for c in candidates)
    min_score = min(c["score"] for c in candidates)

    for c in candidates:
        if max_score != min_score:
            c["score"] = (c["score"] - min_score) / (max_score - min_score)
        else:
            c["score"] = 1.0

    if rerank:
        candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)[:rerankK]

    latency = int((time.time() - start) * 1000)

    return {
        "results": candidates,
        "reranked": rerank,
        "metrics": {
            "latency": latency,
            "totalDocs": 121
        }
    }

