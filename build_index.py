import os, json, time, hashlib
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
INDEX_DIR = "index"
CACHE = f"{INDEX_DIR}/emb_cache.json"
BATCH = 50


def pick_embedding_model(client):
    names = [m.name.replace("models/", "") for m in client.models.list() if "embedding" in m.name]
    for n in names:
        if n.startswith("gemini-embedding"):
            return n
    return names[0] if names else None


def main():
    os.makedirs(INDEX_DIR, exist_ok=True)
    text_chunks = json.load(open("text_chunks.json", encoding="utf-8"))
    visual_raw = json.load(open("visual_chunks.json", encoding="utf-8")) if os.path.exists("visual_chunks.json") else []

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    visual_chunks = []
    for v in visual_raw:
        for j, part in enumerate(splitter.split_text(v["text"])):
            visual_chunks.append({**v, "id": f"{v['id']}_{j}",
                                  "text": part if j == 0 else f"[Visual content on page {v['page']}]\n{part}"})
    chunks = text_chunks + visual_chunks
    print(f"Text chunks: {len(text_chunks)} | Visual chunks: {len(visual_chunks)} | Total: {len(chunks)}")

    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    key = lambda c: hashlib.md5(c["text"].encode("utf-8")).hexdigest()
    todo = [c for c in chunks if key(c) not in cache]
    print(f"Already embedded: {len(chunks) - len(todo)} | To embed now: {len(todo)}")

    model = None
    try:
        client = genai.Client()
        model = pick_embedding_model(client)
        print(f"Embedding model: {model}")
        for i in range(0, len(todo), BATCH):
            batch = todo[i:i + BATCH]
            for attempt in range(3):
                try:
                    res = client.models.embed_content(
                        model=model, contents=[c["text"] for c in batch],
                        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"))
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    print(f"   limit reached, waiting 65s ({str(e)[:60]})")
                    time.sleep(65)
            for c, e in zip(batch, res.embeddings):
                cache[key(c)] = e.values
            json.dump(cache, open(CACHE, "w"))
            print(f"   embedded {min(i + BATCH, len(todo))}/{len(todo)}")
            time.sleep(2)
    except Exception as e:
        print(f"\nGemini embedding stopped ({str(e)[:80]}).")
        print("That's OK: keyword search still works. Re-run later to finish the embeddings.")

    dim = len(next(iter(cache.values()))) if cache else 1
    vecs = np.array([cache.get(key(c), [0.0] * dim) for c in chunks], dtype="float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.where(norms == 0, 1, norms)
    np.save(f"{INDEX_DIR}/vectors.npy", vecs)
    json.dump({"embedding_model": model or "gemini-embedding-001", "chunks": chunks},
              open(f"{INDEX_DIR}/chunks.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    done = sum(1 for c in chunks if key(c) in cache)
    print(f"\nDone. Index saved in '{INDEX_DIR}/' - {len(chunks)} chunks, {done} with embeddings, all with keyword search.")


if __name__ == "__main__":
    main()
