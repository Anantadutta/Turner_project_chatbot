import pymupdf, re, json
from langchain_text_splitters import RecursiveCharacterTextSplitter

doc = pymupdf.open("data/superx.pdf")

# Step 2: extract text page by page
pages = [{"page": i + 1, "text": p.get_text()} for i, p in enumerate(doc)]

# Step 3: clean
def clean(t):
    t = re.sub(r"SUPERX AIHUB|TABLE OF CONTENTS", "", t)   # repeated header
    t = re.sub(r"\.{4,}", " ", t)                           # TOC dotted lines
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n", t)
    return t.strip()

low_text_pages = []
for p in pages:
    p["text"] = clean(p["text"])
    if len(p["text"]) < 200:
        low_text_pages.append(p["page"])   # these go to the visual pipeline later

# Step 4: chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
text_chunks = []
for p in pages:
    for j, c in enumerate(splitter.split_text(p["text"])):
        text_chunks.append({"id": f"t_p{p['page']}_{j}", "text": c,
                            "page": p["page"], "type": "text"})

json.dump(text_chunks, open("text_chunks.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
json.dump(low_text_pages, open("low_text_pages.json", "w"))

print("Pages:", len(pages))
print("Text chunks:", len(text_chunks))
print("Low-text pages (for visual pipeline):", low_text_pages)
print("\nSample chunk:\n", text_chunks[5]["text"][:300])
