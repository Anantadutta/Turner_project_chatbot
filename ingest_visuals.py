"""
Steps 5-7: Visual pipeline
- Finds pages whose content is mainly visual (scanned pages, charts, Gantt charts,
  tables drawn as graphics, large images)
- Renders each such page to a PNG in images/
- Sends each PNG to Gemini, which writes a detailed text description
- Saves descriptions to visual_chunks.json (cached: re-running skips pages already done)

Usage:  python ingest_visuals.py            -> describe all visual pages with Gemini
        python ingest_visuals.py --dry-run  -> only list/render pages, no API calls
"""
import os, sys, json, time
from dotenv import load_dotenv
load_dotenv()
from dotenv import load_dotenv
load_dotenv()
import pymupdf

PDF_PATH = "data/superx.pdf"
IMG_DIR = "images"
OUT_FILE = "visual_chunks.json"
MODEL = "gemini-3.5-flash-lite"      # change here if Google renames/updates the model
PAUSE_SECONDS = 7               # free tier allows ~10 requests/minute

PROMPT = """You are describing one page of a company document so it can be found by a search engine.
1. Say what kind of visual this is (chart, Gantt/schedule, table, org chart, diagram, photo,
   scanned text page, form, etc.) and give its title.
2. Transcribe ALL readable text: headings, labels, legends, notes, names, dates.
3. Charts/plots: describe axes, units, every data value you can read, trends, min/max, outliers.
   Box plots: median, quartiles, whiskers and outliers for each group.
4. Tables and schedules: reproduce them as markdown tables (rows, columns, dates, amounts).
5. Photos/diagrams: describe what they show and why they matter in the document.
Never guess numbers you cannot read clearly - write "unreadable" instead.
Do not add any information that is not on the page."""


def find_visual_pages(doc):
    """Return page numbers (1-based) whose content is mainly visual."""
    visual = []
    for i, page in enumerate(doc):
        text_len = len(page.get_text().strip())
        many_drawings = len(page.get_drawings()) > 200          # vector charts / tables
        page_area = page.rect.width * page.rect.height
        big_image = False
        for img in page.get_images(full=True):
            for r in page.get_image_rects(img[0]):
                if r.width * r.height > 0.03 * page_area:       # image covers >8% of page
                    big_image = True
        if text_len < 200 or many_drawings or big_image:
            visual.append(i + 1)
    return visual


def main():
    dry_run = "--dry-run" in sys.argv
    os.makedirs(IMG_DIR, exist_ok=True)
    doc = pymupdf.open(PDF_PATH)
    pages = find_visual_pages(doc); pages = [p for p in pages if p in (7,14,15,17,18,74,91,116)] if "--new" in sys.argv else pages
    print(f"Visual pages found: {len(pages)}\n{pages}\n")

    for p in pages:
        path = f"{IMG_DIR}/page_{p}.png"
        if not os.path.exists(path):
            doc[p - 1].get_pixmap(dpi=150).save(path)
    print(f"Rendered page images saved in '{IMG_DIR}/'")

    if dry_run:
        print("Dry run: no Gemini calls made.")
        return

    from google import genai
    from google.genai import types

    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("ERROR: GEMINI_API_KEY is not set. See instructions.")
    client = genai.Client()

    chunks = json.load(open(OUT_FILE, encoding="utf-8")) if os.path.exists(OUT_FILE) else []
    done = {c["page"] for c in chunks}

    for n, p in enumerate(pages, 1):
        if p in done:
            continue
        path = f"{IMG_DIR}/page_{p}.png"
        page_text = doc[p - 1].get_text().strip()[:1500]
        print(f"[{n}/{len(pages)}] Describing page {p} ...", end=" ", flush=True)

        for attempt in range(5):
            try:
                resp = client.models.generate_content(
                    model=MODEL,
                    contents=[
                        types.Part.from_bytes(data=open(path, "rb").read(), mime_type="image/png"),
                        PROMPT + f"\n\nText extracted from this page (may be incomplete):\n{page_text}",
                    ],
                )
                desc = (resp.text or "").strip()
                break
            except Exception as e:
                wait = 30 * (attempt + 1)
                print(f"\n   error: {str(e)[:120]} -> retrying in {wait}s", end=" ", flush=True)
                time.sleep(wait)
        else:
            print("skipped after 5 failures")
            continue

        chunks.append({"id": f"v_p{p}", "text": f"[Visual content on page {p}]\n{desc}",
                       "page": p, "type": "image", "image_path": path})
        json.dump(chunks, open(OUT_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"ok ({len(desc)} chars)")
        time.sleep(PAUSE_SECONDS)

    print(f"\nDone. {len(chunks)} visual descriptions saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
