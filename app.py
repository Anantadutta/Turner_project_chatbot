import os, re, json, math
from collections import Counter
import numpy as np
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
CHAT_MODELS = ["gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.5-flash"]
TOP_K = 6
MAX_IMAGES = 2
NOT_FOUND = "I couldn't find this in the company documents."

EXAMPLES = """You can ask me things like:
- Who is this proposal addressed to?
- What are the L1 to L5 commissioning levels?
- What is the total project fee?
- What safety approach does Turner follow?"""

GREETING_RULES = [
    (r"good morning", "Good morning! Hope your day is off to a great start."),
    (r"good afternoon", "Good afternoon! Hope your day is going well."),
    (r"good evening", "Good evening! Hope you had a good day."),
    (r"how are you|how r u|how's it going|whats up|what's up|sup", "I'm doing great, thanks for asking! Ready to dig into the company documents for you."),
    (r"thank|thanks|thx|ty", "You're welcome! Happy to help. Anything else you'd like to know?"),
    (r"bye|goodbye|see you|good night", "Goodbye! Have a great day. Come back anytime you need something from the documents."),
    (r"who are you|what are you|your name", "I'm the Spiderman Company document assistant. I read the company's documents, including charts, tables and scanned pages, and answer questions using only what's in them."),
    (r"what can you do|help", "I can answer questions about the company documents and show you exactly which page each answer comes from, including information inside images and charts."),
    (r"talk to me|chat", "I'm right here! I'm happy to chat about anything in the company documents."),
    (r"^(ok|okay|cool|nice|great|awesome)", "Great! Let me know whenever you have a question."),
    (r"hi+|hello+|hey+|yo|namaste|hola", "Hello! Welcome to the Spiderman Company document assistant. How can I help you today?"),
]

SMALL_TALK = re.compile(
    r"^\s*(hi+|hello+|hey+|yo|namaste|hola|good (morning|afternoon|evening|night)|thanks?( you| a lot)?|thank u|thx|ty|"
    r"ok(ay)?|cool|nice|great|awesome|how are you|how r u|how's it going|what'?s up|sup|who are you|what are you|"
    r"what('?s| is) your name|what can you do|help|(at ?least )?talk to me|let'?s chat|bye|goodbye|see you)\b",
    re.IGNORECASE)


NOT_NAMES = set("a an the not new here back fine good ok okay ready sorry confused looking trying interested working from just also still very so".split())
NAME_INTRO = re.compile(
    r"^\s*(?:(?:hi+|hello+|hey+)[\s,!.]*)?(?:i'?m|i am|my name is|my name's|this is|call me)\s+([a-z]+)(?:\s+([a-z]+))?\s*[!.]*\s*$",
    re.IGNORECASE)


def name_intro(q):
    m = NAME_INTRO.match(q.strip())
    if not m or m.group(1).lower() in NOT_NAMES:
        return None
    return " ".join(w.capitalize() for w in m.groups() if w)
# end name


def is_small_talk(q):
    q = q.strip()
    return name_intro(q) is not None or (bool(SMALL_TALK.match(q)) and len(q.split()) <= 6)


def small_talk_reply(q):
    name = name_intro(q)
    if name:
        st.session_state.user_name = name
        return f"Nice to meet you, {name}! I'm the Spiderman Company document assistant. What would you like to know about the company documents?\n\n" + EXAMPLES
    low = q.lower()
    for pattern, reply in GREETING_RULES:
        if re.search(r"\b(" + pattern + r")\b", low):
            return reply + "\n\n" + EXAMPLES
    return "Hello! How can I help you today?\n\n" + EXAMPLES
# end greetings

STOP = set("the a an and or of to in on for is are was were be by with what which who how does do this that it as at from about explain tell me please".split())

SYSTEM_PROMPT = f"""You are the internal document assistant for Spiderman Company.
Rules:
1. Answer ONLY using the CONTEXT provided and the attached page images. Never use outside knowledge.
2. If the answer is not in the context, reply exactly: "{NOT_FOUND}"
3. Cite the page for every fact, like [p. 12]. If information comes from a chart, table or image, say so.
4. Be concise and precise. Copy numbers, dates and names exactly as they appear.
5. If the context only partly answers the question, answer that part and say what is missing."""


def tokens(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [Counter(tokens(d)) for d in docs]
        self.lens = np.array([sum(d.values()) for d in self.docs], dtype="float32")
        self.avg = self.lens.mean()
        df = Counter(w for d in self.docs for w in d)
        n = len(docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query):
        s = np.zeros(len(self.docs), dtype="float32")
        for w in set(tokens(query)):
            if w not in self.idf:
                continue
            tf = np.array([d.get(w, 0) for d in self.docs], dtype="float32")
            s += self.idf[w] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.lens / self.avg))
        return s


@st.cache_resource
def load_everything():
    client = genai.Client()
    vectors = np.load("index/vectors.npy")
    data = json.load(open("index/chunks.json", encoding="utf-8"))
    bm25 = BM25([c["text"] for c in data["chunks"]])
    return client, vectors, data["chunks"], data["embedding_model"], bm25


def retrieve(client, vectors, chunks, embed_model, bm25, question):
    kw = bm25.scores(question)
    kw_norm = kw / kw.max() if kw.max() > 0 else kw
    sem = np.zeros(len(chunks), dtype="float32")
    mode = "keyword only"
    try:
        res = client.models.embed_content(
            model=embed_model, contents=[question],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"))
        q = np.array(res.embeddings[0].values, dtype="float32")
        if q.shape[0] == vectors.shape[1]:
            sem = vectors @ (q / np.linalg.norm(q))
            mode = "hybrid"
    except Exception:
        pass
    score = 0.5 * kw_norm + 0.5 * sem if mode == "hybrid" else kw_norm
    top = np.argsort(-score)[:TOP_K]
    found = kw.max() > 0 or sem.max() >= 0.30
    return [(chunks[i], float(score[i])) for i in top], found, mode


def answer(client, question, hits, history):
    context = "\n\n".join(
        f"[p. {c['page']}][{'IMAGE' if c['type'] == 'image' else 'TEXT'}]\n{c['text']}" for c, _ in hits)
    parts, used = [], set()
    for c, _ in hits:
        path = c.get("image_path")
        if c["type"] == "image" and path and path not in used and os.path.exists(path) and len(used) < MAX_IMAGES:
            parts.append(types.Part.from_bytes(data=open(path, "rb").read(), mime_type="image/png"))
            used.add(path)
    recent = "\n".join(f"{m['role']}: {m['content']}" for m in history[-4:])
    parts.append(f"CONTEXT:\n{context}\n\nRECENT CONVERSATION:\n{recent}\n\nQUESTION: {question}")
    last_error = ""
    for model in CHAT_MODELS:
        try:
            resp = client.models.generate_content(
                model=model, contents=parts,
                config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0))
            return resp.text.strip()
        except Exception as e:
            last_error = e
    return f"Sorry, the AI service is busy right now. Please try again. ({str(last_error)[:80]})"


st.set_page_config(page_title="Spiderman Company - Document Assistant")
st.title("Spiderman Company Document Assistant")
st.caption("Answers only from company documents, including charts, tables and scanned pages.")

client, vectors, chunks, embed_model, bm25 = load_everything()
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# WELCOME_SHOWN
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("Hello! I'm the Spiderman Company document assistant. Ask me anything about the company documents.\n\n" + EXAMPLES)

question = st.chat_input("Ask a question about the company documents...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the documents..."):
            hits, found, mode = retrieve(client, vectors, chunks, embed_model, bm25, question)
            if is_small_talk(question):
                reply, hits = small_talk_reply(question), []
            elif not found:
                reply, hits = NOT_FOUND, []
            else:
                reply = answer(client, question, hits, st.session_state.messages[:-1])
        st.markdown(reply)
        if hits and reply != NOT_FOUND:
            with st.expander(f"Sources ({mode} search)"):
                for c, score in hits:
                    kind = "Image / chart" if c["type"] == "image" else "Text"
                    st.markdown(f"**Page {c['page']}** - {kind} - score {score:.2f}")
                    if c["type"] == "image" and os.path.exists(c.get("image_path", "")):
                        st.image(Image.open(c["image_path"]), width=350)
                    st.caption(c["text"][:400] + ("..." if len(c["text"]) > 400 else ""))
                    st.divider()
    st.session_state.messages.append({"role": "assistant", "content": reply})

st.markdown('<div style="position:fixed;top:0;left:0;z-index:999999;pointer-events:none;"><svg width="230" height="230" viewBox="0 0 230 230" xmlns="http://www.w3.org/2000/svg"><path d="M0 0L230 8M0 0L221 65M0 0L198 117M0 0L163 163M0 0L117 198M0 0L65 221M0 0L8 230M28 1Q24 4 27 8Q22 10 24 14Q19 15 20 20Q15 19 14 24Q10 22 8 27Q4 24 1 28M54 2Q46 7 52 15Q43 19 46 28Q37 28 38 38Q28 37 28 46Q19 43 15 52Q7 46 2 54M80 3Q68 11 77 22Q63 27 69 41Q54 42 57 57Q42 54 41 69Q27 63 22 77Q11 68 3 80M106 4Q90 15 102 30Q84 36 91 54Q72 56 75 75Q56 72 54 91Q36 84 30 102Q15 90 4 106M132 5Q112 18 127 37Q104 45 114 67Q90 70 93 93Q70 90 67 114Q45 104 37 127Q18 112 5 132M158 6Q134 22 152 44Q125 54 136 81Q107 83 112 112Q83 107 81 136Q54 125 44 152Q22 134 6 158M184 6Q156 25 177 52Q145 63 158 94Q125 97 130 130Q97 125 94 158Q63 145 52 177Q25 156 6 184M210 7Q178 29 202 59Q166 72 181 107Q143 111 148 148Q111 143 107 181Q72 166 59 202Q29 178 7 210" fill="none" stroke="#c9d1d9" stroke-width="1.2" stroke-opacity="0.55"/><line x1="150" y1="150" x2="150" y2="182" stroke="#c9d1d9" stroke-opacity="0.55"/><circle cx="150" cy="188" r="6" fill="#c9d1d9" fill-opacity="0.7"/></svg></div>', unsafe_allow_html=True)
