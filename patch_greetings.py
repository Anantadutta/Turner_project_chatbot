import re

src = open("app.py", encoding="utf-8-sig").read()

src = re.sub(r"\n\nSMALL_TALK_REPLY = .*?len\(q\.split\(\)\) <= 6\n", "\n", src, flags=re.S)
src = re.sub(r"\n\nEXAMPLES = .*?# end greetings\n", "\n", src, flags=re.S)

helper = '''

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
    (r"hi|hello|hey|hii|yo|namaste|hola", "Hello! Welcome to the Spiderman Company document assistant. How can I help you today?"),
]

SMALL_TALK = re.compile(
    r"^\\s*(hi+|hello+|hey+|yo|namaste|hola|good (morning|afternoon|evening|night)|thanks?( you| a lot)?|thank u|thx|ty|"
    r"ok(ay)?|cool|nice|great|awesome|how are you|how r u|how's it going|what'?s up|sup|who are you|what are you|"
    r"what('?s| is) your name|what can you do|help|(at ?least )?talk to me|let'?s chat|bye|goodbye|see you)\\b",
    re.IGNORECASE)


def is_small_talk(q):
    q = q.strip()
    return bool(SMALL_TALK.match(q)) and len(q.split()) <= 6


def small_talk_reply(q):
    low = q.lower()
    for pattern, reply in GREETING_RULES:
        if re.search(r"\\b(" + pattern + r")\\b", low):
            return reply + "\\n\\n" + EXAMPLES
    return "Hello! How can I help you today?\\n\\n" + EXAMPLES
# end greetings
'''

anchor = 'NOT_FOUND = "I couldn\'t find this in the company documents."'
src = src.replace(anchor, anchor + helper, 1)

if "if is_small_talk(question):" not in src:
    src = src.replace(
        "            if not found:\n",
        "            if is_small_talk(question):\n                reply, hits = small_talk_reply(question), []\n            elif not found:\n", 1)
src = src.replace("reply, hits = SMALL_TALK_REPLY, []", "reply, hits = small_talk_reply(question), []")

if "WELCOME_SHOWN" not in src:
    src = src.replace(
        'question = st.chat_input(',
        '# WELCOME_SHOWN\nif not st.session_state.messages:\n    with st.chat_message("assistant"):\n'
        '        st.markdown("Hello! I\'m the Spiderman Company document assistant. Ask me anything about the company documents.\\n\\n" + EXAMPLES)\n\n'
        'question = st.chat_input(', 1)

open("app.py", "w", encoding="utf-8").write(src)
print("Done - friendly greetings added to app.py")
