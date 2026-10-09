import re

src = open("app.py", encoding="utf-8-sig").read()
if "def name_intro" in src:
    print("Already added - nothing to do.")
    raise SystemExit

src = src.replace('(r"hi|hello|hey|hii|yo|namaste|hola",', '(r"hi+|hello+|hey+|yo|namaste|hola",')

helper = '''

NOT_NAMES = set("a an the not new here back fine good ok okay ready sorry confused looking trying interested working from just also still very so".split())
NAME_INTRO = re.compile(
    r"^\\s*(?:(?:hi+|hello+|hey+)[\\s,!.]*)?(?:i'?m|i am|my name is|my name's|this is|call me)\\s+([a-z]+)(?:\\s+([a-z]+))?\\s*[!.]*\\s*$",
    re.IGNORECASE)


def name_intro(q):
    m = NAME_INTRO.match(q.strip())
    if not m or m.group(1).lower() in NOT_NAMES:
        return None
    return " ".join(w.capitalize() for w in m.groups() if w)
# end name
'''
src = src.replace("\n\ndef is_small_talk(q):", helper + "\n\ndef is_small_talk(q):", 1)

src = src.replace(
    "    return bool(SMALL_TALK.match(q)) and len(q.split()) <= 6",
    "    return name_intro(q) is not None or (bool(SMALL_TALK.match(q)) and len(q.split()) <= 6)", 1)

src = src.replace(
    "    low = q.lower()\n    for pattern, reply in GREETING_RULES:",
    "    name = name_intro(q)\n    if name:\n        st.session_state.user_name = name\n"
    "        return f\"Nice to meet you, {name}! I'm the Turner document assistant. What would you like to know about the company documents?\\n\\n\" + EXAMPLES\n"
    "    low = q.lower()\n    for pattern, reply in GREETING_RULES:", 1)

open("app.py", "w", encoding="utf-8").write(src)
print("Done - the bot now recognises your name")
