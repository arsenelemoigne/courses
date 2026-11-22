# ============================================================================
#  ALFRED'S EMAIL SORTER (HF Agents Course, Unit 2.3, "Building Your First LangGraph")
# ============================================================================
#
#  What it does: Alfred receives an email, asks the model "spam or not?", and
#  then either throws it away, or drafts a reply and shows it to Mr. Hugg.
#
#  The graph (boxes = nodes, arrows = edges):
#
#     START
#       |
#   [read_email]          just prints who the email is from
#       |
#   [classify_email]      asks the model: spam or legitimate?
#       |
#   route_email()  ------ a decision, not a box: reads is_spam, picks a path
#     |         |
#   spam    legitimate
#     |         |
#  [handle_spam]  [draft_response]   asks the model to write a reply
#     |         |
#     |      [notify_mr_hugg]        prints the draft
#     |         |
#    END       END
#
#  This is a WORKFLOW, not an agent: the arrows are fixed in advance, and the
#  model never chooses what happens next or uses a tool. Compare with
#  Lupin/pythonLG.py, where the arrow run_tools -> ask_model lets the model go
#  round as many times as it wants. That loop is what makes it an agent.
#
#    PART 1  Imports
#    PART 2  The state      the "form" every node reads and fills in
#    PART 3  The model      which LLM we talk to
#    PART 4  The nodes      one function per box
#    PART 5  The decision   which arrow to follow after classify_email
#    PART 6  The graph      put the boxes and arrows together
#    PART 7  Run it         two test emails
#
#  HOW TO READ THIS FILE
#  Python reads the file from top to bottom, but Parts 1 to 6 only PREPARE
#  things. A `def` line doesn't run the function: it only teaches Python the
#  recipe. The real action starts in Part 7, when invoke() starts the graph,
#  and only THEN do the functions of Part 4 run, one after the other.
#
#  Follow the "▶ STEP" markers to read the code in the order it really runs:
#    ▶ STEP 1-3   Parts 1-3: imports, the form, the model (runs immediately)
#    ▶ STEP 4     Parts 4-5: recipes are stored, nothing runs yet
#    ▶ STEP 5     Part 6:    the graph is built
#    ▶ STEP 6     Part 7:    the fake email is created, invoke() starts the graph
#    ▶ STEP 7     read_email runs
#    ▶ STEP 8     classify_email runs           (model call #1)
#    ▶ STEP 9     route_email decides the path
#    ▶ STEP 10a   spam:       handle_spam runs  -> END
#    ▶ STEP 10b   legitimate: draft_response    (model call #2)
#    ▶ STEP 11    notify_mr_hugg runs           -> END
#    ▶ STEP 12    the second email: steps 6 to 11 again
# ============================================================================


# ============================================================================
#  PART 1: IMPORTS                                              ▶ STEP 1
# ============================================================================
# "Bring in" code other people wrote, so we can use it below.
import os
from typing import TypedDict, List, Dict, Any, Optional  # tools to describe the state's fields (Part 2)
from langgraph.graph import StateGraph, START, END        # the graph building blocks (Part 6)
from langchain_openai import ChatOpenAI                   # LangChain's connector for OpenAI-format models (Part 3)
from langchain_core.messages import HumanMessage          # a "user" message, as a LangChain object


# ============================================================================
#  PART 2: THE STATE (the "form" that travels through the graph)  ▶ STEP 2
# ============================================================================
# This only DESCRIBES the form (which fields it has). No form exists yet:
# the first real one is created in Part 7.
# Think of the state as a form with blank fields. It's handed from box to box.
# Each box (node) reads the fields it needs and fills in new ones.
#
# TypedDict = a dict with fixed, named keys. It just documents which fields
# exist and what type each one holds.
# Optional[str] = "a string, or None (still blank)".
#
# In Lupin, the state was only `messages`. Here it has 6 fields, because this
# workflow needs to remember more than the conversation.
class EmailState(TypedDict):
    email: Dict[str, Any]          # the email itself: {"sender": ..., "subject": ..., "body": ...}
    email_category: Optional[str]  # filled by classify_email, e.g. "inquiry"
    spam_reason: Optional[str]     # filled by classify_email, if it's spam
    is_spam: Optional[bool]        # filled by classify_email: True or False
    email_draft: Optional[str]     # filled by draft_response: the reply Alfred wrote
    messages: List[Dict[str, Any]] # a log of what was sent to the model and what it answered


# ============================================================================
#  PART 3: THE MODEL                                            ▶ STEP 3
# ============================================================================
# Creates the "phone line" to Qwen. Nothing is sent yet: the first call to
# Qwen happens in STEP 8.
# CHANGED from the course: `ChatOpenAI(temperature=0)` calls OpenAI and needs a
# paid key. These settings point it at your local Qwen in Ollama instead
# (same idea as Part 1 of lupin.py).
# temperature=0 = "always pick the most likely words", so answers vary less.
model = ChatOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    model="qwen3:1.7b",
    temperature=0,
)


# ============================================================================
#  PART 4: THE NODES (one function per box)                     ▶ STEP 4
# ============================================================================
# When Python first reads this part, it only STORES these 5 recipes.
# None of them runs now. The graph calls them later (STEPS 7 to 11).
# Every node has the same shape:
#   - it receives the state (the form),
#   - it does its job,
#   - it RETURNS a dict with ONLY the fields it wants to change.
#     LangGraph copies those into the state. Returning {} = "change nothing".

# --- Box 1: read_email ---------------------------------------------- ▶ STEP 7
# Just prints who the email is from. Changes nothing.
def read_email(state: EmailState):
    """Alfred reads and logs the incoming email"""
    email = state["email"]   # take the email out of the form (it's the dict from Part 7)
    # email['sender'] = the "sender" value of that dict, e.g. "john.smith@example.com"
    print(f"Alfred is processing an email from {email['sender']} with subject: {email['subject']}")
    return {}


# --- Box 2: classify_email ------------------------------------------ ▶ STEP 8
# Asks the model "spam or legitimate?", then reads its answer to fill in
# is_spam, spam_reason and email_category.
def classify_email(state: EmailState):
    """Alfred uses an LLM to determine if the email is spam or legitimate"""
    email = state["email"]

    # 1. Write the question for the model. The f"""...""" string lets us drop
    #    the email's fields into the text with {...}.
    prompt = f"""
    As Alfred the butler, analyze this email and determine if it is spam or legitimate.

    Email:
    From: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}

    First, determine if this email is spam. If it is spam, explain why.
    If it is legitimate, categorize it (inquiry, complaint, thank you, etc.).
    """

    # 2. Ask the model. One message, no memory, no tools: a single question.
    messages = [HumanMessage(content=prompt)]   # a list with one "user" message: our question
    response = model.invoke(messages)           # MODEL CALL #1: send it to Qwen, wait for the answer
    # response.content = Qwen's answer, as text

    # 3. Read the answer. The model replies in free text, so the course just
    #    searches it for words. .lower() makes it all lowercase first.
    #    WEAK SPOT: "spam" anywhere in the text and no exact "not spam" = spam.
    #    So an answer like "it's legitimate, it doesn't qualify as spam" is
    #    wrongly marked as spam. (This is what happens with Qwen.)
    response_text = response.content.lower()
    is_spam = "spam" in response_text and "not spam" not in response_text

    # If spam, keep whatever follows "reason:" as the reason (if the model wrote that word).
    spam_reason = None
    if is_spam and "reason:" in response_text:
        spam_reason = response_text.split("reason:")[1].strip()

    # If legitimate, pick the first category word found in the answer.
    email_category = None
    if not is_spam:
        categories = ["inquiry", "complaint", "thank you", "request", "information"]
        for category in categories:
            if category in response_text:
                email_category = category
                break

    # 4. Add this question and answer to the log.
    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.content}
    ]

    # 5. Return the fields this box filled in.
    return {
        "is_spam": is_spam,
        "spam_reason": spam_reason,
        "email_category": email_category,
        "messages": new_messages
    }


# --- Box 3a: handle_spam (spam path) ------------------------------- ▶ STEP 10a
# Prints a note. Changes nothing.
def handle_spam(state: EmailState):
    """Alfred discards spam email with a note"""
    print(f"Alfred has marked the email as spam. Reason: {state['spam_reason']}")
    print("The email has been moved to the spam folder.")
    return {}


# --- Box 3b: draft_response (legitimate path) ---------------------- ▶ STEP 10b
# Asks the model to write a polite reply, and stores it in email_draft.
# Same pattern as classify_email: write the prompt, ask, return the new fields.
def draft_response(state: EmailState):
    """Alfred drafts a preliminary response for legitimate emails"""
    email = state["email"]
    category = state["email_category"] or "general"   # "or": use "general" if the category is None

    prompt = f"""
    As Alfred the butler, draft a polite preliminary response to this email.

    Email:
    From: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}

    This email has been categorized as: {category}

    Draft a brief, professional response that Mr. Hugg can review and personalize before sending.
    """

    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)           # MODEL CALL #2: Qwen writes the reply

    # state.get("messages", []) = the log so far ([] if there's none);
    # "+ [...]" makes a new list with these two entries added at the end.
    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.content}
    ]

    return {
        "email_draft": response.content,
        "messages": new_messages
    }


# --- Box 4: notify_mr_hugg (legitimate path) ------------------------ ▶ STEP 11
# Prints the email's details and Alfred's draft. Changes nothing.
def notify_mr_hugg(state: EmailState):
    """Alfred notifies Mr. Hugg about the email and presents the draft response"""
    email = state["email"]

    print("\n" + "="*50)
    print(f"Sir, you've received an email from {email['sender']}.")
    print(f"Subject: {email['subject']}")
    print(f"Category: {state['email_category']}")
    print("\nI've prepared a draft response for your review:")
    print("-"*50)
    print(state["email_draft"])
    print("="*50 + "\n")
    return {}


# ============================================================================
#  PART 5: THE DECISION (which arrow to follow)                 ▶ STEP 9
# ============================================================================
# (Stored now like the nodes; it runs at STEP 9, right after classify_email.)
# Not a box: a function LangGraph calls right after classify_email. It looks
# at the state and returns a LABEL. Part 6 maps each label to a box.
# (Same role as did_it_answer in Lupin/pythonLG.py.)
def route_email(state: EmailState) -> str:
    """Determine the next step based on spam classification"""
    if state["is_spam"]:
        return "spam"
    else:
        return "legitimate"


# ============================================================================
#  PART 6: THE GRAPH (put the boxes and arrows together)        ▶ STEP 5
# ============================================================================
# This is the drawing at the top of the file, written as code.
# Building the graph doesn't run it: it's like drawing a map before the trip.

email_graph = StateGraph(EmailState)   # an empty graph whose state is an EmailState form

# The boxes: add_node("name", function)
email_graph.add_node("read_email", read_email)
email_graph.add_node("classify_email", classify_email)
email_graph.add_node("handle_spam", handle_spam)
email_graph.add_node("draft_response", draft_response)
email_graph.add_node("notify_mr_hugg", notify_mr_hugg)

# The fixed arrows: add_edge(from, to)
email_graph.add_edge(START, "read_email")             # always start here
email_graph.add_edge("read_email", "classify_email")  # then always classify

# The forking arrow: after classify_email, call route_email(), and follow
# the arrow matching the label it returns.
email_graph.add_conditional_edges(
    "classify_email",
    route_email,
    {
        "spam": "handle_spam",            # label "spam"       -> box handle_spam
        "legitimate": "draft_response"    # label "legitimate" -> box draft_response
    }
)

# The last arrows, to the end.
email_graph.add_edge("handle_spam", END)
email_graph.add_edge("draft_response", "notify_mr_hugg")
email_graph.add_edge("notify_mr_hugg", END)

# Check the drawing and turn it into something we can run.
compiled_graph = email_graph.compile()


# ============================================================================
#  PART 7: RUN IT (two test emails)                             ▶ STEP 6
# ============================================================================
# Everything above was preparation. THIS is where things start happening.
# Two emails as plain dicts: one normal, one obvious spam.
legitimate_email = {
    "sender": "john.smith@example.com",
    "subject": "Question about your services",
    "body": "Dear Mr. Hugg, I was referred to you by a colleague and I'm interested in learning more about your consulting services. Could we schedule a call next week? Best regards, John Smith"
}

spam_email = {
    "sender": "winner@lottery-intl.com",
    "subject": "YOU HAVE WON $5,000,000!!!",
    "body": "CONGRATULATIONS! You have been selected as the winner of our international lottery! To claim your $5,000,000 prize, please send us your bank details and a processing fee of $100."
}

# invoke() = run the graph from START to END. We hand it the starting form:
# the email filled in, every other field blank (None or []).
# It returns the final form, with every field the boxes filled in.
print("\nProcessing legitimate email...")
# This line STARTS the graph: Python now jumps up to read_email (STEP 7),
# and comes back here only when the graph reaches END.
legitimate_result = compiled_graph.invoke({
    "email": legitimate_email,
    "is_spam": None,
    "spam_reason": None,
    "email_category": None,
    "email_draft": None,
    "messages": []
})

# When the line above finishes, the first email went through STEPS 7 to 11.

# ▶ STEP 12: the same again, with the spam email.
print("\nProcessing spam email...")
spam_result = compiled_graph.invoke({
    "email": spam_email,
    "is_spam": None,
    "spam_reason": None,
    "email_category": None,
    "email_draft": None,
    "messages": []
})
