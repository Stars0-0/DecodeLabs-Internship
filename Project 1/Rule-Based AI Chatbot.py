
#KNOWLEDGE BASE: 

import datetime

exact_responses = {
    "hello": "Hello! How can I assist you today?",
    "hi": "Hi there! What can I do for you?",
    "how are you?": "I'm just a program, but I'm functioning as expected! How can I help you?",
    "bye": "Goodbye! Have a great day!",
    "time": "I dont have a clock, but your device does!.(but you can ask a little more formally for the time or date and I can tell you!)",
    "what is ai":    "AI is the simulation of human intelligence by machines — and you're learning to build it!",
    "thank you":      "You're welcome! Happy to help. ",
    "help":          "Topics I know: greetings, ai/ml/nlp,time, date, history",
}

keyword_responses = [
    ("weather",    "I can't check live weather, but try weather.com!"),
    ("time",       lambda: f"The current time is {datetime.datetime.now().strftime('%I:%M %p')}."),
    ("date",       lambda: f"Today is {datetime.datetime.now().strftime('%A, %B %d, %Y')}."),
    ("name",       "My name is DBot — your DecodeLabs Rule-Based Assistant."),
    ("thank",      "You're welcome! Happy to help. "),
    ("sorry",      "No worries at all!"),
    ("what can you do", "I can answer questions about AI/ML/NLP, tell jokes, give the time & date, and chat!"),
    ("project",    "This is Project 1 which is the Rule-Based Chatbot."),
]


print("Welcome to the chatbot! Type 'exit' to end the conversation.")

history = []  # To store conversation history


def log(speaker, message):
    """Save each exchange to the conversation history."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history.append({"time": timestamp, "speaker": speaker, "message": message})
    
def show_history():
    """Display the conversation history."""
    if not history:
        print("No conversation history yet.")
        return
    print("\nConversation History:")
    for entry in history:
        print(f"[{entry['time']}] {entry['speaker']}: {entry['message']}")
    print()  # Add an extra newline for better readability
    
def get_response(clean_input):
    """
    Response engine -3 Tiers:
    1. Exact match -> dictionary 0(1)
    2. Keyword match -> partial scan
    3. Fallback  -> default message

    """
    # Tier 1: Exact match
    if clean_input in exact_responses:
        return exact_responses[clean_input]
    
    # Tier 2: Keyword match
    for keyword, response in keyword_responses:
        if keyword in clean_input:
            return response() if callable(response) else response
    
    # Tier 3: Fallback
    return "I'm sorry, I don't understand that. Type 'help' for a list of things I can do."



# ── MAIN LOOP ────────────────────────────────────────────────
print("=" * 52)
print("   DecodeLabs DBot v2.0 — Rule-Based AI Chatbot")
print("   Type 'exit' to quit | 'history' to review chat")
print("=" * 52 + "\n")


while True:
    
    # PHASE 1: INPUT & SANITIZATION
    raw_input = input("You: ")
    clean_input = raw_input.lower().strip()
    
    
    #skip empty inputs
    if not clean_input:
        continue
    
    #Log user input
    log("User", raw_input)
    
    #Exit command
    if clean_input == "exit":
        print("Chatbot: Goodbye! See you next time!")
        break
    
    if clean_input == "history":
        show_history()
        continue
    
    # PHASE 2 & 3: RESPONSE GENERATION
    response = get_response(clean_input)
    print(f"Chatbot: {response}\n")
    #Log chatbot response
    log("Chatbot", response)