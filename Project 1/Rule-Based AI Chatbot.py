
#KNOWLEDGE BASE: 

responses = {
    "hello": "Hello! How can I assist you today?",
    "hi": "Hi there! What can I do for you?",
    "how are you?": "I'm just a program, but I'm functioning as expected! How can I help you?",
    "bye": "Goodbye! Have a great day!",
    "time": "I dont have a clock, but your device does!.",
}


print("Welcome to the chatbot! Type 'exit' to end the conversation.")


#INPUT LOOP: 
while True:
    
    raw_input = input("You: ")
    clean_input = raw_input.lower().strip()
    
    
    if clean_input == "exit":
        print("Chatbot: Goodbye! See you next time!")
        break
    
    response = responses.get(clean_input, "I'm sorry, I don't understand that. Can you please rephrase?")
    
    print(f"Chatbot: {response}")