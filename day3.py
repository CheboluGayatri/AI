import re

# A dictionary that maps keywords to predefined responses
responses = {
    "hello": "Hello! How can I assist you today?",
    "hi": "Hello! How can I assist you today?",
    "how are you": "I'm just a program, but thanks for asking!",
    "what is your name": "I'm a chatbot created to assist you!",
    "help": "Sure, I'm here to help! What do you need assistance with?",
    "bye": "Goodbye! Have a great day!",
    "thank you": "You're welcome! If you have any more questions, feel free to ask.",
    "default": "I'm sorry, I didn't understand that. Can you please rephrase?"
}

def chatbot_response(user_input: str) -> str:
    # Normalize and remove punctuation so "how are you?" matches "how are you"
    text = user_input.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation

    # Check each keyword (skip the 'default' key)
    for keyword, reply in responses.items():
        if keyword == "default":
            continue
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, text):
            return reply

    return responses.get("default", "I'm sorry, I didn't understand that.")

def chatbot():
    print("Chatbot: Hello! How can I assist you today?")
    try:
        while True:
            user_input = input("You: ")
            # exit on exact 'bye' (case-insensitive, after stripping)
            if user_input.strip().lower() == 'bye':
                print("Chatbot: Goodbye! Have a great day!")
                break

            response = chatbot_response(user_input)
            print(f"Chatbot: {response}")
    except (KeyboardInterrupt, EOFError):
        print("\nChatbot: Goodbye! (exiting)")

if __name__ == "__main__":
    chatbot()
