from claude_chatbot import ClaudeChatbot
import threading
import time
import sys

def main():
    """
    Main function to run the ACR phantom analysis chatbot with progress display
    """
    print("ACR Phantom Analysis Chatbot")
    print("=" * 50)
    print("Type 'quit' or 'exit' to end the session")
    print()
    
    try:
        chatbot = ClaudeChatbot()
        
        print("Welcome! I can help you analyze ACR phantom DICOM images.")
        print("To get started, please provide the directory containing your DICOM files.")
        
        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() in ["quit", "exit", "bye"]:
                print("\nChatbot: Thank you for using the ACR Phantom Analysis Chatbot. Goodbye!")
                break
            
            response = chatbot.process_message(user_input)
            print(f"\nChatbot: {response}")
            
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set the CLAUDE_API_KEY environment variable and try again.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()