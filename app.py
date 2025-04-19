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
            
            # Check if this might be a long-running operation (like loading DICOM files)
            is_long_operation = any(keyword in user_input.lower() for keyword in 
                                   ["load", "analyze", "dicom", "directory", "phantom"])
            is_long_operation=False 
            if is_long_operation:
                # Show immediate acknowledgment
                print("\nChatbot: Processing your request... ")
                
                # Use a separate thread for processing to keep the UI responsive
                response_data = {"response": None, "complete": False}
                
                def process_request():
                    response_data["response"] = chatbot.process_message(user_input)
                    response_data["complete"] = True
                
                # Start processing thread
                processing_thread = threading.Thread(target=process_request)
                processing_thread.daemon = True
                processing_thread.start()
                
                # Show progress indicator while waiting
                spinner = ['-', '\\', '|', '/']
                i = 0
                
                while not response_data["complete"]:
                    # Show spinner and progress updates if available
                    sys.stdout.write("\r" + spinner[i % len(spinner)] + " Processing... ")
                    
                    # Check for progress updates and display them
                    if hasattr(chatbot, 'current_progress') and chatbot.current_progress:
                        # Show the most recent progress update
                        sys.stdout.write(chatbot.current_progress[-1])
                    
                    sys.stdout.flush()
                    time.sleep(0.2)
                    i += 1
                
                # Clear the progress line
                sys.stdout.write("\r" + " " * 80 + "\r")
                
                # Show the complete response
                print(f"\nChatbot: {response_data['response']}")
            else:
                # For regular responses, process normally
                response = chatbot.process_message(user_input)
                print(f"\nChatbot: {response}")
            
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set the CLAUDE_API_KEY environment variable and try again.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()