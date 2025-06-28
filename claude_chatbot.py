import os
import json
from anthropic import Anthropic
import datetime
import logging
import traceback

class ClaudeChatbot:
    def __init__(self):
        self.api_key = os.environ.get("CLAUDE_API_KEY")
        if not self.api_key:
            raise ValueError("CLAUDE_API_KEY environment variable not set")
        
        self.client = Anthropic(api_key=self.api_key)
        self.session_data = {}
        self.messages = []
        
        # Set up logging
        self.setup_logging()
        
        # Add system message
        self.system_prompt = """You are an ACR Phantom Analysis assistant. 
You help users analyze DICOM images of ACR phantoms for PET/CT quality control.
You have access to tools that can load DICOM directories, analyze individual slices, 
and analyze groups of slices. Guide the user through the analysis process step by step.
"""
        # Log the system prompt
        self.logger.info(f"System Prompt: {self.system_prompt}")
    
    def setup_logging(self):
        """Set up logging to save all interactions to a file"""
        # Create logs directory if it doesn't exist
        if not os.path.exists("logs"):
            os.makedirs("logs")
        
        # Create a unique filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/claude_conversation_{timestamp}.log"
        
        # Configure logging
        self.logger = logging.getLogger("claude_chatbot")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler and set format
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
        
        # Also save raw JSON for all exchanges
        self.json_log_file = f"logs/claude_conversation_{timestamp}.json"
        
        # Log start of session
        self.logger.info(f"Starting new conversation session")
        
        # Initialize conversation log
        self.conversation_log = {
            "timestamp": timestamp,
            "exchanges": []
        }
    
    def process_message(self, user_message):
        """
        Process a user message and get a response from Claude
        """
        # Log user message
        self.logger.info(f"User: {user_message}")
        
        # Add user message to conversation history
        self.messages.append({"role": "user", "content": user_message})
        
        # Create message with the updated history
        try:
            # Prepare the request for logging
            request_data = {
                "model": "claude-3-haiku-20240307",
                "max_tokens": 1000,
                "system": self.system_prompt,
                "messages": self.messages,
                # Note: tools field omitted for brevity in log
            }
            
            # Log the request (without API key)
            self.logger.info(f"API Request: {json.dumps(request_data, indent=2)}")
            
            # Make the API call
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                system=self.system_prompt,
                messages=self.messages,
                tools=[
                    {
                        "name": "load_dicom_directory",
                        "description": "Load DICOM files from a directory and extract metadata",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "directory_path": {
                                    "type": "string", 
                                    "description": "Path to directory containing DICOM files"
                                }
                            },
                            "required": ["directory_path"]
                        }
                    },
                    {
                        "name": "display_slices",
                        "description": "Display DICOM slices from a directory",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "directory_path": {
                                    "type": "string", 
                                    "description": "Path to directory containing DICOM files"
                                },
                                "rows": {
                                    "type": "integer", 
                                    "description": "Number of rows in the display",
                                    "default": 10
                                },
                                "cols": {
                                    "type": "integer", 
                                    "description": "Number of columns in the display",
                                    "default": 10
                                },
                                "figsize": {
                                    "type": "array",
                                    "items": {
                                        "type": 
                                        "number"
                                    },
                                    "description": "A tuple of width and height (in inches) for the figure size",
                                    "minItems": 2,
                                    "maxItems": 2,
                                    "default": [10, 10]
                                }
                            },
                            "required": ["directory_path"]
                        }
                    },
                    {
                        "name": "analyze_phantom_group",
                        "description": "Analyze a group of ACR phantom slices by summing them",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "file_paths": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of paths to DICOM files to analyze as a group"
                                },
                                "display_figures": {
                                    "type": "boolean",
                                    "description": "Whether to generate visualization figures"
                                }
                            },
                            "required": ["file_paths"]
                        }
                    }
                ]
            )
            
            # Log the response (convert to dict safely)
            response_dict = self._make_json_serializable(response)
            self.logger.info(f"API Response: {json.dumps(response_dict, indent=2)}")
            
            # Add exchange to JSON log
            exchange = {
                "timestamp": datetime.datetime.now().isoformat(),
                "user_message": user_message,
                "request": request_data,
                "response": response_dict
            }
            self.conversation_log["exchanges"].append(exchange)
            
            # Save the updated JSON log
            with open(self.json_log_file, 'w') as f:
                json.dump(self.conversation_log, f, indent=2)
            
            # Extract text and tool use content
            text_content = None
            tool_use_content = None
            
            # Loop through all content blocks to identify text and tool use content
            for content_block in response.content:
                if content_block.type == "text":
                    text_content = content_block.text
                    self.logger.info(f"Found text content: {text_content}")
                elif content_block.type == "tool_use":
                    tool_use_content = content_block
                    self.logger.info(f"Found tool use content: {tool_use_content.name}")
            
            # Process both the text and tool use if present
            assistant_response = ""
            
            # If there's text content, include it in the response
            if text_content:
                self.logger.info(f"Processing text content")
                assistant_response = text_content
                
                # Add text response to conversation history
                self.messages.append({
                    "role": "assistant",
                    "content": text_content
                })
            
            # If there's a tool use content, process it
            if tool_use_content:
                self.logger.info(f"Processing tool use content")
                
                # Extract tool call details
                tool_id = tool_use_content.id
                tool_name = tool_use_content.name
                tool_input = tool_use_content.input
                
                # Log tool call
                if isinstance(tool_input, dict):
                    tool_input_str = json.dumps(tool_input)
                else:
                    tool_input_str = str(tool_input)
                    
                self.logger.info(f"Tool Call: {tool_name} with input: {tool_input_str}")
                
                # Add a message indicating we're executing the tool
                if assistant_response:
                    assistant_response += "\n\nExecuting the requested tool..."
                else:
                    assistant_response = "Executing the requested tool..."
                
                # Process the tool call
                from acr_tools import handle_tool_call
                
                if isinstance(tool_input, dict):
                    tool_args = tool_input  # It's already a dictionary
                else:
                    # Fall back to JSON parsing if it's a string
                    tool_args = json.loads(tool_input)
                
                print(f"Executing tool: {tool_name} with args: {tool_args}")
                
                # Call the function 
                result = handle_tool_call(tool_name, **tool_args)
                
                # Log tool result
                self.logger.info(f"Tool Result: {json.dumps(result, indent=2)}")
                
                # Store session data if needed
                if tool_name == "load_dicom_directory" and result["status"] == "success":
                    self.session_data["file_paths"] = result.get("file_paths", [])
                    self.session_data["metadata"] = result.get("metadata", {})
                
                # Add to conversation history as a tool use message (from assistant)
                self.messages.append({
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": tool_id,
                            "name": tool_name,
                            "input": tool_input
                        }
                    ]
                })
                
                # Add to conversation history as a tool result message (from user)
                self.messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": json.dumps(result)
                        }
                    ]
                    }
                )
                
                # Get a follow-up response from Claude
                try:
                    # Log the follow-up request
                    follow_up_request = {
                        'model': 'claude-3-haiku-20240307',
                        'max_tokens': 1000,
                        'system': self.system_prompt,
                        'messages': self.messages
                    }
                    self.logger.info(f"Follow-up API Request: {json.dumps(follow_up_request, indent=2)}")
                    
                    # Make the follow-up API call
                    follow_up_response = self.client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=1000,
                        system=self.system_prompt,
                        messages=self.messages
                    )
                    
                    # Log the follow-up response
                    response_dict = self._make_json_serializable(follow_up_response)
                    self.logger.info(f"Follow-up API Response: {json.dumps(response_dict, indent=2)}")
                    
                    # Process the follow-up response
                    if follow_up_response.content and follow_up_response.content[0].type == "text":
                        follow_up_text = follow_up_response.content[0].text
                        
                        # Add to conversation history
                        self.messages.append({
                            "role": "assistant",
                            "content": follow_up_text
                        })
                        
                        # Log the response
                        self.logger.info(f"Assistant follow-up: {follow_up_text}")
                        
                        # Append follow-up to the assistant response
                        assistant_response += "\n\n" + follow_up_text
                        
                except Exception as e:
                    error_message = f"Error in follow-up: {str(e)}"
                    self.logger.error(error_message)
                    self.logger.error(traceback.format_exc())
                    
                    # Add error to the response
                    assistant_response += f"\n\nError getting follow-up response: {str(e)}"
            
            
            return assistant_response
                
        except Exception as e:
            error_message = f"Error communicating with Claude: {str(e)}"
            self.logger.error(error_message)
            self.logger.error(traceback.format_exc())
            return error_message

            
    def _make_json_serializable(self, obj):
        """Convert API response objects to JSON-serializable dictionaries"""
        if hasattr(obj, 'model_dump'):
            # For pydantic models (new Anthropic client)
            return obj.model_dump()
        elif hasattr(obj, '__dict__'):
            # For class instances
            result = {}
            for key, value in obj.__dict__.items():
                if not key.startswith('_'):  # Skip private attributes
                    result[key] = self._make_json_serializable(value)
            return result
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        else:
            # Basic types (str, int, bool, etc.) are already serializable
            return obj