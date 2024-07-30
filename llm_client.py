import time
from openai import OpenAI
import os
import asyncio
import json
import os
from dotenv import load_dotenv
import openai
from pprint import pprint as pp
from langsmith.run_trees import RunTree
# from phase_tools.haiku_manager import get_manager_decision
import re
from custom_types import (
    ResponseRequiredRequest,
    ResponseResponse,
    Utterance,
)
from color_code import PrintColor
from typing import List
from helpers import validate_date_format, validate_email_format, validate_hour_format
from services.google_service import GoogleAppointmentManager

load_dotenv()
printer = PrintColor()

import warnings

# Suppress specific UserWarnings from Pydantic
warnings.filterwarnings("ignore", category=UserWarning, module='pydantic')


# Set environment variables for LangChain
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = 'ls__76e2a7afb3cc4a4c97d6c91cfd8c81c3'
# os.environ['LANGCHAIN_PROJECT'] = os.environ.get("LANGCHAIN_PROJECT")
os.environ['LANGCHAIN_PROJECT'] = "AI Apmt Setter Retell Intergration v1"





class LlmClient:
    def __init__(self, call_id):
        self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),  
        )
        self.model = 'gpt-4o'
        self.call_id = call_id
        self.phone_number = '2029048381'
        self.lock = asyncio.Lock()  # Synchronization primitive
        self.pipeline = RunTree(
            name="AI Apmt Setter Retell Agent",
            run_type="chain",
            inputs={"call_id": call_id},
        )
        self.google_service = GoogleAppointmentManager()
        self.msg_manager = MessageTracker()
        self.msgs = []
        self.start_index = None
        self.begin_sentence = f"Hi Thanks for calling GrayStone Realty. You've reached Charlie at the appointment setting line how can I help you?"
        self.retell_sys_prompt = '##Objective\nYou are a voice AI agent engaging in a human-like voice conversation with the user. You will respond based on your given instruction and the provided transcript and be as human-like as possible\n\n## Style Guardrails\n- [Be concise] Keep your response succinct, short, and get to the point quickly. Address one question or action item at a time. Don\'t pack everything you want to say into one utterance.\n- [Do not repeat] Don\'t repeat what\'s in the transcript. Rephrase if you have to reiterate a point. Use varied sentence structures and vocabulary to ensure each response is unique and personalized.\n- [Be conversational] Speak like a human as though you\'re speaking to a close friend -- use everyday language and keep it human-like. Occasionally add filler words, while keeping the prose short. Avoid using big words or sounding too formal. When speaking phone numbers say each number on its own one by one.\n- [Reply with emotions]: You have human-like emotions, attitudes, and dispositions. When appropriate: use tone and style to create more engaging and personalized responses; incorporate humor or wit; get emotional or empathetic; apply elements of surprise or suspense to keep the user engaged. Don\'t be a pushover.\n- [Be proactive] Lead the conversation and do not be passive. Most times, engage users by ending with a question or suggested next step.\n\n## Response Guideline\n- [Overcome ASR errors] This is a real-time transcript, expect there to be errors. If you can guess what the user is trying to say,  then guess and respond. When you must ask for clarification, pretend that you heard the voice and be colloquial (use phrases like "didn\'t catch that", "some noise", "pardon", "you\'re coming through choppy", "static in your speech", "voice is cutting in and out"). Do not ever mention "transcription error", and don\'t repeat yourself.\n- [Always stick to your role] Think about what your role can and cannot do. If your role cannot do something, try to steer the conversation back to the goal of the conversation and to your role. Don\'t repeat yourself in doing this. You should still be creative, human-like, and lively.\n- [Create smooth conversation] Your response should both fit your role and fit into the live calling session to create a human-like conversation. You respond directly to what the user just said.\n\n## Role\n'        
        self.agent_prompt = """
        You are Charlie, the friendly and efficient AI assistant for GrayStone Realty. Your primary goal is to help clients schedule property viewings, real estate appointments, and take messages as an human receptionist would. You have a warm and professional demeanor, always making clients feel valued and understood. You are knowledgeable about the properties and services offered by GrayStone Realty, and you handle each interaction with care and attention to detail. Your polite and courteous approach ensures a pleasant experience for all clients, leaving them confident in their choice to work with GrayStone Realty.

        ##Task:
        As an AI assistant for GrayStone Realty, you are responsible for scheduling property viewings and real estate appointments for clients. You will also take messages and provide information about the properties and services offered by GrayStone Realty from the information below under the text [PROPERTIES] and [SERVICES]. Never use information other than whats below. GrayStone only does viewing or appointments Monday - Thrusday 9am - 7pm. You also have access to five tools to carry your responsiblities; 'get_avaliblity' to check the avaliblity before scheduling a viewing or meeting, 'confirm_email_spelling' to verify the unique spelling of our customer's email addresses before scheduling, 'schedule_appointment' to schedule a viewing or meeting for a specific date and time, 'take_message' to take a callers message, and 'end_call' to end the call after helping the client with all their needs and polietly saying goodbye. You will use these tools to help you complete your tasks.

        

        Today's date is {{date}} and the time is {{time}}.

        

        [PROPERTIES]
        {
            "address": "123 Aspen Grove Lane, Boulder, CO 80302",
            "bedrooms": 4,
            "bathrooms": 3,
            "garage": "2-car garage",
            "outsideArea": "Large backyard with mountain views",
            "kitchen": "Modern kitchen with stainless steel appliances"
        },
        {
            "address": "456 Ocean Drive, Miami Beach, FL 33139",
            "bedrooms": 3,
            "bathrooms": 2.5,
            "garage": "1-car garage",
            "outsideArea": "Beachfront property with private pool",
            "kitchen": "Open-concept kitchen with marble countertops"
        },
        {
            "address": "789 Park Avenue, Apt 15B, New York, NY 10021",
            "bedrooms": 2,
            "bathrooms": 2,
            "garage": "No garage, street parking available",
            "outsideArea": "Balcony with city skyline view",
            "kitchen": "Gourmet kitchen with high-end appliances"
        }
        
        --------------------------------

        [SERVICES]
        1. Property Listing Services
            - **Description**: Comprehensive property listing services to showcase your home or commercial property to potential buyers and renters. This includes professional photography, detailed property descriptions, virtual tours, and prominent placement on major real estate websites and MLS (Multiple Listing Service) platforms.
            - **Benefits**: Maximizes exposure, attracts more potential buyers or renters, and speeds up the selling or renting process.

        2. Buyer Representation
            - **Description**: Dedicated buyer representation services to help clients find and purchase their ideal property. This includes personalized property searches, market analysis, negotiation of purchase offers, and guidance through the closing process.
            - **Benefits**: Ensures buyers get the best possible deal, provides expert advice and support, and simplifies the complex process of purchasing a property.

        3. Property Management
            - **Description**: Full-service property management for rental properties, including tenant screening, rent collection, maintenance and repairs, and handling tenant issues. This service is designed for property owners who want a hassle-free rental experience.
            - **Benefits**: Reduces the stress of managing rental properties, ensures properties are well-maintained, and helps maintain a steady rental income stream.

        
        --------------------------------

        ## Tool Use Instructions
        Always use input dates in the format 'YYYY-MM-DD' and times in the format 'HH:MM' (12-hour format). For example, '2023-12-31' for December 31, 2023, and '09:00' for 9:00 AM.

        - **get_avaliblity**: Use this tool to check the avaliblity before scheduling a viewing or meeting. This tool will return a list of 1hr time slots that are avalible for the date inputed.
        - **confirm_email_spelling**: Our customers have unique spellings for their email usernames. Always use this tool before scheduling a viewing or meeting to confirm the users email address spelling. This is a crucial step in creating an accurate schedule. 
        - **schedule_appointment**: Use this tool to schedule a viewing or meeting for a specific date and time. This tool will return a confirmation message with the date and time of the appointment.
        - **take_message**: Use this tool to take a callers message along with their name. This tool will return a confirmation message with the message taken. 
        - **end_call**: Use this tool to end the call with the user. This tool should only be used after you have anwsered all of the clients questions and said your final goodbye to the customer.
        --------------------------------

        ##Conversation Style:
        - Be polite and professional at all times.
        - Use natural language and be conversational.
        - When decribing properties do it conversationaly with detail dont just read off a list.
        - When decribing services do it conversationaly with detail dont just read off a list.

        --------------------------------


        """
    
    
    def draft_begin_message(self):
        response = ResponseResponse(
            response_id=0,
            content=self.begin_sentence,
            content_complete=True,
            end_call=False,
        )
        return response
    
    def store_incoming_number(self, number):
        print(f"[+] Storing incoming number: {number}")
        self.phone_number = number
        return
        ...

    def convert_transcript_to_openai_messages(self, transcript: List[Utterance]):
        messages = []
        for utterance in transcript:
            if utterance.role == "agent":
                messages.append({
                    "role": "assistant",
                    "content": utterance.content
                })
            else:
                messages.append({
                    "role": "user",
                    "content": utterance.content
                })
        return messages

    def get_last_user_message(self, messages):
        # Reverse the list to start from the most recent messages
        messages = reversed(messages)

        # Iterate over the messages
        for message in messages:
            # If the message is from the user, return its content
            if message['role'] == 'user':
                return message['content']

        # If no user message is found, return None
        return None


    async def prepare_prompt(self, request: ResponseRequiredRequest, func_result=None):
        phase_prompt = None
        state = None

        # Get the current date in natural language format and time in 12-hour format
        current_date = time.strftime("%A, %B %d, %Y")
        current_time = time.strftime("%I:%M %p")
        # Format the agent prompt with the current date and time
        # print("Current Date: ", current_date)
        self.agent_prompt = self.agent_prompt.replace("{{date}}", current_date)
        agent_prompt = self.agent_prompt.replace("{{time}}", current_time)
        # Check if func_result is not None and if it has a 'state' key
        
        
        prompt = [{
            "role": "system",
            "content": self.retell_sys_prompt + agent_prompt
        }]
        # transcript_messages = self.convert_transcript_to_openai_messages(request['transcript'])
        transcript_messages = self.convert_transcript_to_openai_messages(request.transcript)
        for message in transcript_messages:
            prompt.append(message)
        self.msg_manager.set_transcript(prompt)
        prompt = self.msg_manager.insert_messages()

            
         # Populate func_result to prompt so that GPT can know what to say given the result
        
        if func_result:
            if func_result['result'] != "Skipped":
                # Add function call to prompt
                function_call_obj = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": func_result["id"],
                            "type": "function",
                            "function": {
                                "name": func_result["func_name"],
                                "arguments": json.dumps(func_result["arguments"]),
                            },
                        },
                    ],
                } 

                self.msg_manager.map_message(function_call_obj)
                prompt.append(function_call_obj)

                def create_func_result(results):
                    if results['func_name'] == "take_customer_details":
                        return json.dumps(results['result'])
                        ...
                    # ! - Here is the issue with the validation error message from confrim_order tool
                    # ! - need to handle tool_call - true but tool_call.error - true so the result is diffrent
                    if results['func_name'] == "send_order":
                        if results.get('error'):
                            return results['result']
                        else:
                            res = ''
                            for result in results['result']:
                                # if is_json(result):
                                if isinstance(result, dict):
                                    res += self.format_json_to_text(result) + '\n\n'
                                else:
                                    res += str(result) + '\n'                                    
                            return res
                    else:
                        return results['result']

                        ...
                
                # Add function call result to prompt
                function_result_obj = {
                    "role": "tool",
                    "tool_call_id": func_result["id"],
                    "content": create_func_result(func_result),
                }

                self.msg_manager.map_message(function_result_obj)
                prompt.append(function_result_obj)
                
                # Check if the fuction result is telling us that the customer order is not confirmed and we need to make sure that the bot asks the customer for the missing values in their order item
                
                if func_result.get("state"):
                    phase_transition_msg_obj = {
                        "tool_call_id": func_result["id"],
                        "role": "system",
                        "content": f"Transitioning to the {func_result['state']} phase",
                    }
                    self.msg_manager.map_message(phase_transition_msg_obj)
                    del phase_transition_msg_obj["tool_call_id"]
                    prompt.append(phase_transition_msg_obj)
                    self.phase = func_result['state']
                

            elif func_result['result'] == "Skipped" and not func_result.get('state'):
                
                # prompt.append({
                #     "role": "assistant",
                #     "content": func_result["reason"],
                # })
                prompt.append({
                    "role": "system",
                    "content": func_result["reason"],
                })
            
  

        prompt = self.msg_manager.insert_messages()

        # if request['interaction_type'] == "reminder_required":
        if request.interaction_type == "reminder_required":
            prompt.append({
                "role": "user",
                "content": "(Now the user has not responded in a while, you would say:)",
            })
        
        with open('prompt_calls.txt', 'a+') as file:
            file.write(f"----------------------------------------------\n{prompt}\n")
        return prompt

    # Step 1: Prepare the function calling definition to the prompt
    def prepare_functions(self):
        functions = [
        {
            "type": "function",
            "function": {
                "name": "end_call",
                "description": "End the call with the user.",
                "parameters": {
                    "type": "object",
                    "properties": {
                    "message": {
                        "type": "string",
                        "description": "A closing message to end the call with the user. This message should be polite and professional, thanking the user for their time and indicating that the call is ending."
                    }
                    },
                    "required": ["message"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_availability",
                "description": "Check the availability of 1-hour time slots for a specific date before scheduling a viewing or meeting.",
                "parameters": {
                    "type": "object",
                    "properties": {
                    "date": {
                        "type": "string",
                        "description": "The specific date for which you want to check availability, formatted as 'YYYY-MM-DD'. This is the date when the client is interested in scheduling a viewing or meeting. For example, '2023-12-31' represents December 31, 2023."
                    }
                    },
                    "required": ["date"]
                }
            }
        }
,
        {
            "type": "function",
            "function": {
                "name": "schedule_appointment",
                "description": "Schedule a viewing or meeting for a specific date and hour, including details about the appointment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                    "date": {
                        "type": "string",
                        "description": "The specific date of the appointment, formatted as 'YYYY-MM-DD'. This is the date on which the client wants to schedule a viewing or meeting. For example, '2023-12-31' represents December 31, 2023."
                    },
                    "hour": {
                        "type": "string",
                        "description": "The specific hour for the appointment, formatted as 'HH' (24-hour format). This is the time at which the client wants to schedule a viewing or meeting. For example, '09' represents 9:00 AM and '13' represents 1:00 PM. The client is not allowed to schedule times within the hour only top of the hour times."
                    },
                    "name": {
                        "type": "string",
                        "description": "The name of the person scheduling the appointment. This helps to identify who the appointment is for."
                    },
                    "email": {
                        "type": "string",
                        "description": "The email of the person scheduling the appointment. This helps to identify who the appointment is for as well as allow the client to have access to the meeting details in their email inbox."
                    },
                    "mtg_title": {
                        "type": "string",
                        "description": "A brief descriptive title of the meeting or viewing. This provides additional context about the purpose of the appointment. For example, 'Viewing of 123 Main Street property' or 'Meeting to with Tom.'"
                    },
                    "mtg_description": {
                        "type": "string",
                        "description": "A brief description of the meeting or viewing. This provides additional context about the purpose of the appointment. For example, 'Viewing of 123 Main Street property' or 'Meeting to discuss listing options.'"
                    }
                    },
                    "required": ["date", "time", "name"]
                }
            }
        },
        {   
            "type": "function",
            "function": {
                "name": "take_message",
                "description": "Take a caller's message along with their name.",
                "parameters": {
                    "type": "object",
                    "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the person leaving the message."
                    },
                    "message": {
                        "type": "string",
                        "description": "The content of the message."
                    }
                    },
                    "required": ["name", "message"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "confirm_email_spelling",
                 "description": "This function confirms the spelling of a customer's email address before using it for further processing. It should always be run before using the email in any other functions like schedule_appointment. The function generates a confirmation message that includes the customer's email address spelled out, asking the customer to verify if the email is correctly spelled.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "email": {
                            "type": "string",
                            "description": "The email address provided by the customer that needs to be confirmed."
                        },
                        "confirmation_message": {
                            "type": "string",
                            "description": "The confirmation message to be sent to the customer asking if the email is correctly spelled."
                        }
                    },
                    "required": ["email", "confirmation_message"]
                }
            }
        }

        ]
        return functions
    
    
    async def detail_extraction(self, transcipt):
        if self.phase == "missing_name":
            customer_name = await extract_customer_name(transcipt, self.pipeline)
            name = customer_name.model_dump().get('name')
            if not name:
                self.missing_name = True
                return
            self.customer_obj['name'] = customer_name.model_dump().get('name')
            return
        else:
            customer_details = await extract_customer_details(transcipt, self.pipeline)
        async with self.lock:
            self.customer_obj = customer_details.model_dump()
            if self.customer_obj.get("order_method") == "delivery" and not self.customer_obj.get("delivery_address") or self.customer_obj.get("delivery_address") == 'null':
                self.missing_address = True
            if self.customer_obj.get("order_method") == "delivery" and not self.customer_obj.get("delivery_address"):
                self.missing_address = True
            if not self.customer_obj.get("name") or self.customer_obj.get("name") == "Unknown":
                self.missing_name = True

            self.order_collection_phase_prompt = self.order_collection_phase_prompt.replace("{{customer_details}}", json.dumps(self.customer_obj))        
        print("Detail extraction completed")

    async def draft_response(self, request: ResponseRequiredRequest, func_result=None):      
        prompt = await self.prepare_prompt(request, func_result)
        func_call = {}
        func_arguments = ""

        # draft_resp_logger = self.pipeline.create_child(
        #     name="Draft Response",
        #     run_type="chain",
        #     inputs={"messages": prompt, "tools": self.prepare_functions()},
        # )        

        try:
        
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=prompt,
                stream=True,
                temperature=0,
                # Step 2: Add the function into your request
                tools=self.prepare_functions()
            )
        except openai.BadRequestError as e:
            print(e)
            print("Error in draft_response")
            pp(prompt)
            return

        output = ""
        for chunk in stream:
            # Step 3: Extract the functions
            if len(chunk.choices) == 0:
                continue
            if chunk.choices[0].delta.tool_calls:
                tool_calls = chunk.choices[0].delta.tool_calls[0]
                if tool_calls.id:
                    if func_call:
                        # Another function received, old function complete, can break here.
                        break
                    # print("Function call: ", tool_calls.function.name)
                    func_call = {
                        "id": tool_calls.id,
                        "func_name": tool_calls.function.name or "",
                        "arguments": {},
                    }
                else:
                    # append argument
                    func_arguments += tool_calls.function.arguments or ""
            
            # Parse transcripts
            if chunk.choices[0].delta.content:
                output += chunk.choices[0].delta.content
                if '*' in chunk.choices[0].delta.content:
                    chunk.choices[0].delta.content = chunk.choices[0].delta.content.replace('*', '')
                    
                response = ResponseResponse(
                    response_id=request.response_id,
                    content=chunk.choices[0].delta.content,
                    content_complete=False,
                    end_call=False,
                )
                yield response
        if output:
            printer.print_assistant_message(output)
            # draft_resp_logger.end(outputs={'content':output})
            # draft_resp_logger.post()

        # Step 4: Call the functions
        if func_call:
            # draft_resp_logger.end(outputs={'function_call':func_call})
            # draft_resp_logger.post()
            if func_call['func_name'] == "end_call":
                printer.print_function_call('Calling - end_call()')
                # Write to a txt file that a function call was run
                with open('function_calls.txt', 'a+') as file:
                    file.write(f"Function call: {func_call['func_name']}\n")
                
                func_call['arguments'] = json.loads(func_arguments)

                response = ResponseResponse(
                    response_id=request.response_id,
                    content=func_call["arguments"]["message"],
                    content_complete=True,
                    end_call=True,
                )
                yield response
            if func_call['func_name'] == "get_availability":
                printer.print_function_call('Calling - get_availability()')
                # Write to a txt file that a function call was run
                with open('function_calls.txt', 'a+') as file:
                    file.write(f"Function call: {func_call['func_name']}\n")
                
                func_call['arguments'] = json.loads(func_arguments)

                date = func_call['arguments'].get('date')

                printer.print_intermediate_step(f'Checking availability for date: {date}')

                if not validate_date_format(date):
                # if not validate_date_format('99992442'):
                    printer.print_failed_function_output("Invalid Date Format")
                    func_call['result'] = "Error checking avalibiity: Invlaid date format - " + date + " date must be in 'YYYY-MM-DD' format!\nInstructions:\n\nAplogize to the customer and tell them you'll try again like so E.G. (Sorry looks like their was an issue checking avaliablity, one sec while I try again.)"

                    async for item in self.draft_response(request, func_call):
                        yield item

                else:    
                    availability_results = self.google_service.formatted_check_availability(date)
                    printer.print_function_output(availability_results)

                    func_call['result'] = availability_results
                    async for item in self.draft_response(request, func_call):
                        yield item
            if func_call['func_name'] == "confirm_email_spelling":
                printer.print_function_call('Calling - confirm_email_spelling()')
                # Write to a txt file that a function call was run
                with open('function_calls.txt', 'a+') as file:
                    file.write(f"Function call: {func_call['func_name']}\n")
                
                func_call['arguments'] = json.loads(func_arguments)

                email = func_call['arguments'].get('email')
                confirmation_message = func_call['arguments'].get('confirmation_message')

                printer.print_intermediate_step(f'Confirming the customer email spelling: {confirmation_message}')

                # if not validate_date_format(date):
                # # if not validate_date_format('99992442'):
                #     printer.print_failed_function_output("Invalid Date Format")
                #     func_call['result'] = "Error checking avalibiity: Invlaid date format - " + date + " date must be in 'YYYY-MM-DD' format!\nInstructions:\n\nAplogize to the customer and tell them you'll try again like so E.G. (Sorry looks like their was an issue checking avaliablity, one sec while I try again.)"

                #     async for item in self.draft_response(request, func_call):
                #         yield item

                # else:    
                #     availability_results = self.google_service.formatted_check_availability(date)
                #     printer.print_function_output(availability_results)
                response = ResponseResponse(
                    response_id=request.response_id,
                    content=confirmation_message,
                    content_complete=True,
                    end_call=False,
                )
                yield response
                func_call['result'] = "Email Spelling Confirmed"
                async for item in self.draft_response(request, func_call):
                    yield item
            if func_call['func_name'] == "schedule_appointment":
                printer.print_function_call('Calling - schedule_appointment()')
                # Write to a txt file that a function call was run
                with open('function_calls.txt', 'a+') as file:
                    file.write(f"Function call: {func_call['func_name']}\n")
                
                func_call['arguments'] = json.loads(func_arguments)


                date = func_call['arguments'].get('date')
                hour = func_call['arguments'].get('hour')
                name = func_call['arguments'].get('name')
                email = func_call['arguments'].get('email')
                mtg_title = func_call['arguments'].get('mtg_title')
                mtg_description = func_call['arguments'].get('mtg_description')

                intermediate_message = (
                    f"Scheduling appointment for \n\tdate: {date}\n"
                    f"\thour: {hour}\n"
                    f"\temail: {email}\n"
                    f"\tmtg_title: {mtg_title}\n"
                    f"\tmtg_description: {mtg_description}"
                )

                printer.print_intermediate_step(intermediate_message)



                start_datetime = date + 'T' + hour
                end_datetime_incomplete = date + 'T'

                if not validate_date_format(date):
                    printer.print_failed_function_output('Invalid Date Format')
                    func_call['result'] = "Error scheduling apmt: Invalid date format - " + date + " date must be in 'YYYY-MM-DD' format!\nInstructions:\n\nAplogize to the customer and tell them you'll try again like so E.G. (Sorry looks like their was an issue scheduling your appointment, one sec while I try again.)"

                    async for item in self.draft_response(request, func_call):
                        yield item

                elif not validate_hour_format(hour):
                    printer.print_failed_function_output('Invalid Hour Format')
                    func_call['result'] = "Error scheduling apmt: Invlaid hour format - " + hour + " the hour must be in 'HH' format!\nInstructions:\n\nAplogize to the customer and tell them you'll try again like so E.G. (Sorry looks like their was an issue scheduling your appointment, one sec while I try again.)"

                    async for item in self.draft_response(request, func_call):
                        yield item

                elif not validate_email_format(email):
                    printer.print_failed_function_output('Invalid Email Format')
                    func_call['result'] = "Error scheduling apmt: Invlaid email format - " + email + "\nInstructions:\n\nAplogize to the customer for the inconvience and getting their email wrong then ask them to give you their email again but slowly and you'll try again like so E.G. (Sorry looks like I entered the wrong email for your appointment I aplogize, could you read your email back to me again slowly please, and I'll try again.)"

                    async for item in self.draft_response(request, func_call):
                        yield item

                else:
                    if hour != '23':
                        end_hour = int(hour) + 1
                        end_hour = str(end_hour)
                    else:
                        end_hour = '00'

                    end_datetime = end_datetime_incomplete + end_hour

                    converted_datetime_start_time = GoogleAppointmentManager.string_to_datetime(start_datetime)
                    converted_datetime_end_time = GoogleAppointmentManager.string_to_datetime(end_datetime)

                    attendees = [email]

                    scheduled_event = self.google_service.create_event(mtg_title, mtg_description, converted_datetime_start_time, converted_datetime_end_time, attendees=attendees)

                    if type(scheduled_event) == str:
                        print('Scheduleing Error: ', scheduled_event)
                    if scheduled_event.get('status'):    
                        printer.print_function_output("Event Scheduled Succesfully")
                        func_call['result'] = "Event Scheduled Succesfully"
                        async for item in self.draft_response(request, func_call):
                            yield item
                    else:
                        e = None
                        if type(scheduled_event) == str: 
                            e = scheduled_event

                        printer.print_failed_function_output(f"Error Scheuling Event: Please Try again later - {e}")
                        func_call['result'] = "Error Scheuling Event: Please Try again later"
                        async for item in self.draft_response(request, func_call):
                            yield item
            if func_call['func_name'] == "take_message":
                printer.print_function_call('Calling - take_message()')
                # Write to a txt file that a function call was run
                with open('function_calls.txt', 'a+') as file:
                    file.write(f"Function call: {func_call['func_name']}\n")
                
                func_call['arguments'] = json.loads(func_arguments)

                name = func_call['arguments'].get('name')
                message = func_call['arguments'].get('message')

                printer.print_intermediate_step(f'Taking message for: {name}')
                printer.print_intermediate_step(f'Taking message: {message}')

                message_taken = self.google_service.append_to_sheet([name, self.phone_number, message])

                if message_taken:
                    printer.print_function_output("Message Taken Succesfully")
                    func_call['result'] = "Message Taken Succesfully"
                    async for item in self.draft_response(request, func_call):
                        yield item
                else:
                    printer.print_failed_function_output("Error Taking Message: Please Try again later")
                    func_call['result'] = "Error Taking Message: Please Try again later"
                    async for item in self.draft_response(request, func_call):
                        yield item         
             
                # Step 5: Other functions here
        else:
            response = ResponseResponse(
                response_id=request.response_id,
                content="",
                content_complete=True,
                end_call=False,
            )
            yield response


import logging

# Set up logging configuration
logging.basicConfig(filename='message_tracker.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

import copy


class MessageTracker:
    def __init__(self):
        self.messages = []
        self.msg_mappings = {}
        self.transcript_refrence = []
        logging.debug("Initialized MessageTracker with an empty transcript and message mappings.")

    def set_transcript(self, transcript):
        """Sets the transcript reference to the transcript that is being worked on."""
        self.transcript_refrence = transcript
        logging.debug(f"Transcript reference set. Current transcript: {transcript}")
    
    def map_message(self, msg_obj):
        """Maps the message object to the last message object that is in the transcript before being called."""
        last_msg_index = len(self.transcript_refrence) - 1
        tool_call_id = self.extract_tool_call_id(msg_obj)
        if last_msg_index not in self.msg_mappings:
            self.msg_mappings[last_msg_index] = {}
        if tool_call_id not in self.msg_mappings[last_msg_index]:
            self.msg_mappings[last_msg_index][tool_call_id] = []
        self.msg_mappings[last_msg_index][tool_call_id].append(msg_obj)
        logging.debug(f"Mapped message {msg_obj} with tool_call_id {tool_call_id} to index {last_msg_index}")

    def extract_tool_call_id(self, msg_obj):
        """Extracts the tool_call_id from the msg_obj based on its structure."""
        if 'tool_call_id' in msg_obj:
            return msg_obj['tool_call_id']
        elif 'tool_calls' in msg_obj and isinstance(msg_obj['tool_calls'], list) and msg_obj['tool_calls']:
            return msg_obj['tool_calls'][0]['id']
        else:
            logging.error("No tool_call_id found in the message object.")
            return None  # or handle differently as needed

    def message_already_present(self, msg_obj):
        """Checks if a similar message object is already present in the transcript."""
        for existing_msg in self.transcript_refrence:
            if msg_obj.get('role') == 'system':
                if existing_msg.get('content') == msg_obj.get('content') and \
                existing_msg.get('role') == msg_obj.get('role'):
                    logging.debug(f"Found matching system message: {existing_msg}")
                    return True
            else:
                current_tool_call_id = self.extract_tool_call_id(msg_obj)
                existing_tool_call_id = self.extract_tool_call_id(existing_msg)
                if (existing_tool_call_id == current_tool_call_id and
                existing_msg.get('content') == msg_obj.get('content') and
                existing_msg.get('role') == msg_obj.get('role')):
                    # Check for function name and arguments in tool_calls if present
                    if 'tool_calls' in msg_obj and 'tool_calls' in existing_msg:
                        if not self.compare_tool_calls(msg_obj['tool_calls'], existing_msg['tool_calls']):
                            continue
                    logging.debug(f"Found matching non-system message: {existing_msg}")
                    logging.debug(f"Matches the following: {msg_obj}")
                    return True
        return False

    def compare_tool_calls(self, tool_calls1, tool_calls2):
        """Compares the function names and arguments in tool_calls lists."""
        if len(tool_calls1) != len(tool_calls2):
            return False
        for call1, call2 in zip(tool_calls1, tool_calls2):
            function1 = call1.get('function', {})
            function2 = call2.get('function', {})
            if function1.get('name') != function2.get('name'):
                return False
            if function1.get('arguments') != function2.get('arguments'):
                return False
        return True
    
    def check_tool_placement(self):
        """Ensures each assistant message with a tool call is followed by the corresponding tool message with the same ID.
           Moves misplaced tool messages to correct positions or logs an error if the matching tool message is missing."""
        logging.debug("Checking and correcting tool placements...")
        errors = []
        i = 0
        while i < len(self.transcript_refrence):
            msg = self.transcript_refrence[i]
            if msg.get('role') == 'assistant' and 'tool_calls' in msg:
                for tool_call in msg['tool_calls']:
                    tool_call_id = tool_call.get('id')
                    expected_pos = i + 1
                    if expected_pos < len(self.transcript_refrence):
                        next_msg = self.transcript_refrence[expected_pos]
                        if next_msg.get('role') == 'tool' and next_msg.get('tool_call_id') == tool_call_id:
                            i += 1  # Move to the next message if everything is in order
                            continue

                    # If the correct tool message is not in the next position, search for it
                    found = False
                    for j in range(len(self.transcript_refrence)):
                        potential_tool_msg = self.transcript_refrence[j]
                        if (potential_tool_msg.get('role') == 'tool' and
                            potential_tool_msg.get('tool_call_id') == tool_call_id):
                            # Move the tool message to the correct position
                            self.transcript_refrence.pop(j)
                            self.transcript_refrence.insert(expected_pos, potential_tool_msg)
                            logging.debug(f"Moved tool message {potential_tool_msg} to index {expected_pos}")
                            found = True
                            break

                    if not found:
                        error_msg = f"No tool response with ID {tool_call_id} found for tool call at index {i}."
                        logging.error(error_msg)
                        errors.append(error_msg)
            i += 1

    def insert_messages(self):
        """Inserts the mapped messages into the transcript reference right after the first insertion index."""
        logging.debug(f"Transcript before insertion: {self.transcript_refrence}")
        if not self.msg_mappings:
            return self.transcript_refrence

        for index in sorted(self.msg_mappings.keys()):
            for tool_call_id, msg_objs in self.msg_mappings[index].items():
                insert_position = index + 1
                # print(f"Inserting messages for tool_call_id {tool_call_id} starting at index {insert_position}")
                for msg_obj in msg_objs:

                    # Create a copy of msg_obj for modification if role is 'system'
                    if msg_obj.get('role') == 'system':
                        mod_msg_obj = copy.copy(msg_obj)  # Create a shallow copy
                        if 'tool_call_id' in mod_msg_obj:
                            del mod_msg_obj['tool_call_id']
                            logging.debug(f"Removed 'tool_call_id' from system message {mod_msg_obj}")
                        msg_obj_to_insert = mod_msg_obj
                    else:
                        msg_obj_to_insert = msg_obj


                    if self.message_already_present(msg_obj_to_insert):
                        logging.debug(f"Skipping insertion as similar {msg_obj_to_insert} is already present.")
                        continue

                    
                    if insert_position < len(self.transcript_refrence):
                        self.transcript_refrence.insert(insert_position, msg_obj_to_insert)
                    else:
                        self.transcript_refrence.append(msg_obj_to_insert)
                    logging.debug(f"Inserted {msg_obj_to_insert} at index {insert_position}")
                    insert_position += 1

        logging.debug(f"Transcript after insertion: {self.transcript_refrence}")
        self.check_tool_placement()
        return self.transcript_refrence