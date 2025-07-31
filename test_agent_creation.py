import os
import requests
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import service_account
import json
import datetime
import traceback
import re
import time # For potential retries

# --- Configuration (MUST MATCH your project) ---
PROJECT_ID = "mirrormind12-706ee" 
SERVICE_ACCOUNT_KEY_PATH = "mirrormind12-706ee-d101a30d0f05.json" # This file must be in the same directory or provide full path
DEFAULT_LANGUAGE_CODE = "en"
GEMINI_API_KEY = "AIzaSyDjUVilLCAJM_vvj9NvNMr6k2nprFy9U0E" 
# --- End Configuration ---

def create_and_configure_agent_comprehensive(gemini_api_key):
    """
    Creates a Conversational Agent, creates essential intents,
    uses LLM to generate a comprehensive generative prompt,
    sets the Generative Model, enables Generative Fallback with the custom prompt.
    This version uses ONLY the requests library and focuses on the generative prompt.
    It requires MANUAL setting of "Start Resource" in the console if a specific playbook start is desired.
    """
    try:
        # --- Dummy Agent Data for LLM Prompt (from your reference) ---
        dynamic_agent_purpose = "To provide basic fitness advice and track workout progress."
        dynamic_agent_goal = "Empower users to maintain a healthy lifestyle."
        dynamic_core_behaviors = "Be polite, provide safe exercise recommendations, encourage consistency."
        dynamic_fallback_strategy = "If a query is out of scope, politely state limitations."
        dynamic_conversation_success_metrics = "User satisfaction and query resolution rate."
        dynamic_key_topics = "Workout routines, Exercise suggestions, Nutrition basics, Hydration tracking"
        dynamic_forbidden_topics = "Medical treatment, Specific diet plans, Supplement recommendations"
        dynamic_desired_tone = "Encouraging, knowledgeable, supportive"
        dynamic_target_audience = "Individuals new to fitness, people seeking workout ideas"
        dynamic_context_url = "https://www.myfitnessguide.com/faq"

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        purpose_hint = dynamic_agent_purpose.split(' ')[0] if dynamic_agent_purpose else "Agent"
        agent_display_name = f"PlaybookGenerativeAgent-{purpose_hint}-{timestamp}" 

        description = (
            f"Purpose: {dynamic_agent_purpose}\n"
            f"Goal: {dynamic_agent_goal}\n"
            f"Core Behaviors: {dynamic_core_behaviors}\n"
            f"Fallback Strategy: {dynamic_fallback_strategy}\n"
            f"Conversation Success Metrics: {dynamic_conversation_success_metrics}\n"
            f"Key Topics: {dynamic_key_topics}\n" 
            f"Forbidden Topics: {dynamic_forbidden_topics}\n" 
            f"Tone: {dynamic_desired_tone}\n"
            f"Audience: {dynamic_target_audience}"
        )
        if dynamic_context_url:
            description += f"\nContext URL: {dynamic_context_url}"
        # --- End Dummy Agent Data ---

        # --- AUTHENTICATION ---
        print("INFO: Attempting to load service account credentials...")
        if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            print(f"WARNING: GOOGLE_APPLICATION_CREDENTIALS environment variable not set. Attempting to use SERVICE_ACCOUNT_KEY_PATH: {SERVICE_ACCOUNT_KEY_PATH}")
            credentials = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_KEY_PATH,
                scopes=['https://www.googleapis.com/auth/cloud-platform'] 
            )
        else:
            print("INFO: Using GOOGLE_APPLICATION_CREDENTIALS environment variable for authentication.")
            credentials, project = google.auth.default(
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            if project and project != PROJECT_ID:
                print(f"WARNING: GOOGLE_APPLICATION_CREDENTIALS default project '{project}' does not match script's PROJECT_ID '{PROJECT_ID}'. Using script's PROJECT_ID.")
        
        if not credentials.token or credentials.expired:
            auth_req = GoogleAuthRequest()
            credentials.refresh(auth_req)
        access_token = credentials.token
        print("INFO: Successfully loaded credentials.")

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "X-Goog-User-Project": PROJECT_ID
        }

        # STEP 1: Create the Conversational Agent
        print(f"INFO: Creating Conversational Agent: '{agent_display_name}'...")
        agent_creation_url = f"https://dialogflow.googleapis.com/v3/projects/{PROJECT_ID}/locations/global/agents"
        agent_payload = {
            "displayName": agent_display_name,
            "defaultLanguageCode": DEFAULT_LANGUAGE_CODE,
            "timeZone": "America/New_York",
            "description": description
        }
        response = requests.post(agent_creation_url, headers=headers, json=agent_payload)
        response.raise_for_status()
        agent_data = response.json()
        agent_name_full = agent_data.get("name") 
        agent_id = agent_name_full.split('/')[-1] 
        print(f"SUCCESS: Agent created: '{agent_display_name}' (Full Name: '{agent_name_full}')")

        # --- STEP 2: Call LLM to Generate Intents AND Playbooks AND the Comprehensive Generative Prompt ---
        print("--- Starting LLM Generation Process for Intents, Playbooks, and Generative Prompt ---")
        
        gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_api_key}" 
        print(f"INFO: Gemini API URL: {gemini_api_url}")

        llm_prompt = f"""
        You are an expert Dialogflow CX agent designer. Your task is to generate a JSON object containing
        a list of initial conversational intents, a list of simple Playbooks, and a comprehensive generative prompt for an AI agent.

        **Crucially, you MUST generate one playbook with the exact display name "Default Generative Playbook".**
        This playbook should have a generic goal like "Serve as the main entry point for the agent and greet users." and simple steps
        like a greeting response. All other playbooks should have unique names.

        For each intent: include 3-5 diverse training phrases for each, and a concise default fulfillment response.
        For each Playbook: define a clear goal, specify an initial intent that triggers it, and outline
        2-3 simple conversational steps. Steps can be of type "response" (just text) or "question" (text
        and a parameter to collect).

        Focus strictly on the agent's defined scope and constraints.

        Agent Profile:
        - Name: {agent_display_name}
        - Purpose: {dynamic_agent_purpose}
        - Goal: {dynamic_agent_goal}
        - Desired Tone: {dynamic_desired_tone}
        - Target Audience: {dynamic_target_audience}
        - Context URL: {dynamic_context_url if dynamic_context_url else 'N/A'}

        Core Behaviors/Rules: {dynamic_core_behaviors}

        Fallback Strategy (Out-of-Scope Handling): {dynamic_fallback_strategy}
        Specifically, if a query is out of scope, the agent should politely state its limitations,
        e.g., "I can only assist with {dynamic_key_topics.lower()}. Is there anything related to that I can help with?"
        Only offer human assistance if the user explicitly asks for it.

        Conversation Success Metrics: {dynamic_conversation_success_metrics}

        Key Topics/Domains (MUST cover these): {dynamic_key_topics}

        Forbidden Topics/Restrictions (MUST NOT discuss these): {dynamic_forbidden_topics}

        Generate intents and responses ONLY related to the 'Key Topics/Domains' and strictly
        adhere to 'Forbidden Topics/Restrictions'. If a user query falls outside the 'Key Topics',
        it should be handled by a specific 'out_of_scope' intent following the 'Fallback Strategy'.
        Always include a 'greeting', 'thank_you', 'out_of_scope', and 'human_handoff' intent.

        Example JSON format for the complete output:
        {{
          "intents": [
            {{
              "intentName": "greeting",
              "trainingPhrases": ["hi", "hello", "hey there", "good morning"],
              "response": "Hello! How can I assist you today?"
            }},
            {{
              "intentName": "thank_you",
              "trainingPhrases": ["thank you", "thanks", "appreciate it", "cheers"],
              "response": "You're very welcome!"
            }},
            {{
              "intentName": "out_of_scope",
              "trainingPhrases": ["tell me about the weather", "what is your name", "how old are you"],
              "response": "I can only assist with fitness and workout tracking. Is there anything related to that I can help with?"
            }},
            {{
              "intentName": "human_handoff",
              "trainingPhrases": ["talk to a person", "connect me to support", "I need human help"],
              "response": "I understand. Please hold while I connect you to a human agent."
            }},
            {{
              "intentName": "book_appointment",
              "trainingPhrases": ["I want to book an appointment", "schedule a meeting", "can I set up a time"],
              "response": "Certainly! What type of appointment are you looking to book?"
            }}
          ],
          "playbooks": [
            {{
              "playbookName": "Default Generative Playbook",
              "goal": "Serve as the main entry point for the agent and greet users.",
              "initialIntent": "greeting",
              "steps": [
                {{
                  "type": "response",
                  "text": "Hello! I'm your AI assistant. How can I help you today?"
                }}
              ]
            }},
            {{
              "playbookName": "SimpleAppointmentBooking",
              "goal": "Collect patient details and preferred appointment time.",
              "initialIntent": "book_appointment",
              "steps": [
                {{
                  "type": "question",
                  "text": "Okay, what's your full name?",
                  "parameter": "patient_name"
                }},
                {{
                  "type": "question",
                  "text": "And what's the best phone number or email to reach you?",
                  "parameter": "contact_info"
                }},
                {{
                  "type": "response",
                  "text": "Thank you! I've noted your details. A staff member will contact you shortly to confirm your appointment."
                }}
              ]
            }}
          ],
          "generativePrompt": "You are a friendly, knowledgeable, and supportive fitness assistant. Your goal is to empower users to maintain a healthy lifestyle by providing safe exercise recommendations and encouraging consistency. ALWAYS be polite. ONLY provide advice on workout routines, exercise suggestions, nutrition basics, and hydration tracking. NEVER discuss medical treatments, specific diet plans, or supplement recommendations. If a user asks about forbidden topics or anything outside of your scope, politely state: 'I can only assist with fitness and workout tracking. Is there anything related to that I can help with?' Encourage users to ask more questions about fitness. Start conversations by offering help related to fitness. Keep responses concise and actionable. The user's query was: {{query}}. Please provide a helpful response based on your persona and rules."
        }}

        Please generate at least 5-7 distinct intents (including the required ones), 2-3 simple Playbooks relevant to the agent's description, and a detailed `generativePrompt`.
        Ensure that any Playbook's `initialIntent` matches an `intentName` in the generated intents list.
        Ensure the `generativePrompt` is a single, well-formatted string.
        """
        print(f"INFO: LLM Prompt (first 200 chars): {llm_prompt[:200]}...")

        llm_response_schema = {
            "type": "OBJECT",
            "properties": {
                "intents": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "intentName": { "type": "STRING" },
                            "trainingPhrases": {
                                "type": "ARRAY",
                                "items": { "type": "STRING" }
                            },
                            "response": { "type": "STRING" }
                        },
                        "propertyOrdering": ["intentName", "trainingPhrases", "response"]
                    }
                },
                "playbooks": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "playbookName": { "type": "STRING" },
                            "goal": { "type": "STRING" },
                            "initialIntent": { "type": "STRING" },
                            "steps": {
                                "type": "ARRAY",
                                "items": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "type": { "type": "STRING", "enum": ["question", "response"] },
                                        "text": { "type": "STRING" },
                                        "parameter": { "type": "STRING" }
                                    },
                                    "required": ["type", "text"]
                                }
                            }
                        },
                        "required": ["playbookName", "goal", "initialIntent", "steps"]
                    }
                },
                "generativePrompt": { "type": "STRING" }
            },
            "required": ["intents", "playbooks", "generativePrompt"]
        }

        llm_payload = {
            "contents": [{"role": "user", "parts": [{"text": llm_prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": llm_response_schema
            }
        }
        
        print("INFO: Sending request to Gemini API...")
        llm_response = requests.post(gemini_api_url, headers={'Content-Type': 'application/json'}, json=llm_payload)
        print(f"INFO: Received response from Gemini API with status: {llm_response.status_code}")
        llm_response.raise_for_status()

        llm_result = llm_response.json()
        
        generated_data = json.loads(llm_result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "{}"))
        generated_intents = generated_data.get("intents", [])
        generated_playbooks = generated_data.get("playbooks", [])
        generated_generative_prompt = generated_data.get("generativePrompt", "")
        
        print(f"INFO: LLM generated intents: {json.dumps(generated_intents, indent=2)}")
        print(f"INFO: LLM generated playbooks: {json.dumps(generated_playbooks, indent=2)}")
        print(f"INFO: LLM generated generative prompt (first 200 chars): {generated_generative_prompt[:200]}...")
        print("--- LLM Generation Process Completed ---")

        # --- STEP 3: Create Intents in Conversational Agent based on LLM output ---
        print("--- Starting Conversational Agent Intent Creation ---")
        df_cx_intents_api_base_url = f"https://dialogflow.googleapis.com/v3/{agent_name_full}/intents"
        
        SYSTEM_DEFINED_INTENTS = ["Default Welcome Intent"] # We will let this one exist and not touch it.

        valid_intents_to_create = []
        for intent_data in generated_intents:
            intent_display_name = intent_data.get('intentName')
            fulfillment_response = intent_data.get('response')

            if not intent_display_name:
                print(f"WARNING: Skipping intent due to missing 'intentName': {json.dumps(intent_data)}")
                continue
            
            if intent_display_name in SYSTEM_DEFINED_INTENTS:
                print(f"INFO: Skipping creation of system-defined intent: '{intent_display_name}'. Conversational Agents provides this by default.")
                continue

            training_phrases = intent_data.get('trainingPhrases', [])
            
            cleaned_training_phrases = []
            for phrase in training_phrases:
                if isinstance(phrase, str):
                    processed_phrase = re.sub(r'\[.*?\]', ' ', phrase)
                    processed_phrase = re.sub(r'\s+', ' ', processed_phrase).strip()
                    
                    if processed_phrase and len(processed_phrase) >= 2 and not processed_phrase.isdigit() and any(char.isalnum() for char in processed_phrase):
                        cleaned_training_phrases.append(processed_phrase)
                    else:
                        print(f"WARNING: Skipping potentially problematic training phrase for intent '{intent_display_name}': '{phrase}' (processed: '{processed_phrase}')")
                else:
                    print(f"WARNING: Skipping non-string training phrase for intent '{intent_display_name}': {phrase}")
            
            if not cleaned_training_phrases:
                default_phrase = f"{intent_display_name.replace('_', ' ')} inquiry" 
                print(f"WARNING: Intent '{intent_display_name}' has no valid training phrases after aggressive cleaning. Forcing generic default: '{default_phrase}'")
                cleaned_training_phrases = [default_phrase]

            cx_training_phrases = []
            for phrase in cleaned_training_phrases:
                cx_training_phrases.append({
                    "parts": [{"text": phrase}],
                    "repeatCount": 1 
                })

            cx_fulfillment_messages = [
                {"text": {"text": [fulfillment_response]}}
            ]

            intent_payload = {
                "displayName": intent_display_name,
                "trainingPhrases": cx_training_phrases,
                "fulfillment": {
                    "messages": cx_fulfillment_messages
                },
                "priority": 0
            }
            valid_intents_to_create.append(intent_payload)
            print(f"INFO: Intent payload being sent to Conversational Agent for '{intent_display_name}': {json.dumps(intent_payload, indent=2)}")
            
        if not valid_intents_to_create:
            print("WARNING: No valid custom intents generated by LLM or after filtering. Agent will only have default intents.")

        for intent_payload in valid_intents_to_create:
            intent_display_name = intent_payload['displayName']
            print(f"INFO: Attempting to create intent '{intent_display_name}' for agent {agent_name_full}")
            intent_creation_response = requests.post(df_cx_intents_api_base_url, headers=headers, json=intent_payload)
            intent_creation_response.raise_for_status() 
            print(f"SUCCESS: Successfully created intent '{intent_display_name}'")
        print("--- Conversational Agent Intent Creation Completed ---")

        # --- STEP 4: Create Playbooks in Dialogflow CX based on LLM output ---
        print("--- Starting Conversational Agent Playbook Creation ---")
        df_cx_playbooks_api_base_url = f"https://dialogflow.googleapis.com/v3/{agent_name_full}/playbooks"
        
        # Store the resource name of the "Default Generative Playbook"
        default_playbook_resource_name = None

        for playbook_data in generated_playbooks:
            playbook_display_name = playbook_data.get('playbookName')
            playbook_goal = playbook_data.get('goal')
            playbook_initial_intent_name = playbook_data.get('initialIntent')
            playbook_steps = playbook_data.get('steps', [])

            if not (playbook_display_name and playbook_goal and playbook_initial_intent_name and playbook_steps):
                print(f"WARNING: Skipping playbook due to missing required fields: {json.dumps(playbook_data)}")
                continue

            # Construct Playbook steps for API
            cx_playbook_steps = []
            for step in playbook_steps:
                step_type = step.get('type')
                step_text = step.get('text')
                step_parameter = step.get('parameter') # Optional

                if step_type == "response":
                    cx_playbook_steps.append({
                        "text": {"text": [step_text]}
                    })
                elif step_type == "question" and step_parameter:
                    # For questions, we need to define a parameter to collect
                    cx_playbook_steps.append({
                        "text": {"text": [step_text]},
                        "parameters": [
                            {
                                "id": step_parameter, # Parameter ID
                                "type": "STRING", # Assuming string for simplicity
                                "required": True
                            }
                        ]
                    })
                else:
                    print(f"WARNING: Skipping invalid playbook step: {json.dumps(step)}")
                    continue

            playbook_payload = {
                "displayName": playbook_display_name,
                "goal": playbook_goal,
                "initialIntent": playbook_initial_intent_name, # This needs to be the display name
                "steps": cx_playbook_steps
            }

            print(f"INFO: Playbook payload being sent to Conversational Agent for '{playbook_display_name}': {json.dumps(playbook_payload, indent=2)}")

            playbook_creation_response = requests.post(df_cx_playbooks_api_base_url, headers=headers, json=playbook_payload)
            playbook_creation_response.raise_for_status()
            
            created_playbook_data = playbook_creation_response.json()
            created_playbook_name = created_playbook_data.get("name") # Get the full resource name

            if playbook_display_name == "Default Generative Playbook":
                default_playbook_resource_name = created_playbook_name
                print(f"INFO: Captured resource name for 'Default Generative Playbook': {default_playbook_resource_name}")

            print(f"SUCCESS: Successfully created playbook '{playbook_display_name}'")
        print("--- Conversational Agent Playbook Creation Completed ---")

        # --- STEP 5: Update Generative Settings to Enable Playbook Mode, Set Generative Model, and Enable Generative Fallback with Custom Prompt ---
        print("INFO: Updating Generative Settings (enable playbook, set model, enable generative fallback with custom prompt)...")
        generative_settings_url = f"https://dialogflow.googleapis.com/v3/{agent_name_full}/generativeSettings"
        generative_settings_payload = {
            "playbookConfig": {"enabled": True}, # Keep playbook config enabled, even if no custom playbooks are made
            "generativeModelConfig": {"model": "gemini-2.5-flash"},
            "fallbackSettings": { # Generative Fallback settings
                "enabled": True,
                "model": "gemini-2.5-flash", # Use the same model for fallback
                "promptText": generated_generative_prompt, # Use the LLM-generated comprehensive prompt here
                "maxTokens": 200,
                "temperature": 0.5
            },
            "languageCode": DEFAULT_LANGUAGE_CODE
        }
        # Update mask now includes fallbackSettings fields
        update_mask = "playbookConfig.enabled,generativeModelConfig.model,fallbackSettings.enabled,fallbackSettings.model,fallbackSettings.promptText,fallbackSettings.maxTokens,fallbackSettings.temperature,languageCode"
        
        response = requests.patch(
            generative_settings_url, 
            headers=headers, 
            json=generative_settings_payload, 
            params={"updateMask": update_mask} 
        )
        response.raise_for_status()
        print("SUCCESS: Generative Settings updated with Playbook and Generative Fallback enabled and custom prompt applied.")

        # --- Removed STEP 6: No patching of Default Welcome Intent to invoke playbook ---
        # The agent will start with Default Start Flow, and general queries will use the generative fallback prompt.

        console_url = f"https://dialogflow.cloud.google.com/cx/projects/{PROJECT_ID}/locations/global/agents/{agent_id}/overview"

        print("\n--- Conversational Agent Setup Completed ---")
        print(f"Agent Display Name: {agent_display_name}")
        print(f"Agent ID: {agent_id}")
        print(f"Dialogflow CX Console URL: {console_url}")
        print("\n*** IMPORTANT: In the console, 'Start Resource' will still show 'Default Start Flow'. ***")
        print("However, general queries (including greetings) should now be handled by the LLM using the custom prompt.")
        print("Verify 'Models' is 'gemini-2.5-flash' under Generative AI settings, and 'Generative Fallback' is enabled with your custom prompt.")
        print("\n*** Then, test the agent in the simulator by typing 'hi' or 'hello'. ***")
        print("It should respond based on the generative prompt you provided.")
        print("\n*** Also, test with an out-of-scope query (e.g., 'tell me about quantum physics') to see Generative Fallback in action, governed by your custom prompt. ***")


        return {"status": "success", "agent_url": console_url, "agent_id": agent_id}

    except requests.exceptions.RequestException as req_e:
        print(f"ERROR: HTTP Request Error: {req_e}")
        error_body = req_e.response.text if req_e.response is not None else "No response body."
        status_code = req_e.response.status_code if req_e.response is not None else "N/A"
        print(f"ERROR: HTTP Response Status Code: {status_code}")
        print(f"ERROR: HTTP Response Body: {error_body}") 
        print(f"TRACEBACK: {traceback.format_exc()}")
        return {"status": "error", "message": f"Dialogflow CX API error: {error_body}"}
    except json.JSONDecodeError as json_e:
        print(f"ERROR: JSON Decoding Error from LLM response: {json_e}")
        print(f"TRACEBACK: {traceback.format_exc()}")
        return {"status": "error", "message": f"LLM response parsing error: {str(json_e)}"}
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {str(e)}")
        print(f"TRACEBACK: {traceback.format_exc()}")
        return {"status": "error", "message": f"An unexpected error occurred: {str(e)}"}

if __name__ == "__main__":
    result = create_and_configure_agent_comprehensive(GEMINI_API_KEY)
    print(json.dumps(result, indent=2))
