import os
import requests
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import service_account
import json
import datetime
import traceback
import re
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS 

app = Flask(__name__)
CORS(app) 

# --- Configuration ---
PROJECT_ID = "mirrormind12-706ee" 
GEMINI_API_KEY = "AIzaSyDjUVilLCAJM_vvj9NvNMr6k2nprFy9U0E" 
SERVICE_ACCOUNT_KEY_PATH = "mirrormind12-706ee-d101a30d0f05.json" 
# --- End Configuration ---

@app.route('/')
def serve_index():
    """Serve the main HTML page from the 'static' directory."""
    return send_from_directory('static', 'index.html')

@app.route('/create_agent', methods=['POST'])
def create_df_agent_endpoint():
    """
    Flask endpoint to trigger the Dialogflow CX agent creation process.
    This endpoint expects a POST request from the frontend with agent details.
    """
    try:
        request_data = request.get_json(silent=True) or {} 

        # Extract dynamic inputs, with fallbacks to generic defaults if not provided
        dynamic_agent_purpose = request_data.get('purpose', "To assist users with general inquiries.")
        dynamic_agent_goal = request_data.get('goal', "Provide accurate information and resolve queries efficiently.")
        dynamic_core_behaviors = request_data.get('core_behaviors', "Be polite, provide clear answers, and offer help.")
        dynamic_fallback_strategy = request_data.get('fallback_strategy', "If a query is out of scope, politely state limitations and suggest relevant topics.")
        dynamic_conversation_success_metrics = request_data.get('conversation_success_metrics', "User satisfaction and query resolution rate.")
        
        dynamic_key_topics_list = request_data.get('key_topics', ["general inquiries", "support"])
        if isinstance(dynamic_key_topics_list, list):
            dynamic_key_topics = ", ".join(dynamic_key_topics_list)
        else:
            dynamic_key_topics = dynamic_key_topics_list 

        dynamic_forbidden_topics_list = request_data.get('forbidden_topics', ["personal financial advice", "medical diagnoses"])
        if isinstance(dynamic_forbidden_topics_list, list):
            dynamic_forbidden_topics = ", ".join(dynamic_forbidden_topics_list)
        else:
            dynamic_forbidden_topics = dynamic_forbidden_topics_list

        dynamic_desired_tone = request_data.get('desired_tone', "Helpful and informative.")
        dynamic_target_audience = request_data.get('target_audience', "General users.")
        dynamic_context_url = request_data.get('context_url', "") 

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        purpose_hint = dynamic_agent_purpose.split(' ')[0] if dynamic_agent_purpose else "Agent"
        agent_display_name = f"PlaybookDynamicAgent-{purpose_hint}-{timestamp}" 
        
        # Define default language code for the agent
        DEFAULT_LANGUAGE_CODE = "en" # Define this once

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

        # --- AUTHENTICATION FOR DIALOGFLOW CX API ---
        try:
            credentials = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_KEY_PATH,
                scopes=['https://www.googleapis.com/auth/cloud-platform'] 
            )
            if not credentials.token or credentials.expired:
                auth_req = GoogleAuthRequest()
                credentials.refresh(auth_req)
            access_token = credentials.token
            print(f"INFO: Successfully loaded credentials from {SERVICE_ACCOUNT_KEY_PATH}")
        except Exception as e:
            print(f"ERROR: Failed to load service account key from {SERVICE_ACCOUNT_KEY_PATH}: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return jsonify({"status": "error", "message": f"Authentication failed: {str(e)}"}), 500

        # STEP 1: Create the Dialogflow CX Agent
        dialogflow_api_url = f"https://dialogflow.googleapis.com/v3/projects/{PROJECT_ID}/locations/global/agents"
        print(f"Dialogflow API URL being used for agent creation: {dialogflow_api_url}")
        
        agent_payload = {
            "displayName": agent_display_name,
            "defaultLanguageCode": DEFAULT_LANGUAGE_CODE, # Use the defined language code
            "timeZone": "America/New_York",
            "description": description
            # generativeSettings removed from initial create, will be patched
        }

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "X-Goog-User-Project": PROJECT_ID
        }

        print(f"Attempting to create agent via REST API with display_name: {agent_display_name} in project: {PROJECT_ID}")
        
        response_api = requests.post(dialogflow_api_url, headers=headers, json=agent_payload)
        response_api.raise_for_status() 

        response_data = response_api.json()
        agent_name_full = response_data.get("name") 
        agent_id = agent_name_full.split('/')[-1] 

        if agent_name_full is None:
            error_msg = f"Dialogflow API response missing 'name' field. Full response: {json.dumps(response_data, indent=2)}"
            print(f"ERROR: {error_msg}")
            return jsonify({"status": "error", "message": "Failed to retrieve agent ID from API response."}), 500

        print(f"Agent created: {agent_display_name} (ID: {agent_name_full})")

        # --- NEW STEP: Set the start playbook ---
        print(f"Attempting to set the start playbook for agent {agent_name_full}...")
        agent_patch_url = f"https://dialogflow.googleapis.com/v3/{agent_name_full}"

        agent_patch_payload = {
            "startPlaybook": f"projects/{PROJECT_ID}/locations/global/agents/{agent_id}/playbooks/00000000-0000-0000-0000-000000000000"
        }

        patch_agent_response = requests.patch(
            agent_patch_url,
            headers=headers,
            json=agent_patch_payload,
            params={"updateMask": "start_playbook"}
        )
        patch_agent_response.raise_for_status()
        print(f"Successfully set the start playbook for agent {agent_name_full}.")
        # --- END NEW STEP ---

        # --- NEW STEP: Call updateGenerativeSettings to set the LLM model ---
        print(f"Attempting to set the LLM model for agent {agent_name_full} using updateGenerativeSettings...")
        generative_settings_url = f"https://dialogflow.googleapis.com/v3/{agent_name_full}/generativeSettings"
        
        generative_settings_payload = {
            "llmModelSettings": {
                "model": "gemini-1.0-pro"
            }
        }
        
        patch_generative_settings_response = requests.patch(
            generative_settings_url, 
            headers=headers, 
            json=generative_settings_payload, 
            params={"updateMask": "llm_model_settings.model"}
        )
        patch_generative_settings_response.raise_for_status()
        print(f"Successfully set the LLM model for agent {agent_name_full}.")
        # --- END NEW STEP ---

        # STEP 2: Call LLM to Generate Intents AND Playbooks
        print("--- Starting LLM Generation Process for Intents and Playbooks ---")
        
        gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
        print(f"Gemini API URL: {gemini_api_url}")

        llm_prompt = f"""
        You are an expert Dialogflow CX agent designer. Your task is to generate a JSON object containing
        a list of initial conversational intents and a list of simple Playbooks.

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
              "response": "I can only assist with {dynamic_key_topics.lower()}. Is there anything related to that I can help with?"
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
          ]
        }}

        Please generate at least 5-7 distinct intents (including the required ones) and 2-3 simple Playbooks relevant to the agent's description.
        Ensure that any Playbook's `initialIntent` matches an `intentName` in the generated intents list.
        """
        print(f"LLM Prompt (first 200 chars): {llm_prompt[:200]}...")

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
                }
            },
            "required": ["intents", "playbooks"]
        }

        llm_payload = {
            "contents": [{"role": "user", "parts": [{"text": llm_prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": llm_response_schema
            }
        }
        
        print("Sending request to Gemini API...")
        llm_response = requests.post(gemini_api_url, headers={'Content-Type': 'application/json'}, json=llm_payload)
        print(f"Received response from Gemini API with status: {llm_response.status_code}")
        print(f"Raw LLM response: {llm_response.text}")
        llm_response.raise_for_status()

        llm_result = llm_response.json()
        
        generated_data = json.loads(llm_result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "{}"))
        generated_intents = generated_data.get("intents", [])
        generated_playbooks = generated_data.get("playbooks", [])
        
        print(f"LLM generated intents: {json.dumps(generated_intents, indent=2)}")
        print(f"LLM generated playbooks: {json.dumps(generated_playbooks, indent=2)}")
        print("--- LLM Generation Process Completed ---")

        # STEP 3: Create Intents in Dialogflow CX based on LLM output
        print("--- Starting Dialogflow CX Intent Creation ---")
        df_cx_intents_api_base_url = f"https://dialogflow.googleapis.com/v3/{agent_name_full}/intents"
        
        SYSTEM_DEFINED_INTENTS = ["greeting", "thank_you", "out_of_scope", "human_handoff"]

        valid_intents_to_create = []
        for intent_data in generated_intents:
            intent_display_name = intent_data.get('intentName')
            fulfillment_response = intent_data.get('response')

            if not intent_display_name:
                print(f"WARNING: Skipping intent due to missing 'intentName': {json.dumps(intent_data)}")
                continue
            
            if intent_display_name in SYSTEM_DEFINED_INTENTS:
                print(f"INFO: Skipping creation of system-defined intent: '{intent_display_name}'. Dialogflow CX provides this by default.")
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
            print(f"Intent payload being sent to DF CX for '{intent_display_name}': {json.dumps(intent_payload, indent=2)}")
            
        if not valid_intents_to_create:
            print("WARNING: No valid custom intents generated by LLM or after filtering. Agent will only have default intents.")

        for intent_payload in valid_intents_to_create:
            intent_display_name = intent_payload['displayName']
            print(f"Attempting to create intent '{intent_display_name}' for agent {agent_name_full}")
            intent_creation_response = requests.post(df_cx_intents_api_base_url, headers=headers, json=intent_payload)
            intent_creation_response.raise_for_status() 
            print(f"Successfully created intent '{intent_display_name}'")
        print("--- Dialogflow CX Intent Creation Completed ---")

        # STEP 4: Create Playbooks in Dialogflow CX based on LLM output
        print("--- Starting Dialogflow CX Playbook Creation ---")
        df_cx_playbooks_api_base_url = f"https://dialogflow.googleapis.com/v3/{agent_name_full}/playbooks"

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

            print(f"Playbook payload being sent to DF CX for '{playbook_display_name}': {json.dumps(playbook_payload, indent=2)}")

            playbook_creation_response = requests.post(df_cx_playbooks_api_base_url, headers=headers, json=playbook_payload)
            playbook_creation_response.raise_for_status()
            print(f"Successfully created playbook '{playbook_display_name}'")
        print("--- Dialogflow CX Playbook Creation Completed ---")


        return jsonify({
            "status": "success",
            "message": f"Agent '{agent_display_name}' created successfully with custom intents and playbooks!",
            "agent_id": agent_id,
            "agent_url": f"https://dialogflow.cloud.google.com/cx/projects/{PROJECT_ID}/locations/global/agents/{agent_id}/overview"
        }), 200

    except requests.exceptions.RequestException as req_e:
        print(f"HTTP Request Error: {req_e}")
        error_body = req_e.response.text if req_e.response is not None else "No response body."
        print(f"HTTP Response Status Code: {req_e.response.status_code}")
        print(f"HTTP Response Body: {error_body}") 
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({"status": "error", "message": f"Dialogflow CX API error: {error_body}"}), req_e.response.status_code if req_e.response is not None else 500
    except json.JSONDecodeError as json_e:
        print(f"JSON Decoding Error from LLM response: {json_e}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({"status": "error", "message": f"LLM response parsing error: {str(json_e)}"}), 500
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({"status": "error", "message": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
