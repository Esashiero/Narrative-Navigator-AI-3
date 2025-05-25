import json
import re
import threading
import time
import ollama
from PyQt5.QtCore import QThread, pyqtSignal
from constants import TRANSCRIPT_CHUNK_DURATION_SECONDS
from llm_prompts import base_system_prompt

# Define the JSON schema for the expected entity output format
entity_list_schema = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string"
                    },
                    "type": {
                        "type": "string"
                    },
                    "description": {
                        "type": "string"
                    },
                    "base_importance_score": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10
                    }
                },
                "required": [
                    "name",
                    "type",
                    "description",
                    "base_importance_score"
                ]
            }
        }
    },
    "required": [
        "entities"
    ]
}

class LLMThread(QThread):
    llm_log = pyqtSignal(dict) 
    entities_updated = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        try:
            self.model = "llama3.2:latest"
            ollama.chat(model=self.model, messages=[{"role": "user", "content": "hi"}], stream=False)
        except Exception as e:
            self.llm_log.emit({"type": "error", "message": f"Failed to connect to Ollama or model '{self.model}' not found: {e}"})
            self.model = None

        self.transcriptions = []
        self.entities = [] 
        self.running = False
        self.external_context = ""
        self.content_title = "Unknown Content"

        self.base_system_prompt_template = base_system_prompt
        # Debug prints to inspect template and formatted string
        print(f"DEBUG: Initializing LLMThread")
        print(f"DEBUG: base_system_prompt_template (first 500 chars):\n{self.base_system_prompt_template[:500]}...")
        print(f"DEBUG: Keys provided for initial format: content_title='{self.content_title}', external_context='No external context loaded yet...'")
        # Format the system prompt with only the expected keys
        self.system_prompt = self.base_system_prompt_template.format(
            content_title=self.content_title,
            external_context="No external context loaded yet. Please wait for the application to gather information."
        )
        print(f"DEBUG: Formatted self.system_prompt (first 500 chars):\n{self.system_prompt[:500]}...")
        self.last_transcript_processed_idx = -1
        
        self.VALID_ENTITY_TYPES = ["Characters", "Locations", "Organizations", "Key Objects", "Concepts/Events"]

    def set_content_title(self, title):
        self.content_title = title
        self._update_system_prompt()

    def set_external_context(self, context):
        self.external_context = context
        self._update_system_prompt()

    def _update_system_prompt(self):
        # Debug prints to inspect template and formatted string
        print(f"DEBUG: Updating system prompt")
        print(f"DEBUG: base_system_prompt_template (first 500 chars):\n{self.base_system_prompt_template[:500]}...")
        print(f"DEBUG: Keys provided for update format: content_title='{self.content_title}', external_context='{self.external_context[:50]}...'")
        self.system_prompt = self.base_system_prompt_template.format(
            content_title=self.content_title,
            external_context=self.external_context
        )
        print(f"DEBUG: Formatted self.system_prompt (first 500 chars):\n{self.system_prompt[:500]}...")
    
    def _normalize_entity_type(self, type_str):
        """Normalizes LLM output type strings to our canonical types."""
        if not type_str:
            return None
        type_str_lower = type_str.lower().strip()
        
        # Handle compound types by splitting and taking the first part as a primary hint
        if ',' in type_str_lower:
            type_str_lower = type_str_lower.split(',')[0].strip()
        
        # Explicitly handle combined types or singular/plural mismatches from LLM
        if type_str_lower == "concept/event":
            return "Concepts/Events"
        if type_str_lower == "locations/organizations":
            # Heuristic: if a place name, it's probably a location first.
            # We'll normalize name separately, and use the normalized name to guide this
            # For "USA" or "United States" cases, "Locations" is more appropriate.
            return "Locations" # Default to Locations for this combo
        
        # Handle new combined type seen in logs
        if type_str_lower == "locations/concepts/events":
            return "Locations" # Defaulting based on observed example ('Marines')
        
        if type_str_lower in ["characters", "characters/individuals", "character", "individual"]:
            return "Characters"
        elif type_str_lower in ["locations", "location", "countries", "country", "cities", "city", "places", "place"]:
            return "Locations"
        elif type_str_lower in ["organizations", "organization", "agencies", "agency", "governments", "government", "corporations", "corporation", "groups", "group", "factions", "faction", "allies", "powers"]:
            return "Organizations"
        elif type_str_lower in ["key objects", "key object", "objects", "object", "artifacts", "artifact", "weapons", "weapon"]:
            return "Key Objects"
        elif type_str_lower in ["concepts/events", "concepts", "concept", "events", "event", "historical events", "dates", "years", "periods", "period", "projects", "programs", "wars", "conflicts", "eras", "era", "ages", "age", "campaigns", "campaign"]:
            return "Concepts/Events"
        return None

    def _normalize_for_comparison(self, name):
        """
        Normalizes entity names for robust comparison and deduplication.
        Includes a very comprehensive mapping for common synonyms, acronyms, and known typos/variations.
        The goal is to map various inputs to a single, canonical string.
        """
        name_lower = name.lower().strip()

        # Comprehensive canonical mapping
        canonical_map = {
            # Characters
            "easterman": "hendrick joliet easterman",
            "eastman": "hendrick joliet easterman",
            "hendrick joliet easterman": "hendrick joliet easterman",
            "wernick": "rudolf gustav wernicke",
            "wernicke": "rudolf gustav wernicke",
            "rudolf gustav wernicke": "rudolf gustav wernicke",
            "jameson lawler": "jameson lawler",
            "jameson lawler (cia agent)": "jameson lawler",
            "abe bradley aviano": "abe bradley aviano",
            "abe bradley aviano's": "abe bradley aviano", # Possessive
            "stalin": "joseph stalin",
            "joseph stalin": "joseph stalin",
            "truman": "harry s truman",
            "harry s truman": "harry s truman",
            "billy": "billy", # The Outlast character
            "miles": "miles", # The Outlast character
            "hope": "hope", # The Outlast character
            "mother gooseberry": "mother gooseberry",

            # Locations
            "us": "united states",
            "usa": "united states",
            "united states": "united states",
            "united states (usa)": "united states",
            "united states of america": "united states",
            "the usa": "united states",
            "e usa (united states)": "united states", # From screenshot
            "los alamos": "los alamos",
            "los alamos national laboratory": "los alamos",
            "hong kong": "hong kong",
            "hiroshima": "hiroshima",
            "nagasaki": "nagasaki",
            "poland": "poland",
            "germany": "germany",
            "italy": "italy",
            "japan": "japan",
            "soviet union": "union of soviet socialist republics", # Map to full name
            "chicago": "chicago",
            "eniwetok atoll": "eniwetok atoll",
            "enno atoq atoll": "eniwetok atoll", # Typo from transcript
            "mount massive asylum": "mount massive asylum",

            # Organizations
            "ussr": "union of soviet socialist republics",
            "union of soviet socialist republics": "union of soviet socialist republics",
            "oss": "office of strategic services",
            "office of strategic services": "office of strategic services",
            "cia": "central intelligence agency",
            "central intelligence agency": "central intelligence agency",
            "nazi germany": "nazi germany",
            "kingdom of italy": "kingdom of italy",
            "empire of japan": "empire of japan",
            "murkoff corporation": "murkoff corporation",
            "murkov corporation": "murkoff corporation", # Typo from transcript
            "red barrels": "red barrels", # Game developer
            "axis powers": "axis powers",
            "korean people's army": "korean people's army",

            # Key Objects
            "atomic bomb": "atomic bomb",
            "hydrogen bomb": "hydrogen bomb",
            "walrider": "walrider",
            "lsd": "lsd",

            # Concepts/Events
            "cold war": "cold war era",
            "cold war era": "cold war era",
            "1939": "1939",
            "1945": "1945",
            "1947": "1947",
            "1949": "1949",
            "1951": "1951",
            "1953": "1953",
            "world war ii": "world war ii",
            "second world war": "world war ii",
            "nuclear age": "nuclear age",
            "arms race": "arms race",
            "operation paperclip": "operation paperclip",
            "project bluebird": "project bluebird",
            "project artichoke": "project artichoke",
            "project bluebird, project artichoke (cia projects)": "project bluebird", # Pick one as primary, description can be combined later
            "project bluebird, project artichoke": "project bluebird",
            "space race": "space race",
            "the outlast trials": "the outlast trials", # Canonical game title
            "the outlast trials video games": "the outlast trials",
            "outlast trials game": "the outlast trials",
            "season 1 of the outlast trials": "season 1 of the outlast trials", # Canonical season name
            "season 1 of the trials": "season 1 of the outlast trials", # Shorter form
            "season 2 of the outlast series": "season 2 of the outlast series", # Canonical season name
            "season 2 of the outlast": "season 2 of the outlast series", # Shorter form
            "season 2": "season 2 of the outlast series", # Maps to full season
            "korean war": "korean war",
        }
        
        # Check for direct map first
        if name_lower in canonical_map:
            return canonical_map[name_lower]

        # Attempt to clean further if not directly mapped.
        # This is a fallback and less reliable than explicit mappings.
        cleaned_name = re.sub(r'\s*\([^)]*\)', '', name_lower).strip() # remove parenthesized text (e.g., "(US)")
        for article in ['the ', 'a ', 'an ']:
            if cleaned_name.startswith(article):
                cleaned_name = cleaned_name[len(article):].strip()
        cleaned_name = re.sub(r"'s\b", '', cleaned_name).strip() # remove 's at end of word
        cleaned_name = re.sub(r"s'\b", 's', cleaned_name).strip() # remove s' at end of word
        cleaned_name = re.sub(r'[^a-z0-9\s]', '', cleaned_name).strip() # remove non-alphanumeric (keep spaces)
        cleaned_name = re.sub(r'\s+', ' ', cleaned_name).strip() # reduce multiple spaces

        # After cleaning, check if it now matches a canonical form
        if cleaned_name in canonical_map:
            return canonical_map[cleaned_name]

        return cleaned_name # Return cleaned name if no canonical mapping is found

    def _normalize_for_mention_check(self, text):
        """
        More aggressive normalization for checking mentions in raw text.
        Converts to lowercase, removes most punctuation, handles common plural/possessive endings.
        """
        text = text.lower()
        text = re.sub(r"['’]\s*s?\b", '', text) # Handles 's and s' (e.g., 'character's' or 'characters')
        text = re.sub(r'[^a-z0-9\s]', ' ', text) # Replace non-alphanumeric with space
        text = re.sub(r'\s+', ' ', text).strip() # Reduce multiple spaces to single space
        return text

    def _is_similar_entity(self, entity1, entity2):
        """
        Compares two entity dicts for similarity based on normalized name and canonical type.
        This function is crucial for internal deduplication.
        """
        type1 = self._normalize_entity_type(entity1.get("type", ""))
        type2 = entity2.get("type", "") 
        
        if type1 != type2 or type1 is None:
            return False
            
        name1_norm = self._normalize_for_comparison(entity1["name"])
        name2_norm = self._normalize_for_comparison(entity2["name"])
        return name1_norm == name2_norm

    def _update_importance_from_transcript(self, transcript_text):
        """
        Increments mention_count for existing entities found in the new transcript.
        Uses canonical names for matching to handle variations/typos.
        """
        normalized_transcript = self._normalize_for_mention_check(transcript_text)
        
        for entity in self.entities:
            # Get the canonical name of the *existing* entity for matching
            entity_canonical_name = self._normalize_for_comparison(entity["name"]) 
            
            # Prepare a regex pattern for the canonical name, ensuring whole word match
            # re.escape is important if entity name contains special regex characters
            pattern = r'\b' + re.escape(entity_canonical_name) + r'\b'
            
            if re.search(pattern, normalized_transcript):
                entity["mention_count"] += 1
                self.llm_log.emit({"type": "debug", "message": f"Entity '{entity['name']}' (canonical: '{entity_canonical_name}') ({entity['type']}) mention_count incremented to {entity['mention_count']}"})

    def run(self):
        self.running = True
        if self.model is None:
            self.llm_log.emit({"type": "error", "message": "LLM model not loaded, cannot process entities."})
            self.running = False
            return

        if not self.external_context:
            self.llm_log.emit({"type": "status", "message": "LLM Thread waiting for external context..."})
            while self.running and not self.external_context:
                time.sleep(1)
            if not self.running:
                return

        self.llm_log.emit({"type": "status", "message": "LLM Thread started with external context."})
        while self.running:
            if len(self.transcriptions) > self.last_transcript_processed_idx + 1:
                transcript_to_process = self.transcriptions[self.last_transcript_processed_idx + 1] # Process the next unprocessed transcript
                current_transcript_idx = self.last_transcript_processed_idx + 1 
                
                # Take recent transcripts for better context for LLM
                # Use a sliding window of the last 5 transcripts (including current)
                recent_transcripts = self.transcriptions[max(0, current_transcript_idx - 4):current_transcript_idx + 1]
                
                current_entities_for_llm_prompt = [
                    {"name": e["name"], "type": e["type"], "description": e["description"]}
                    for e in self.entities
                ]

                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content":
                     f"Current transcript snippet: {transcript_to_process}\n"
                     f"Recent context (last {len(recent_transcripts)} snippets): {recent_transcripts}\n"
                     f"Current narrative cheat sheet: {json.dumps(current_entities_for_llm_prompt, indent=2)}\n"
                     f"Based on ALL information (current transcript, recent context, full cheat sheet), identify *all identifiable* entities. For EVERY entity you return (new or existing), provide its 'base_importance_score' (1-10) re-evaluated based on its inherent narrative relevance. Remember to include historical dates/years and specific organizations like OSS if mentioned. Ensure to correctly categorize and canonicalize names like 'Easterman'/'Eastman' to 'Hendrick Joliet Easterman', 'Wernick'/'Wernicke' to 'Rudolf Gustav Wernicke', and 'Operation Paperclip'."}
                ]
                self.llm_log.emit({"type": "prompt", "message": f"Prompt for transcript index {current_transcript_idx}", "data": messages})
                
                llm_identified_entities = []
                raw_content_from_llm = ""
                try:
                    response = ollama.chat(
                        model=self.model,
                        messages=messages,
                        stream=False,
                        # Use the new format parameter to enforce JSON schema
                        format=entity_list_schema, # Pass the schema dictionary here
                        options={
                            "temperature": 0.1, # Keep low for deterministic output
                            "top_p": 0.9,       # Keep relatively high
                            "top_k": 40,        # Default is fine
                            "repeat_penalty": 1.0 # Avoid penalizing repeated keys
                        } # Keep these options for output quality
                    )
                    raw_content_from_llm = response['message']['content']
                    self.llm_log.emit({"type": "raw_response", "message": f"Raw LLM response for transcript index {current_transcript_idx}", "data": raw_content_from_llm})

                    # --- Robust JSON Parsing ---
                    parsed_llm_output = None
                    json_string_to_parse = ""

                    try:
                        # Attempt 1: Try to parse directly (most common good case)
                        try:
                            json_string_to_parse = raw_content_from_llm.strip()
                            parsed_llm_output = json.loads(json_string_to_parse)
                        except json.JSONDecodeError:
                            # Attempt 2: Search for a JSON block (e.g., if wrapped in markdown)
                            json_match = re.search(r"```json\s*(.*?)\s*```", raw_content_from_llm, re.DOTALL)
                            if json_match:
                                json_string_to_parse = json_match.group(1).strip()
                                parsed_llm_output = json.loads(json_string_to_parse)
                            else:
                                # Attempt 3: Search for the first valid JSON object (from first { to last })
                                json_match_loose = re.search(r"\{.*\}", raw_content_from_llm, re.DOTALL)
                                if json_match_loose:
                                    json_string_to_parse = json_match_loose.group(0).strip()
                                    parsed_llm_output = json.loads(json_string_to_parse)
                                else:
                                    # Attempt 4: Search for the first valid JSON array (from first [ to last ])
                                    json_match_array_loose = re.search(r"\[.*\]", raw_content_from_llm, re.DOTALL)
                                    if json_match_array_loose:
                                        json_string_to_parse = json_match_array_loose.group(0).strip()
                                        parsed_llm_output = json.loads(json_string_to_parse)
                                    else:
                                        raise json.JSONDecodeError("No recognizable JSON structure found.", raw_content_from_llm, 0)

                    except (json.JSONDecodeError, ValueError) as e:
                        # Handle JSON parsing and structure errors
                        self.llm_log.emit({"type": "error", "message": f"JSON parsing/structure error: {str(e)}\nAttempted to parse:\n{json_string_to_parse if json_string_to_parse else raw_content_from_llm}", "data": raw_content_from_llm})
                        # Proceed with an empty entities list to avoid crashing
                        llm_identified_entities = []
                    except Exception as e:
                        # Handle other potential errors during the LLM call or processing
                        self.llm_log.emit({"type": "error", "message": f"LLM processing failed: {str(e)}"})
                        llm_identified_entities = [] # Ensure it's always a list

                    # Now process the parsed_llm_output
                    if parsed_llm_output is None:
                        self.llm_log.emit({"type": "warning", "message": f"JSON parsing attempts found no valid structure.\nAttempted to parse:\n{raw_content_from_llm}", "data": raw_content_from_llm})
                        llm_identified_entities = [] # Ensure empty list if nothing was parsed
                    elif isinstance(parsed_llm_output, list):
                        # LLM sometimes returns a list of entities directly instead of {"entities": [...]}
                        llm_identified_entities = parsed_llm_output
                    elif isinstance(parsed_llm_output, dict) and 'entities' in parsed_llm_output:
                        llm_identified_entities = parsed_llm_output.get('entities', [])
                    else:
                        raise ValueError(f"Unexpected JSON structure after parsing: {type(parsed_llm_output)} - {json_string_to_parse}")

                    if not isinstance(llm_identified_entities, list):
                        raise ValueError("Expected 'entities' to be a list after parsing.")
                            
                    self.llm_log.emit({"type": "parsed_entities", "message": f"Parsed entities from LLM for transcript index {current_transcript_idx}", "data": llm_identified_entities})

                except (json.JSONDecodeError, ValueError) as e:
                    self.llm_log.emit({"type": "error", "message": f"JSON parsing error: {str(e)}\nAttempted to parse:\n{json_string_to_parse if json_string_to_parse else raw_content_from_llm}", "data": raw_content_from_llm})
                    # Continue with empty entities list if parsing fails, but log the issue
                    llm_identified_entities = []
                except Exception as e:
                    self.llm_log.emit({"type": "error", "message": f"LLM request failed: {str(e)}"})
                    llm_identified_entities = [] # Ensure it's always a list

                # --- Entity Reconciliation and Update ---
                temp_entities_dict = {}
                for e in self.entities:
                    # Use the canonical name and type as the primary key for existing entities
                    normalized_name = self._normalize_for_comparison(e["name"])
                    temp_entities_dict[(normalized_name, e["type"])] = e

                for llm_entity_data in llm_identified_entities:
                    name = llm_entity_data.get("name")
                    raw_type = llm_entity_data.get("type")
                    description = llm_entity_data.get("description")
                    base_importance_score = llm_entity_data.get("base_importance_score")

                    canonical_type = self._normalize_entity_type(raw_type)
                    if canonical_type is None:
                        self.llm_log.emit({"type": "warning", "message": f"Skipping entity with unrecognized type '{raw_type}'.", "entity_data": llm_entity_data})
                        continue
                    type_ = canonical_type

                    if not name:
                        self.llm_log.emit({"type": "warning", "message": f"Skipping invalid entity from LLM (missing name).", "entity_data": llm_entity_data})
                        continue
                    
                    # Ensure base_importance_score is an int within range
                    if not isinstance(base_importance_score, int) or not (1 <= base_importance_score <= 10):
                        self.llm_log.emit({"type": "warning", "message": f"Invalid base_importance_score for '{name}'. Defaulting to 1.", "entity_data": llm_entity_data})
                        base_importance_score = 1

                    # Use the normalized LLM-provided name and canonical type as the key for lookup
                    normalized_key_for_llm_entity = (self._normalize_for_comparison(name), type_)
                    existing_entity = temp_entities_dict.get(normalized_key_for_llm_entity)

                    if existing_entity:
                        # Update existing entity
                        existing_entity["description"] = description if description else existing_entity["description"]
                        
                        # Prioritize update if new name is more complete or has better casing
                        # Only update if the normalized names are the same (already guaranteed by normalized_key)
                        current_normalized_name = self._normalize_for_comparison(existing_entity["name"])
                        new_normalized_name = self._normalize_for_comparison(name)

                        if current_normalized_name == new_normalized_name:
                            # Heuristic: Prefer longer name (more complete) or better casing
                            if len(name) > len(existing_entity["name"]) or \
                               (name and name[0].isupper() and not (existing_entity["name"] and existing_entity["name"][0].isupper())):
                                existing_entity["name"] = name # Update to the better raw name
                        
                        existing_entity["base_importance_score"] = base_importance_score
                        # mention_count is NOT incremented here; it's done by _update_importance_from_transcript
                    else:
                        # Add new entity
                        new_entity = {
                            "name": name,
                            "type": type_, # Use canonical type
                            "description": description if description else "",
                            "base_importance_score": base_importance_score,
                            "mention_count": 0, # Initialize to 0, will be updated by _update_importance_from_transcript later
                            "first_mentioned_idx": current_transcript_idx
                        }
                        temp_entities_dict[normalized_key_for_llm_entity] = new_entity
                
                self.entities = list(temp_entities_dict.values())
                # Now that the entities list is updated, trigger mention count update for the *current* transcript.
                # This ensures any newly identified entities in this round get their first mention counted,
                # and existing entities also get their count incremented if mentioned.
                self._update_importance_from_transcript(transcript_to_process)
                
                self.entities_updated.emit(self.entities)
                self.last_transcript_processed_idx = current_transcript_idx # Mark this index as processed

            time.sleep(2) # Wait a bit before checking for next transcript

    def add_transcription(self, text):
        # Transcriptions are appended here, but processed sequentially in run()
        self.transcriptions.append(text) 

    def get_transcriptions(self):
        return self.transcriptions

    def get_entities(self):
        return self.entities

    def stop(self):
        self.running = False