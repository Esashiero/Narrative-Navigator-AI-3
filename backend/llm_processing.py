import json
import re
import threading
import time
import ollama
import Levenshtein # Used for robust string similarity comparison
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
                    },
                    "aliases": { # NEW: Optional list of aliases
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "default": []
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
            # Attempt a simple chat to ensure Ollama is running and model exists
            ollama.chat(model=self.model, messages=[{"role": "user", "content": "hi"}], stream=False)
            self.llm_log.emit({"type": "status", "message": f"Successfully connected to Ollama model '{self.model}'."})
        except Exception as e:
            self.llm_log.emit({"type": "error", "message": f"Failed to connect to Ollama or model '{self.model}' not found: {e}"})
            self.model = None

        self.transcriptions = []
        self.entities = [] # This will hold the canonical entities (with combined info)
        self.dynamic_alias_map = {} # Maps alias (normalized string) -> canonical name (actual string from entities list)
        self.running = False
        self.external_context = ""
        self.content_title = "Unknown Content"

        self.base_system_prompt_template = base_system_prompt
        
        # Initial formatting of the system prompt
        self.system_prompt = self.base_system_prompt_template.format(
            content_title=self.content_title,
            external_context="No external context loaded yet. Please wait for the application to gather information."
        )
        self.last_transcript_processed_idx = -1
        
        self.VALID_ENTITY_TYPES = ["Characters", "Locations", "Organizations", "Key Objects", "Concepts/Events"]

    def set_content_title(self, title):
        self.content_title = title
        self._update_system_prompt()

    def set_external_context(self, context):
        self.external_context = context
        self._update_system_prompt()

    def _update_system_prompt(self):
        self.system_prompt = self.base_system_prompt_template.format(
            content_title=self.content_title,
            external_context=self.external_context
        )
    
    def _normalize_entity_type(self, type_str):
        """Normalizes LLM output type strings to our canonical types."""
        if not type_str:
            return None
        type_str_lower = type_str.lower().strip()
        
        # Handle compound types by splitting and taking the first part as a primary hint
        if ',' in type_str_lower:
            type_str_lower = type_str_lower.split(',')[0].strip()
        
        if type_str_lower == "concept/event":
            return "Concepts/Events"
        if type_str_lower == "locations/organizations":
            return "Locations" 
        if type_str_lower == "locations/concepts/events":
            return "Locations" 
        
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

    def _normalize_for_comparison(self, name, entity_type=None):
        """
        Normalizes entity names for robust comparison and deduplication.
        It uses a dynamic alias map, then string similarity, and finally a fallback cleaning.
        `entity_type` (canonical type) is crucial for disambiguation.
        Returns the canonical name (actual string) or a cleaned string if no match found.
        """
        if not name:
            return ""

        # Step 1: Basic cleaning for lookup in alias map and similarity comparison
        # Remove parenthesized content for a cleaner base name
        cleaned_base_name = re.sub(r'\s*\([^)]*\)', '', name).strip()
        name_lower_stripped = re.sub(r'[^a-z0-9\s]', '', cleaned_base_name.lower()).strip()
        name_lower_stripped = re.sub(r'\s+', ' ', name_lower_stripped).strip()

        # Step 2: Check dynamic alias map for direct lookup
        if name_lower_stripped in self.dynamic_alias_map:
            # We found a canonical name, return it
            canonical_name = self.dynamic_alias_map[name_lower_stripped]
            self.llm_log.emit({"type": "debug", "message": f"Alias map hit: '{name}' (cleaned: '{name_lower_stripped}') mapped to '{canonical_name}'."})
            return canonical_name

        # Step 3: If not in alias map, try to find a similar existing canonical entity
        # This acts as a fallback for aliases LLM might miss, or minor transcription errors.
        best_match_canonical_name = None
        highest_similarity = 0.88 # Threshold for considering a strong match (0.0 to 1.0)

        for existing_entity in self.entities:
            # Ensure entity type is comparable or ignored if not provided
            if entity_type is not None and self._normalize_entity_type(existing_entity["type"]) != self._normalize_entity_type(entity_type):
                continue # Skip if types are known and don't match

            existing_canonical_name_cleaned = re.sub(r'[^a-z0-9\s]', '', existing_entity["name"].lower()).strip()
            existing_canonical_name_cleaned = re.sub(r'\s+', ' ', existing_canonical_name_cleaned).strip()
            
            # Use Levenshtein distance ratio for similarity
            similarity = Levenshtein.ratio(name_lower_stripped, existing_canonical_name_cleaned)
            
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match_canonical_name = existing_entity["name"] # Keep the actual canonical name from our entity list

        if best_match_canonical_name:
            # If a strong similarity match is found, add the current name as an alias
            # to the best_match_canonical_name for future faster lookups.
            self.dynamic_alias_map[name_lower_stripped] = best_match_canonical_name
            self.llm_log.emit({"type": "debug", "message": f"Similarity match: '{name}' (cleaned: '{name_lower_stripped}') strongly similar to '{best_match_canonical_name}' (similarity: {highest_similarity:.2f}). Added to alias map."})
            return best_match_canonical_name
        
        # Step 4: No direct alias or strong similarity match found.
        # This is a new potential canonical entity name. Apply more aggressive cleaning.
        # This cleaning is done *after* alias/similarity check to preserve LLM's raw name if it's the canonical one.
        final_canonical_candidate = cleaned_base_name # Start with the name without parenthesized text

        # Remove common articles and possessive endings for robust canonicalization
        for article in ['the ', 'a ', 'an ']:
            if final_canonical_candidate.lower().startswith(article):
                final_canonical_candidate = final_canonical_candidate[len(article):].strip()
        final_canonical_candidate = re.sub(r"'s?\b", '', final_canonical_candidate, flags=re.IGNORECASE).strip() # remove 's or s' at end of word
        final_canonical_candidate = re.sub(r'[^a-zA-Z0-9\s]', '', final_canonical_candidate).strip() # remove non-alphanumeric (keep spaces)
        final_canonical_candidate = re.sub(r'\s+', ' ', final_canonical_candidate).strip() # reduce multiple spaces

        if not final_canonical_candidate: # If cleaning resulted in empty string, use original
            final_canonical_candidate = name.strip()
            
        # Add this new canonical candidate to the alias map, pointing to itself
        self.dynamic_alias_map[final_canonical_candidate.lower()] = final_canonical_candidate
        self.llm_log.emit({"type": "debug", "message": f"New canonical candidate: '{name}' mapped to cleaned '{final_canonical_candidate}'. Added to alias map."})
        return final_canonical_candidate

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

    # _is_similar_entity is removed as its logic is now primarily handled by _normalize_for_comparison
    # during entity reconciliation for deduplication.

    def _update_importance_from_transcript(self, transcript_text):
        """
        Increments mention_count for existing entities found in the new transcript.
        Uses canonical names for matching to handle variations/typos.
        """
        normalized_transcript_for_check = self._normalize_for_mention_check(transcript_text)
        
        for entity in self.entities:
            # Get the canonical name of the *existing* entity for matching
            # Ensure it's in the form used for mention checking.
            entity_canonical_name_for_check = self._normalize_for_mention_check(entity["name"]) 
            
            # Prepare a regex pattern for the canonical name, ensuring whole word match
            # re.escape is important if entity name contains special regex characters
            pattern = r'\b' + re.escape(entity_canonical_name_for_check) + r'\b'
            
            if re.search(pattern, normalized_transcript_for_check):
                entity["mention_count"] += 1
                self.llm_log.emit({"type": "debug", "message": f"Entity '{entity['name']}' (canonical: '{entity_canonical_name_for_check}') ({entity['type']}) mention_count incremented to {entity['mention_count']}"})

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
                transcript_to_process = self.transcriptions[self.last_transcript_processed_idx + 1] 
                current_transcript_idx = self.last_transcript_processed_idx + 1 
                
                recent_transcripts = self.transcriptions[max(0, current_transcript_idx - 4):current_transcript_idx + 1]
                
                current_entities_for_llm_prompt = [
                    {"name": e["name"], "type": e["type"], "description": e["description"]}
                    for e in self.entities
                ]

                # Combine current and recent transcripts for a robust mention check later
                all_relevant_transcript_text = "\n".join(recent_transcripts)
                normalized_all_relevant_transcript_text = self._normalize_for_mention_check(all_relevant_transcript_text)


                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content":
                     f"Current transcript snippet: {transcript_to_process}\n"
                     f"Recent context (last {len(recent_transcripts)} snippets): {recent_transcripts}\n"
                     f"Current narrative cheat sheet: {json.dumps(current_entities_for_llm_prompt, indent=2)}\n"
                     f"Based on ALL information (current transcript, recent context, full cheat sheet), identify *all identifiable* entities. For EVERY entity you return (new or existing), provide its 'base_importance_score' (1-10) re-evaluated based on its inherent narrative relevance. Also, include an 'aliases' array for any recognized alternative names, nicknames, or common variations. Remember to include historical dates/years and specific organizations if mentioned. Ensure to correctly categorize and canonicalize names."}
                ]
                self.llm_log.emit({"type": "prompt", "message": f"Prompt for transcript index {current_transcript_idx}", "data": messages})
                
                llm_identified_entities = []
                raw_content_from_llm = ""
                try:
                    response = ollama.chat(
                        model=self.model,
                        messages=messages,
                        stream=False,
                        format=entity_list_schema, 
                        options={
                            "temperature": 0.1, 
                            "top_p": 0.9,       
                            "top_k": 40,        
                            "repeat_penalty": 1.0 
                        }
                    )
                    raw_content_from_llm = response['message']['content']
                    self.llm_log.emit({"type": "raw_response", "message": f"Raw LLM response for transcript index {current_transcript_idx}", "data": raw_content_from_llm})

                    # --- Robust JSON Parsing ---
                    parsed_llm_output = None
                    json_string_to_parse = ""

                    try:
                        json_string_to_parse = raw_content_from_llm.strip()
                        parsed_llm_output = json.loads(json_string_to_parse)
                    except json.JSONDecodeError:
                        json_match = re.search(r"```json\s*(.*?)\s*```", raw_content_from_llm, re.DOTALL)
                        if json_match:
                            json_string_to_parse = json_match.group(1).strip()
                            parsed_llm_output = json.loads(json_string_to_parse)
                        else:
                            json_match_loose = re.search(r"\{.*\}", raw_content_from_llm, re.DOTALL)
                            if json_match_loose:
                                json_string_to_parse = json_match_loose.group(0).strip()
                                parsed_llm_output = json.loads(json_string_to_parse)
                            else:
                                json_match_array_loose = re.search(r"\[.*\]", raw_content_from_llm, re.DOTALL)
                                if json_match_array_loose:
                                    json_string_to_parse = json_match_array_loose.group(0).strip()
                                    parsed_llm_output = json.loads(json_string_to_parse)
                                else:
                                    raise json.JSONDecodeError("No recognizable JSON structure found.", raw_content_from_llm, 0)

                    if parsed_llm_output is None:
                        self.llm_log.emit({"type": "warning", "message": f"JSON parsing attempts found no valid structure.\nAttempted to parse:\n{raw_content_from_llm}", "data": raw_content_from_llm})
                        llm_identified_entities = [] 
                    elif isinstance(parsed_llm_output, list):
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
                    llm_identified_entities = []
                except Exception as e:
                    self.llm_log.emit({"type": "error", "message": f"LLM request failed: {str(e)}"})
                    llm_identified_entities = [] 

                # --- Entity Reconciliation and Update ---
                temp_entities_dict = {}
                for e in self.entities:
                    # Key by canonical name and canonical type to ensure uniqueness
                    normalized_name_key = self._normalize_for_comparison(e["name"], e["type"])
                    canonical_type = self._normalize_entity_type(e["type"]) # Ensure type is canonical too
                    temp_entities_dict[(normalized_name_key, canonical_type)] = e

                filtered_llm_identified_entities = []
                for llm_entity_data in llm_identified_entities:
                    name = llm_entity_data.get("name")
                    raw_type = llm_entity_data.get("type")
                    
                    if not name:
                        self.llm_log.emit({"type": "warning", "message": f"Skipping invalid entity from LLM (missing name).", "entity_data": llm_entity_data})
                        continue

                    canonical_type = self._normalize_entity_type(raw_type)
                    if canonical_type is None:
                        self.llm_log.emit({"type": "warning", "message": f"Skipping entity with unrecognized type '{raw_type}'.", "entity_data": llm_entity_data})
                        continue
                    
                    # New strict mention filter:
                    # Check if the LLM-provided name (or its normalized form) is in the transcript text
                    # We check both the raw name and the more aggressively normalized name
                    mention_found = False
                    
                    # Check raw LLM name directly (case-insensitive, basic cleaning)
                    # This targets the actual string mentioned by the LLM
                    normalized_llm_name = self._normalize_for_mention_check(name)
                    if normalized_llm_name and normalized_llm_name in normalized_all_relevant_transcript_text:
                        mention_found = True
                    else:
                        # Also check any aliases provided by the LLM for mention
                        for alias in llm_entity_data.get("aliases", []):
                            normalized_alias = self._normalize_for_mention_check(alias)
                            if normalized_alias and normalized_alias in normalized_all_relevant_transcript_text:
                                mention_found = True
                                break # Found a mention via an alias, no need to check further aliases for THIS entity
                            
                    if not mention_found:
                        self.llm_log.emit({"type": "warning", "message": f"LLM proposed entity '{name}' ({raw_type}) not found explicitly in transcript or its aliases. Skipping.", "entity_data": llm_entity_data})
                        continue # Skip this entity if not actually mentioned
                    
                    filtered_llm_identified_entities.append(llm_entity_data)

                # Process only the entities that were actually mentioned in the transcript
                for llm_entity_data in filtered_llm_identified_entities:
                    name = llm_entity_data.get("name")
                    raw_type = llm_entity_data.get("type")
                    description = llm_entity_data.get("description")
                    base_importance_score = llm_entity_data.get("base_importance_score")
                    aliases = llm_entity_data.get("aliases", []) 

                    canonical_type = self._normalize_entity_type(raw_type) # Already checked above, but keep for clarity
                    
                    if not isinstance(base_importance_score, int) or not (1 <= base_importance_score <= 10):
                        self.llm_log.emit({"type": "warning", "message": f"Invalid base_importance_score for '{name}'. Defaulting to 1.", "entity_data": llm_entity_data})
                        base_importance_score = 1

                    # Get the canonical name for the LLM-provided name
                    # IMPORTANT: Use the _normalize_for_comparison method here for the LLM's primary name
                    llm_provided_canonical_name = self._normalize_for_comparison(name, canonical_type)
                    
                    # Use this canonical name and type as the key for lookup in our temp dict
                    entity_key = (llm_provided_canonical_name, canonical_type)
                    existing_entity = temp_entities_dict.get(entity_key)

                    if existing_entity:
                        # Update existing entity
                        # Prioritize update if new name is more complete or has better casing
                        # Only update if the normalized names are the same (already guaranteed by entity_key)
                        current_normalized_name = self._normalize_for_comparison(existing_entity["name"], existing_entity["type"])
                        new_normalized_name = self._normalize_for_comparison(name, canonical_type)

                        if current_normalized_name == new_normalized_name:
                            # Heuristic: Prefer longer name (more complete) or better casing
                            if len(name) > len(existing_entity["name"]) or \
                               (name and name[0].isupper() and not (existing_entity["name"] and existing_entity["name"][0].isupper())):
                                existing_entity["name"] = name # Update to the better raw name
                        
                        existing_entity["description"] = description if description else existing_entity["description"]
                        existing_entity["base_importance_score"] = max(existing_entity["base_importance_score"], base_importance_score) # Take max score

                        # Add all provided aliases to the dynamic alias map for the existing entity's canonical name
                        # Ensure the existing canonical name itself is mapped to its cleaned form
                        # This ensures the canonical name always maps to itself in its cleaned form
                        self.dynamic_alias_map[self._normalize_for_comparison(existing_entity["name"], existing_entity["type"]).lower()] = existing_entity["name"]
                        for alias_name in aliases:
                            alias_lower = self._normalize_for_comparison(alias_name, canonical_type).lower() # Normalize alias string as well
                            # Add alias to map IF it's not already pointing to a different canonical entity
                            # This prevents "Apple" (fruit) mapping to "Apple" (company) if both exist.
                            if alias_lower not in self.dynamic_alias_map or \
                               self.dynamic_alias_map[alias_lower] == existing_entity["name"]:
                                self.dynamic_alias_map[alias_lower] = existing_entity["name"]
                                self.llm_log.emit({"type": "debug", "message": f"Added alias '{alias_name}' (norm: '{alias_lower}') for existing entity '{existing_entity['name']}'"})

                    else:
                        # Add new entity
                        new_entity_canonical_name = self._normalize_for_comparison(name, canonical_type) # Already computed
                        new_entity = {
                            "name": new_entity_canonical_name, # Use the derived canonical name for the new entity
                            "type": canonical_type,
                            "description": description if description else "",
                            "base_importance_score": base_importance_score,
                            "mention_count": 0, # Initialize to 0, will be updated by _update_importance_from_transcript later (if also mentioned in current transcript)
                            "first_mentioned_idx": current_transcript_idx
                        }
                        temp_entities_dict[entity_key] = new_entity

                        # Add current name and all its aliases to the dynamic alias map
                        self.dynamic_alias_map[new_entity_canonical_name.lower()] = new_entity_canonical_name
                        for alias_name in aliases:
                            alias_lower = self._normalize_for_comparison(alias_name, canonical_type).lower() # Normalize alias string as well
                            if alias_lower not in self.dynamic_alias_map or \
                               self.dynamic_alias_map[alias_lower] == new_entity_canonical_name: # Check to avoid overwriting
                                self.dynamic_alias_map[alias_lower] = new_entity_canonical_name
                                self.llm_log.emit({"type": "debug", "message": f"Added alias '{alias_name}' (norm: '{alias_lower}') for new entity '{new_entity_canonical_name}'"})

                self.entities = list(temp_entities_dict.values())
                # Now that the entities list is updated, trigger mention count update for the *current* transcript.
                # This ensures any newly identified entities in this round get their first mention counted,
                # and existing entities also get their count incremented if mentioned.
                # The _update_importance_from_transcript method handles the mention_count, including setting 1 for first mention.
                self._update_importance_from_transcript(transcript_to_process)
                
                self.entities_updated.emit(self.entities)
                self.last_transcript_processed_idx = current_transcript_idx 

            time.sleep(2) 

    def add_transcription(self, text):
        self.transcriptions.append(text) 

    def get_transcriptions(self):
        return self.transcriptions

    def get_entities(self):
        return self.entities

    def get_alias_map(self):
        """Returns the current dynamic alias map."""
        return self.dynamic_alias_map

    def stop(self):
        self.running = False